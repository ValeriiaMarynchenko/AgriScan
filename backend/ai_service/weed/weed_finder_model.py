import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image

# --- 1. КОНФІГУРАЦІЯ ---
# ЗАДАЧА: Бінарна сегментація бур'янів (Weed Segmentation)
# Кількість класів: 2 (0: Фон/Культура, 1: Бур'ян)
NUM_CLASSES = 2
# Вхідні канали: 4 (R, G, B, NIR)
IN_CHANNELS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 3
PIN_MEMORY = True
NUM_WORKERS = 2
LOAD_MODEL = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Кореневий каталог датасету weedMap_2018.
# !!! УВАГА: ЗМІНІТЬ ЦЕЙ ШЛЯХ на кореневий каталог, де знаходиться папка 'weedMap_2018' !!!
# Використовуйте прямі слеші '/' для кращої сумісності або подвійні '\\'
DATA_ROOT = "D:\\datasets"


# --- 2. UNET АРХІТЕКТУРА ---
class DoubleConv(nn.Module):
    """Блок подвійної згортки для UNET."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """Модель UNET з 4 вхідними каналами та NUM_CLASSES вихідними класами."""

    def __init__(
            self, in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse the list

        # Up path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # Обрізаємо або доповнюємо, щоб форми збігалися
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# --- 3. КЛАС ДАТАСЕТУ (АДАПТОВАНИЙ ДЛЯ WEEDMAP_2018) ---
class WeedMapDataset(Dataset):
    """
    Датасет, адаптований для завантаження багатоспектральних даних (RGB + NIR)
    та бінарних масок бур'янів зі структури weedMap_2018 (Tiles).

    ОЧІКУВАНА СТРУКТУРА ПАПОК (оновлено для використання папки 'mask'):
    {DATA_ROOT}/weedMap_2018/Tiles/{sensor}/[tile_folder]/
        ├── mask/         <- Бінарні маски (Ground Truth)
        └── tile/
            ├── RGB/      <- RGB зображення
            └── NIR/      <- NIR зображення

    Де [tile_folder] — це '000', '001', і т.д.
    """

    def __init__(self, root_dir=DATA_ROOT, sensor='RedEdge', transform=None):
        self.root_dir = os.path.join(root_dir, 'weedMap_2018', 'Tiles', sensor)
        self.transform = transform
        self.image_paths = []

        print(f"ОЧІКУВАНА КОРЕНЕВА ПАПКА ДАТАСЕТУ: {self.root_dir}")

        # Перевірка наявності кореневої папки
        if not os.path.exists(self.root_dir):
            print(f"Помилка: Коренева папка не знайдена: {self.root_dir}")
            return

        print(f"Сканування датасету {sensor}...")
        extensions = ('.png', '.jpg', '.tif', '.tiff')

        # Скануємо всі папки-плитки (000, 001, ...)
        for tile_folder in os.listdir(self.root_dir):
            tile_path = os.path.join(self.root_dir, tile_folder)

            if os.path.isdir(tile_path):
                # Формуємо шляхи до підпапок
                rgb_path = os.path.join(tile_path, 'tile', 'RGB')
                nir_path = os.path.join(tile_path, 'tile', 'NIR')
                # !!! ЗМІНЕНО: Використовуємо папку 'mask' замість 'groundtruth' !!!
                gt_path = os.path.join(tile_path, 'mask')

                # Перевіряємо, чи існують всі необхідні підпапки
                if os.path.exists(rgb_path) and os.path.exists(nir_path) and os.path.exists(gt_path):

                    try:
                        # Шукаємо файли за розширенням
                        # Ключ словника - це ID файлу без розширення (наприклад, 'frame0000')
                        rgb_files = {os.path.splitext(f)[0]: f for f in os.listdir(rgb_path) if
                                     f.lower().endswith(extensions)}
                        nir_files = {os.path.splitext(f)[0]: f for f in os.listdir(nir_path) if
                                     f.lower().endswith(extensions)}
                        gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_path) if
                                    f.lower().endswith(extensions)}
                    except Exception as e:
                        print(f"    - Помилка читання файлів у плитці '{tile_folder}': {e}")
                        continue

                    # Вибираємо лише ті ID, які мають всі три компоненти
                    common_ids = set(rgb_files.keys()) & set(nir_files.keys()) & set(gt_files.keys())

                    print(
                        f"  -> Плитка '{tile_folder}': RGB({len(rgb_files)}) | NIR({len(nir_files)}) | GT({len(gt_files)}) | Спільні({len(common_ids)})")

                    if len(common_ids) == 0:
                        print(
                            f"    - Попередження: У плитці '{tile_folder}' не знайдено спільних зображень RGB, NIR та GT. Перевірте, чи назви файлів збігаються (наприклад, 'frame0000')!")

                    for img_id in common_ids:
                        self.image_paths.append({
                            'rgb': os.path.join(rgb_path, rgb_files[img_id]),
                            'nir': os.path.join(nir_path, nir_files[img_id]),
                            'gt': os.path.join(gt_path, gt_files[img_id]),
                        })
                else:
                    # Повідомлення про відсутність очікуваних підпапок
                    missing_folders = []
                    # Зверніть увагу: якщо відсутня папка 'RGB' або 'NIR', переконайтеся, що вони присутні у вашій структурі!
                    if not os.path.exists(rgb_path): missing_folders.append(f"'{os.path.join('tile', 'RGB')}'")
                    if not os.path.exists(nir_path): missing_folders.append(f"'{os.path.join('tile', 'NIR')}'")
                    if not os.path.exists(gt_path): missing_folders.append(f"'mask'")

                    print(
                        f"  -> Пропуск плитки '{tile_folder}': Відсутні необхідні підпапки: {', '.join(missing_folders)}")

        print(f"\nЗнайдено {len(self.image_paths)} зразків у {sensor} для навчання.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        paths = self.image_paths[index]

        # 1. Завантаження RGB (3 канали). cv2.IMREAD_COLOR (1)
        rgb_img = cv2.imread(paths['rgb'], cv2.IMREAD_COLOR)
        if rgb_img is None: raise Exception(f"Помилка завантаження RGB: {paths['rgb']}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        # 2. Завантаження NIR (1 канал). cv2.IMREAD_GRAYSCALE (0)
        # Це має зчитати один канал, навіть якщо файл .png чи .tif
        nir_img = cv2.imread(paths['nir'], cv2.IMREAD_GRAYSCALE)
        if nir_img is None: raise Exception(f"Помилка завантаження NIR: {paths['nir']}")

        # 3. Завантаження Ground Truth (маска). cv2.IMREAD_UNCHANGED (-1)
        gt_mask = cv2.imread(paths['gt'], cv2.IMREAD_UNCHANGED)
        if gt_mask is None: raise Exception(f"Помилка завантаження GT: {paths['gt']}")

        # 4. Зміна розміру

        # Зміна розміру вхідних зображень (лінійна інтерполяція)
        rgb_img = cv2.resize(rgb_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        nir_img = cv2.resize(nir_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Зміна розміру маски (інтерполяція Nearest Neighbor - КРАЙНЄ ВАЖЛИВО для масок)
        target_mask = cv2.resize(gt_mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # 5. Об'єднання в 4-канальне зображення (H, W, 4)
        # Додаємо NIR канал до RGB (розширення розмірності NIR, якщо необхідно)
        if len(nir_img.shape) == 2:  # Якщо nir_img має форму (H, W), додаємо розмірність для dstack
            nir_img = np.expand_dims(nir_img, axis=-1)

        input_image = np.dstack((rgb_img, nir_img))

        # 6. Обробка бінарної маски (перетворення на класи 0 або 1)
        # Будь-яке значення маски > 0 позначається як клас 1 (Бур'ян), інакше клас 0 (Фон/Культура)
        # Маска з папки 'mask' має бути бінарною або сірою, де бур'ян має високе значення.
        target_mask = (target_mask > 0).astype(np.uint8)

        # 7. Трансформації та нормалізація

        # Перетворення numpy (H, W, C) в tensor (C, H, W)
        image_tensor = TF.to_tensor(input_image).float()

        # Нормалізація (припускаємо загальні середні та STD для 4 каналів [0.5, 0.5, 0.5, 0.5])
        mean = [0.5] * IN_CHANNELS
        std = [0.5] * IN_CHANNELS

        for i in range(IN_CHANNELS):
            image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]

        # Маска міток повинна бути типу Long для CrossEntropyLoss
        mask_tensor = torch.from_numpy(target_mask).long()  # (H, W)

        return image_tensor, mask_tensor


# --- 4. ДОПОМІЖНІ ФУНКЦІЇ (без змін) ---

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Зберігає стан моделі та оптимізатора."""
    print("=> Збереження контрольної точки")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """Завантажує контрольні точки моделі."""
    print("=> Завантаження контрольної точки")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device):
    """Перевіряє точність сегментації (заглушка)."""
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # torch.argmax(dim=1) для вибору класу з logits
            preds = torch.argmax(model(x), dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    accuracy = (num_correct / num_pixels) * 100
    print(f"Отримано {num_correct}/{num_pixels} правильних пікселів (Точність: {accuracy:.2f}%)")
    model.train()
    return accuracy


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Функція для одного етапу навчання."""
    loop = tqdm(loader)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)  # 4 канали (N, 4, H, W)
        targets = targets.to(DEVICE)  # 1 канал (N, H, W)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # CrossEntropyLoss: predictions (N, C, H, W), targets (N, H, W)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    # Навчання UNET для бінарної сегментації бур'янів (класи 0 і 1)

    # Використовуємо лише RedEdge (за замовчуванням)
    train_dataset = WeedMapDataset(root_dir=DATA_ROOT, sensor='RedEdge')

    # Перевірка, чи знайдено дані
    if len(train_dataset) == 0:
        print(
            "Неможливо розпочати: Не знайдено жодного зразка даних. Перевірте DATA_ROOT та структуру папок WeedMapDataset.")
        return

    # Розділення на навчальний та валідаційний набори (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Ініціалізація моделі: 4 вхідні канали, 2 вихідні класи
    model = UNET(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(DEVICE)
    # CrossEntropyLoss ідеально підходить для багатокласової (і бінарної) сегментації
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    # Головний цикл навчання
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Епоха {epoch + 1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Перевірка точності на валідаційному наборі
        accuracy = check_accuracy(val_loader, model, DEVICE)
        print(f"Валідаційна точність: {accuracy:.2f}%")

        # Зберігаємо контрольну точку
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # Використовуємо окрему назву, щоб не перезаписувати файл на кожній епосі
        save_checkpoint(checkpoint, filename=f"unet_weedmap_epoch_{epoch + 1}.pth.tar")

    print("\nНавчання завершено.")


if __name__ == "__main__":
    main()