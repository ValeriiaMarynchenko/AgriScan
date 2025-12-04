import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import sys

# --- 1. КОНФІГУРАЦІЯ ПРОЕКТУ КАРТОПЛІ ---
# Класи хвороб картоплі, визначені за назвами ваших папок
TRAIN_DIRS = [
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight"
]
CLASSES = TRAIN_DIRS
NUM_CLASSES = len(CLASSES)  # Загальна кількість класів: 3

# РОЗМІРИ ЗОБРАЖЕННЯ для навчання
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# ШЛЯХИ ДО ДАНИХ ТА ГІПЕРПАРАМЕТРИ
# КОРЕНЕВА ДИРЕКТОРІЯ ДЛЯ КЛАСІВ КАРТОПЛІ
DATA_ROOT = "D:\\datasets\\corn_potato_wheat\\Potato"

# Гіперпараметри навчання
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. АРХІТЕКТУРА CNN (ЗБЕРЕЖЕНО АРХІТЕКТУРУ) ---

class ClassificationCNN(nn.Module):
    """Проста згорткова мережа для класифікації хвороб картоплі."""

    def __init__(self, num_classes):
        super(ClassificationCNN, self).__init__()
        # Виділення ознак
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Глобальне усереднююче пулінгування
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Класифікатор
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # num_classes = 3 для картоплі
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# --- 3. ТРАНСФОРМАЦІЇ І ДАТАСЕТ (АДАПТОВАНО) ---

def get_transforms(image):
    """Базові трансформації для класифікації."""
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image_tensor = TF.to_tensor(image).float()

    mean = [0.5] * 3
    std = [0.5] * 3
    for i in range(3):
        image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]

    return image_tensor


class PotatoDiseaseDataset(Dataset):
    """Клас для завантаження RGB зображень картоплі для класифікації."""

    def __init__(self, rgb_dir, class_dirs, transform=get_transforms):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}

        print("Сканування директорій...")
        for disease_name in class_dirs:
            rgb_path = os.path.join(rgb_dir, disease_name)
            class_id = self.class_to_idx[disease_name]

            # Пошук усіх PNG та JPG файлів у папках
            file_names = glob.glob(os.path.join(rgb_path, "*.png")) + glob.glob(os.path.join(rgb_path, "*.jpg"))

            for file_name in file_names:
                self.images.append(file_name)
                self.labels.append(class_id)

        print(f"Завантажено {len(self.images)} зображень для класифікації картоплі.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_path = self.images[index]
        class_id = self.labels[index]

        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

        if image_bgr is None:
            # Обробка помилки
            print(f"Помилка завантаження файлу: {rgb_path}. Пропускаємо.")
            return self.__getitem__((index - 1) % len(self))

            # Перетворення BGR -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Застосування трансформацій
        image_tensor = self.transform(image_rgb)

        # Створення цільового тензора (ID класу)
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return image_tensor, label_tensor


# --- 4. ДОПОМІЖНІ ФУНКЦІЇ ---

def save_checkpoint(state, filename="cnn_potato_classification_best.pth.tar"):
    """Зберігає контрольну точку моделі."""
    print("=> Збереження контрольної точки")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, device):
    """Завантажує контрольні точки моделі."""
    if not os.path.exists(checkpoint_path):
        print(f"Помилка: Файл контрольної точки не знайдено за шляхом: {checkpoint_path}. Пропускаємо завантаження.")
        return

    print(f"=> Завантаження контрольної точки з {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as e:
        print(f"Помилка завантаження стану моделі: {e}. Перевірте архітектуру.")
    model.eval()


# --- 5. ЦИКЛ НАВЧАННЯ (НЕ ЗМІНЕНО) ---

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """Виконує один цикл навчання."""
    loop = tqdm(loader, desc="Training Potato")
    total_loss = 0

    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# --- 6. ОСНОВНА ФУНКЦІЯ ---

def main():
    print(f"Використовується пристрій: {DEVICE}")
    print(f"Завдання: Класифікація Картоплі ({NUM_CLASSES} класів)")

    # 1. Ініціалізація Даталоадера
    train_dataset = PotatoDiseaseDataset(
        rgb_dir=DATA_ROOT,
        class_dirs=TRAIN_DIRS,
        transform=get_transforms
    )

    if len(train_dataset) == 0:
        print("\nПОМИЛКА: Датасет порожній (0 зображень).")
        print("==> Перевірте шлях та структуру папок:")
        print(f"  - Коренева директорія: {DATA_ROOT}")
        print(f"  - Очікувані підпапки: {TRAIN_DIRS}")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
    )

    # 2. Ініціалізація Моделі, Оптимізатора та Функції Втрат
    model = ClassificationCNN(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # 3. Запуск Циклу Навчання
    best_loss = float('inf')
    CHECKPOINT_PATH = "cnn_potato_classification_best.pth.tar"

    print("Початок навчання...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nЕпоха {epoch + 1}/{NUM_EPOCHS}")

        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=CHECKPOINT_PATH)

    print(f"\nНавчання завершено. Найкращі втрати: {best_loss:.4f}")

    pass


if __name__ == "__main__":
    main()