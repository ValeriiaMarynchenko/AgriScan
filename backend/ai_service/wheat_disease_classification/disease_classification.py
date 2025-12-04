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
import sys  # Додаємо sys для коректного виходу

# --- 1. КОНФІГУРАЦІЯ ПРОЕКТУ ---
# Класифікація хвороб пшениці
TRAIN_DIRS = [
    "Black Rust", "Blast", "Brown Rust", "Fusarium Head Blight",
    "Leaf Blight", "Mildew", "Septoria", "Smut", "Tan spot",
    "Yellow Rust", "Healthy"  # Порядок у цьому списку визначає індекс класу (0 до 10)
]
CLASSES = TRAIN_DIRS
NUM_CLASSES = len(CLASSES)  # Загальна кількість класів: 11

# РОЗМІРИ ЗОБРАЖЕННЯ для навчання
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# ШЛЯХИ ДО ДАНИХ ТА ГІПЕРПАРАМЕТРИ
# MASK_ROOT та COLOR_MAP більше не потрібні для класифікації
DATA_ROOT = "D:\\datasets\\Wheat_desizes\\RGB"

# Гіперпараметри навчання
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. АРХІТЕКТУРА CNN (ЗМІНЕНО НА КЛАСИФІКАЦІЮ) ---

class ClassificationCNN(nn.Module):
    """Проста згорткова мережа для класифікації хвороб пшениці."""

    def __init__(self, num_classes):
        super(ClassificationCNN, self).__init__()
        # Виділення ознак: Згорткові шари для зменшення розмірності та вилучення ознак
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Вихід: 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Вихід: 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Вихід: 32x32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Вихід: 16x16
        )
        # Глобальне усереднююче пулінгування для перетворення на вектор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Класифікатор: Повнозв'язні шари
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),  # 256 - кількість каналів після feature, 1x1 - розмір
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Регуляризація
            nn.Linear(512, num_classes)  # Фінальний шар для 11 класів
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# --- 3. ТРАНСФОРМАЦІЇ І ДАТАСЕТ ---

def get_transforms(image):
    """
    Базові трансформації для класифікації: ресайз, нормалізація RGB та перетворення на тензор.
    """
    # 1. Ресайз (cv2.resize приймає W, H)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # 2. Нормалізація RGB та перетворення на тензор
    image_tensor = TF.to_tensor(image).float()

    # Використовуємо нормалізацію [0.5, 0.5, 0.5]
    mean = [0.5] * 3
    std = [0.5] * 3
    for i in range(3):
        image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]

    return image_tensor


class WheatDiseaseDataset(Dataset):
    """Клас для завантаження RGB зображень для класифікації."""

    def __init__(self, rgb_dir, class_dirs, transform=get_transforms):
        self.transform = transform
        self.images = []
        self.labels = []  # Змінено: тепер зберігає ID класу
        self.class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}

        for disease_name in class_dirs:
            rgb_path = os.path.join(rgb_dir, disease_name)
            class_id = self.class_to_idx[disease_name]

            # Пошук усіх PNG та JPG файлів у папках
            file_names = glob.glob(os.path.join(rgb_path, "*.png")) + glob.glob(os.path.join(rgb_path, "*.jpg"))

            for file_name in file_names:
                self.images.append(file_name)
                self.labels.append(class_id)  # Зберігаємо ID класу

        print(f"Завантажено {len(self.images)} зображень для класифікації.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 1. Завантаження даних
        rgb_path = self.images[index]
        class_id = self.labels[index]

        # Завантажуємо зображення (BGR)
        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

        if image_bgr is None:
            # Обробка помилки: якщо файл пошкоджений або відсутній
            print(f"Помилка завантаження файлу: {rgb_path}. Ігноруємо та пропускаємо.")
            # Повертаємо останній елемент, щоб уникнути IndexError
            # NOTE: Це просте рішення для уникнення збою. У реальному коді краще
            # використовувати try/except або виключити пошкоджений файл під час ініціалізації.
            return self.__getitem__((index - 1) % len(self))

            # 2. Перетворення зображення BGR -> RGB (для PyTorch)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 3. Застосування трансформацій
        image_tensor = self.transform(image_rgb)

        # 4. Створення цільового тензора (ID класу)
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return image_tensor, label_tensor


# --- 4. ДОПОМІЖНІ ФУНКЦІЇ ---

def save_checkpoint(state, filename="my_checkpoint_classification.pth.tar"):
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as e:
        print(f"Помилка завантаження стану моделі: {e}. Перевірте архітектуру.")
    model.eval()


# --- 5. ЦИКЛ НАВЧАННЯ ---

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """Виконує один цикл навчання."""
    loop = tqdm(loader, desc="Training")
    total_loss = 0

    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass з автоматичним змішуванням точності (AMP)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # CrossEntropyLoss вимагає: predictions (N, C), targets (N) LongTensor
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Оновлення tqdm
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# --- 6. ОСНОВНА ФУНКЦІЯ ---

def main():
    print(f"Використовується пристрій: {DEVICE}")
    print(f"Завдання: Класифікація ({NUM_CLASSES} класів)")

    # 1. Ініціалізація Даталоадера
    # Зверніть увагу, що MASK_ROOT більше не використовується!
    train_dataset = WheatDiseaseDataset(
        rgb_dir=DATA_ROOT,
        class_dirs=TRAIN_DIRS,
        transform=get_transforms
    )

    # Перевірка на порожній датасет
    if len(train_dataset) == 0:
        print("\nПОМИЛКА: Датасет порожній (0 зображень).")
        print("Це спричиняє помилку 'ValueError: num_samples=0' у DataLoader.")
        print("==> Перевірте шлях та структуру папок:")
        print(f"  - RGB (Зображення): {DATA_ROOT}")
        print(f"  - Очікувані підпапки: {TRAIN_DIRS}")
        print("Переконайтеся, що в кожній підпапці є файли (.png або .jpg).")
        sys.exit(1)  # Зупиняємо виконання

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
    )

    # 2. Ініціалізація Моделі, Оптимізатора та Функції Втрат
    model = ClassificationCNN(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()  # Стандартна втрата для класифікації
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()  # Для Automatic Mixed Precision

    # 3. Запуск Циклу Навчання
    best_loss = float('inf')
    CHECKPOINT_PATH = "cnn_disease_classification_best.pth.tar"

    print("Початок навчання...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nЕпоха {epoch + 1}/{NUM_EPOCHS}")

        # Навчання
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)

        # Збереження найкращої моделі
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