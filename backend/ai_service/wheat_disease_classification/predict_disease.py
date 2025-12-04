import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
import sys
from typing import List
from disease_classification import ClassificationCNN

# --- 1. КОНФІГУРАЦІЯ ТА КЛАСИ (МАЮТЬ БУТИ ІДЕНТИЧНІ disease_classification.py) ---
# Класифікація хвороб пшениці
TRAIN_DIRS = [
    "Black Rust", "Blast", "Brown Rust", "Fusarium Head Blight",
    "Leaf Blight", "Mildew", "Septoria", "Smut", "Tan spot",
    "Yellow Rust", "Healthy"
]
CLASSES = TRAIN_DIRS
NUM_CLASSES = len(CLASSES)  # Загальна кількість класів: 11
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "cnn_disease_classification_best.pth.tar"


# --- 2. АРХІТЕКТУРА CNN (ПОВТОРЕННЯ З disease_classification.py) ---
#
# class ClassificationCNN(nn.Module):
#     """Проста згорткова мережа для класифікації хвороб пшениці."""
#
#     def __init__(self, num_classes):
#         super(ClassificationCNN, self).__init__()
#         # Виділення ознак
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         # Глобальне усереднююче пулінгування
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # Класифікатор
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 1 * 1, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = self.classifier(x)
#         return x


# --- 3. ФУНКЦІЇ ДОПОМОГИ ТА ТРАНСФОРМАЦІЇ ---

def load_and_transform_image(image_path: str) -> torch.Tensor:
    """Завантажує, трансформує та нормалізує зображення для інференсу."""
    print(f"Завантаження зображення: {image_path}")

    # 1. Завантаження (cv2 читає як BGR)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Не вдалося завантажити зображення. Перевірте шлях: {image_path}")

    # 2. Перетворення BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. Ресайз
    image_resized = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # 4. Нормалізація та перетворення на тензор (H, W, C) -> (C, H, W)
    image_tensor = TF.to_tensor(image_resized).float()

    # 5. Стандартна нормалізація, використана під час навчання
    mean = [0.5] * 3
    std = [0.5] * 3
    for i in range(3):
        image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]

    # 6. Додавання розмірності батчу (C, H, W) -> (1, C, H, W)
    return image_tensor.unsqueeze(0)


def predict_disease(model: nn.Module, image_tensor: torch.Tensor, classes: List[str]) -> str:
    """Виконує передбачення та повертає назву класу."""

    model.eval()  # Переведення моделі в режим оцінки

    with torch.no_grad():
        # Переміщення тензора на пристрій
        image_tensor = image_tensor.to(DEVICE)

        # Отримання логітів
        outputs = model(image_tensor)

        # Перетворення логітів на ймовірності за допомогою Softmax
        probabilities = torch.softmax(outputs, dim=1)

        # Визначення найімовірнішого класу
        confidence, predicted_class_idx = torch.max(probabilities, 1)

        # Отримання назви класу
        predicted_class = classes[predicted_class_idx.item()]
        confidence_score = confidence.item() * 100

        print(f"\n--- РЕЗУЛЬТАТ КЛАСИФІКАЦІЇ ---")
        print(f"Передбачений клас: {predicted_class}")
        print(f"Впевненість: {confidence_score:.2f}%")

        # Виведення всіх ймовірностей (для налагодження)
        print("\nУсі ймовірності:")
        for idx, (cls_name, prob) in enumerate(zip(classes, probabilities[0])):
            print(f"  {cls_name:<20}: {prob.item() * 100:.2f}%")

        return predicted_class


# --- 4. ГОЛОВНА ФУНКЦІЯ ІНФЕРЕНСУ ---

def main():
    if len(sys.argv) < 2:
        print("Використання: python predict_disease.py <шлях_до_зображення>")
        print("Приклад: python predict_disease.py D:\\datasets\\Wheat_desizes\\RGB\\septoria_test\\test_wheat.jpg")
        # python predict_disease.py "D:\datasets\Wheat\RGB\Wheat___Healthy\Healthy086.jpg"
        sys.exit(1)

    input_image_path = sys.argv[1]

    # 1. Ініціалізація та завантаження моделі
    model = ClassificationCNN(num_classes=NUM_CLASSES).to(DEVICE)

    try:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Файл контрольної точки '{CHECKPOINT_PATH}' не знайдено. "
                "Спочатку запустіть disease_classification.py для навчання."
            )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Модель успішно завантажена з {CHECKPOINT_PATH}")

    except Exception as e:
        print(f"Критична помилка завантаження моделі: {e}")
        sys.exit(1)

    # 2. Завантаження та підготовка зображення
    try:
        image_tensor = load_and_transform_image(input_image_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Критична помилка обробки зображення: {e}")
        sys.exit(1)

    # 3. Виконання передбачення
    predict_disease(model, image_tensor, CLASSES)


if __name__ == "__main__":
    main()