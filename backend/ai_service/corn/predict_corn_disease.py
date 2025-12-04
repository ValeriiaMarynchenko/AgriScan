import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from typing import List, Tuple
from corn_disease_classification import ClassificationCNN

# --- 1. КОНФІГУРАЦІЯ ТА КЛАСИ ---
CLASSES = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight"
]
NUM_CLASSES = len(CLASSES)  # 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "ai_service/corn/cnn_corn_classification_best.pth.tar"


# --- 2. АРХІТЕКТУРА CNN (ВИКОРИСТОВУЄТЬСЯ ДЛЯ ЗАВАНТАЖЕННЯ) ---

# class ClassificationCNN(nn.Module):
#     """Проста згорткова мережа для класифікації хвороб кукурудзи."""
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


# --- 3. ФУНКЦІЇ ПІДГОТОВКИ ЗОБРАЖЕННЯ ---

def load_and_transform_image(image_bytes: bytes) -> torch.Tensor:
    """Декодує байтове зображення, трансформує та нормалізує для інференсу."""

    # 1. Декодування байтів у масив NumPy (BGR)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Не вдалося декодувати зображення. Недійсний формат файлу.")

    # 2. Перетворення BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. Ресайз
    image_resized = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # 4. Нормалізація та перетворення на тензор
    image_tensor = TF.to_tensor(image_resized).float()

    # 5. Стандартна нормалізація
    mean = [0.5] * 3
    std = [0.5] * 3
    for i in range(3):
        image_tensor[i] = (image_tensor[i] - mean[i]) / std[i]

    # 6. Додавання розмірності батчу
    return image_tensor.unsqueeze(0)


# --- 4. ОСНОВНА ФУНКЦІЯ ПЕРЕДБАЧЕННЯ ---

def predict_corn(model: ClassificationCNN, image_bytes: bytes) -> Tuple[str, float]:
    """Виконує передбачення хвороби кукурудзи, приймаючи байтову послідовність зображення."""

    # Завантаження та підготовка зображення
    image_tensor = load_and_transform_image(image_bytes)

    # Переміщення тензора на пристрій
    image_tensor = image_tensor.to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

        predicted_class = CLASSES[predicted_class_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score