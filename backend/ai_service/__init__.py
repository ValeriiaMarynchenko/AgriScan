import torch
import torch.nn as nn
import os
import sys

# ІМПОРТИ ДЛЯ КУКУРУДЗИ
from corn.predict_corn_disease import ClassificationCNN as CornCNN, predict_corn, NUM_CLASSES as CORN_NUM_CLASSES
# ІМПОРТИ ДЛЯ КАРТОПЛІ
from potato.predict_potato_disease import ClassificationCNN as PotatoCNN, predict_potato, \
    NUM_CLASSES as POTATO_NUM_CLASSES
# ІМПОРТИ ДЛЯ БУР'ЯНУ
from weed.weed_interface import predict_weed_segmentation

# Припустимо, що UNet модель знаходиться тут (ПОТРІБНО ДОДАТИ КЛАС UNET)
# from .weed.weed_finder_model import UNet

# Конфігурація
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Шляхи до контрольних точок (оновлено шляхи відносно кореня backend)
CORN_CHECKPOINT = "ai_service/corn/cnn_corn_classification_best.pth.tar"
POTATO_CHECKPOINT = "ai_service/potato/cnn_potato_classification_best.pth.tar"
WEED_CHECKPOINT = "ai_service/weed/unet_weedmap_epoch_3.pth.tar"

# Глобальні змінні для моделей
CORN_MODEL = None
POTATO_MODEL = None
WEED_MODEL = None


def load_model(model_class: nn.Module, checkpoint_path: str, num_classes: int) -> nn.Module:
    """Завантажує модель та її ваги."""
    print(f"[{model_class.__name__}] Завантаження моделі з {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        print(f"ПОМИЛКА: Контрольна точка не знайдена: {checkpoint_path}")
        return None

    try:
        model = model_class(num_classes=num_classes).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        # Завантаження стану моделі
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        print(f"[{model_class.__name__}] Успішно завантажено на {DEVICE}.")
        return model

    except Exception as e:
        print(f"Критична помилка завантаження моделі {model_class.__name__}: {e}")
        return None


async def initialize_ai_services():
    """Ініціалізує всі AI моделі під час запуску FastAPI."""
    global CORN_MODEL, POTATO_MODEL, WEED_MODEL

    # Завантаження моделі Кукурудзи
    CORN_MODEL = load_model(CornCNN, CORN_CHECKPOINT, CORN_NUM_CLASSES)

    # Завантаження моделі Картоплі
    POTATO_MODEL = load_model(PotatoCNN, POTATO_CHECKPOINT, POTATO_NUM_CLASSES)

    # Завантаження моделі Бур'яну (ВИКОРИСТОВУЄТЬСЯ ЗАГЛУШКА)
    try:
        # Тут має бути завантаження UNet моделі.
        # Оскільки UNet клас не надано, використовуємо заглушку
        # Якщо клас UNet буде визначено, замініть рядок нижче на реальне завантаження:
        # WEED_MODEL = load_model(UNet, WEED_CHECKPOINT, num_classes=1)
        WEED_MODEL = "Stub_UNet_Ready"  # Placeholder для імітації готовності
        print(f"[UNet Model] Використовується заглушка. UNet готовий до роботи.")

    except Exception as e:
        print(f"[UNet Model] Помилка: {e}")
        WEED_MODEL = None

    if CORN_MODEL and POTATO_MODEL and WEED_MODEL:
        print("Усі AI-сервіси успішно ініціалізовано.")
    else:
        print("Помилка ініціалізації одного або кількох AI-сервісів.")