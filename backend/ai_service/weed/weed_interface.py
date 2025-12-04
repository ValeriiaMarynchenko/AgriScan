import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import os
from typing import Tuple, List
from weed_finder_model import UNET

# --- КОНСТАНТИ, ЩО ВІДПОВІДАЮТЬ МОДЕЛІ ---
IN_CHANNELS = 4
NUM_CLASSES = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "unet_weedmap_epoch_3.pth.tar"  # Приклад назви контрольної точки
# Стандартна нормалізація [0.5, 0.5, 0.5, 0.5]
MEAN = [0.5] * IN_CHANNELS
STD = [0.5] * IN_CHANNELS


def load_and_transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Декодує байтове зображення, трансформує та нормалізує для інференсу UNET.

    ВАЖЛИВО: Оскільки модель очікує 4 канали (RGB + NIR), але на вхід
    подаються лише RGB байти (3 канали), ми імітуємо 4-й канал (NIR)
    дублюванням Червоного каналу (це дуже спрощена імітація).
    """

    # 1. Декодування байтів у масив NumPy (BGR)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Не вдалося декодувати зображення. Недійсний формат файлу.")

    # 2. Перетворення BGR -> RGB та ресайз
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3)
    image_resized = cv2.resize(image_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # 3. Створення 4-канального входу (RGB + NIR імітація)
    # Копіюємо Червоний канал (індекс 0) як імітацію NIR.
    R_channel = image_resized[:, :, 0]

    # Додаємо NIR канал як 4-й
    # dstack об'єднує (H, W, 3) та (H, W) -> (H, W, 4)
    input_image_4ch = np.dstack((image_resized, R_channel))

    # 4. Нормалізація та перетворення на тензор (H, W, C) -> (C, H, W)
    image_tensor = TF.to_tensor(input_image_4ch).float()

    # 5. Стандартна нормалізація для 4 каналів
    for i in range(IN_CHANNELS):
        image_tensor[i] = (image_tensor[i] - MEAN[i]) / STD[i]

    # 6. Додавання розмірності батчу (C, H, W) -> (1, C, H, W)
    return image_tensor.unsqueeze(0)


def create_mask_overlay(original_image_bytes: bytes, mask_logits: torch.Tensor) -> bytes:
    """
    Створює накладене зображення: оригінал + напівпрозора кольорова маска бур'янів.
    Приймає байти оригінального зображення та вихідний тензор логітів UNET (N, 2, H, W).
    Повертає байтову послідовність накладеного PNG зображення.
    """

    # 1. Декодування оригінального зображення для фону (BGR)
    nparr = np.frombuffer(original_image_bytes, np.uint8)
    original_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_bgr is None:
        raise ValueError("Не вдалося декодувати оригінальне зображення.")

    H_orig, W_orig = original_bgr.shape[:2]

    # 2. Обробка вихідного тензора логітів (N, C=2, H, W)
    # Softmax для отримання ймовірностей та Argmax для визначення класу
    probabilities = torch.softmax(mask_logits, dim=1).squeeze().cpu().numpy()  # (C=2, H, W)

    # Знаходимо індекс класу з найвищою ймовірністю для кожного пікселя
    # (H, W). Клас 0: Фон/Культура, Клас 1: Бур'ян
    predicted_mask_idx = np.argmax(probabilities, axis=0).astype(np.uint8)

    # 3. Вибираємо лише маску бур'янів (клас 1)
    weed_mask_256 = (predicted_mask_idx == 1).astype(np.uint8)

    # Ресайз маски до розмірів оригінального зображення
    resized_mask = cv2.resize(weed_mask_256, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

    # 4. Створення кольорового накладення (зелений для бур'янів)
    weed_color = [0, 255, 0]  # BGR зелений
    colored_overlay = np.zeros_like(original_bgr, dtype=np.uint8)
    # Накладення лише на пікселі, де resized_mask == 1
    colored_overlay[resized_mask == 1] = weed_color

    # 5. Змішування (накладення)
    alpha = 0.5  # Прозорість маски
    # Додаємо зваженість: 1.0 * оригінал + 0.5 * кольорове накладення + 0 (гама-корекція)
    overlay_result = cv2.addWeighted(original_bgr, 1.0, colored_overlay, alpha, 0)

    # 6. Кодування результату у байтову послідовність PNG
    is_success, buffer = cv2.imencode(".png", overlay_result)
    if not is_success:
        raise Exception("Помилка кодування PNG результату накладення.")

    return buffer.tobytes()


def predict_weed_segmentation(model: UNET, rgb_image_bytes: bytes) -> bytes:
    """
    Виконує передбачення маски бур'янів та повертає байтову послідовність
    накладеного PNG зображення (оригінал + маска бур'янів).
    """

    # 1. Ініціалізація та завантаження моделі (якщо вона не була передана як об'єкт)
    if model is None or model == "Stub_UNet_Ready":
        print("INFO: Ініціалізація моделі UNET (4-канальний вхід, 2 класи).")
        model = UNET(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)

        # Завантаження контрольної точки, якщо вона існує
        try:
            if os.path.exists(CHECKPOINT_PATH):
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                model.load_state_dict(checkpoint["state_dict"])
                print(f"Модель успішно завантажена з {CHECKPOINT_PATH}")
            else:
                print(
                    f"Попередження: Файл контрольної точки '{CHECKPOINT_PATH}' не знайдено. Використовується нетренована модель.")
        except Exception as e:
            print(f"Помилка завантаження контрольної точки: {e}. Використовується нетренована модель.")

        model.to(DEVICE)

    # 2. Підготовка зображення (створює 4-канальний тензор)
    try:
        image_tensor = load_and_transform_image(rgb_image_bytes)
    except Exception as e:
        print(f"Помилка обробки зображення: {e}")
        # Повертаємо оригінальні байти, якщо обробка не вдалася
        return rgb_image_bytes

        # 3. Інференс
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        # Отримання логітів маски (1, 2, H, W)
        mask_logits = model(image_tensor)

    # 4. Створення накладеного зображення (Overlay)
    return create_mask_overlay(rgb_image_bytes, mask_logits)