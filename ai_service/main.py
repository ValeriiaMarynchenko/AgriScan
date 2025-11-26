# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torchvision.transforms.functional as F

# const and settings, example, have no correct data yet

TRAIN_DATA_FOLDERS = [
    (0, "data/train/healthy_fields"),
    (1, "data/train/disease"),
    (2, "data/train/dehydration"),
    (3, "data/train/micro_deficiency"),
]

VAL_DATA_FOLDERS = [
    (0, "data/val/healthy_fields"),
    (1, "data/val/disease"),
    (2, "data/val/dehydration"),
    (3, "data/val/micro_deficiency"),
]

NUM_CLASSES = 4

LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_MAP_BGR = {
    0: (0, 255, 0),  # green
    1: (0, 0, 255),  # red
    2: (0, 255, 255),  # yellow
    3: (255, 0, 0),  # blue
}


# -----------------------------------------------------------
# DATASET & AUGMENTATIONS
# -----------------------------------------------------------

class CornDiseaseDataset(Dataset):
    """
    Class PyTorch Dataset for multiclass segmentation task.
    Automatically generate masks with class index,
    thought images grouped by folders.
    """

    def __init__(self, data_folders, transform=None):
        self.transform = transform
        # path to image, class index
        self.data_list = []

        for class_index, folder_path in data_folders:
            if not os.path.isdir(folder_path):
                print(f"WARNING: folder not found: {folder_path}. skipping.")
                continue

            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    self.data_list.append((img_path, class_index))

        if not self.data_list:
            raise FileNotFoundError(
                "images not found. check paths TRAIN_DATA_FOLDERS/VAL_DATA_FOLDERS.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path, class_index = self.data_list[index]

        # convert RGB -> BGR
        image = cv2.imread(img_path)
        if image is None:
            print(f"file read error: {img_path}. skipping.")
            # Повертаємо перший елемент як запасний варіант
            return self.__getitem__((index + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # create one-channel mask
        # class_index for every pixel
        mask = np.full((h, w), class_index, dtype=np.uint8)

        # augmentation
        if self.transform is not None:
            # make mask as NumPy-list (H, W) with indexes
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # PyTorch
        # image: HxWxC -> CxHxW, float32, normalization to [0, 1]
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        # mask: HxW, LongTensor (for CrossEntropyLoss)
        mask = torch.from_numpy(mask).long()

        return image, mask


# Augmentayion policies (Albumentations)
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            # Normalize using forv -score standardisation #TODO maybe no sens
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # avarage ImageNet
                std=[0.229, 0.224, 0.225],  # lambda ImageNet
                max_pixel_value=255.0,
            ),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ])


#  U-NET architecture

class DoubleConv(nn.Module):
    """
    Блок подвійного згортання: Conv -> BatchNorm -> ReLU (2 рази)
    """

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
    """
    Багатокласова архітектура UNET.
    """

    def __init__(self, in_channels=3, out_channels=NUM_CLASSES, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder Path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Перевірка розміру (важливо, якщо вхідний розмір не ділиться на 16/32)
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# Learning and validation

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Function for one-step training in epoche.
    """
    model.train()
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(DEVICE)  # (N, C, H, W)
        targets = targets.to(DEVICE)  # (N, H, W)

        # Forward Pass (N, NUM_CLASSES, H, W)
        predictions = model(data)

        # CrossEntropyLoss for multiclass segmentation
        loss = loss_fn(predictions, targets)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"training loss per epoch: {avg_loss:.4f}")


def check_accuracy(loader, model, device=DEVICE):
    """
    check model accuracy on validation or test set,
    using Multi-Class Dice Score.
    """
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # (N, H, W)

            preds = model(x)  # (N, C, H, W)

            # get index classes -> (N, H, W)
            preds_labels = torch.argmax(preds, dim=1)

            # calculate pixel accuracy
            num_correct += (preds_labels == y).sum()
            num_pixels += torch.numel(preds_labels)

            # calculate Dice Score
            for cls in range(1, NUM_CLASSES):
                pred_cls = (preds_labels == cls).float()
                target_cls = (y == cls).float()

                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()

                if union > 0:
                    dice_score += (2. * intersection) / union

    pixel_accuracy = num_correct / num_pixels * 100
    avg_dice_score = dice_score / (len(loader) * (NUM_CLASSES - 1))  # -1 because don't use background (0)

    print(f"  Піксельна точність: {pixel_accuracy:.2f}%")
    print(f"  Середній Dice Score (без фону): {avg_dice_score:.4f}")

    model.train()
    return avg_dice_score

#TODO start check here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def save_checkpoint(state, filename="my_unet_checkpoint.pth.tar"):
    """Зберігає ваги моделі."""
    print("=> Збереження контрольної точки")
    torch.save(state, filename)



# visualisation of results by colors #TODO check it

def visualize_multi_class_results(model, image_tensor, device, color_map_bgr):
    """
    Отримує прогнози моделі та накладає кольорову маску відповідно до класу хвороби.
    """
    # Перемикаємо модель у режим оцінки
    model.eval()

    # Створюємо копію оригінального зображення для OpenCV (H x W x C, uint8, BGR)
    # Зворотне перетворення: CxHxW -> HxWxC, знімаємо нормалізацію (* 255)
    original_image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    original_image_np = original_image_np.astype(np.uint8)
    original_image_bgr = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        # Додаємо батч-розмір для одного зображення (1, C, H, W)
        data = image_tensor.unsqueeze(0).to(device)
        predictions = model(data)  # (1, NUM_CLASSES, H, W)

    # 1. Argmax для отримання індексу класу (H, W)
    pred_mask_np = torch.argmax(predictions.squeeze(0), dim=0).cpu().numpy()

    # 2. Створення триканальної кольорової маски
    colored_mask = np.zeros(original_image_bgr.shape, dtype=np.uint8)

    # 3. Застосування кольорів
    for class_idx, color in color_map_bgr.items():
        # Знаходимо всі пікселі, що належать до цього класу
        pixels_to_color = (pred_mask_np == class_idx)

        # Застосовуємо колір BGR до цих пікселів
        if np.any(pixels_to_color):
            colored_mask[pixels_to_color] = color

    # 4. Змішування (Blending)
    alpha = 0.7  # Вага оригінального зображення
    beta = 0.3  # Вага кольорової маски

    highlighted_image = cv2.addWeighted(original_image_bgr, alpha, colored_mask, beta, 0)

    # print("Візуалізація завершена. Збереження результату...")
    # cv2.imwrite("highlighted_result.png", highlighted_image)

    model.train()  # Повертаємо модель у режим навчання
    return highlighted_image


# main func #TODO check it

def main():
    # 1. Ініціалізація DataLoader'ів
    train_ds = CornDiseaseDataset(
        data_folders=TRAIN_DATA_FOLDERS,
        transform=get_transforms(train=True)
    )
    val_ds = CornDiseaseDataset(
        data_folders=VAL_DATA_FOLDERS,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    # 2. Ініціалізація Моделі, Оптимізатора, Втрат
    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Для використання змішаної точності (Mixed Precision Training)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    print(f"Модель UNET ініціалізована. Класів: {NUM_CLASSES}. Пристрій: {DEVICE}")

    # 3. Навчання
    best_dice = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Епоха {epoch + 1}/{NUM_EPOCHS} ---")

        # Крок тренування
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Крок валідації та перевірка точності
        dice_score = check_accuracy(val_loader, model, device=DEVICE)

        # Збереження найкращої моделі
        if dice_score > best_dice:
            best_dice = dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="best_multi_class_unet.pth.tar")

        # Додатково: Візуалізація результатів для першого зображення (для перевірки)
        # first_image, _ = val_ds[0]
        # visualized_img = visualize_multi_class_results(model, first_image, DEVICE, COLOR_MAP_BGR)
        # Відображення або збереження visualized_img


if __name__ == "__main__":
    main()