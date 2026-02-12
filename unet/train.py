"""
U-Net Training Script

Segmentation 데이터셋으로 U-Net을 학습하는 예제
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os

from model import UNet


class DummySegmentationDataset(Dataset):
    """
    더미 Segmentation 데이터셋

    실제 사용 시에는 자신의 데이터셋으로 교체:
    - COCO, PASCAL VOC, Cityscapes 등
    - 이미지 + segmentation mask 페어로 구성
    """

    def __init__(self, num_samples=1000, img_size=256, num_classes=2):
        """
        Args:
            num_samples: 데이터셋 크기
            img_size: 이미지 크기
            num_classes: Segmentation 클래스 수
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            image: [3, H, W] 텐서
            mask: [H, W] 텐서 (픽셀별 클래스 인덱스)
        """
        # 더미 이미지와 마스크 생성
        image = torch.randn(3, self.img_size, self.img_size)
        mask = torch.randint(0, self.num_classes, (self.img_size, self.img_size))

        return image, mask


def dice_coefficient(pred, target, num_classes):
    """
    Dice Coefficient 계산

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)

    Segmentation에서 널리 사용되는 metric
    IoU와 유사하지만 조화 평균 사용

    Args:
        pred: 예측 [batch, H, W] (클래스 인덱스)
        target: 정답 [batch, H, W] (클래스 인덱스)
        num_classes: 클래스 수

    Returns:
        평균 Dice coefficient
    """
    dice_scores = []

    # TODO: 각 클래스별 Dice coefficient 계산
    # 1. pred와 target을 one-hot encoding
    # 2. intersection = (pred == c) & (target == c)의 합
    # 3. dice = 2 * intersection / (pred_sum + target_sum)
    # 힌트: (pred == c).sum()으로 각 클래스의 픽셀 수 계산

    for c in range(num_classes):
        pred_mask = (pred==c)
        target_mask = (target == c)

        intersection = (pred_mask & target_mask).sum().item()

        pred_sum = pred_mask.sum().item()
        target_sum = target_mask.sum().item()

        if pred_sum + target_sum > 0:
            dice = 2.0 * intersection / (pred_sum + target_sum)
            dice_scores.append(dice)

    return np.mean(dice_scores) if dice_scores else 0.0


def iou_score(pred, target, num_classes):
    """
    IoU (Intersection over Union) 계산

    IoU = |pred ∩ target| / |pred ∪ target|

    Args:
        pred: 예측 [batch, H, W]
        target: 정답 [batch, H, W]
        num_classes: 클래스 수

    Returns:
        평균 IoU
    """
    iou_scores = []

    # TODO: 각 클래스별 IoU 계산
    # 1. intersection = (pred == c) & (target == c)의 합
    # 2. union = (pred == c) | (target == c)의 합
    # 3. iou = intersection / union

    for c in range(num_classes):
        pred_mask = (pred == c)
        target_mask = (target == c)

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union > 0:
            iou = intersection / union
            iou_scores.append(iou)

    return np.mean(iou_scores) if iou_scores else 0.0


def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    """
    한 epoch 학습

    Returns:
        평균 loss, Dice coefficient, IoU
    """
    model.train()
    running_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).long()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # 통계
        running_loss += loss.item()

        # Prediction
        # outputs에서 가장 높은 확률의 클래스 선택
        preds = torch.argmax(outputs, dim=1)  # [batch, H, W]

        # Metrics 계산
        dice = dice_coefficient(preds.cpu(), masks.cpu(), num_classes)
        iou = iou_score(preds.cpu(), masks.cpu(), num_classes)

        dice_total += dice
        iou_total += iou

        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'dice': dice_total / (pbar.n + 1),
            'iou': iou_total / (pbar.n + 1)
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_dice = dice_total / len(train_loader)
    epoch_iou = iou_total / len(train_loader)

    return epoch_loss, epoch_dice, epoch_iou


def validate(model, val_loader, criterion, device, num_classes):
    """
    검증/테스트

    Returns:
        평균 loss, Dice coefficient, IoU
    """
    model.eval()
    running_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    # Validation loop
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            dice = dice_coefficient(preds.cpu(), masks.cpu(), num_classes)
            iou = iou_score(preds.cpu(), masks.cpu(), num_classes)

            dice_total += dice
            iou_total += iou

    epoch_loss = running_loss / len(val_loader)
    epoch_dice = dice_total / len(val_loader)
    epoch_iou = iou_total / len(val_loader)

    return epoch_loss, epoch_dice, epoch_iou


def main():
    # 하이퍼파라미터
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    num_classes = 2  # Binary segmentation
    img_size = 256

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TODO: 데이터셋 생성
    # 실제로는 자신의 segmentation 데이터셋으로 교체
    # 예: datasets.VOCSegmentation, datasets.CocoDetection 등
    train_dataset = DummySegmentationDataset(
        num_samples=1000,
        img_size=img_size,
        num_classes=num_classes
    )

    val_dataset = DummySegmentationDataset(
        num_samples=200,
        img_size=img_size,
        num_classes=num_classes
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 모델 생성 및 device로 이동
    model = UNet(in_channels=3, num_classes=num_classes, bilinear=True).to(device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function 정의
    # Segmentation에는 주로 다음을 사용:
    # - nn.CrossEntropyLoss() (일반적)
    # - Dice Loss (의료 영상)
    # - Focal Loss (불균형 데이터)
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    # Adam optimizer 추천 (lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (선택적)
    # ReduceLROnPlateau: validation loss가 개선되지 않으면 lr 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # 학습 루프
    best_dice = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, num_classes
        )

        # Scheduler step
        scheduler.step(val_loss)

        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # Best model 저장
        if val_dice > best_dice:
            best_dice = val_dice
            print(f"Saving best model (Dice: {best_dice:.4f})...")

            # 모델 저장
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_unet.pth')

    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
