"""
ResNet Training Script

CIFAR-10 데이터셋으로 ResNet을 학습하는 예제
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from model import resnet18, resnet34, resnet50


def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """
    CIFAR-10 데이터로더 생성

    데이터 증강 (Data Augmentation):
    - RandomCrop: 랜덤하게 crop하여 위치 변화에 강건하게
    - RandomHorizontalFlip: 좌우 반전
    - Normalize: 표준화
    """
    # TODO: Training 데이터 transform 정의
    # 1. transforms.RandomCrop(32, padding=4)
    # 2. transforms.RandomHorizontalFlip()
    # 3. transforms.ToTensor()
    # 4. transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO: Test 데이터 transform 정의
    # 1. transforms.ToTensor()
    # 2. transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 주의: Test 데이터는 augmentation 하지 않음!
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # TODO: DataLoader 생성
    # - batch_size, shuffle, num_workers 설정
    # - train은 shuffle=True, test는 shuffle=False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 epoch 학습

    Returns:
        평균 loss, 정확도
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # TODO: Forward pass
        # 1. optimizer.zero_grad()
        # 2. outputs = model(inputs)
        # 3. loss = criterion(outputs, targets)
        # 4. loss.backward()
        # 5. optimizer.step()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 통계
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """
    검증/테스트

    Returns:
        평균 loss, 정확도
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # TODO: Validation loop
    # 1. torch.no_grad() 사용
    # 2. model(inputs)로 예측
    # 3. loss, accuracy 계산

    with torch.no_grad():  # 여기를 채우세요
        for inputs, targets in tqdm(test_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def main():
    # 하이퍼파라미터
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.1
    num_classes = 10  # CIFAR-10

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로더
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)

    # TODO: 모델 생성 및 device로 이동
    # - resnet18(num_classes=10) 또는 다른 모델 선택
    # - .to(device)로 이동
    model = resnet18(num_classes=10).to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # TODO: Loss function 정의
    # CrossEntropyLoss 사용
    criterion = nn.CrossEntropyLoss()

    # TODO: Optimizer 정의
    # SGD with momentum
    # - lr=learning_rate
    # - momentum=0.9
    # - weight_decay=5e-4 (L2 regularization)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # TODO: Learning rate scheduler 정의
    # - MultiStepLR: epoch [50, 75]에서 lr을 0.1배로 감소
    # - 또는 CosineAnnealingLR 사용 가능
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    # 학습 루프
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )

        # TODO: Scheduler step
        # scheduler.step() 호출
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Best model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Saving best model (acc: {best_acc:.2f}%)...")

            # TODO: 모델 저장
           
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
