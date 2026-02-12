# ============================================================
# AlexNet 구현 템플릿
# 논문: ImageNet Classification with Deep Convolutional
#        Neural Networks (Krizhevsky et al., 2012)
#
# 📝 규칙:
#   - TODO: 직접 구현해야 할 부분
#   - 힌트: 막히면 참고하세요
#   - 참고: 우리가 공부한 내용 복귀용
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. AlexNet 모델 정의
# ============================================================
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # ────────────────────────────────────────────────────
        # TODO 1: self.features 정의 (Conv Layers 5개)
        # ────────────────────────────────────────────────────
        # 참고: 이 구조를 따라라
        #
        # Conv1 → ReLU → LRN → Pool
        # Conv2 → ReLU → LRN → Pool
        # Conv3 → ReLU
        # Conv4 → ReLU
        # Conv5 → ReLU → Pool
        #
        # 힌트 (각 Conv 설정):
        # ┌─────────┬────────┬──────────┬────────┬─────────┐
        # │  Layer  │in_ch   │  out_ch  │  k     │ stride  │
        # ├─────────┼────────┼──────────┼────────┼─────────┤
        # │  Conv1  │  3     │   96     │  11    │    4    │
        # │  Conv2  │  96    │  256     │   5    │    1    │
        # │  Conv3  │ 256    │  384     │   3    │    1    │
        # │  Conv4  │ 384    │  384     │   3    │    1    │
        # │  Conv5  │ 384    │  256     │   3    │    1    │
        # └─────────┴────────┴──────────┴────────┴─────────┘
        #
        # 힌트 (padding):
        #   Conv1: padding=0
        #   Conv2~5: padding = kernel_size // 2  (출력 크기 유지)
        #
        # 힌트 (LRN):
        #   nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        #   → Conv1, Conv2 후에만!
        #
        # 힌트 (Pool):
        #   nn.MaxPool2d(kernel_size=3, stride=2)
        #   → Conv1, Conv2, Conv5 후에만!

        self.features = nn.Sequential(
            # === Conv Layer 1 ===
            # 입력: 224×224×3 → 출력: 54×54×96
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
    
            # === Conv Layer 2 ===
            # 출력: 26×26×256
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # === Conv Layer 3 ===
            # 출력: 12×12×384
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            # === Conv Layer 4 ===
            # 출력: 12×12×384
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            # === Conv Layer 5 ===
            # 출력: 5×5×256
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # ────────────────────────────────────────────────────
        # TODO 2: self.avgpool 정의
        # ────────────────────────────────────────────────────
        # 힌트: nn.AdaptiveAvgPool2d 사용
        # 출력 크기를 6×6으로 고정하라
        # → FC Layer 입력 크기가 항상 동일하게 됨
        self.avgpool = nn.AdaptiveAvgPool2d(6)

        # ────────────────────────────────────────────────────
        # TODO 3: self.classifier 정의 (FC Layers 3개)
        # ────────────────────────────────────────────────────
        # 참고: 이 구조를 따라라
        #
        # Dropout → FC1 → ReLU
        # Dropout → FC2 → ReLU
        # FC3
        #
        # 힌트:
        #   FC1 입력: avgpool 출력을 Flatten한 후 크기
        #          = 256 × 6 × 6 = ?
        #   FC1 출력: 4096
        #   FC2 출력: 4096
        #   FC3 출력: num_classes
        #   Dropout: p=0.5 (FC1, FC2 앞에만!)

        self.classifier = nn.Sequential(
            # === FC Layer 1 ===
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            # === FC Layer 2 ===
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # === FC Layer 3 (출력층) ===
            nn.Linear(4096, num_classes)
            # ※ Softmax는 아래 Loss에 포함 → 여기서는 안 씀!
        )

    def forward(self, x):
        # ────────────────────────────────────────────────────
        # TODO 4: Forward Pass 구현
        # ────────────────────────────────────────────────────
        # 힌트: 순서대로 통과시키라
        #   1. self.features(x)
        #   2. self.avgpool(x)
        #   3. torch.flatten(x, 1)   ← 1번째 dim부터 flatten
        #   4. self.classifier(x)
        #   5. return x
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


# ============================================================
# 2. Training Loop
# ============================================================
def train(model, dataloader, criterion, optimizer):
    # ────────────────────────────────────────────────────────
    # TODO 5: 학습 루프 구현
    # ────────────────────────────────────────────────────────
    # 힌트: 이 순서로 구현하라
    #
    #   1. model.train()           → 학습 모드 ON (Dropout 작동!)
    #   2. for data, target in dataloader:
    #   3.     optimizer.zero_grad()   → gradient 초기화
    #   4.     output = model(data)    → 순방향
    #   5.     loss = criterion(output, target)  → loss 계산
    #   6.     loss.backward()         → 역전파
    #   7.     optimizer.step()        → 가중치 업데이트
    #   8.     loss 출력
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")


def test(model, dataloader, criterion):
    # ────────────────────────────────────────────────────────
    # TODO 6: 테스트 루프 구현
    # ────────────────────────────────────────────────────────
    # 힌트: train과 다른 점 2가지!
    #   1. model.eval()            → 테스트 모드 (Dropout 끄기!)
    #   2. with torch.no_grad():   → gradient 안 계산 (속도↑ 메모리↓)
    #
    # 나머지는 train과 동일하게:
    #   output → loss 계산 → 출력
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            print(f"Test Loss: {loss.item():.4f}")
# ============================================================
# 3. Main
# ============================================================
if __name__ == '__main__':

    # ────────────────────────────────────────────────────────
    # TODO 7: 아래 순서대로 구현하라
    # ────────────────────────────────────────────────────────

    # 1. 모델 생성
    #    힌트: model = AlexNet(num_classes=10)
    #    참고: 원본 논문은 1000개, 테스트용으로 10개
    model = AlexNet(num_classes=10)

    # 2. 더미 데이터 생성
    #    힌트:
    #      train_data   = torch.randn(32, 3, 224, 224)
    #      train_labels = torch.randint(0, 10, (32,))
    #      test_data    = torch.randn(8, 3, 224, 224)
    #      test_labels  = torch.randint(0, 10, (8,))
    #
    #    DataLoader로 감싸기:
    #      train_loader = DataLoader(
    #          TensorDataset(train_data, train_labels),
    #          batch_size=8, shuffle=True
    #      )
    #      test_loader  = DataLoader(...)
    train_data = torch.randn(32, 3, 224, 224)
    train_labels = torch.randint(0, 10, (32,))
    test_data    = torch.randn(8, 3, 224, 224)
    test_labels  = torch.randint(0, 10, (8,))

    train_loader = DataLoader(
        TensorDataset(train_data, train_labels),
        batch_size = 8, shuffle= True
    )
    test_loader = DataLoader(
        TensorDataset(test_data, test_labels),
        batch_size=8, shuffle=True
    )

    # 3. Loss 정의
    #    힌트: nn.CrossEntropyLoss()
    #    참고: Softmax + NLLLoss 포함!
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer 정의
    #    힌트: optim.SGD(
    #        model.parameters(),
    #        lr=0.01,
    #        momentum=0.9,
    #        weight_decay=5e-4
    #    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )

    # 5. 학습 루프
    #    힌트:
    #    EPOCHS = 3
    #    for epoch in range(EPOCHS):
    #        train(...)
    #        test(...)
    EPOCHS = 3
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train(model, train_loader, criterion, optimizer)
        test(model, test_loader, criterion)