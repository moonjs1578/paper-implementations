"""
ResNet Model Architecture

다양한 깊이의 ResNet 모델을 정의합니다:
- ResNet-18, 34: BasicBlock 사용
- ResNet-50, 101, 152: Bottleneck 사용
"""

import torch
import torch.nn as nn
from .blocks import BasicBlock, Bottleneck


class ResNet(nn.Module):
    """
    ResNet Architecture

    전체 구조:
    1. Initial conv (7x7, stride=2) + maxpool
    2. 4개의 layer (각 layer는 여러 개의 residual block으로 구성)
    3. Global average pooling
    4. Fully connected layer
    """

    def __init__(self, block, layers, num_classes=1000):
        """
        Args:
            block: BasicBlock 또는 Bottleneck
            layers: 각 layer의 block 개수 리스트 [layer1, layer2, layer3, layer4]
                   예) ResNet-50: [3, 4, 6, 3]
            num_classes: 분류할 클래스 수
        """
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution layer (ImageNet용 큰 이미지에 적합)
        # TODO: 7x7 convolution 정의
        # - 3 (RGB) -> 64 channels
        # - kernel_size=7, stride=2, padding=3
        # - bias=False
        self.conv1 = None  # 여기를 채우세요

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # TODO: Max pooling 정의
        # - kernel_size=3, stride=2, padding=1
        # 입력 이미지 크기를 절반으로 줄임
        self.maxpool = None  # 여기를 채우세요

        # ResNet의 4개 layer
        # 각 layer는 여러 개의 residual block으로 구성
        # stride=2를 사용하여 spatial dimension을 줄임 (첫 block에서만)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # TODO: Global Average Pooling 정의
        # - AdaptiveAvgPool2d를 사용하여 출력을 (1, 1)로 만듦
        # - 어떤 입력 크기든 고정된 크기의 feature를 얻을 수 있음
        self.avgpool = None  # 여기를 채우세요

        # TODO: Fully Connected Layer 정의
        # - 512 * block.expansion -> num_classes
        # - block.expansion: BasicBlock=1, Bottleneck=4
        self.fc = None  # 여기를 채우세요

        # Weight 초기화
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        하나의 layer를 구성하는 메서드

        각 layer는 여러 개의 residual block으로 구성됨
        첫 번째 block에서만 stride를 적용하여 downsampling

        Args:
            block: BasicBlock 또는 Bottleneck
            out_channels: 출력 채널 수 (Bottleneck의 경우 중간 채널 수)
            num_blocks: 이 layer에 포함될 block의 개수
            stride: 첫 번째 block의 stride (보통 1 또는 2)

        Returns:
            nn.Sequential로 구성된 layer
        """
        downsample = None

        # TODO: Downsample layer 필요 여부 판단
        # downsample이 필요한 경우:
        # 1. stride != 1 (spatial dimension 변경)
        # 2. self.in_channels != out_channels * block.expansion (채널 수 변경)
        #
        # downsample layer 구성:
        # - 1x1 conv: self.in_channels -> out_channels * block.expansion
        # - stride=stride
        # - BatchNorm2d
        # nn.Sequential로 묶어서 생성
        if None:  # 조건을 채우세요
            downsample = nn.Sequential(
                None,  # 1x1 conv를 채우세요
                None,  # BatchNorm을 채우세요
            )

        layers = []

        # TODO: 첫 번째 block 추가
        # - self.in_channels, out_channels, stride, downsample를 인자로 전달
        layers.append(None)  # 여기를 채우세요

        # TODO: in_channels 업데이트
        # 첫 번째 block 이후 채널 수가 변경됨
        self.in_channels = None  # 여기를 채우세요

        # TODO: 나머지 block들 추가
        # stride=1, downsample=None (이미 크기/채널이 맞음)
        for _ in range(1, num_blocks):
            layers.append(None)  # 여기를 채우세요

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 텐서 [batch_size, 3, H, W]

        Returns:
            출력 logits [batch_size, num_classes]
        """
        # TODO: Initial layers
        # conv1 -> bn1 -> relu -> maxpool
        x = None  # 여기를 채우세요

        # TODO: ResNet의 4개 layer 통과
        x = None  # layer1을 채우세요
        x = None  # layer2를 채우세요
        x = None  # layer3을 채우세요
        x = None  # layer4를 채우세요

        # TODO: Global average pooling
        x = None  # 여기를 채우세요

        # TODO: Flatten
        # [batch, channels, 1, 1] -> [batch, channels]
        x = None  # 여기를 채우세요

        # TODO: Fully connected layer
        x = None  # 여기를 채우세요

        return x

    def _initialize_weights(self):
        """
        Weight 초기화 (Kaiming He initialization)

        ResNet 논문에서 제안한 방법:
        - Conv layers: Kaiming normal initialization
        - BatchNorm: weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TODO: Kaiming He initialization
                # nn.init.kaiming_normal_ 사용
                # - mode='fan_out'
                # - nonlinearity='relu'
                pass  # 여기를 채우세요

            elif isinstance(m, nn.BatchNorm2d):
                # TODO: BatchNorm 초기화
                # - weight를 1로
                # - bias를 0으로
                pass  # 여기를 채우세요

            elif isinstance(m, nn.Linear):
                # TODO: Linear layer 초기화
                # - weight: normal distribution (mean=0, std=0.01)
                # - bias: 0
                pass  # 여기를 채우세요


# 다양한 깊이의 ResNet을 만드는 helper 함수들

def resnet18(num_classes=1000):
    """ResNet-18: [2, 2, 2, 2] BasicBlocks"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    """ResNet-34: [3, 4, 6, 3] BasicBlocks"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    """ResNet-50: [3, 4, 6, 3] Bottlenecks"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=1000):
    """ResNet-101: [3, 4, 23, 3] Bottlenecks"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=1000):
    """ResNet-152: [3, 8, 36, 3] Bottlenecks"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    # 모델 테스트
    print("Testing ResNet models...")

    # 더미 입력 (ImageNet 크기)
    x = torch.randn(1, 3, 224, 224)

    models = {
        'ResNet-18': resnet18(num_classes=1000),
        'ResNet-34': resnet34(num_classes=1000),
        'ResNet-50': resnet50(num_classes=1000),
    }

    for name, model in models.items():
        output = model(x)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {num_params:,}")
        print()
