"""
U-Net Model Architecture

U자 모양의 대칭적인 Encoder-Decoder 구조를 정의합니다.
"""

import torch
import torch.nn as nn
from blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    U-Net Architecture

    전체 구조:
    1. Encoder (Contracting Path): 4개의 Down blocks
       - 이미지를 점진적으로 다운샘플링하며 high-level features 추출
       - 각 단계에서 채널 수를 2배로 늘림: 64 → 128 → 256 → 512

    2. Bottleneck: 가장 깊은 층
       - 512 → 1024 channels

    3. Decoder (Expanding Path): 4개의 Up blocks
       - 업샘플링하며 spatial resolution 복원
       - Skip connections로 encoder의 features와 결합
       - 각 단계에서 채널 수를 절반으로 줄임: 1024 → 512 → 256 → 128 → 64

    4. Output: 1x1 conv로 최종 segmentation map 생성
    """

    def __init__(self, in_channels=3, num_classes=2, bilinear=True):
        """
        Args:
            in_channels: 입력 이미지 채널 수 (RGB: 3, Grayscale: 1)
            num_classes: Segmentation 클래스 수
            bilinear: True면 bilinear upsampling, False면 transposed conv
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # TODO: Initial DoubleConv
        # in_channels → 64
        self.inc = DoubleConv(in_channels, 64)

        # TODO: Encoder (Contracting Path)
        # 4개의 Down blocks
        # 64 → 128 → 256 → 512 → 1024
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck (가장 깊은 층)
        # bilinear=True면 채널 수를 절반으로 (upsampling 후 concat 고려)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # TODO: Decoder (Expanding Path)
        # 4개의 Up blocks
        # 1024 → 512 → 256 → 128 → 64
        # 주의: concat 후 채널 수는 2배가 됨
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # TODO: Output Convolution
        # 64 → num_classes
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 [batch, in_channels, H, W]

        Returns:
            segmentation map [batch, num_classes, H, W]
        """
        # TODO: Encoder path
        # 각 단계의 출력을 저장 (skip connection용)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # TODO: Decoder path
        # Skip connections: encoder의 해당 level feature와 concat
        # up1: x5 (decoder) + x4 (encoder)
        # up2: up1 결과 + x3 (encoder)
        # up3: up2 결과 + x2 (encoder)
        # up4: up3 결과 + x1 (encoder)
        x = self.up1(x4, x5)  # up1(x4, x5)를 채우세요
        x = self.up2(x3, x)  # up2(x3, x)를 채우세요
        x = self.up3(x2, x) # up3(x2, x)를 채우세요
        x = self.up4(x1, x)  # up4(x1, x)를 채우세요

        # TODO: Output layer
        # 최종 segmentation map 생성
        logits = self.outc(x)

        return logits


if __name__ == "__main__":
    # 모델 테스트
    print("Testing U-Net model...")

    # Binary segmentation (2 classes)
    model = UNet(in_channels=3, num_classes=2, bilinear=True)

    # 더미 입력
    x = torch.randn(1, 3, 256, 256)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Multi-class segmentation (21 classes, PASCAL VOC)
    model_multiclass = UNet(in_channels=3, num_classes=21, bilinear=True)
    output_multiclass = model_multiclass(x)

    print(f"\nMulti-class output shape: {output_multiclass.shape}")
    print(f"Multi-class parameters: {sum(p.numel() for p in model_multiclass.parameters()):,}")
