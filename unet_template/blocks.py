"""
U-Net Building Blocks

이 파일은 U-Net의 핵심 빌딩 블록들을 정의합니다:
- DoubleConv: U-Net의 기본 convolution 블록
- Down: Encoder의 downsampling 블록
- Up: Decoder의 upsampling 블록
- OutConv: 최종 출력 convolution
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    U-Net의 기본 빌딩 블록: (Conv → BatchNorm → ReLU) × 2

    구조:
    x → [3x3 conv] → [BatchNorm] → [ReLU] → [3x3 conv] → [BatchNorm] → [ReLU] → output

    원본 논문에서는 BatchNorm이 없었지만, 현대적 구현에서는 추가하여 학습 안정성 향상
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            mid_channels: 중간 채널 수 (기본값: out_channels)
        """
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # TODO: DoubleConv 구현
        # 1. 첫 번째 convolution
        #    - in_channels → mid_channels
        #    - kernel_size=3, padding=1 (크기 유지)
        #    - bias=False (BatchNorm 사용 시)
        # 2. BatchNorm2d(mid_channels)
        # 3. ReLU(inplace=True)
        # 4. 두 번째 convolution
        #    - mid_channels → out_channels
        #    - kernel_size=3, padding=1
        #    - bias=False
        # 5. BatchNorm2d(out_channels)
        # 6. ReLU(inplace=True)
        # nn.Sequential로 묶어서 정의

        self.double_conv = nn.Sequential(
            None,  # 첫 번째 Conv2d를 채우세요
            None,  # BatchNorm2d를 채우세요
            None,  # ReLU를 채우세요
            None,  # 두 번째 Conv2d를 채우세요
            None,  # BatchNorm2d를 채우세요
            None,  # ReLU를 채우세요
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            출력 텐서 [batch, out_channels, H, W]
        """
        # TODO: DoubleConv forward 구현
        return None  # 여기를 채우세요


class Down(nn.Module):
    """
    Encoder의 Downsampling 블록

    구조:
    x → [MaxPool 2x2] → [DoubleConv] → output

    MaxPooling으로 spatial dimension을 절반으로 줄이고,
    DoubleConv로 채널 수를 늘림
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
        """
        super(Down, self).__init__()

        # TODO: Down 블록 구현
        # 1. MaxPool2d(kernel_size=2, stride=2)
        #    - 크기를 절반으로 줄임
        # 2. DoubleConv(in_channels, out_channels)
        # nn.Sequential로 묶어서 정의

        self.maxpool_conv = nn.Sequential(
            None,  # MaxPool2d를 채우세요
            None,  # DoubleConv를 채우세요
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            출력 텐서 [batch, out_channels, H/2, W/2]
        """
        # TODO: Down forward 구현
        return None  # 여기를 채우세요


class Up(nn.Module):
    """
    Decoder의 Upsampling 블록

    구조:
    x1 (encoder) ──────────────┐
                               ↓ (concatenate)
    x2 (decoder) → [Upsample] → [Concat] → [DoubleConv] → output

    Upsampling 방법:
    1. Transposed Convolution (학습 가능)
    2. Bilinear Upsampling + Conv (더 부드러운 결과)
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Args:
            in_channels: 입력 채널 수 (decoder의 채널 수)
            out_channels: 출력 채널 수
            bilinear: True면 bilinear upsampling, False면 transposed conv
        """
        super(Up, self).__init__()

        # TODO: Upsampling layer 정의
        # bilinear=True인 경우:
        #   - nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #   - 그 후 1x1 conv로 채널 조정: in_channels → in_channels // 2
        # bilinear=False인 경우:
        #   - nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        if bilinear:
            self.up = nn.Sequential(
                None,  # Upsample을 채우세요
                None,  # 1x1 Conv2d를 채우세요 (in_channels → in_channels // 2)
            )
        else:
            self.up = None  # ConvTranspose2d를 채우세요

        # TODO: DoubleConv 정의
        # Concatenation 후 채널 수: in_channels (up된 것 + skip connection)
        # 출력 채널: out_channels
        self.conv = None  # 여기를 채우세요

    def forward(self, x1, x2):
        """
        Forward pass with skip connection

        Args:
            x1: Encoder의 feature map [batch, C1, H, W] (skip connection)
            x2: Decoder의 feature map [batch, C2, H/2, W/2] (upsampling할 것)

        Returns:
            출력 텐서 [batch, out_channels, H, W]
        """
        # TODO: Upsampling
        x2 = None  # x2를 upsample하세요

        # TODO: 크기 차이 처리 (선택적)
        # 원본 U-Net은 크기가 정확히 맞지 않을 수 있음
        # 필요시 padding으로 크기 맞추기
        # diffY = x1.size()[2] - x2.size()[2]
        # diffX = x1.size()[3] - x2.size()[3]
        # x2 = nn.functional.pad(x2, [diffX // 2, diffX - diffX // 2,
        #                             diffY // 2, diffY - diffY // 2])

        # TODO: Concatenation
        # x1 (encoder)과 x2 (upsampled decoder)를 channel dimension에서 concat
        # torch.cat([x1, x2], dim=1)
        x = None  # 여기를 채우세요

        # TODO: DoubleConv 통과
        x = None  # 여기를 채우세요

        return x


class OutConv(nn.Module):
    """
    최종 출력 Convolution

    1x1 convolution으로 채널 수를 num_classes로 변환
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수 (num_classes)
        """
        super(OutConv, self).__init__()

        # TODO: 1x1 Convolution 정의
        # - in_channels → out_channels
        # - kernel_size=1
        self.conv = None  # 여기를 채우세요

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            출력 텐서 [batch, out_channels, H, W]
        """
        # TODO: 1x1 Conv forward
        return None  # 여기를 채우세요
