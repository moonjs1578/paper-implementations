"""
ResNet Building Blocks

이 파일은 ResNet의 핵심 빌딩 블록들을 정의합니다:
- BasicBlock: ResNet-18, ResNet-34에서 사용
- Bottleneck: ResNet-50, ResNet-101, ResNet-152에서 사용
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet의 기본 블록 (ResNet-18, 34에서 사용)

    구조:
    x -> [3x3 conv] -> [3x3 conv] -> output
    |                                   |
    +------- skip connection -----------+

    expansion: 출력 채널이 입력 채널의 몇 배인지 (BasicBlock은 1배)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            stride: 첫 번째 conv의 stride (downsampling용)
            downsample: skip connection을 조정하기 위한 layer (채널/크기 불일치 시)
        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass with residual connection

        핵심: F(x) + x 계산
        - F(x): conv-bn-relu-conv-bn을 거친 출력
        - x: skip connection (identity 또는 downsampled)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet의 Bottleneck 블록 (ResNet-50, 101, 152에서 사용)

    구조:
    x -> [1x1 conv] -> [3x3 conv] -> [1x1 conv] -> output
    |                                                 |
    +------------- skip connection ------------------+

    특징:
    - 1x1 conv로 채널을 줄였다가 (dimensionality reduction)
    - 3x3 conv로 처리하고
    - 1x1 conv로 채널을 다시 늘림 (dimensionality restoration)
    - 계산량을 줄이면서 깊게 쌓을 수 있음

    expansion: 출력 채널이 중간 채널의 4배
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 중간 채널 수 (최종 출력은 out_channels * 4)
            stride: 3x3 conv의 stride
            downsample: skip connection 조정용
        """
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass with residual connection

        핵심: F(x) + x 계산
        - F(x): 1x1-3x3-1x1 conv를 거친 출력
        - x: skip connection
        """
        identity = x

        # TODO: Bottleneck path 구현
        # 1. conv1 -> bn1 -> relu (1x1 conv, reduction)
        # 2. conv2 -> bn2 -> relu (3x3 conv, main)
        # 3. conv3 -> bn3 (1x1 conv, expansion)
        # 주의: 마지막 relu는 skip connection 더한 후에!

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # TODO: Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # TODO: Residual connection
        out += identity
        out = self.relu(out)

        return out
