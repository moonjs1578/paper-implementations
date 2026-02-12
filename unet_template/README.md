# U-Net Implementation

PyTorch implementation of **U-Net: Convolutional Networks for Biomedical Image Segmentation** (MICCAI 2015)

## 📄 Paper Information

- **Title**: U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Authors**: Olaf Ronneberger, Philipp Fischer, Thomas Brox (University of Freiburg)
- **Conference**: MICCAI 2015
- **Paper**: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

## 🎯 Key Contributions

### 1. U-Shaped Architecture
- **Encoder (Contracting Path)**: 특징 추출 및 다운샘플링
- **Decoder (Expanding Path)**: 업샘플링 및 정밀한 위치 정보 복원
- **Skip Connections**: Encoder의 high-resolution features를 Decoder로 전달

### 2. Architecture Overview

```
Input (1×572×572)
       ↓
┌──────────────── Encoder (Contracting Path) ────────────────┐
│  Conv-Conv → MaxPool (64 channels, 568×568 → 284×284)     │──┐
│  Conv-Conv → MaxPool (128 channels, 280×280 → 140×140)    │  │
│  Conv-Conv → MaxPool (256 channels, 136×136 → 68×68)      │  │
│  Conv-Conv → MaxPool (512 channels, 64×64 → 32×32)        │  │
│  Conv-Conv (1024 channels, 28×28) ← Bottleneck            │  │
└────────────────────────────────────────────────────────────┘  │
                                                                 │
┌──────────────── Decoder (Expanding Path) ──────────────────┐  │
│  UpConv → Concat ← Skip Connection ────────────────────────┼──┘
│  Conv-Conv (512 channels, 52×52)                           │  │
│  UpConv → Concat ← Skip Connection ────────────────────────┼──┘
│  Conv-Conv (256 channels, 100×100)                         │  │
│  UpConv → Concat ← Skip Connection ────────────────────────┼──┘
│  Conv-Conv (128 channels, 196×196)                         │  │
│  UpConv → Concat ← Skip Connection ────────────────────────┼──┘
│  Conv-Conv (64 channels, 388×388)                          │
│  1×1 Conv → Output (num_classes, 388×388)                  │
└────────────────────────────────────────────────────────────┘
```

### 3. Why U-Net Works?

1. **Skip Connections**: Low-level spatial information + High-level semantic information
2. **Symmetric Architecture**: Encoder와 Decoder의 균형있는 구조
3. **Data Augmentation**: 적은 데이터로도 학습 가능 (elastic deformation)
4. **Overlap-Tile Strategy**: 큰 이미지를 패치 단위로 처리

## 🏗️ Model Variants

| Variant | Input Size | Channels | Use Case |
|---------|-----------|----------|----------|
| U-Net (Original) | 572×572 | [64,128,256,512,1024] | Medical Image |
| U-Net (Modified) | 256×256 | [64,128,256,512,1024] | General Segmentation |
| U-Net Small | 128×128 | [32,64,128,256,512] | Fast Inference |

## 📁 File Structure

```
unet/
├── __init__.py          # Package initialization
├── blocks.py            # DoubleConv, Down, Up, OutConv 정의
├── model.py             # U-Net 모델 정의
├── train.py             # 학습 스크립트
└── README.md            # This file
```

## 🚀 Usage

### 1. Import Model
```python
from unet import UNet

# Create model
model = UNet(in_channels=3, num_classes=2)  # Binary segmentation
model = UNet(in_channels=1, num_classes=21)  # 21-class segmentation
```

### 2. Forward Pass
```python
import torch

x = torch.randn(1, 3, 256, 256)  # [batch, channels, height, width]
output = model(x)  # [1, num_classes, 256, 256]

# For binary segmentation
probs = torch.sigmoid(output)  # Pixel-wise probabilities
```

### 3. Training (Custom Dataset)
```bash
cd unet
python train.py
```

## 💡 Learning Points (TODO 구현하면서 배울 내용)

### 1. Encoder-Decoder Architecture
- Contracting path의 역할 (특징 추출)
- Expanding path의 역할 (위치 정보 복원)
- 대칭 구조의 의미

### 2. Skip Connections
- ResNet의 residual connection과의 차이
- Concatenation vs Addition
- High-resolution features 보존의 중요성

### 3. Upsampling Methods
- Transposed Convolution (Deconvolution)
- Upsampling + Convolution
- Bilinear Interpolation
- 각 방법의 장단점

### 4. Segmentation Techniques
- Pixel-wise classification
- Cross-Entropy Loss for segmentation
- IoU (Intersection over Union) metric
- Dice Coefficient

## 🔬 Implementation Details

### Network Architecture (Original Paper)

```
Encoder:
- Double Conv (3→64): 3×3 conv, ReLU, 3×3 conv, ReLU
- MaxPool 2×2
- Double Conv (64→128)
- MaxPool 2×2
- Double Conv (128→256)
- MaxPool 2×2
- Double Conv (256→512)
- MaxPool 2×2

Bottleneck:
- Double Conv (512→1024)

Decoder:
- Up-conv 2×2 (1024→512)
- Concatenate with encoder features
- Double Conv (1024→512)
- Up-conv 2×2 (512→256)
- Concatenate with encoder features
- Double Conv (512→256)
- Up-conv 2×2 (256→128)
- Concatenate with encoder features
- Double Conv (256→128)
- Up-conv 2×2 (128→64)
- Concatenate with encoder features
- Double Conv (128→64)
- 1×1 conv (64→num_classes)
```

### Training Hyperparameters

- **Optimizer**: SGD with momentum (0.99)
- **Loss**: Cross-Entropy (or Dice Loss for medical imaging)
- **Learning rate**: High momentum, no learning rate decay
- **Data augmentation**:
  - Elastic deformation
  - Random rotations, shifts
  - Brightness/contrast adjustment

### Modified U-Net (Practical Version)

Original U-Net은 입력 크기가 줄어드는 문제가 있음 (572→388).
실제로는 padding을 추가하여 입력과 출력 크기를 동일하게 유지:

- 모든 conv에 `padding=1` 추가
- Input size = Output size (e.g., 256×256 → 256×256)

## 📊 Expected Results

### ISBI Cell Tracking Challenge
- **IoU**: > 0.92
- **Warping Error**: < 1.5

### Common Datasets
- **Carvana (Car Segmentation)**: Dice > 0.99
- **Cityscapes (Street Scene)**: mIoU > 65%

## 🎓 Key Takeaways

1. **Symmetric Design**: Encoder-Decoder 구조의 균형이 중요
2. **Skip Connections**: Spatial information 보존의 핵심
3. **Flexible Architecture**: 다양한 해상도/채널 수에 적용 가능
4. **Medical Imaging**: 적은 데이터로도 강력한 성능

## 🔗 References

- [Original Paper](https://arxiv.org/abs/1505.04597)
- [PyTorch Example](https://github.com/milesial/Pytorch-UNet)
- [TensorFlow Implementation](https://www.tensorflow.org/tutorials/images/segmentation)

## 📝 TODO Checklist

학습을 위해 다음 부분들을 직접 구현해보세요:

### blocks.py
- [ ] DoubleConv: 3x3 conv → ReLU → 3x3 conv → ReLU
- [ ] Down: MaxPool → DoubleConv
- [ ] Up: Upsampling → Conv → Concatenate → DoubleConv
- [ ] OutConv: 1x1 convolution for final output

### model.py
- [ ] Encoder path (4개의 Down blocks)
- [ ] Bottleneck (가장 깊은 층)
- [ ] Decoder path (4개의 Up blocks)
- [ ] Skip connections (concatenation)
- [ ] Forward pass 전체 구현

### train.py
- [ ] Segmentation dataset 로딩
- [ ] IoU/Dice metric 구현
- [ ] Training loop
- [ ] Validation loop
- [ ] Segmentation 결과 시각화

---

⭐ **Tip**: 구현하면서 막히면 논문의 Figure 1을 참고하세요!
