# ResNet Implementation

PyTorch implementation of **Deep Residual Learning for Image Recognition** (CVPR 2015)

## 📄 Paper Information

- **Title**: Deep Residual Learning for Image Recognition
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- **Conference**: CVPR 2015 (Best Paper Award 🏆)
- **Paper**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

## 🎯 Key Contributions

### 1. Residual Learning
- **문제**: 네트워크가 깊어질수록 degradation 발생 (성능 저하)
- **해결**: Skip connection (shortcut connection)을 통한 residual learning
- **핵심 아이디어**: `F(x) + x` - 잔차(residual)를 학습하는 것이 더 쉬움

### 2. Network Architecture

#### BasicBlock (ResNet-18, 34)
```
x → [3x3 conv] → [3x3 conv] → + → output
|                                |
+--------- skip connection ------+
```

#### Bottleneck (ResNet-50, 101, 152)
```
x → [1x1 conv] → [3x3 conv] → [1x1 conv] → + → output
|  (reduce)       (process)      (expand)    |
+--------------- skip connection -------------+
```

### 3. Why ResNet Works?

1. **Gradient Flow**: Skip connection이 gradient를 직접 전파
2. **Identity Mapping**: 최악의 경우에도 입력을 그대로 전달 가능
3. **Ensemble Effect**: 다양한 깊이의 네트워크를 앙상블하는 효과

## 🏗️ Model Variants

| Model | Blocks | Layers | Parameters | Top-1 Error (ImageNet) |
|-------|--------|--------|------------|----------------------|
| ResNet-18 | BasicBlock | [2,2,2,2] | 11.7M | 30.24% |
| ResNet-34 | BasicBlock | [3,4,6,3] | 21.8M | 26.70% |
| ResNet-50 | Bottleneck | [3,4,6,3] | 25.6M | 24.01% |
| ResNet-101 | Bottleneck | [3,4,23,3] | 44.5M | 22.44% |
| ResNet-152 | Bottleneck | [3,8,36,3] | 60.2M | 21.69% |

## 📁 File Structure

```
resnet/
├── __init__.py          # Package initialization
├── blocks.py            # BasicBlock & Bottleneck 정의
├── model.py             # ResNet 모델 정의
├── train.py             # 학습 스크립트
└── README.md            # This file
```

## 🚀 Usage

### 1. Import Model
```python
from resnet import resnet18, resnet34, resnet50

# Create model
model = resnet18(num_classes=10)  # for CIFAR-10
model = resnet50(num_classes=1000)  # for ImageNet
```

### 2. Forward Pass
```python
import torch

x = torch.randn(1, 3, 224, 224)  # [batch, channels, height, width]
output = model(x)  # [1, num_classes]
```

### 3. Training (CIFAR-10)
```bash
cd resnet
python train.py
```

## 💡 Learning Points (TODO 구현하면서 배울 내용)

### 1. Residual Connection
- `F(x) + x`의 의미와 구현
- Skip connection이 gradient flow에 미치는 영향
- Identity mapping의 중요성

### 2. Bottleneck Design
- 1x1 convolution의 역할 (dimensionality reduction/expansion)
- 계산량 절감 효과
- 깊은 네트워크를 효율적으로 만드는 방법

### 3. Downsampling
- Stride를 이용한 spatial dimension 축소
- Skip connection의 크기/채널 맞추기
- 1x1 convolution을 이용한 projection

### 4. Training Techniques
- Kaiming He initialization
- Batch Normalization의 위치
- Learning rate scheduling
- Data augmentation

## 🔬 Implementation Details

### Network Architecture
```
Input (3×224×224)
    ↓
Conv1: 7×7, 64, stride=2
BatchNorm + ReLU
MaxPool: 3×3, stride=2
    ↓
Layer1: [64, 64, 64, ...] (stride=1)
Layer2: [128, 128, 128, ...] (stride=2)  ← downsampling
Layer3: [256, 256, 256, ...] (stride=2)  ← downsampling
Layer4: [512, 512, 512, ...] (stride=2)  ← downsampling
    ↓
Global Average Pooling
Fully Connected (→ num_classes)
```

### Training Hyperparameters (ImageNet)
- **Optimizer**: SGD with momentum (0.9)
- **Weight decay**: 0.0001
- **Batch size**: 256
- **Learning rate**: 0.1, divided by 10 at 30k, 60k iterations
- **Epochs**: 90
- **Data augmentation**: Random crop, horizontal flip

### CIFAR-10 Adaptation
- 더 작은 입력 크기 (32×32)
- 첫 번째 conv: 3×3, stride=1 (7×7 대신)
- MaxPool 제거
- 더 긴 학습 (200 epochs)

## 📊 Expected Results (CIFAR-10)

| Model | Parameters | Test Accuracy |
|-------|------------|---------------|
| ResNet-18 | ~11M | ~95% |
| ResNet-34 | ~21M | ~95.5% |
| ResNet-50 | ~23M | ~96% |

## 🎓 Key Takeaways

1. **Depth Matters**: 적절한 구조(residual connection)만 있다면 깊은 네트워크가 더 좋음
2. **Skip Connections**: Gradient vanishing 문제 해결의 핵심
3. **Bottleneck Design**: 계산 효율성과 표현력의 균형
4. **Simplicity**: 복잡한 기법 없이도 강력한 성능

## 🔗 References

- [Original Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Official Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [Author's Slides](https://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf)

## 📝 TODO Checklist

학습을 위해 다음 부분들을 직접 구현해보세요:

### blocks.py
- [ ] BasicBlock의 convolution layers
- [ ] BasicBlock의 forward pass (residual connection)
- [ ] Bottleneck의 1x1-3x3-1x1 구조
- [ ] Bottleneck의 forward pass

### model.py
- [ ] ResNet의 initial layers (conv1, maxpool)
- [ ] `_make_layer` 메서드의 downsample 로직
- [ ] ResNet의 forward pass
- [ ] Weight initialization

### train.py
- [ ] Data augmentation transforms
- [ ] Training loop (forward, backward, optimizer step)
- [ ] Validation loop
- [ ] Learning rate scheduler

---

⭐ **Tip**: 구현하면서 막히면 논문의 Figure 3, Table 1을 참고하세요!
