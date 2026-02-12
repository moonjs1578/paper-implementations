# AlexNet Implementation (2012)

PyTorch implementation of "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton.

## 📝 Paper

**Title**: ImageNet Classification with Deep Convolutional Neural Networks  
**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Published**: NIPS 2012  
**Paper Link**: [PDF](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
**Blog Review**: [논문 리뷰 (Korean)](https://velog.io/@moonjs1578/논문-리뷰-AlexNet-2012-딥러닝-혁명의-시작)

## 🎯 Key Contributions

1. **ReLU Activation**: First major use of ReLU instead of tanh/sigmoid
2. **Dropout**: Regularization technique to prevent overfitting
3. **GPU Training**: Parallel training on multiple GPUs
4. **Local Response Normalization (LRN)**: Normalization scheme
5. **Data Augmentation**: Random crops, horizontal flips, color jittering

## 🏗️ Architecture

```
Input (227×227×3)
    ↓
Conv1 (96 filters, 11×11, stride 4) + ReLU + MaxPool + LRN
    ↓
Conv2 (256 filters, 5×5, pad 2) + ReLU + MaxPool + LRN
    ↓
Conv3 (384 filters, 3×3, pad 1) + ReLU
    ↓
Conv4 (384 filters, 3×3, pad 1) + ReLU
    ↓
Conv5 (256 filters, 3×3, pad 1) + ReLU + MaxPool
    ↓
FC6 (4096 neurons) + ReLU + Dropout
    ↓
FC7 (4096 neurons) + ReLU + Dropout
    ↓
FC8 (1000 classes)
```

## 🚀 Usage

### Basic Usage

```python
from alexnet_complete import AlexNet

# Create model
model = AlexNet(num_classes=1000)

# Forward pass
import torch
x = torch.randn(1, 3, 227, 227)
output = model(x)
```

### Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from alexnet_complete import AlexNet

# Setup
model = AlexNet(num_classes=1000)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

# Training loop
model.train()
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## 📂 Files

- `alexnet_complete.py`: Full implementation with training utilities
- `alexnet_template.py`: Basic template for learning

## 🔍 Implementation Details

### Original Paper vs This Implementation

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Input Size | 224×224 | 227×227 (corrected) |
| GPUs | 2 GTX 580 | Single GPU/CPU |
| LRN | Yes | Optional (deprecated) |
| Optimizer | SGD + momentum | Configurable |

### Notes

- Input size is 227×227 (not 224×224 as commonly stated) to match the math
- LRN is included but optional (BatchNorm often preferred in modern implementations)
- Dropout rate: 0.5 as in original paper

## 📊 Results (Original Paper)

- **Top-1 Error**: 37.5%
- **Top-5 Error**: 17.0%
- **Dataset**: ImageNet ILSVRC-2010
- **Training**: ~6 days on 2 GPUs

## 🛠️ Requirements

```bash
pip install torch torchvision
```

## 📚 References

- Original Paper: [Krizhevsky et al., 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Blog Review: [논문 리뷰 (Korean)](https://velog.io/@moonjs1578/논문-리뷰-AlexNet-2012-딥러닝-혁명의-시작)

## 📄 License

MIT License
