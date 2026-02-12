# Paper Implementations

PyTorch implementations of classic deep learning papers for learning and reference.

## 📚 Implemented Papers

### Computer Vision
- **[AlexNet (2012)](alexnet/)** - ImageNet Classification with Deep Convolutional Neural Networks
  - Status: ✅ Completed
  - Blog: 📝 [논문 리뷰](https://velog.io/@moonjs1578/논문-리뷰-AlexNet-2012-딥러닝-혁명의-시작)
  - Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
  - Key contributions: ReLU, Dropout, GPU training, LRN

- **[ResNet (2015)](resnet/)** - Deep Residual Learning for Image Recognition
  - Status: ✅ Completed
  - Conference: CVPR 2015 (Best Paper Award 🏆)
  - Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
  - Key contributions: Residual learning, Skip connections, Very deep networks (152 layers)

### Medical Imaging / Segmentation
- **[U-Net (2015)](unet/)** - Convolutional Networks for Biomedical Image Segmentation
  - Status: ✅ Completed
  - Conference: MICCAI 2015
  - Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox (University of Freiburg)
  - Key contributions: U-shaped encoder-decoder, Skip connections, Efficient with small datasets

### Coming Soon
- [ ] **VGGNet (2014)** - Very Deep Convolutional Networks
- [ ] **GoogLeNet (2014)** - Going Deeper with Convolutions

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/moonjs1578/paper-implementations.git
cd paper-implementations

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

#### AlexNet

```bash
cd alexnet
python alexnet_complete.py
```

#### ResNet (CIFAR-10)

```bash
cd resnet
python train.py
```

#### U-Net (Segmentation)

```bash
cd unet
python train.py
```

## 🎯 Goals

This repository aims to:
- 📖 Understand classic deep learning architectures through implementation
- 💻 Provide clean, readable PyTorch code
- 📝 Document key insights and learning points
- 🔗 Connect theory (papers) with practice (code)

## 📝 Blog Series

Detailed paper reviews and implementation notes (in Korean):
- [AlexNet (2012) 논문 리뷰 - 딥러닝 혁명의 시작](https://velog.io/@moonjs1578/논문-리뷰-AlexNet-2012-딥러닝-혁명의-시작) ✅

## 🛠️ Tech Stack

- **Framework**: PyTorch 2.0+
- **Language**: Python 3.8+
- **Tools**: NumPy, tqdm

## 🤝 Contributing

Feel free to:
- Open issues for bugs or suggestions
- Submit PRs for improvements
- Share your learning experience

## 📄 License

MIT License - Feel free to use for learning and reference!

## 👤 Author

**moonjs1578**
- GitHub: [@moonjs1578](https://github.com/moonjs1578)
- Blog: [velog.io/@moonjs1578](https://velog.io/@moonjs1578)

---

⭐ Star this repo if you find it helpful for learning!
