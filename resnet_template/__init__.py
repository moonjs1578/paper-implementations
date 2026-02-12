"""
ResNet Implementation
Paper: Deep Residual Learning for Image Recognition (2015)
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
"""

from .model import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
