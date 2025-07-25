# src/model.py

import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=2, pretrained=True):
    """
    Build and return a ResNet18 model.
    
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use ImageNet pretrained weights.
    
    Returns:
        model (nn.Module): Modified ResNet18 model.
    """
    model = models.resnet18(pretrained=pretrained)

    # Freeze all layers if using pretrained
    for param in model.parameters():
        param.requires_grad = True  # Set False if you want feature extraction

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

