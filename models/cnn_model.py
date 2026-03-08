"""
Baseline CNN Model — ResNet50 for Binary Skin Lesion Classification.

Uses a pretrained ResNet50 backbone from ImageNet.
The final fully-connected layer is replaced for binary classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """ResNet50-based binary classifier."""

    def __init__(self, num_classes=2, pretrained=True):
        super(BaselineCNN, self).__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace the final FC layer
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_cnn_model(num_classes=2, pretrained=True):
    """Helper function to create the model."""
    model = BaselineCNN(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Quick test
    model = build_cnn_model()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
