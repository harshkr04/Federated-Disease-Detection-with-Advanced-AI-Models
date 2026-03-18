"""
Hybrid CNN + Transformer Model for Skin Lesion Classification.

Architecture:
    Image → ResNet50 (backbone) → Feature map → Transformer Encoder → Classification Head

The ResNet50 extracts spatial features, which are reshaped into a sequence
and fed to a Transformer encoder for global attention-based classification.
"""

import torch
import torch.nn as nn
from torchvision import models
import math


class HybridCNNTransformer(nn.Module):
    """
    Hybrid model combining ResNet50 feature extraction
    with a Transformer encoder for classification.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super(HybridCNNTransformer, self).__init__()

        # --- CNN Backbone (ResNet50 without final FC & avgpool) ---
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Keep everything except avgpool and fc
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output: (B, 2048, 7, 7) for 224x224 input

        # --- Projection to transformer dimension ---
        self.feature_proj = nn.Conv2d(2048, d_model, kernel_size=1)

        # --- Positional encoding ---
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)
        # 49 = 7x7 spatial positions

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def get_embedding(self, x):
        """
        Extract the d_model-dimensional embedding before the classification head.
        Used by MOON for contrastive learning.
        """
        features = self.cnn_backbone(x)
        features = self.feature_proj(features)
        B, C, H, W = features.shape
        features = features.flatten(2).permute(0, 2, 1)
        features = features + self.pos_embedding
        features = self.transformer_encoder(features)
        features = features.mean(dim=1)  # (B, d_model)
        return features

    def forward(self, x, return_embedding=False):
        # CNN feature extraction
        features = self.cnn_backbone(x)          # (B, 2048, 7, 7)

        # Project to transformer dimension
        features = self.feature_proj(features)    # (B, d_model, 7, 7)

        B, C, H, W = features.shape
        # Reshape to sequence: (B, H*W, d_model)
        features = features.flatten(2).permute(0, 2, 1)  # (B, 49, d_model)

        # Add positional encoding
        features = features + self.pos_embedding

        # Transformer encoder
        features = self.transformer_encoder(features)  # (B, 49, d_model)

        # Global average pooling over the sequence
        embedding = features.mean(dim=1)  # (B, d_model)

        # Classification
        out = self.classifier(embedding)  # (B, num_classes)

        if return_embedding:
            return out, embedding
        return out


def build_hybrid_model(num_classes=2, pretrained=True):
    """Helper function to create the Hybrid CNN+Transformer model."""
    model = HybridCNNTransformer(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Quick test
    model = build_hybrid_model()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
