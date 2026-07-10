"""Siamese network for writer-independent signature verification.

A Siamese network is one embedding network applied to both images of a
pair (shared weights). Verification happens in embedding space: the
Euclidean distance between the two embeddings is compared to a threshold.

The baseline backbone is a compact CNN sized for 150x220 grayscale
inputs and CPU training. Alternative backbones (ResNet, EfficientNet,
ViT) plug in later via sigver.models.backbones without changing the
Siamese wrapper or the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Compact convolutional backbone -> embedding vector.

    Input:  (B, 1, 150, 220) float32 in [0, 1]
    Output: (B, embedding_dim)
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # 150x220 -> 75x110
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 75x110 -> 37x55
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 37x55 -> 18x27
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 18x27 -> 9x13
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, 128, 1, 1) regardless of input size
            nn.Flatten(),              # (B, 128)
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SiameseNetwork(nn.Module):
    """Applies one shared backbone to both images of a pair."""

    def __init__(self, backbone: nn.Module | None = None,
                 embedding_dim: int = 128):
        super().__init__()
        self.backbone = backbone if backbone is not None else SmallCNN(embedding_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        return self.embed(x_a), self.embed(x_b)