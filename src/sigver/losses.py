"""Loss functions for signature verification."""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings_a, embeddings_b, labels):
        distance = torch.norm(embeddings_a - embeddings_b, dim=1)
        loss = labels.float() * distance.pow(2) + (1 - labels.float()) * torch.clamp(self.margin - distance, min=0.0).pow(2)
        return loss.mean()
