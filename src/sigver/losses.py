"""Loss functions for metric-learning signature verification.

Label convention (must match sigver.data.pairs):
    y = 1.0  -> similar pair (two genuine signatures, same writer)
    y = 0.0  -> dissimilar pair (skilled forgery or different writer)

Contrastive loss (Hadsell, Chopra & LeCun, 2006):
    L = y * d^2  +  (1 - y) * max(0, margin - d)^2
where d is the Euclidean distance between the two embeddings.
Similar pairs are pulled together (d -> 0); dissimilar pairs are pushed
apart until they exceed the margin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self,
                emb_a: torch.Tensor,
                emb_b: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """emb_a, emb_b: (B, D) embeddings; label: (B,) in {0.0, 1.0}."""
        d = F.pairwise_distance(emb_a, emb_b)  # (B,)
        loss_similar = label * d.pow(2)
        loss_dissimilar = (1.0 - label) * F.relu(self.margin - d).pow(2)
        return (loss_similar + loss_dissimilar).mean()