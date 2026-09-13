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


class TripletLoss(nn.Module):
    """Triplet loss (Schroff, Kalenichenko & Philbin, 2015; see also
    Maergner et al.'s graph-edit-distance + triplet-network offline
    signature work in the reading list, which motivates this arm):

        L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Triplets carry no explicit label: the ordering of the three inputs
    (anchor, positive, negative) encodes the same-writer / different-
    writer relationship that ContrastiveLoss instead reads from a
    separate label tensor. Uses F.pairwise_distance, the same distance
    definition as ContrastiveLoss, so the two losses are comparable.

    margin defaults to 1.0 to match the contrastive arm for a clean
    single-variable (loss-only) comparison; the source paper's margin
    value could not be verified from the PDF, so this is a comparability
    choice, not a paper-derived one.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self,
                emb_a: torch.Tensor,
                emb_p: torch.Tensor,
                emb_n: torch.Tensor) -> torch.Tensor:
        """emb_a, emb_p, emb_n: (B, D) anchor/positive/negative embeddings."""
        d_ap = F.pairwise_distance(emb_a, emb_p)
        d_an = F.pairwise_distance(emb_a, emb_n)
        return F.relu(d_ap - d_an + self.margin).mean()