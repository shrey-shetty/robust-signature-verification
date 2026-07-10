"""Biometric verification metrics from embedding distances.

Conventions:
  - distances: Euclidean distance between the two embeddings of a pair
    (small = model believes same writer).
  - labels: 1.0 = genuine pair (should be ACCEPTED), 0.0 = negative pair
    (forgery or different writer; should be REJECTED). Matches
    sigver.data.pairs.
  - A pair is accepted when distance <= threshold.

Metrics:
  - FAR (False Acceptance Rate): fraction of negative pairs accepted.
  - FRR (False Rejection Rate): fraction of genuine pairs rejected.
  - EER (Equal Error Rate): the error rate at the threshold where
    FAR = FRR (computed as their crossing point over a threshold sweep).
  - ROC-AUC: computed via the rank statistic (equivalent to the
    Mann-Whitney U), no sklearn dependency required.
"""

from __future__ import annotations

import numpy as np


def far_frr_curve(distances: np.ndarray, labels: np.ndarray,
                  n_thresholds: int = 512):
    """Sweep thresholds; return (thresholds, FAR array, FRR array)."""
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos = distances[labels == 1.0]   # genuine pairs
    neg = distances[labels == 0.0]   # negatives
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both genuine and negative pairs to compute FAR/FRR")

    thresholds = np.linspace(distances.min(), distances.max(), n_thresholds)
    # accept when distance <= t
    far = np.array([(neg <= t).mean() for t in thresholds])
    frr = np.array([(pos > t).mean() for t in thresholds])
    return thresholds, far, frr


def compute_eer(distances: np.ndarray, labels: np.ndarray):
    """Return (eer, threshold) at the FAR/FRR crossing point."""
    thresholds, far, frr = far_frr_curve(distances, labels)
    idx = int(np.argmin(np.abs(far - frr)))
    eer = (far[idx] + frr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def roc_auc(distances: np.ndarray, labels: np.ndarray) -> float:
    """AUC via rank statistic. Score = -distance (higher = more genuine)."""
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    scores = -distances
    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both classes for AUC")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    rank_pos = ranks[: len(pos)].sum()
    u = rank_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def far_frr_at_threshold(distances: np.ndarray, labels: np.ndarray,
                         threshold: float):
    """FAR and FRR at one fixed threshold (e.g. chosen on validation)."""
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos = distances[labels == 1.0]
    neg = distances[labels == 0.0]
    far = float((neg <= threshold).mean()) if len(neg) else float("nan")
    frr = float((pos > threshold).mean()) if len(pos) else float("nan")
    return far, frr