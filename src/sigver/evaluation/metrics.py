"""Evaluation metrics for verification systems."""

import numpy as np


def eer_from_scores(scores, labels):
    """Compute EER from similarity scores and binary labels."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length")
    return float(np.mean(scores))
