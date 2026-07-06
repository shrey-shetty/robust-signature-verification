import numpy as np

from sigver.evaluation.metrics import eer_from_scores


def test_eer_from_scores_returns_mean_for_placeholder_impl():
    scores = np.array([0.2, 0.8])
    labels = np.array([0, 1])
    assert eer_from_scores(scores, labels) == np.mean(scores)
