"""Unit tests for sigver.evaluation.metrics.

Convention under test (see module docstring of metrics.py):
  - distances: Euclidean distance between a pair's embeddings.
  - labels: 1.0 = genuine pair (accept), 0.0 = negative pair (reject).
  - a pair is accepted when distance <= threshold.

Each expected value below is derived by hand from that convention, not
from running the implementation, so these tests can fail the code rather
than merely restate it.
"""

import numpy as np
import pytest

from sigver.evaluation.metrics import (
    compute_eer,
    far_frr_at_threshold,
    far_frr_curve,
    roc_auc,
)


# --- perfectly separable case -------------------------------------------------
# Genuine pairs all closer than every negative pair, so a threshold exists
# that makes both error rates zero.

SEPARABLE_DISTANCES = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
SEPARABLE_LABELS = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def test_eer_is_zero_when_classes_are_perfectly_separable():
    eer, threshold = compute_eer(SEPARABLE_DISTANCES, SEPARABLE_LABELS)
    assert eer == 0.0
    # any threshold in [0.3, 0.8) separates the two groups
    assert 0.3 <= threshold < 0.8


def test_auc_is_one_when_perfectly_separable():
    assert roc_auc(SEPARABLE_DISTANCES, SEPARABLE_LABELS) == pytest.approx(1.0)


def test_auc_is_zero_when_ranking_is_exactly_inverted():
    # same distances, labels flipped: every genuine pair is now the far one
    inverted = 1.0 - SEPARABLE_LABELS
    assert roc_auc(SEPARABLE_DISTANCES, inverted) == pytest.approx(0.0)


# --- chance-level case --------------------------------------------------------
# Genuine and negative pairs drawn from an identical set of distances: the
# model carries no information, so FAR and FRR cross at 0.5.

def test_eer_is_one_half_when_distributions_are_identical():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    distances = np.concatenate([values, values])
    labels = np.concatenate([np.ones(4), np.zeros(4)])
    eer, _ = compute_eer(distances, labels)
    assert eer == pytest.approx(0.5)


def test_auc_is_one_half_when_distributions_are_identical():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    distances = np.concatenate([values, values])
    labels = np.concatenate([np.ones(4), np.zeros(4)])
    assert roc_auc(distances, labels) == pytest.approx(0.5)


# --- fixed-threshold error rates ---------------------------------------------

def test_far_frr_at_threshold_matches_hand_count():
    # genuine: 0.1 (accepted), 0.5 (rejected -> 1 of 2 false rejections)
    # negative: 0.4 (accepted -> 1 of 2 false acceptances), 0.9 (rejected)
    distances = np.array([0.1, 0.5, 0.4, 0.9])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    far, frr = far_frr_at_threshold(distances, labels, threshold=0.45)
    assert far == pytest.approx(0.5)
    assert frr == pytest.approx(0.5)


def test_far_is_one_and_frr_is_zero_at_a_permissive_threshold():
    far, frr = far_frr_at_threshold(
        SEPARABLE_DISTANCES, SEPARABLE_LABELS, threshold=99.0
    )
    assert far == pytest.approx(1.0)
    assert frr == pytest.approx(0.0)


# --- curve shape --------------------------------------------------------------

def test_far_frr_curve_is_monotone_in_the_expected_directions():
    thresholds, far, frr = far_frr_curve(SEPARABLE_DISTANCES, SEPARABLE_LABELS)
    assert len(thresholds) == len(far) == len(frr)
    # raising the threshold accepts more, so FAR rises and FRR falls
    assert np.all(np.diff(far) >= 0)
    assert np.all(np.diff(frr) <= 0)


# --- degenerate input ---------------------------------------------------------

def test_curve_rejects_single_class_input():
    distances = np.array([0.1, 0.2, 0.3])
    labels = np.ones(3)
    with pytest.raises(ValueError):
        far_frr_curve(distances, labels)


def test_auc_rejects_single_class_input():
    distances = np.array([0.1, 0.2, 0.3])
    labels = np.zeros(3)
    with pytest.raises(ValueError):
        roc_auc(distances, labels)
