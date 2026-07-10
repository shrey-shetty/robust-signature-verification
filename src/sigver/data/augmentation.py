"""Training-time augmentation for signature images.

Motivation (CEDAR shortcut audit, see preproc_cedar_shortcut_eer.csv):
after binarization removed the photometric artifact, a residual
class-conditional stroke-width difference remained (single-statistic
EER ~0.36 via mean intensity of the anti-aliased binary output). Its
attribution is uncertain (corr with raw ink darkness -0.212 => mostly
NOT the scanner artifact, plausibly behavioral), so stroke width is not
deleted in preprocessing. Instead it is decorrelated from the class
label at TRAINING time: each sample is randomly eroded / kept /
dilated, so global stroke width cannot function as a class cue, while
width information remains intact at inference.

Apply to TRAINING data only. Val/test stay deterministic.

Input/output contract: float32 array, shape (H, W), values in [0, 1]
(matches sigver.data.preprocessing output).
"""

from __future__ import annotations

import cv2
import numpy as np

# 3x3 cross (4-neighborhood): the gentlest structuring element cv2
# offers. Full 3x3 square erosion is too aggressive for the ~1-2 px
# strokes that survive the 150x220 resize.
_KERNEL_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


class MorphAugment:
    """Random stroke-width perturbation: erode / identity / dilate.

    p_erode + p_dilate must be <= 1; the remainder is identity.
    Erosion safety: if erosion removes more than `max_ink_loss` of the
    ink mass (thin strokes can vanish), the original image is returned
    instead — a destroyed signature teaches the model nothing.

    Deterministic given `seed` in single-process loading. With
    DataLoader num_workers > 0, each worker forks a copy of this RNG:
    pass a worker_init_fn that reseeds (not needed for the current
    single-process setup).
    """

    def __init__(self, p_erode: float = 0.25, p_dilate: float = 0.25,
                 max_ink_loss: float = 0.5, seed: int = 42):
        if p_erode + p_dilate > 1.0:
            raise ValueError("p_erode + p_dilate must be <= 1")
        self.p_erode = p_erode
        self.p_dilate = p_dilate
        self.max_ink_loss = max_ink_loss
        self.rng = np.random.default_rng(seed)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        u = self.rng.random()
        if u < self.p_erode:
            eroded = cv2.erode(arr, _KERNEL_CROSS, iterations=1)
            before = float(arr.sum())
            if before > 0 and float(eroded.sum()) / before < (1 - self.max_ink_loss):
                return arr  # erosion destroyed the stroke; skip
            return eroded
        if u < self.p_erode + self.p_dilate:
            return cv2.dilate(arr, _KERNEL_CROSS, iterations=1)
        return arr

    def __repr__(self) -> str:  # lands in config snapshots
        return (f"MorphAugment(p_erode={self.p_erode}, "
                f"p_dilate={self.p_dilate}, "
                f"max_ink_loss={self.max_ink_loss})")