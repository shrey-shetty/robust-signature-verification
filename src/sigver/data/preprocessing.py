"""Signature image preprocessing.

Pipeline (per image), following Hafemann et al. (2017) with adaptations
for the heterogeneous resolutions across our five datasets:

  1. Convert to grayscale (institutional images are RGB; others are 'L').
  2. Estimate background with Otsu's threshold; set background pixels to
     white, keep foreground (ink) in grayscale.
  3. Invert intensities so background = 0 and ink is bright. Networks
     train better when the informative pixels are the nonzero ones.
  4. Tight-crop to the bounding box of the ink.
  5. Pad to the target aspect ratio (centered), then resize to the
     target size (default H=150, W=220, as in Hafemann et al.).

Output arrays are float32 in [0, 1], shape (H, W).

Usage as a module:
    from sigver.data.preprocessing import preprocess_image
    arr = preprocess_image("path/to/signature.png")

Notes:
  - Otsu is implemented directly on the grayscale histogram (NumPy only)
    to avoid an extra dependency on scikit-image.
  - If an image is blank (no ink found after thresholding), a ValueError
    is raised rather than silently returning an empty array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

TARGET_HEIGHT = 150
TARGET_WIDTH = 220


def otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu's threshold for a uint8 grayscale image.

    Returns the threshold t such that pixels <= t are considered
    foreground (ink is dark in the raw scans).
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size

    sum_all = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    best_t, best_var = 0, -1.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_t = t
    return best_t


def preprocess_array(gray: np.ndarray,
                     target_h: int = TARGET_HEIGHT,
                     target_w: int = TARGET_WIDTH) -> np.ndarray:
    """Run steps 2-5 of the pipeline on a uint8 grayscale array."""
    if gray.dtype != np.uint8:
        raise ValueError(f"expected uint8 grayscale array, got {gray.dtype}")

    # 2. Background removal: ink is dark, background is light.
    t = otsu_threshold(gray)
    ink_mask = gray <= t

    if not ink_mask.any():
        raise ValueError("no ink found after Otsu thresholding (blank image?)")

    # 3. Invert: background -> 0, ink stays grayscale (bright).
    inverted = 255 - gray.astype(np.int16)
    inverted[~ink_mask] = 0
    inverted = inverted.astype(np.uint8)

    # 4. Tight crop to ink bounding box.
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    cropped = inverted[r0:r1, c0:c1]

    # 5. Pad (centered) to target aspect ratio, then resize.
    h, w = cropped.shape
    target_ratio = target_w / target_h
    ratio = w / h
    if ratio > target_ratio:
        # too wide -> pad height
        new_h = int(round(w / target_ratio))
        pad = new_h - h
        top, bottom = pad // 2, pad - pad // 2
        padded = np.pad(cropped, ((top, bottom), (0, 0)))
    else:
        # too tall -> pad width
        new_w = int(round(h * target_ratio))
        pad = new_w - w
        left, right = pad // 2, pad - pad // 2
        padded = np.pad(cropped, ((0, 0), (left, right)))

    resized = Image.fromarray(padded).resize(
        (target_w, target_h), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def preprocess_image(path: str | Path,
                     target_h: int = TARGET_HEIGHT,
                     target_w: int = TARGET_WIDTH) -> np.ndarray:
    """Full pipeline: load from disk -> float32 (H, W) in [0, 1]."""
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return preprocess_array(gray, target_h=target_h, target_w=target_w)