"""Signature image preprocessing.

Pipeline (per image), following Hafemann et al. (2017) with adaptations
for the heterogeneous resolutions across our five datasets:

  1. Convert to grayscale (institutional images are RGB; others are 'L').
  2. Estimate background with Otsu's threshold; set background pixels to
     white, keep foreground (ink) in grayscale.
  3. Binarize: background = 0, ink = 255. Grayscale ink intensity is
     deliberately discarded here, before the tight crop — CEDAR carries
     a class-conditional brightness artifact that acts as a shortcut
     feature if preserved (see preproc_cedar_shortcut_eer.csv).
  4. Tight-crop to the bounding box of the ink.
  5. Pad to the target aspect ratio (centered), then resize to the
     target size (default H=150, W=220, as in Hafemann et al.) using
     bilinear interpolation for accurate sub-pixel mask geometry, then
     re-threshold the resized array back to strict binary. Resizing a
     0/255 mask with bilinear interpolation reintroduces intermediate
     gray values at stroke edges (anti-aliasing); left alone, that
     residual gradient is enough to reconstruct a per-image mean
     intensity that still correlates with the class label (measured
     EER ~0.36 on CEDAR even with step-3 binarization applied). The
     post-resize re-threshold removes it, restoring the "ink axis
     removed by construction" guarantee for the array this function
     actually returns.

Output arrays are float32, shape (H, W), values strictly in {0.0, 1.0}
(background=0.0, ink=1.0) — binary end to end, not merely binary before
the resize step. No per-pixel grayscale statistic of the output (mean,
std, ...) can carry ink-intensity information; only stroke shape/geometry
survives. This is PREPROCESSING_VERSION 2 (v1 kept ink in grayscale;
v1.5 — the CEDAR-freeze-diagnostic commit — binarized before the crop
but not after the resize, leaving the edge-antialiasing leak above).

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

# Bumped whenever preprocess_array's output contract changes (dtype,
# value range, or the pixels it produces for the same input). Consumers
# that persist preprocessed arrays to disk should fold this into their
# cache key/filename so a stale on-disk cache from an older version
# can never be silently reused.
PREPROCESSING_VERSION = 2


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

    # 3. Binarize (pass 1 of 2, pre-crop): background -> 0, ink -> 255.
    # Ink is deliberately NOT kept in grayscale: CEDAR scans carry a
    # class-conditional brightness artifact (genuine vs forged mean ink
    # intensity differs), which survives grayscale-preserving
    # preprocessing and is exploitable as a shortcut (see
    # preproc_cedar_shortcut_eer.csv). This pass alone is not sufficient
    # -- the resize below reintroduces gray values at edges -- see the
    # second binarization pass after resize.
    inverted = np.zeros_like(gray)
    inverted[ink_mask] = 255

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

    # Bilinear (not nearest-neighbor) so the downsampled/upsampled mask
    # keeps accurate sub-pixel stroke geometry -- nearest-neighbor would
    # alias thin strokes into jagged or broken lines. This is the step
    # that reintroduces intermediate gray values at ink edges.
    resized = Image.fromarray(padded).resize(
        (target_w, target_h), Image.Resampling.BILINEAR
    )
    resized_arr = np.asarray(resized, dtype=np.uint8)

    # Binarize (pass 2 of 2, post-resize): collapse the anti-aliased
    # edges from the resize back to strict {0, 255} at the midpoint
    # threshold. Without this pass the returned array is only "mostly"
    # binary, and the residual edge gradient alone is enough to recover
    # a per-image mean-intensity statistic that correlates with the
    # class label (the CEDAR shortcut this binarization exists to kill).
    binary = (resized_arr > 127).astype(np.float32)
    return binary


def preprocess_image(path: str | Path,
                     target_h: int = TARGET_HEIGHT,
                     target_w: int = TARGET_WIDTH) -> np.ndarray:
    """Full pipeline: load from disk -> float32 (H, W), values in {0.0, 1.0}."""
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return preprocess_array(gray, target_h=target_h, target_w=target_w)
