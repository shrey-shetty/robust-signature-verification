"""Pixel-level check of Hamming-0 phash collisions between institutional and
GPDS Synthetic 4000.

scripts/verify_inst_gpds_overlap.py flagged 41,220 institutional-vs-GPDS
image pairs (same writer ID) with perceptual-hash Hamming distance <= 4,
including 10,665 at distance 0. A distance-0 phash match with different
MD5s (already confirmed: 0 exact byte duplicates) could mean either:
  (a) the same underlying signature, re-encoded/re-scanned differently, or
  (b) two distinct signatures that happen to share the same coarse
      background/stroke-density structure that phash's low-frequency DCT
      is sensitive to (plausible for signature images: mostly blank
      background, thin dark strokes in a similar spatial region).

This script distinguishes (a) from (b) with actual pixel comparison, plus a
filename-index sanity check (does the institutional sample index within its
writer match the GPDS sample index?).

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\inspect_inst_gpds_hamming0.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "report" / "dataset_checks"
FIG_DIR = PROJECT_ROOT / "report" / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NEAR_DUP_CSV = TABLE_DIR / "eda_inst_gpds_near_duplicates.csv"
OUT_CSV = TABLE_DIR / "eda_inst_gpds_hamming0_pixelcheck.csv"

RNG_SEED = 42
SAMPLE_CAP = 500
NCC_THRESHOLD = 0.99
FRAC_DIFF_THRESHOLD = 0.01
N_FIGURES_PER_VERDICT = 5

_INST_RE = re.compile(r"^(original|forgeries)_(\d+)_(\d+)$", re.IGNORECASE)
_GPDS_RE = re.compile(r"^(c|cf)-(\d+)-(\d+)$", re.IGNORECASE)


def parse_inst_index(path: str) -> Optional[int]:
    """Extract the trailing sample index from original_N_k / forgeries_N_k."""
    m = _INST_RE.match(Path(path).stem)
    return int(m.group(3)) if m else None


def parse_gpds_index(path: str) -> Optional[int]:
    """Extract the trailing sample index from c-NNN-kk / cf-NNN-kk."""
    m = _GPDS_RE.match(Path(path).stem)
    return int(m.group(3)) if m else None


def load_gray(path: str) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def align_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize the larger image down to the smaller image's shape (INTER_AREA)."""
    if a.shape == b.shape:
        return a, b
    if a.size >= b.size:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, b


def pixel_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return (mean abs diff, fraction differing by >10 levels, NCC)."""
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    diff = np.abs(af - bf)
    mad = float(diff.mean())
    frac_diff_gt10 = float((diff > 10).mean())

    a_c = af - af.mean()
    b_c = bf - bf.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
    ncc = float((a_c * b_c).sum() / denom) if denom > 0 else 0.0
    return mad, frac_diff_gt10, ncc


def classify(ncc: float, frac_diff_gt10: float) -> str:
    if ncc > NCC_THRESHOLD and frac_diff_gt10 < FRAC_DIFF_THRESHOLD:
        return "same image (re-encoded)"
    return "distinct"


def plot_pair(row: pd.Series, n: int) -> Path:
    img_i = load_gray(row["inst_path"])
    img_g = load_gray(row["gpds_path"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    for ax, img, tag in [(axes[0], img_g, "gpds"), (axes[1], img_i, "institutional")]:
        if img is not None:
            ax.imshow(img, cmap="gray")
        ax.set_title(tag, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"writer {row['writer_id']} — {row['verdict']}\n"
        f"mad={row['mad']:.2f} frac_diff>10={row['frac_diff_gt10']:.3f} "
        f"ncc={row['ncc']:.4f}",
        fontsize=9,
    )
    fig.tight_layout()
    out = FIG_DIR / f"inst_gpds_hamming0_pair_{n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    if not NEAR_DUP_CSV.is_file():
        print(f"ERROR: {NEAR_DUP_CSV} not found. Run "
              f"scripts/verify_inst_gpds_overlap.py first.")
        return 1

    near = pd.read_csv(NEAR_DUP_CSV)
    ham0 = near[near["hamming"] == 0].copy()
    print(f"Hamming-0 pairs: {len(ham0):,}")
    print(f"Distinct writers affected: {ham0['writer_id'].nunique():,}")

    if len(ham0) > SAMPLE_CAP:
        sample = ham0.sample(SAMPLE_CAP, random_state=RNG_SEED).reset_index(drop=True)
        print(f"Sampling {SAMPLE_CAP} of {len(ham0):,} pairs (seed={RNG_SEED})")
    else:
        sample = ham0.reset_index(drop=True)
        print(f"Checking all {len(sample)} pairs (below sample cap)")

    records: list[dict] = []
    for i, row in sample.iterrows():
        img_i = load_gray(row["inst_path"])
        img_g = load_gray(row["gpds_path"])
        if img_i is None or img_g is None:
            print(f"  [SKIP] unreadable image: inst={row['inst_path']} "
                  f"gpds={row['gpds_path']}")
            continue
        shape_inst, shape_gpds = img_i.shape, img_g.shape
        img_i_aligned, img_g_aligned = align_shapes(img_i, img_g)
        mad, frac_diff_gt10, ncc = pixel_metrics(img_i_aligned, img_g_aligned)
        verdict = classify(ncc, frac_diff_gt10)
        records.append({
            "writer_id": row["writer_id"],
            "inst_path": row["inst_path"],
            "gpds_path": row["gpds_path"],
            "shape_inst": f"{shape_inst[1]}x{shape_inst[0]}",
            "shape_gpds": f"{shape_gpds[1]}x{shape_gpds[0]}",
            "mad": round(mad, 4),
            "frac_diff_gt10": round(frac_diff_gt10, 4),
            "ncc": round(ncc, 4),
            "verdict": verdict,
        })
        if (i + 1) % 100 == 0:
            print(f"  checked {i + 1}/{len(sample)}")

    result = pd.DataFrame(records, columns=[
        "writer_id", "inst_path", "gpds_path", "shape_inst", "shape_gpds",
        "mad", "frac_diff_gt10", "ncc", "verdict",
    ])
    result.to_csv(OUT_CSV, index=False)
    print(f"\nWritten -> {OUT_CSV.relative_to(PROJECT_ROOT)}")

    n_checked = len(result)
    n_same = (result["verdict"] == "same image (re-encoded)").sum()
    n_distinct = (result["verdict"] == "distinct").sum()

    # --- Filename-index alignment check --------------------------------
    result["inst_index"] = result["inst_path"].map(parse_inst_index)
    result["gpds_index"] = result["gpds_path"].map(parse_gpds_index)
    both_parsed = result.dropna(subset=["inst_index", "gpds_index"])
    index_match = (both_parsed["inst_index"] == both_parsed["gpds_index"])
    print(f"\nFilename index alignment: {index_match.sum():,}/{len(both_parsed):,} "
          f"checked pairs have matching (original_N_k / c-NNN-k) sample index")

    # Cross-tab: index match vs. pixel verdict.
    if not both_parsed.empty:
        xtab = pd.crosstab(
            both_parsed["verdict"], index_match.rename("index_matches")
        )
        print(xtab.to_string())

    # --- Figures ----------------------------------------------------------
    print(f"\nSaving up to {N_FIGURES_PER_VERDICT} figures per verdict...")
    n = 0
    same_rows = result[result["verdict"] == "same image (re-encoded)"].head(
        N_FIGURES_PER_VERDICT
    )
    distinct_rows = result[result["verdict"] == "distinct"].head(
        N_FIGURES_PER_VERDICT
    )
    for _, row in pd.concat([same_rows, distinct_rows]).iterrows():
        out = plot_pair(row, n)
        print(f"  saved -> {out.relative_to(PROJECT_ROOT)}")
        n += 1
    if n == 0:
        print("  no pairs available to plot.")

    # --- Summary ------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"Pairs checked           : {n_checked:,}")
    if n_checked:
        print(f"Same image (re-encoded) : {n_same:,} ({100 * n_same / n_checked:.1f}%)")
        print(f"Distinct                : {n_distinct:,} "
              f"({100 * n_distinct / n_checked:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
