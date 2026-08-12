"""Per-image pixel-statistics extraction for the institutional dataset (EDA cache).

Computes ink-mask statistics from RAW images (preprocess_image() is never
called — this characterizes source data before the pipeline touches it).
Thresholding reuses sigver.data.preprocessing.otsu_threshold, the same
function the v2 pipeline uses, so EDA ink-mask detection agrees with what
training actually sees.

Per image: ink_fraction, ink bbox width/height/aspect ratio, mean_intensity
(whole image, raw grayscale), mean_ink_intensity (raw grayscale, ink pixels
only). Images with no ink after Otsu thresholding (blank) are skipped and
counted, not silently dropped.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\extract_institutional_pixel_stats.py
    .\\.venv\\Scripts\\python.exe scripts\\extract_institutional_pixel_stats.py --limit 500
    .\\.venv\\Scripts\\python.exe scripts\\extract_institutional_pixel_stats.py --out path.csv

With no --limit, the script first times a 500-image probe and extrapolates
the full-corpus runtime. If that extrapolation exceeds ~20 minutes, it
stops and reports the estimate instead of running to completion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sigver.data.datasets import list_samples  # noqa: E402
from sigver.data.preprocessing import otsu_threshold  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT = PROJECT_ROOT / "report" / "dataset_checks" / "eda_institutional_pixel_stats.csv"

PROBE_N = 500
TIME_BUDGET_S = 20 * 60


def process_one(sample) -> dict | None:
    """Return one pixel-stats record, or None if the image has no ink (blank)."""
    with Image.open(sample.path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)

    t = otsu_threshold(gray)
    ink_mask = gray <= t  # raw-domain convention: ink is dark (see preprocessing.py)
    if not ink_mask.any():
        return None

    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    bbox_h, bbox_w = int(r1 - r0), int(c1 - c0)

    return {
        "path": Path(sample.path).resolve().relative_to(DATA_ROOT).as_posix(),
        "writer_id": sample.writer_id,
        "label": "forgery" if sample.is_forgery else "genuine",
        "ink_fraction": float(ink_mask.mean()),
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_aspect_ratio": float(bbox_w / bbox_h) if bbox_h else float("nan"),
        "mean_intensity": float(gray.mean()),
        "mean_ink_intensity": float(gray[ink_mask].mean()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="institutional",
                    help="dataset name understood by sigver.data.datasets.list_samples")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N samples (deterministic order). "
                         "Default: no limit (full corpus, with a timing gate).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output CSV path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    samples = list_samples(args.dataset, split=None)
    print(f"{args.dataset}: {len(samples):,} samples total")

    records: list[dict] = []
    n_blank = 0
    n_unreadable = 0

    if args.limit is not None:
        todo = samples[: args.limit]
        print(f"--limit {args.limit}: processing first {len(todo):,} samples, no timing gate")
    else:
        probe = samples[:PROBE_N]
        t0 = time.perf_counter()
        for s in probe:
            try:
                rec = process_one(s)
            except (UnidentifiedImageError, OSError):
                n_unreadable += 1
                continue
            if rec is None:
                n_blank += 1
            else:
                records.append(rec)
        t1 = time.perf_counter()
        rate = (t1 - t0) / len(probe)
        est_total_s = rate * len(samples)
        print(f"timing probe: {len(probe)} images in {t1 - t0:.1f}s "
              f"({rate * 1000:.2f} ms/img) -> extrapolated full run "
              f"({len(samples):,} images): {est_total_s / 60:.1f} min")
        if est_total_s > TIME_BUDGET_S:
            print(f"STOPPING: extrapolated {est_total_s / 60:.1f} min exceeds the "
                  f"~{TIME_BUDGET_S / 60:.0f}-min local budget. Not running the full pass. "
                  f"Re-run with --limit for a partial cache, or move this to Kaggle.")
            return 1
        todo = samples[PROBE_N:]

    t_full0 = time.perf_counter()
    for s in todo:
        try:
            rec = process_one(s)
        except (UnidentifiedImageError, OSError):
            n_unreadable += 1
            continue
        if rec is None:
            n_blank += 1
        else:
            records.append(rec)
    t_full1 = time.perf_counter()

    stats = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out, index=False)

    print(f"processed {len(records):,} images with ink "
          f"({n_blank} blank, {n_unreadable} unreadable) in "
          f"{t_full1 - t_full0:.1f}s (this segment)")
    print(f"saved -> {args.out.relative_to(PROJECT_ROOT) if args.out.is_relative_to(PROJECT_ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
