"""Generate writer-independent train/val/test splits for all datasets.

Splits are made BY WRITER (not by image): every writer's signatures land
entirely in one split. This is the core requirement of the
writer-independent protocol (test writers never seen in training, and all
model/hyperparameter decisions made on validation writers only).

Ratios: 70% train / 10% val / 20% test (by writer count), fixed seed.
Output: one CSV per dataset in data/splits/ with columns
    writer_id, split
plus a summary file recording ratios, seed, and writer counts, so the
exact split configuration is documented for the report.

Ground-truth rule: writer identity comes from the FOLDER name (or, for
the flat institutional layout, the ID embedded in filenames — its only
available source). This makes the BHSig260-Hindi writer-123 filename
anomaly harmless, per the documented decision.

The generated CSVs are meant to be committed to git for reproducibility.
Re-running with the same seed and same data reproduces identical splits;
the script refuses to overwrite existing splits unless --force is given,
to protect against silently invalidating results computed on old splits.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\make_splits.py
    .\\.venv\\Scripts\\python.exe scripts\\make_splits.py --force
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

SEED = 42
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


def writers_from_folders(root: Path) -> list[int]:
    """Layout A: writer IDs are numeric folder names under root."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    ids = []
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            ids.append(int(p.name))
    return sorted(ids)


def writers_from_filenames(folder: Path, pattern: re.Pattern) -> list[int]:
    """Layout B (flat): writer IDs embedded in filenames."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    ids = set()
    for f in folder.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                ids.add(int(m.group("writer")))
    return sorted(ids)


def collect_writers() -> dict[str, list[int]]:
    return {
        "gpds_synthetic_4000": writers_from_folders(
            DATA_RAW / "SignatureGPDSSyntheticOffLine4000" / "firmasSINTESISmanuscritas"
        ),
        "bhsig260_bengali": writers_from_folders(DATA_RAW / "BHSig260-Bengali"),
        "bhsig260_hindi": writers_from_folders(DATA_RAW / "BHSig260-Hindi"),
        "cedar": writers_from_folders(DATA_RAW / "CEDAR"),
        "institutional": writers_from_filenames(
            DATA_RAW / "signature_verification" / "full_org",
            re.compile(r"^original_(?P<writer>\d+)_\d+\.jpg$", re.IGNORECASE),
        ),
    }


def split_writers(writers: list[int], seed: int) -> dict[int, str]:
    """Shuffle deterministically, then cut into train/val/test."""
    rng = random.Random(seed)
    shuffled = writers.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * RATIOS["train"])
    n_val = round(n * RATIOS["val"])
    # test gets the remainder, so the three parts always sum to n

    assignment: dict[int, str] = {}
    for i, w in enumerate(shuffled):
        if i < n_train:
            assignment[w] = "train"
        elif i < n_train + n_val:
            assignment[w] = "val"
        else:
            assignment[w] = "test"
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing split files")
    args = ap.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(SPLITS_DIR.glob("split_*.csv"))
    if existing and not args.force:
        print("Split files already exist in data/splits/ — refusing to "
              "overwrite (results computed on old splits would become "
              "irreproducible). Re-run with --force to regenerate.")
        for f in existing:
            print(f"  exists: {f.relative_to(PROJECT_ROOT)}")
        return 1

    try:
        all_writers = collect_writers()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Check dataset paths (or adjust them in collect_writers).")
        return 1

    summary_lines = [
        f"seed: {SEED}",
        f"ratios: train={RATIOS['train']}, val={RATIOS['val']}, test={RATIOS['test']}",
        "note: splits are by writer (writer-independent protocol); "
        "writer identity from folder ID (BHSig260-Hindi writer 123 "
        "filename anomaly documented separately)",
        "",
    ]

    gpds_assignment = split_writers(all_writers["gpds_synthetic_4000"], SEED)

    for name, writers in all_writers.items():
        if name == "institutional":
            # Institutional shares writer identities with GPDS Synthetic
            # (confirmed by visual inspection of multiple writer pairs);
            # reuse the GPDS assignment to prevent writer leakage across
            # the two datasets in combined-training experiments.
            assignment = {w: gpds_assignment[w] for w in writers}
        elif name == "gpds_synthetic_4000":
            assignment = gpds_assignment
        else:
            assignment = split_writers(writers, SEED)
        counts = {"train": 0, "val": 0, "test": 0}
        out = SPLITS_DIR / f"split_{name}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["writer_id", "split"])
            for wid in sorted(assignment):
                w.writerow([wid, assignment[wid]])
                counts[assignment[wid]] += 1
        line = (f"{name}: {len(writers)} writers -> "
                f"{counts['train']} train / {counts['val']} val / "
                f"{counts['test']} test")
        print(line)
        print(f"  written: {out.relative_to(PROJECT_ROOT)}")
        summary_lines.append(line)

    summary_path = SPLITS_DIR / "SPLITS_INFO.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nSummary written: {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())