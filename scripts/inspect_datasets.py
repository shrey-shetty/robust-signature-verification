"""
Dataset inspection script v2 - signature verification project.
Handles both flat folders (CEDAR-style) and nested per-writer folders
(BHSig260-style). Run:  python inspect_datasets.py

Edit BASE below, then run. Requires: pip install pillow
"""

import os
import re
import random
from collections import defaultdict
from PIL import Image

# ----------------------------------------------------------------------
# EDIT THIS to your data folder (the one containing the 4 dataset folders)
# ----------------------------------------------------------------------
BASE = r"C:\Users\nehas\Downloads\new\data\raw"

DATASET_DIRS = {
    "professor_dataset": os.path.join(BASE, "signature_verification"),
    "cedar":             os.path.join(BASE, "CEDAR"),
    "bhsig_hindi":       os.path.join(BASE, "BHSig260-Hindi"),
    "bhsig_bengali":     os.path.join(BASE, "BHSig260-Bengali"),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DIMENSION_SAMPLE_SIZE = 150  # images opened per dataset for size check

# pattern A: original_12_3.png / forgeries_1000_5.jpg  (CEDAR-style)
PAT_A = re.compile(r"^([A-Za-z]+)_(\d+)_(\d+)\.\w+$")
# pattern B: B-S-001-G-01.tif / H-S-160-F-24.tif  (common BHSig260-style)
PAT_B = re.compile(r"^([A-Z])-S-(\d+)-([GF])-(\d+)\.\w+$", re.IGNORECASE)


def collect_images(root):
    """Recursively collect (relative_dir, filename) for all images."""
    items = []
    for dirpath, _, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                items.append((rel, f))
    return items


def inspect(name, root):
    print("=" * 70)
    print(f"DATASET: {name}")
    print(f"PATH:    {root}")
    if not os.path.isdir(root):
        print("  !! Folder not found - fix the path\n")
        return

    items = collect_images(root)
    print(f"  Total image files (recursive): {len(items)}")
    if not items:
        # maybe zips not extracted, or unexpected extensions
        all_files = []
        for dirpath, _, filenames in os.walk(root):
            all_files.extend(filenames)
        print(f"  (Found {len(all_files)} non-image files; "
              f"examples: {all_files[:5]})\n")
        return

    # directory structure summary
    subdirs = sorted({rel for rel, _ in items if rel != "."})
    if subdirs:
        print(f"  Images live in {len(subdirs)} subfolder(s). "
              f"Examples: {subdirs[:5]}")
    else:
        print("  Flat structure: all images directly in the folder.")

    # extension counts
    ext_counts = defaultdict(int)
    for _, f in items:
        ext_counts[os.path.splitext(f)[1].lower()] += 1
    print(f"  Extensions: {dict(ext_counts)}")

    # try to parse naming conventions
    writers_A = defaultdict(lambda: defaultdict(int))  # prefix -> writer -> n
    writers_B = defaultdict(lambda: defaultdict(int))  # G/F -> writer -> n
    unmatched = []
    for rel, f in items:
        mA = PAT_A.match(f)
        mB = PAT_B.match(f)
        if mA:
            prefix, wid, _ = mA.group(1), int(mA.group(2)), mA.group(3)
            writers_A[prefix.lower()][wid] += 1
        elif mB:
            gf, wid = mB.group(3).upper(), int(mB.group(2))
            writers_B[gf][wid] += 1
        else:
            unmatched.append(os.path.join(rel, f))

    for prefix, wmap in writers_A.items():
        counts = list(wmap.values())
        print(f"  [pattern prefix_writer_sample] '{prefix}': "
              f"{len(wmap)} writers, IDs {min(wmap)}..{max(wmap)}, "
              f"samples/writer min={min(counts)} max={max(counts)}")
    label = {"G": "genuine", "F": "forged"}
    for gf, wmap in writers_B.items():
        counts = list(wmap.values())
        print(f"  [pattern X-S-writer-G/F-n] {label.get(gf, gf)}: "
              f"{len(wmap)} writers, IDs {min(wmap)}..{max(wmap)}, "
              f"samples/writer min={min(counts)} max={max(counts)}")
    if unmatched:
        print(f"  !! {len(unmatched)} files match neither known pattern. "
              f"Examples: {unmatched[:5]}")

    # dimensions + color mode on random sample
    sample = random.sample(items, min(DIMENSION_SAMPLE_SIZE, len(items)))
    dims, modes, errors = defaultdict(int), defaultdict(int), 0
    for rel, f in sample:
        try:
            with Image.open(os.path.join(root, rel, f)) as img:
                dims[img.size] += 1
                modes[img.mode] += 1
        except Exception:
            errors += 1
    print(f"  Dimension check on {len(sample)} random images "
          f"({len(dims)} distinct sizes). Top 5:")
    for (w, h), c in sorted(dims.items(), key=lambda x: -x[1])[:5]:
        print(f"     {w}x{h}: {c}")
    print(f"  Color modes: {dict(modes)}")
    if errors:
        print(f"  !! {errors} images failed to open")
    print()


if __name__ == "__main__":
    random.seed(42)
    for name, path in DATASET_DIRS.items():
        inspect(name, path)
    print("Done. Paste this full output back to Claude.")