"""Verify institutional vs. GPDS Synthetic 4000 dataset independence.

The institutional ``signature_verification`` corpus (full_org/full_forg) and
GPDS Synthetic 4000 both use writer IDs 1..4000. This script checks that
this is a *coincidence of ID numbering*, not accidental data leakage: the
two corpora should share writer identities (same ID range) while containing
zero overlapping image samples.

Checks:
  1. Exact duplicates — MD5 every image in both datasets; any matching hash
     is a byte-identical file shared between corpora (should be zero).
  2. Near duplicates — perceptual hash (imagehash.phash, hash_size=16) per
     image; within each shared writer ID, flag institutional-vs-GPDS pairs
     with Hamming distance <= 4 (near-identical images, e.g. re-encoded or
     lightly re-scanned copies).
  3. Visual identity spot-check — for 10 random writer IDs, a montage of
     GPDS genuine (top) vs. institutional genuine (bottom) samples, to
     confirm "writer 42 in GPDS" and "writer 42 in institutional" are
     plausibly the same physical signer style (or, if the IDs are simply
     independent 1..4000 counters, at least not contradictory).

NOT checked: whether the two datasets are the *same underlying population*
in a legal/provenance sense — only content-level overlap.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\verify_inst_gpds_overlap.py
"""

from __future__ import annotations

import hashlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imagehash  # noqa: E402
from sigver.data.catalog import DATASET_SPECS, build_catalog  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
TABLE_DIR = PROJECT_ROOT / "report" / "dataset_checks"
FIG_DIR = PROJECT_ROOT / "report" / "figures" / "eda"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
HASH_SIZE = 16
HAMMING_THRESHOLD = 4
N_MONTAGE_WRITERS = 10
PROGRESS_EVERY = 20_000

DATASETS = ("institutional", "gpds_synthetic_4000")


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: str) -> Optional[np.ndarray]:
    """Return a flat bool array (256 bits for hash_size=16), or None on failure."""
    try:
        with Image.open(path) as im:
            hsh = imagehash.phash(im, hash_size=HASH_SIZE)
        return hsh.hash.flatten()
    except Exception:
        return None


def parallel_md5(paths: list[str], label: str) -> dict[str, str]:
    out: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        for path, digest in zip(paths, ex.map(md5_file, paths, chunksize=200)):
            out[path] = digest
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"    [{label}] md5 {done:,}/{len(paths):,}")
    return out


def parallel_phash(paths: list[str], label: str) -> dict[str, Optional[np.ndarray]]:
    out: dict[str, Optional[np.ndarray]] = {}
    done = 0
    with ProcessPoolExecutor() as ex:
        for path, hsh in zip(paths, ex.map(compute_phash, paths, chunksize=200)):
            out[path] = hsh
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"    [{label}] phash {done:,}/{len(paths):,}")
    return out


def load_gray(path: str) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def plot_identity_montage(
    gpds_paths: list[str], inst_paths: list[str], writer_id: str
) -> Path:
    """4 GPDS genuine (top) vs. 4 institutional genuine (bottom) for one writer."""
    n = 4
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 4.2))
    for row_i, (paths, tag, color) in enumerate(
        [(gpds_paths, "gpds genuine", "#2a9d8f"),
         (inst_paths, "institutional genuine", "#264653")]
    ):
        for col in range(n):
            ax = axes[row_i, col]
            if col < len(paths):
                img = load_gray(paths[col])
                if img is not None:
                    ax.imshow(img, cmap="gray")
                ax.set_title(tag, fontsize=8, color=color)
            ax.axis("off")
    fig.suptitle(f"writer {writer_id}: GPDS genuine (top) vs. institutional "
                 f"genuine (bottom)", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / f"inst_gpds_identity_check_{writer_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    specs = tuple(s for s in DATASET_SPECS if s.name in DATASETS)
    catalog = build_catalog(RAW_ROOT, specs=specs)
    df = catalog.dropna(subset=["label"]).copy()
    df["writer_id"] = df["writer_id"].astype(str)

    inst = df[df["dataset"] == "institutional"].reset_index(drop=True)
    gpds = df[df["dataset"] == "gpds_synthetic_4000"].reset_index(drop=True)

    print(f"institutional      : {len(inst):,} images, "
          f"{inst['writer_id'].nunique():,} writers")
    print(f"gpds_synthetic_4000: {len(gpds):,} images, "
          f"{gpds['writer_id'].nunique():,} writers")

    inst_ids = set(inst["writer_id"])
    gpds_ids = set(gpds["writer_id"])
    ids_equal = inst_ids == gpds_ids
    shared_ids = inst_ids & gpds_ids
    print(f"Writer ID sets equal: {ids_equal} "
          f"(shared: {len(shared_ids):,}, "
          f"institutional-only: {len(inst_ids - gpds_ids):,}, "
          f"gpds-only: {len(gpds_ids - inst_ids):,})")

    # --- 1. Exact duplicates via MD5 -------------------------------------
    print("\n[1/3] MD5 hashing institutional images...")
    inst_md5 = parallel_md5(inst["path"].tolist(), "institutional")
    print("[1/3] MD5 hashing GPDS images...")
    gpds_md5 = parallel_md5(gpds["path"].tolist(), "gpds")

    inst["md5"] = inst["path"].map(inst_md5)
    gpds["md5"] = gpds["path"].map(gpds_md5)

    md5_to_inst_rows: dict[str, list[tuple[str, str]]] = {}
    for _, row in inst.iterrows():
        md5_to_inst_rows.setdefault(row["md5"], []).append(
            (row["writer_id"], row["path"])
        )

    exact_dupes: list[dict] = []
    for _, row in gpds.iterrows():
        matches = md5_to_inst_rows.get(row["md5"])
        if not matches:
            continue
        for inst_writer, inst_path in matches:
            exact_dupes.append({
                "writer_id": inst_writer,
                "inst_path": inst_path,
                "gpds_path": row["path"],
            })

    exact_dupes_df = pd.DataFrame(
        exact_dupes, columns=["writer_id", "inst_path", "gpds_path"]
    )
    exact_out = TABLE_DIR / "eda_inst_gpds_exact_duplicates.csv"
    exact_dupes_df.to_csv(exact_out, index=False)
    print(f"Exact duplicate hashes (institutional <-> gpds): {len(exact_dupes_df):,}")
    print(f"  written -> {exact_out.relative_to(PROJECT_ROOT)}")

    # --- 2. Near duplicates via perceptual hash, within shared writers ---
    print("\n[2/3] Perceptual hashing (phash, hash_size=16) institutional images...")
    inst_shared = inst[inst["writer_id"].isin(shared_ids)]
    gpds_shared = gpds[gpds["writer_id"].isin(shared_ids)]
    inst_phash = parallel_phash(inst_shared["path"].tolist(), "institutional")
    print("[2/3] Perceptual hashing GPDS images...")
    gpds_phash = parallel_phash(gpds_shared["path"].tolist(), "gpds")

    near_dupes: list[dict] = []
    n_writers_checked = 0
    for writer_id in sorted(shared_ids, key=lambda w: (len(w), w)):
        i_paths = inst_shared.loc[
            inst_shared["writer_id"] == writer_id, "path"
        ].tolist()
        g_paths = gpds_shared.loc[
            gpds_shared["writer_id"] == writer_id, "path"
        ].tolist()
        i_hashes = [inst_phash[p] for p in i_paths if inst_phash[p] is not None]
        i_valid_paths = [p for p in i_paths if inst_phash[p] is not None]
        g_hashes = [gpds_phash[p] for p in g_paths if gpds_phash[p] is not None]
        g_valid_paths = [p for p in g_paths if gpds_phash[p] is not None]
        if not i_hashes or not g_hashes:
            continue
        n_writers_checked += 1

        i_mat = np.stack(i_hashes)  # (n_inst, 256) bool
        g_mat = np.stack(g_hashes)  # (n_gpds, 256) bool
        # Pairwise Hamming distance via broadcasting.
        dist = np.count_nonzero(
            i_mat[:, None, :] != g_mat[None, :, :], axis=2
        )  # (n_inst, n_gpds)
        flagged = np.argwhere(dist <= HAMMING_THRESHOLD)
        for i_idx, g_idx in flagged:
            near_dupes.append({
                "writer_id": writer_id,
                "inst_path": i_valid_paths[i_idx],
                "gpds_path": g_valid_paths[g_idx],
                "hamming": int(dist[i_idx, g_idx]),
            })

    near_dupes_df = pd.DataFrame(
        near_dupes, columns=["writer_id", "inst_path", "gpds_path", "hamming"]
    )
    near_out = TABLE_DIR / "eda_inst_gpds_near_duplicates.csv"
    near_dupes_df.to_csv(near_out, index=False)
    print(f"Writers checked for near-duplicates: {n_writers_checked:,}")
    print(f"Near-duplicate flags (Hamming <= {HAMMING_THRESHOLD}): "
          f"{len(near_dupes_df):,}")
    print(f"  written -> {near_out.relative_to(PROJECT_ROOT)}")

    # --- 3. Visual identity spot-check montages ---------------------------
    print(f"\n[3/3] Saving {N_MONTAGE_WRITERS} identity-check montages...")
    montage_candidates = sorted(shared_ids, key=lambda w: (len(w), w))
    sample_ids = random.Random(RNG_SEED).sample(
        montage_candidates, min(N_MONTAGE_WRITERS, len(montage_candidates))
    )
    for writer_id in sample_ids:
        g_gen = gpds[
            (gpds["writer_id"] == writer_id) & (gpds["label"] == "genuine")
        ]["path"].tolist()
        i_gen = inst[
            (inst["writer_id"] == writer_id) & (inst["label"] == "genuine")
        ]["path"].tolist()
        if not g_gen or not i_gen:
            print(f"  [SKIP] writer {writer_id}: missing genuine samples "
                  f"(gpds={len(g_gen)}, inst={len(i_gen)})")
            continue
        rng = random.Random(RNG_SEED)
        g_sample = rng.sample(g_gen, min(4, len(g_gen)))
        i_sample = rng.sample(i_gen, min(4, len(i_gen)))
        out = plot_identity_montage(g_sample, i_sample, writer_id)
        print(f"  saved -> {out.relative_to(PROJECT_ROOT)}")

    # --- Summary ------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"institutional images       : {len(inst):,} "
          f"({inst['writer_id'].nunique():,} writers)")
    print(f"gpds_synthetic_4000 images : {len(gpds):,} "
          f"({gpds['writer_id'].nunique():,} writers)")
    print(f"Writer ID sets equal       : {ids_equal}")
    print(f"Exact duplicates (MD5)     : {len(exact_dupes_df):,}")
    print(f"Near-duplicate flags       : {len(near_dupes_df):,} "
          f"(Hamming <= {HAMMING_THRESHOLD}, hash_size={HASH_SIZE})")

    ok = ids_equal and len(exact_dupes_df) == 0
    print("\nResult: CLEAN — shared writer IDs, no overlapping samples" if ok
          else "\nResult: REVIEW NEEDED — see tables above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
