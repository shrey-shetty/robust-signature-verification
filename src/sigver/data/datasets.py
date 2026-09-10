"""PyTorch datasets for signature verification.

Ties together:
  - the verified raw-data layouts (per-writer folders / flat org-forg),
  - the committed writer-independent split CSVs in data/splits/,
  - the preprocessing pipeline in sigver.data.preprocessing.

Ground-truth rule: writer identity comes from the FOLDER name for
per-writer layouts (this neutralizes the BHSig260-Hindi writer-123
filename anomaly), and from the filename for the flat institutional
layout (its only source of identity).

Main entry points:
    samples = list_samples("cedar", split="train")
    ds = SignatureDataset(samples)          # yields (image, writer_id, is_forgery)

`SignatureDataset` supports an optional in-memory cache (uint8) that is
practical for the small datasets (CEDAR: ~2.6k images ~= 90 MB cached)
and should be left off for GPDS/institutional scale.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sigver.data.preprocessing import preprocess_image, TARGET_HEIGHT, TARGET_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


@dataclass(frozen=True)
class Sample:
    path: Path
    writer_id: int
    is_forgery: bool  # False = genuine, True = skilled forgery


# ---------------------------------------------------------------------------
# Dataset layout specs (mirrors scripts/inspect_datasets.py, kept in sync
# manually; both were validated against the actual data).
#
# Built as functions of `raw_root` (default DATA_RAW) rather than baked-in
# module-level paths, so callers (e.g. train_baseline.py --raw-root on
# Kaggle, where data is mounted read-only under /kaggle/input/<dataset>/)
# can point at a different raw-data location without editing this file.
# ---------------------------------------------------------------------------


def _per_writer_specs(raw_root: Path) -> dict:
    return {
        "cedar": {
            "root": raw_root / "CEDAR",
            "genuine": re.compile(r"^original_\d+_\d+\.png$", re.IGNORECASE),
            "forgery": re.compile(r"^forgeries_\d+_\d+\.png$", re.IGNORECASE),
        },
        "bhsig260_bengali": {
            "root": raw_root / "BHSig260-Bengali",
            "genuine": re.compile(r"^B-S-\d+-G-\d+\.tif$", re.IGNORECASE),
            "forgery": re.compile(r"^B-S-\d+-F-\d+\.tif$", re.IGNORECASE),
        },
        "bhsig260_hindi": {
            "root": raw_root / "BHSig260-Hindi",
            "genuine": re.compile(r"^H-S-\d+-G-\d+\.tif$", re.IGNORECASE),
            "forgery": re.compile(r"^H-S-\d+-F-\d+\.tif$", re.IGNORECASE),
        },
        "gpds_synthetic_4000": {
            "root": raw_root / "SignatureGPDSSyntheticOffLine4000" / "firmasSINTESISmanuscritas",
            # order matters: test forgery ('cf-') before genuine ('c-')
            "genuine": re.compile(r"^c-\d+-\d+\.jpg$", re.IGNORECASE),
            "forgery": re.compile(r"^cf-\d+-\d+\.jpg$", re.IGNORECASE),
        },
    }


def _flat_specs(raw_root: Path) -> dict:
    return {
        "institutional": {
            "genuine_dir": raw_root / "signature_verification" / "full_org",
            "forgery_dir": raw_root / "signature_verification" / "full_forg",
            "genuine": re.compile(r"^original_(?P<writer>\d+)_\d+\.jpg$", re.IGNORECASE),
            "forgery": re.compile(r"^forgeries_(?P<writer>\d+)_\d+\.jpg$", re.IGNORECASE),
        },
    }


# Fixed dataset names (independent of raw_root) — used for validation/listing.
DATASET_NAMES = sorted(list(_per_writer_specs(DATA_RAW)) + list(_flat_specs(DATA_RAW)))


def load_split(dataset: str) -> dict[int, str]:
    """Read data/splits/split_<dataset>.csv -> {writer_id: 'train'|'val'|'test'}."""
    path = SPLITS_DIR / f"split_{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path} — run scripts/make_splits.py first."
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(r["writer_id"]): r["split"] for r in csv.DictReader(fh)}


def _list_per_writer(name: str, raw_root: Path) -> list[Sample]:
    spec = _per_writer_specs(raw_root)[name]
    root: Path = spec["root"]
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    samples: list[Sample] = []
    for wdir in root.iterdir():
        if not (wdir.is_dir() and re.fullmatch(r"\d+", wdir.name)):
            continue
        wid = int(wdir.name)  # folder ID is ground truth
        for f in wdir.iterdir():
            if not f.is_file():
                continue
            if spec["forgery"].match(f.name):
                samples.append(Sample(f, wid, True))
            elif spec["genuine"].match(f.name):
                samples.append(Sample(f, wid, False))
            # unrecognized files (e.g. GPDS ParamsUser*.mat) are skipped;
            # inspect_datasets.py is the tool that audits those.
    return samples


def _list_flat(name: str, raw_root: Path) -> list[Sample]:
    spec = _flat_specs(raw_root)[name]
    samples: list[Sample] = []
    for key, is_forgery in (("genuine_dir", False), ("forgery_dir", True)):
        folder: Path = spec[key]
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        pattern = spec["forgery" if is_forgery else "genuine"]
        for f in folder.iterdir():
            if f.is_file():
                m = pattern.match(f.name)
                if m:
                    samples.append(Sample(f, int(m.group("writer")), is_forgery))
    return samples


def list_samples(dataset: str, split: str | None = None,
                 raw_root: Path | None = None) -> list[Sample]:
    """Enumerate samples for a dataset, optionally filtered to one split.

    split: 'train', 'val', 'test', or None for all samples.
    raw_root: override for the raw-data root (default: DATA_RAW, i.e.
        <project_root>/data/raw). Use this to point at a read-only mount
        such as Kaggle's /kaggle/input/<dataset-name>/ without touching
        this module.
    """
    root = raw_root if raw_root is not None else DATA_RAW
    if dataset in _per_writer_specs(root):
        samples = _list_per_writer(dataset, root)
    elif dataset in _flat_specs(root):
        samples = _list_flat(dataset, root)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {DATASET_NAMES}")

    if split is not None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        assignment = load_split(dataset)
        samples = [s for s in samples if assignment.get(s.writer_id) == split]

    # deterministic order (path-sorted) so downstream shuffling is the only
    # source of randomness
    return sorted(samples, key=lambda s: str(s.path))


# ---------------------------------------------------------------------------
# Multi-dataset (RQ3): combining corpora requires namespacing writer_id so
# distinct datasets' writers never collide (pairs.py groups purely on the
# integer writer_id, with no dataset qualifier).
# ---------------------------------------------------------------------------

# Largest single corpus is 4,000 writers, so this step is comfortably
# collision-free between any two datasets' raw writer_id ranges.
DATASET_ID_OFFSET_STEP = 1_000_000


def build_dataset_offsets(dataset_names: list[str]) -> dict[str, int]:
    """Deterministic per-dataset writer-id namespace offset.

    dataset_index is the position of each name in the sorted order of the
    *given* dataset_names (not a fixed global catalog order), so the same
    requested set of datasets always yields the same mapping regardless of
    the order they were requested in.
    """
    ordered = sorted(set(dataset_names))
    return {name: i * DATASET_ID_OFFSET_STEP for i, name in enumerate(ordered)}


def resolve_namespaced_writer(namespaced_id: int, offsets: dict[str, int]) -> tuple[str, int]:
    """Invert build_dataset_offsets: namespaced writer_id -> (dataset, raw writer_id)."""
    for name, offset in sorted(offsets.items(), key=lambda kv: kv[1], reverse=True):
        if namespaced_id >= offset:
            return name, namespaced_id - offset
    raise ValueError(f"No dataset offset covers namespaced id {namespaced_id} "
                     f"(offsets: {offsets})")


def writer_counts_by_dataset(samples: list[Sample], offsets: dict[str, int]) -> dict[str, int]:
    """Count distinct writers per source dataset in a namespaced sample list."""
    writers_by_name: dict[str, set[int]] = {name: set() for name in offsets}
    for s in samples:
        name, raw_id = resolve_namespaced_writer(s.writer_id, offsets)
        writers_by_name[name].add(raw_id)
    return {name: len(writers_by_name[name]) for name in offsets}


def list_samples_multi(dataset_names: list[str], split: str | None = None,
                       raw_root: Path | None = None,
                       ) -> tuple[list[Sample], dict[str, int]]:
    """Concatenate list_samples(...) across multiple datasets with namespaced writer_ids.

    Each dataset keeps its own split CSV in data/splits/ — this does not
    invent a new split scheme, it just takes the union of each dataset's
    per-split writers. Samples are namespaced (see build_dataset_offsets)
    before concatenation so two datasets' raw writer_id ranges can never
    collide downstream in pairs.py's writer grouping.

    Returns (samples, offsets); offsets maps dataset name -> the integer
    added to that dataset's raw writer_id (invert with
    resolve_namespaced_writer to recover the source dataset + raw id).

    Raises ValueError if any requested dataset has zero writers in the
    requested split (a silent empty contribution would be worse than a
    loud failure here).
    """
    offsets = build_dataset_offsets(dataset_names)
    combined: list[Sample] = []
    for name in dataset_names:
        base = list_samples(name, split, raw_root=raw_root)
        if not base:
            raise ValueError(
                f"Dataset '{name}' has zero samples in split {split!r} — "
                f"refusing to silently combine an empty contribution."
            )
        offset = offsets[name]
        combined.extend(Sample(s.path, s.writer_id + offset, s.is_forgery) for s in base)
    return combined, offsets


class SignatureDataset(Dataset):
    """Yields (image_tensor, writer_id, is_forgery) triples.

    image_tensor: float32, shape (1, H, W), values in [0, 1]
    writer_id:    int64 tensor (scalar)
    is_forgery:   float32 tensor (scalar; 0.0 genuine, 1.0 forgery)

    cache_in_memory: preprocess every image once up front and keep it as
    uint8 in RAM. Sensible for CEDAR/BHSig scale; do NOT enable for
    GPDS/institutional (hundreds of thousands of images).

    transform: optional callable applied to the float32 [0, 1] array
    AFTER cache retrieval (or after on-the-fly preprocessing when caching
    is off), so the cache itself always stays clean/deterministic and
    every __getitem__ call gets fresh, independent randomness. By
    convention this is used for training-only augmentation (e.g.
    sigver.data.augmentation.MorphAugment); val/test datasets should be
    constructed with transform=None so evaluation reflects the raw
    preprocessing pipeline output.
    """

    def __init__(self, samples: list[Sample],
                 target_h: int = TARGET_HEIGHT,
                 target_w: int = TARGET_WIDTH,
                 cache_in_memory: bool = False,
                 transform=None):
        self.samples = samples
        self.target_h = target_h
        self.target_w = target_w
        self.transform = transform
        self._cache: list[np.ndarray] | None = None
        if cache_in_memory:
            self._cache = [
                (preprocess_image(s.path, target_h, target_w) * 255).astype(np.uint8)
                for s in samples
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if self._cache is not None:
            arr = self._cache[idx].astype(np.float32) / 255.0
        else:
            arr = preprocess_image(s.path, self.target_h, self.target_w)
        if self.transform is not None:
            arr = self.transform(arr)
        image = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        writer = torch.tensor(s.writer_id, dtype=torch.int64)
        label = torch.tensor(1.0 if s.is_forgery else 0.0, dtype=torch.float32)
        return image, writer, label