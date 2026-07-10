"""Unified dataset catalog for all signature-verification corpora.

Builds a single pandas DataFrame indexing every signature image across the
five datasets used in this project:

    - CEDAR                                   (per-writer folders, .png)
    - BHSig260-Bengali / BHSig260-Hindi       (per-writer folders, .tif)
    - GPDS Synthetic 4000                     (per-writer folders, .jpg + .mat metadata)
    - institutional ``signature_verification``(flat full_org / full_forg, .jpg)

Design decisions (documented in the final report):

1.  **Folder ID is the ground truth for writer identity** in per-writer
    layouts. This guards against the known BHSig260-Hindi writer-123 anomaly,
    where filenames inside folder ``123`` claim writer ``124``. Filename IDs
    are still parsed and any mismatch is flagged in the ``id_mismatch``
    column so anomalies surface in the EDA instead of silently corrupting
    labels.
2.  **GPDS ``ParamsUserNNNN.mat`` files are skipped** via an ignore regex —
    they are known synthesis metadata, not images.
3.  The catalog never modifies raw data; it is a read-only index.

Typical usage::

    from sigver.data.catalog import build_catalog, DATASET_SPECS
    df = build_catalog(project_root / "data" / "raw")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd

# Image extensions we treat as signature samples.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
)

# Files that are expected in the raw tree but are NOT signature images.
# GPDS ships one ParamsUserNNNN.mat per writer (synthesis metadata).
IGNORE_FILE_RE = re.compile(r"^ParamsUser\d+\.mat$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Per-dataset filename parsers
# ---------------------------------------------------------------------------
# Each parser receives the file path plus (for per-writer layouts) the folder
# name, and returns (label, filename_writer_id) where label is "genuine" or
# "forgery". Returning None for label means "unrecognized file".

_BHSIG_RE = re.compile(r"^[BH]-S-(\d+)-([FG])-\d+$", re.IGNORECASE)
_CEDAR_STYLE_RE = re.compile(r"^(original|forgeries)_(\d+)_\d+$", re.IGNORECASE)
_GPDS_RE = re.compile(r"^(c|cf)-(\d+)-\d+$", re.IGNORECASE)


def parse_bhsig(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Parse BHSig260 filenames like ``B-S-1-G-04`` / ``H-S-123-F-17``.

    Returns:
        (label, writer_id_from_filename) — label is "genuine" for -G-,
        "forgery" for -F-, or (None, None) if the name is unrecognized.
    """
    m = _BHSIG_RE.match(stem)
    if not m:
        return None, None
    label = "genuine" if m.group(2).upper() == "G" else "forgery"
    return label, m.group(1)


def parse_cedar_style(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Parse CEDAR/institutional names like ``original_12_3`` / ``forgeries_12_3``."""
    m = _CEDAR_STYLE_RE.match(stem)
    if not m:
        return None, None
    label = "genuine" if m.group(1).lower() == "original" else "forgery"
    return label, m.group(2)


def parse_gpds(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Parse GPDS Synthetic names like ``c-001-05`` (genuine) / ``cf-001-05`` (forgery)."""
    m = _GPDS_RE.match(stem)
    if not m:
        return None, None
    label = "genuine" if m.group(1).lower() == "c" else "forgery"
    return label, m.group(2)


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Describes how one dataset is laid out under ``data/raw``.

    Attributes:
        name: Canonical short name used throughout the EDA.
        rel_root: Path of the dataset root relative to ``data/raw``.
        layout: "per_writer" (writer = subfolder name) or "flat"
            (writer parsed from filename; folder encodes only the label).
        parser: Filename-stem parser returning (label, filename_writer_id).
        flat_label_dirs: For flat layouts, maps subfolder name -> label.
    """

    name: str
    rel_root: str
    layout: str  # "per_writer" | "flat"
    parser: Callable[[str], tuple[Optional[str], Optional[str]]]
    flat_label_dirs: dict[str, str] = field(default_factory=dict)


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="cedar",
        rel_root="CEDAR",
        layout="per_writer",
        parser=parse_cedar_style,
    ),
    DatasetSpec(
        name="bhsig260_bengali",
        rel_root="BHSig260-Bengali",
        layout="per_writer",
        parser=parse_bhsig,
    ),
    DatasetSpec(
        name="bhsig260_hindi",
        rel_root="BHSig260-Hindi",
        layout="per_writer",
        parser=parse_bhsig,
    ),
    DatasetSpec(
        name="gpds_synthetic_4000",
        rel_root="SignatureGPDSSyntheticOffLine4000/firmasSINTESISmanuscritas",
        layout="per_writer",
        parser=parse_gpds,
    ),
    DatasetSpec(
        name="institutional",
        rel_root="signature_verification",
        layout="flat",
        parser=parse_cedar_style,
        flat_label_dirs={"full_org": "genuine", "full_forg": "forgery"},
    ),
)


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def _iter_dataset_rows(spec: DatasetSpec, raw_root: Path) -> Iterator[dict]:
    """Yield one catalog row per image file for a single dataset.

    Unrecognized files (bad stems, unexpected extensions) are yielded with
    ``label=None`` so they can be reported rather than silently dropped.
    """
    ds_root = raw_root / spec.rel_root
    if not ds_root.is_dir():
        return  # Dataset not present on this machine — caller reports it.

    if spec.layout == "per_writer":
        for writer_dir in sorted(p for p in ds_root.iterdir() if p.is_dir()):
            folder_writer_id = writer_dir.name.lstrip("0") or writer_dir.name
            for f in sorted(writer_dir.iterdir()):
                if not f.is_file() or IGNORE_FILE_RE.match(f.name):
                    continue
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    yield _row(spec, f, None, folder_writer_id, None)
                    continue
                label, fname_writer = spec.parser(f.stem)
                fname_writer_norm = (
                    fname_writer.lstrip("0") or fname_writer
                ) if fname_writer else None
                yield _row(spec, f, label, folder_writer_id, fname_writer_norm)

    elif spec.layout == "flat":
        for sub_name, sub_label in spec.flat_label_dirs.items():
            sub_dir = ds_root / sub_name
            if not sub_dir.is_dir():
                continue
            for f in sorted(sub_dir.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    yield _row(spec, f, None, None, None)
                    continue
                fname_label, fname_writer = spec.parser(f.stem)
                # Folder decides the label; filename label is cross-checked.
                yield {
                    **_row(spec, f, sub_label, fname_writer, fname_writer),
                    "label_mismatch": (
                        fname_label is not None and fname_label != sub_label
                    ),
                }
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown layout: {spec.layout!r}")


def _row(
    spec: DatasetSpec,
    path: Path,
    label: Optional[str],
    writer_id: Optional[str],
    fname_writer_id: Optional[str],
) -> dict:
    """Assemble one catalog row. Writer ID = folder ID for per-writer layouts."""
    return {
        "dataset": spec.name,
        "writer_id": writer_id,
        "label": label,
        "path": str(path),
        "filename": path.name,
        "ext": path.suffix.lower(),
        "filename_writer_id": fname_writer_id,
        "id_mismatch": (
            writer_id is not None
            and fname_writer_id is not None
            and writer_id != fname_writer_id
        ),
        "label_mismatch": False,
    }


def build_catalog(
    raw_root: Path,
    specs: tuple[DatasetSpec, ...] = DATASET_SPECS,
) -> pd.DataFrame:
    """Build the unified image catalog across all datasets under ``raw_root``.

    Args:
        raw_root: Path to ``data/raw``.
        specs: Dataset specifications to index (defaults to all five).

    Returns:
        DataFrame with columns: dataset, writer_id, label, path, filename,
        ext, filename_writer_id, id_mismatch, label_mismatch. Rows with
        ``label`` NaN correspond to unrecognized files and should be
        reported by the EDA, then excluded from analysis.
    """
    rows: list[dict] = []
    missing: list[str] = []
    for spec in specs:
        before = len(rows)
        rows.extend(_iter_dataset_rows(spec, raw_root))
        if len(rows) == before and not (raw_root / spec.rel_root).is_dir():
            missing.append(spec.name)

    df = pd.DataFrame(rows)
    if missing:
        print(f"[catalog] WARNING: datasets not found under {raw_root}: {missing}")
    if not df.empty:
        df["label"] = df["label"].astype("category")
        df["dataset"] = df["dataset"].astype("category")
    return df