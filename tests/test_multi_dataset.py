"""Tests for the RQ3 combined-dataset training path.

Covers sigver.data.datasets.list_samples_multi (namespaced writer_id
concatenation across corpora) and scripts/train_baseline.py's --dataset
CLI parsing.

Fixtures build tiny synthetic raw_root layouts rather than depending on the
real (gigabytes, not present in every environment) raw data, using the
same filename conventions as sigver.data.datasets._per_writer_specs /
_flat_specs (verified by reading datasets.py before writing this file).
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

import sigver.data.datasets as datasets
from sigver.data.datasets import (
    Sample,
    build_dataset_offsets,
    list_samples,
    list_samples_multi,
    resolve_namespaced_writer,
    writer_counts_by_dataset,
)
from sigver.data.pairs import generate_pairs


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_cedar_writer(raw_root: Path, writer_id: int, n_genuine: int, n_forgery: int) -> None:
    wdir = raw_root / "CEDAR" / str(writer_id)
    wdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_genuine + 1):
        (wdir / f"original_{writer_id}_{i}.png").write_bytes(b"")
    for i in range(1, n_forgery + 1):
        (wdir / f"forgeries_{writer_id}_{i}.png").write_bytes(b"")


def _make_institutional_writer(raw_root: Path, writer_id: int, n_genuine: int, n_forgery: int) -> None:
    org_dir = raw_root / "signature_verification" / "full_org"
    forg_dir = raw_root / "signature_verification" / "full_forg"
    org_dir.mkdir(parents=True, exist_ok=True)
    forg_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_genuine + 1):
        (org_dir / f"original_{writer_id}_{i}.jpg").write_bytes(b"")
    for i in range(1, n_forgery + 1):
        (forg_dir / f"forgeries_{writer_id}_{i}.jpg").write_bytes(b"")


def _write_split_csv(splits_dir: Path, dataset: str, writer_to_split: dict[int, str]) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = splits_dir / f"split_{dataset}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["writer_id", "split"])
        for wid, split in writer_to_split.items():
            writer.writerow([wid, split])


@pytest.fixture
def tmp_splits(tmp_path, monkeypatch):
    """Point sigver.data.datasets.SPLITS_DIR at a scratch directory.

    load_split() looks up the module-level SPLITS_DIR by name at call time,
    so monkeypatching the module attribute redirects it without touching
    the real committed split CSVs under data/splits/.
    """
    splits_dir = tmp_path / "splits"
    monkeypatch.setattr(datasets, "SPLITS_DIR", splits_dir)
    return splits_dir


# ---------------------------------------------------------------------------
# 1. Regression: list_samples('cedar', 'train', ...) is unchanged.
# ---------------------------------------------------------------------------

def test_list_samples_cedar_train_is_a_golden_fixed_result(tmp_path, tmp_splits):
    raw_root = tmp_path / "raw"
    _make_cedar_writer(raw_root, writer_id=1, n_genuine=2, n_forgery=1)  # train
    _make_cedar_writer(raw_root, writer_id=2, n_genuine=1, n_forgery=0)  # train
    _make_cedar_writer(raw_root, writer_id=3, n_genuine=1, n_forgery=1)  # val (excluded)
    _write_split_csv(tmp_splits, "cedar", {1: "train", 2: "train", 3: "val"})

    result = list_samples("cedar", "train", raw_root=raw_root)

    expected = sorted(
        [
            Sample(raw_root / "CEDAR" / "1" / "original_1_1.png", 1, False),
            Sample(raw_root / "CEDAR" / "1" / "original_1_2.png", 1, False),
            Sample(raw_root / "CEDAR" / "1" / "forgeries_1_1.png", 1, True),
            Sample(raw_root / "CEDAR" / "2" / "original_2_1.png", 2, False),
        ],
        key=lambda s: str(s.path),
    )
    assert result == expected
    # writer 3 (val) must not leak into the train split
    assert all(s.writer_id != 3 for s in result)


def test_list_samples_multi_with_one_dataset_matches_list_samples(tmp_path, tmp_splits):
    """list_samples_multi must not change single-dataset behaviour (offset 0)."""
    raw_root = tmp_path / "raw"
    _make_cedar_writer(raw_root, writer_id=1, n_genuine=2, n_forgery=1)
    _make_cedar_writer(raw_root, writer_id=2, n_genuine=1, n_forgery=0)
    _write_split_csv(tmp_splits, "cedar", {1: "train", 2: "train"})

    direct = list_samples("cedar", "train", raw_root=raw_root)
    combined, offsets = list_samples_multi(["cedar"], "train", raw_root=raw_root)

    assert offsets == {"cedar": 0}
    assert combined == direct


# ---------------------------------------------------------------------------
# 2. Collision: namespacing keeps two corpora's overlapping raw writer IDs distinct.
# ---------------------------------------------------------------------------

def test_namespacing_prevents_writer_id_collision_across_datasets(tmp_path, tmp_splits):
    raw_root = tmp_path / "raw"
    # Same raw writer IDs {1, 2, 3} used in both corpora on purpose.
    for wid in (1, 2, 3):
        _make_cedar_writer(raw_root, writer_id=wid, n_genuine=2, n_forgery=1)
        _make_institutional_writer(raw_root, writer_id=wid, n_genuine=2, n_forgery=1)
    _write_split_csv(tmp_splits, "cedar", {1: "train", 2: "train", 3: "train"})
    _write_split_csv(tmp_splits, "institutional", {1: "train", 2: "train", 3: "train"})

    combined, offsets = list_samples_multi(["cedar", "institutional"], "train", raw_root=raw_root)

    distinct_namespaced = {s.writer_id for s in combined}
    cedar_raw = {s.writer_id for s in list_samples("cedar", "train", raw_root=raw_root)}
    inst_raw = {s.writer_id for s in list_samples("institutional", "train", raw_root=raw_root)}

    assert len(distinct_namespaced) == len(cedar_raw) + len(inst_raw) == 6
    assert offsets["cedar"] != offsets["institutional"]

    counts = writer_counts_by_dataset(combined, offsets)
    assert counts == {"cedar": 3, "institutional": 3}


def test_build_dataset_offsets_is_reproducible_regardless_of_request_order():
    assert build_dataset_offsets(["institutional", "cedar"]) == build_dataset_offsets(["cedar", "institutional"])


def test_resolve_namespaced_writer_inverts_the_offset():
    offsets = build_dataset_offsets(["cedar", "institutional"])
    for name, raw_id in (("cedar", 7), ("institutional", 42)):
        namespaced = raw_id + offsets[name]
        assert resolve_namespaced_writer(namespaced, offsets) == (name, raw_id)


def test_list_samples_multi_raises_on_a_dataset_with_no_writers_in_split(tmp_path, tmp_splits):
    raw_root = tmp_path / "raw"
    _make_cedar_writer(raw_root, writer_id=1, n_genuine=2, n_forgery=1)
    _make_institutional_writer(raw_root, writer_id=1, n_genuine=2, n_forgery=1)
    _write_split_csv(tmp_splits, "cedar", {1: "train"})
    _write_split_csv(tmp_splits, "institutional", {1: "val"})  # no train writers

    with pytest.raises(ValueError, match="institutional"):
        list_samples_multi(["cedar", "institutional"], "train", raw_root=raw_root)


# ---------------------------------------------------------------------------
# 3. Pair integrity: positive pairs never cross a dataset boundary.
# ---------------------------------------------------------------------------

def test_positive_pairs_never_cross_dataset_boundaries(tmp_path, tmp_splits):
    raw_root = tmp_path / "raw"
    for wid in (1, 2, 3):
        _make_cedar_writer(raw_root, writer_id=wid, n_genuine=4, n_forgery=2)
        _make_institutional_writer(raw_root, writer_id=wid, n_genuine=4, n_forgery=2)
    _write_split_csv(tmp_splits, "cedar", {1: "train", 2: "train", 3: "train"})
    _write_split_csv(tmp_splits, "institutional", {1: "train", 2: "train", 3: "train"})

    combined, offsets = list_samples_multi(["cedar", "institutional"], "train", raw_root=raw_root)
    pairs = generate_pairs(combined, seed=42)

    positives = [p for p in pairs if p.kind == "positive"]
    assert positives, "fixture must actually produce positive pairs to test anything"

    for p in positives:
        wa, _ = resolve_namespaced_writer(combined[p.i].writer_id, offsets)
        wb, _ = resolve_namespaced_writer(combined[p.j].writer_id, offsets)
        assert wa == wb, (
            f"positive pair spans datasets {wa} vs {wb} — writer-disjoint "
            f"protocol broken by an un-namespaced writer_id collision"
        )
        # and, more strongly: same writer entirely, not just same dataset
        assert combined[p.i].writer_id == combined[p.j].writer_id


# ---------------------------------------------------------------------------
# 4. CLI parsing: --dataset accepts a comma-separated list.
# ---------------------------------------------------------------------------

def _load_train_baseline_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline.py"
    spec = importlib.util.spec_from_file_location("train_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_baseline_module():
    return _load_train_baseline_module()


def test_cli_dataset_flag_parses_comma_separated_list(train_baseline_module):
    args = train_baseline_module.build_arg_parser().parse_args(
        ["--dataset", "institutional,cedar"]
    )
    assert args.dataset == ["institutional", "cedar"]


def test_cli_dataset_flag_still_parses_a_single_name(train_baseline_module):
    args = train_baseline_module.build_arg_parser().parse_args(["--dataset", "institutional"])
    assert args.dataset == ["institutional"]


def test_cli_dataset_flag_default_is_a_single_element_list(train_baseline_module):
    args = train_baseline_module.build_arg_parser().parse_args([])
    assert args.dataset == ["cedar"]


def test_cli_dataset_flag_rejects_an_empty_name(train_baseline_module):
    with pytest.raises(SystemExit):
        train_baseline_module.build_arg_parser().parse_args(["--dataset", "cedar,"])
