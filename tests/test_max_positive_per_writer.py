"""Tests for --max-positive-per-writer (RQ3 combined-dataset pair balance).

Covers sigver.data.pairs.generate_pairs's existing max_positive_per_writer
parameter (unchanged, only plumbed through here) and its wiring into
scripts/train_baseline.py's CLI and generate_pairs() call sites.

Expected values below are derived by hand from the combinatorics
(C(n, 2) genuine-genuine pairs per writer) and from generate_pairs's
documented skilled/random split (n_skilled = n_neg // 2 if the writer has
forgeries, n_random = n_neg - n_skilled), not from running the
implementation — see the same convention in tests/test_metrics.py.

Verified by reading src/sigver/data/pairs.py before writing this file:
the cap is applied to `pos` *after* `rng.shuffle(pos)`, so it is an
unbiased, deterministic random subsample, not a truncation of generation
order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sigver.data.datasets import Sample
from sigver.data.pairs import generate_pairs, pair_summary


# ---------------------------------------------------------------------------
# Fixture builders: plain Sample lists, no disk I/O needed for pairs.py tests.
# ---------------------------------------------------------------------------

def _writer_samples(writer_id: int, n_genuine: int, n_forgery: int, start_idx: int = 0):
    samples = []
    for i in range(n_genuine):
        samples.append(Sample(Path(f"w{writer_id}_gen_{i}.png"), writer_id, False))
    for i in range(n_forgery):
        samples.append(Sample(Path(f"w{writer_id}_forg_{i}.png"), writer_id, True))
    return samples


# ---------------------------------------------------------------------------
# 2. Cap applies: 24 genuine -> 276 uncapped, 66 with cap=66.
# ---------------------------------------------------------------------------

def test_cap_reduces_positive_pairs_to_the_cap_value():
    # writer A: 24 genuine (C(24,2) = 276 uncapped); writer B: minimal, just
    # to satisfy generate_pairs's >=2-writers-with-genuine requirement.
    samples = _writer_samples(1, n_genuine=24, n_forgery=6) + _writer_samples(2, n_genuine=2, n_forgery=0)

    uncapped = generate_pairs(samples, seed=42, max_positive_per_writer=None)
    capped = generate_pairs(samples, seed=42, max_positive_per_writer=66)

    uncapped_pos_a = [p for p in uncapped if p.kind == "positive"
                      and samples[p.i].writer_id == 1]
    capped_pos_a = [p for p in capped if p.kind == "positive"
                    and samples[p.i].writer_id == 1]

    assert len(uncapped_pos_a) == 276  # C(24, 2)
    assert len(capped_pos_a) == 66


# ---------------------------------------------------------------------------
# 3. No-op below the cap: 12 genuine -> 66 pairs either way, identical lists.
#    This is the test proving institutional-only runs (12 genuine/writer,
#    C(12,2)=66) are unaffected by a cap of 66.
# ---------------------------------------------------------------------------

def test_cap_is_a_no_op_when_writers_are_already_at_or_below_it():
    # Two writers, each with exactly 12 genuine (C(12,2) = 66), mirroring
    # the institutional corpus's 12-genuine-per-writer layout.
    samples = (
        _writer_samples(1, n_genuine=12, n_forgery=4)
        + _writer_samples(2, n_genuine=12, n_forgery=4)
    )

    uncapped = generate_pairs(samples, seed=42, max_positive_per_writer=None)
    capped = generate_pairs(samples, seed=42, max_positive_per_writer=66)

    assert uncapped == capped  # identical pair lists, not just identical length

    positives = [p for p in uncapped if p.kind == "positive"]
    assert len(positives) == 2 * 66  # C(12, 2) per writer, two writers


# ---------------------------------------------------------------------------
# 4. Negative-pair balance holds under a cap.
# ---------------------------------------------------------------------------

def test_negative_pair_balance_holds_under_a_cap():
    # writer A: 24 genuine + forgeries, capped 276 -> 66; n_skilled=33, n_random=33
    # writer B: 2 genuine, no forgeries, uncapped by 66 (only 1 combination);
    #           n_skilled=0, n_random=1
    samples = _writer_samples(1, n_genuine=24, n_forgery=6) + _writer_samples(2, n_genuine=2, n_forgery=0)

    pairs = generate_pairs(samples, seed=42, max_positive_per_writer=66)
    summary = pair_summary(pairs)

    assert summary == {"positive": 67, "skilled": 33, "random": 34, "total": 134}
    # the defining balance rule from pairs.py: n_neg == n_skilled + n_random == n_pos
    assert summary["skilled"] + summary["random"] == summary["positive"]


# ---------------------------------------------------------------------------
# 5. Determinism: same seed + cap -> identical pair list across calls.
# ---------------------------------------------------------------------------

def test_same_seed_and_cap_produce_identical_pairs_across_calls():
    samples = _writer_samples(1, n_genuine=24, n_forgery=6) + _writer_samples(2, n_genuine=12, n_forgery=4)

    first = generate_pairs(samples, seed=42, max_positive_per_writer=66)
    second = generate_pairs(samples, seed=42, max_positive_per_writer=66)

    assert first == second


# ---------------------------------------------------------------------------
# 6. CLI parse: --max-positive-per-writer parses to int; absent -> None.
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


def test_cli_max_positive_per_writer_parses_to_int(train_baseline_module):
    args = train_baseline_module.build_arg_parser().parse_args(
        ["--max-positive-per-writer", "66"]
    )
    assert args.max_positive_per_writer == 66
    assert isinstance(args.max_positive_per_writer, int)


def test_cli_max_positive_per_writer_defaults_to_none(train_baseline_module):
    args = train_baseline_module.build_arg_parser().parse_args([])
    assert args.max_positive_per_writer is None


# ---------------------------------------------------------------------------
# 1. Default unchanged: with no flag, main() calls generate_pairs with
#    max_positive_per_writer=None. Asserted on the call itself (via a spy
#    that delegates to the real implementation), not just on the output --
#    a None default and a 10,000 default could produce the same output on
#    a tiny fixture where no writer is anywhere near either cap.
# ---------------------------------------------------------------------------

def _make_signature_image(path: Path, seed: int) -> None:
    from PIL import Image, ImageDraw
    img = Image.new("L", (220, 150), color=255)
    draw = ImageDraw.Draw(img)
    x0, y0 = 10 + seed * 3, 75
    for i in range(6):
        x1 = x0 + 15
        y1 = y0 + (10 if i % 2 == 0 else -10)
        draw.line([(x0, y0), (x1, y1)], fill=0, width=3)
        x0, y0 = x1, y1
    img.save(path)


def _make_cedar_writer(raw_root: Path, writer_id: int, n_genuine: int, n_forgery: int) -> None:
    wdir = raw_root / "CEDAR" / str(writer_id)
    wdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_genuine + 1):
        _make_signature_image(wdir / f"original_{writer_id}_{i}.png", i)
    for i in range(1, n_forgery + 1):
        _make_signature_image(wdir / f"forgeries_{writer_id}_{i}.png", i + 10)


def _write_split_csv(splits_dir: Path, dataset: str, writer_to_split: dict[int, str]) -> None:
    import csv
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = splits_dir / f"split_{dataset}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["writer_id", "split"])
        for wid, split in writer_to_split.items():
            writer.writerow([wid, split])


def test_default_run_calls_generate_pairs_with_max_positive_per_writer_none(
    tmp_path, monkeypatch, train_baseline_module
):
    import sigver.data.datasets as datasets_mod

    raw_root = tmp_path / "raw"
    splits_dir = tmp_path / "splits"
    monkeypatch.setattr(datasets_mod, "SPLITS_DIR", splits_dir)

    _make_cedar_writer(raw_root, writer_id=1, n_genuine=3, n_forgery=2)
    _make_cedar_writer(raw_root, writer_id=2, n_genuine=3, n_forgery=2)
    _make_cedar_writer(raw_root, writer_id=3, n_genuine=3, n_forgery=2)
    _make_cedar_writer(raw_root, writer_id=4, n_genuine=3, n_forgery=2)
    _write_split_csv(splits_dir, "cedar", {1: "train", 2: "train", 3: "val", 4: "val"})

    real_generate_pairs = generate_pairs
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_generate_pairs(*args, **kwargs)

    out_dir = tmp_path / "out"
    argv = [
        "train_baseline.py",
        "--dataset", "cedar",
        "--raw-root", str(raw_root),
        "--out", str(out_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--no-cache",
        "--num-workers", "0",
        "--device", "cpu",
    ]

    with patch.object(train_baseline_module, "generate_pairs", side_effect=spy):
        with patch.object(sys, "argv", argv):
            rc = train_baseline_module.main()

    assert rc == 0
    assert len(calls) == 2  # train call, val call
    for call_kwargs in calls:
        assert call_kwargs["max_positive_per_writer"] is None
