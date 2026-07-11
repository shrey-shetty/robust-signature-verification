"""Evaluate a trained checkpoint on the held-out TEST split.

Writer-independent protocol: this is the one deliberate use of the test
split. Reports EER (all pairs and skilled-only), ROC-AUC, and FAR/FRR at
a FIXED operating threshold selected on the validation set (pass it via
--threshold; for exp01 CEDAR this is epoch 1's val threshold, 0.3635).

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_checkpoint.py ^
        --dataset cedar ^
        --checkpoint experiments\\siamese_smallcnn_cedar\\best_model.pt ^
        --threshold 0.3635

Path/device overrides (defaults reproduce local behavior exactly):
    --raw-root <dir>   raw-data root (default data/raw under project root;
                       e.g. /kaggle/input/<dataset-name> on Kaggle)
    --out / --out-dir <dir>  where to write outputs (default: alongside
                       the checkpoint, i.e. Path(checkpoint).parent)
    --device cpu|cuda  override autodetection (default: cuda if available)

Outputs (next to the checkpoint, unless --out-dir overrides it):
    test_metrics.json   EER/AUC/FAR/FRR + run details
    test_scores.csv     per-pair distance, label, kind (for ROC plots later)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigver.data.datasets import list_samples, SignatureDataset  # noqa: E402
from sigver.data.pairs import generate_pairs, pair_summary, PairDataset  # noqa: E402
from sigver.models.siamese import SiameseNetwork  # noqa: E402
from sigver.evaluation.metrics import compute_eer, roc_auc  # noqa: E402


def collect_distances(model, loader, device):
    """Return (distances, labels) over a pair loader."""
    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for a, b, y in loader:
            a, b = a.to(device), b.to(device)
            ea, eb = model(a, b)
            d = torch.nn.functional.pairwise_distance(ea, eb)
            dists.append(d.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(dists), np.concatenate(labels)


def far_frr_at_threshold(dists: np.ndarray, labels: np.ndarray,
                         threshold: float) -> tuple[float, float]:
    """FAR/FRR at a fixed threshold.

    Convention (matches pairs.py: label 1.0 = similar/genuine pair):
      accept  <=> distance <  threshold
      FAR = accepted negatives / all negatives   (forgeries let in)
      FRR = rejected positives / all positives   (genuines kept out)
    """
    pos = labels >= 0.5
    neg = ~pos
    far = float((dists[neg] < threshold).mean()) if neg.any() else float("nan")
    frr = float((dists[pos] >= threshold).mean()) if pos.any() else float("nan")
    return far, frr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cedar")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, required=True,
                    help="operating threshold selected on the VALIDATION set")
    ap.add_argument("--split", default="test",
                    help="split to evaluate (default: test)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action="store_true",
                    help="cache preprocessed images in RAM")
    ap.add_argument("--raw-root", default=None,
                    help="override raw-data root (default data/raw under the "
                         "project root; e.g. /kaggle/input/<dataset-name> on Kaggle)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="override device autodetection (default: cuda if available)")
    ap.add_argument("--out", "--out-dir", dest="out", default=None,
                    help="output dir for test_metrics.json/test_scores.csv "
                         "(default: alongside the checkpoint)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root) if args.raw_root else None

    # ---- data ---------------------------------------------------------
    print(f"[data] loading {args.dataset} / {args.split} split ...")
    t0 = time.time()
    samples = list_samples(args.dataset, args.split, raw_root=raw_root)
    # seed offset 2: distinct from train (seed) and val (seed+1) pair RNG
    pairs = generate_pairs(samples, seed=args.seed + 2)
    print(f"[data] {args.split} pairs: {pair_summary(pairs)}")

    ds = PairDataset(SignatureDataset(samples, cache_in_memory=args.cache), pairs)
    loader = DataLoader(ds, batch_size=args.batch_size)
    print(f"[data] ready in {time.time() - t0:.1f}s")

    # ---- model --------------------------------------------------------
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"[model] loaded {ckpt_path} on {device}")

    # ---- evaluate -----------------------------------------------------
    dists, labels = collect_distances(model, loader, device)

    kinds = np.array([p.kind for p in pairs])
    skilled_mask = (kinds == "positive") | (kinds == "skilled")

    # Pair.i/.j index into `samples`, which already carries writer_id
    # (folder-ID ground truth) -- no change to pairs.py needed to expose it.
    writer_a = np.array([samples[p.i].writer_id for p in pairs])
    writer_b = np.array([samples[p.j].writer_id for p in pairs])
    # sanity check: positive/skilled pairs are constructed within one
    # writer, so writer_a should equal writer_b for every non-random pair.
    bad = int(((writer_a != writer_b) & (kinds != "random")).sum())
    if bad:
        print(f"WARNING: {bad} non-random pairs have writer_a != writer_b "
              "-- check pairs.generate_pairs")

    eer_all, thr_test = compute_eer(dists, labels)
    eer_skilled, _ = compute_eer(dists[skilled_mask], labels[skilled_mask])
    auc = roc_auc(dists, labels)

    far, frr = far_frr_at_threshold(dists, labels, args.threshold)
    far_sk, frr_sk = far_frr_at_threshold(
        dists[skilled_mask], labels[skilled_mask], args.threshold)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "n_pairs": int(len(labels)),
        "n_pairs_skilled_subset": int(skilled_mask.sum()),
        # threshold-free metrics
        "test_eer_all": round(float(eer_all), 4),
        "test_eer_skilled": round(float(eer_skilled), 4),
        "test_auc": round(float(auc), 4),
        "test_eer_threshold": round(float(thr_test), 4),  # diagnostic only
        # fixed operating point (chosen on validation — the honest number)
        "operating_threshold_from_val": args.threshold,
        "far_at_val_threshold": round(far, 4),
        "frr_at_val_threshold": round(frr, 4),
        "far_skilled_at_val_threshold": round(far_sk, 4),
        "frr_skilled_at_val_threshold": round(frr_sk, 4),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # per-pair scores for later ROC/DET plotting in 03_results_analysis.
    # writer_a/writer_b: folder-ID ground truth for the reference/
    # questioned side of the pair (older test_scores.csv files won't have
    # these columns -- 03_results_analysis guards on their presence).
    import csv
    with open(out_dir / "test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "label", "kind", "writer_a", "writer_b"])
        for d, y, k, wa, wb in zip(dists, labels, kinds, writer_a, writer_b):
            w.writerow([f"{d:.6f}", int(y), k, int(wa), int(wb)])

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved test_metrics.json and test_scores.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())