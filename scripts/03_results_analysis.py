"""Results analysis: rigorous test-set evaluation of a trained checkpoint.

Mirrors the CEDAR canonical-baseline methodology:
    - EER / AUC on the held-out TEST split (not val - val was used for
      model selection during training; test is untouched until now).
    - ROC and DET (log-scale) curves.
    - FAR/FRR decision table at standard operating points.
    - Genuine-Genuine vs Genuine-Forgery distance distributions.
    - Writer-level bootstrap (resample TEST WRITER IDENTITIES, not pairs)
      for 95% CIs on EER and AUC, matching the documented CEDAR method.

IMPORTANT CAVEATS (read before trusting output):
  - This reimplements EER/AUC/FAR/FRR directly rather than importing
    sigver.evaluation.metrics, because that module's exact function
    signatures were not available when this script was written. If your
    project has a canonical compute_eer()/roc_auc(), cross-check this
    script's numbers against it once before treating these as final -
    they should match, but "should" is not "verified against your code."
  - SiameseNetwork's constructor is assumed to be
    SiameseNetwork(embedding_dim=...) based on train_baseline.py's usage.
    If the real signature differs, the model-loading block will need a
    small edit.
  - Verify config.json in your run's --out directory actually contains
    "embedding_dim" and "margin" keys before running - this script reads
    them from there rather than hard-coding.

Usage (from project root, same conventions as train_baseline.py):
    .\\.venv\\Scripts\\python.exe scripts\\03_results_analysis.py \\
        --exp-dir /kaggle/working/exp_institutional_v1 \\
        --dataset institutional \\
        --raw-root /kaggle/working/data_raw

Outputs (written under --exp-dir/plots/ and --exp-dir/):
    plots/roc_curve.png
    plots/det_curve.png
    plots/distance_distributions.png
    metrics_table.md          (EER, AUC, avg contrastive loss + 95% CIs)
    far_frr_table.md          (FAR/FRR at 1%/5%/10% FAR + EER threshold)
    bootstrap_raw.json        (all 1000 bootstrap EER/AUC values, for audit)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # headless-safe for Kaggle
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigver.data.datasets import list_samples, SignatureDataset  # noqa: E402
from sigver.data.pairs import generate_pairs, pair_summary, PairDataset, Pair  # noqa: E402
from sigver.models.siamese import SiameseNetwork  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics (reimplemented directly - see caveat in module docstring)
# ---------------------------------------------------------------------------

def roc_points(dists: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, far, frr) swept over all distinct distance values.

    Convention: predict "genuine" (same-writer) if dist <= threshold.
    FAR(t) = fraction of label==0 (dissimilar) pairs with dist <= t  (wrongly accepted)
    FRR(t) = fraction of label==1 (similar)    pairs with dist >  t  (wrongly rejected)
    """
    thresholds = np.unique(dists)
    neg = dists[labels == 0.0]
    pos = dists[labels == 1.0]
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("need both positive and negative pairs to compute ROC/EER")

    # vectorized: for each threshold, count via searchsorted on sorted arrays
    neg_sorted = np.sort(neg)
    pos_sorted = np.sort(pos)
    far = np.searchsorted(neg_sorted, thresholds, side="right") / len(neg_sorted)
    frr = 1.0 - (np.searchsorted(pos_sorted, thresholds, side="right") / len(pos_sorted))
    return thresholds, far, frr


def compute_eer(dists: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (eer, eer_threshold) via linear interpolation at the FAR/FRR crossing."""
    thresholds, far, frr = roc_points(dists, labels)
    diff = far - frr
    # find sign change (crossing point)
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        # no crossing found (degenerate case) - return closest point
        i = np.argmin(np.abs(diff))
        return float((far[i] + frr[i]) / 2), float(thresholds[i])
    i = idx[0]
    # linear interpolation between i and i+1
    x0, x1 = thresholds[i], thresholds[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    if d1 == d0:
        t_star = x0
    else:
        t_star = x0 - d0 * (x1 - x0) / (d1 - d0)
    far_at, frr_at = np.interp(t_star, thresholds, far), np.interp(t_star, thresholds, frr)
    eer = float((far_at + frr_at) / 2)
    return eer, float(t_star)


def compute_auc(dists: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC using -distance as the 'genuine score' (higher = more similar)."""
    score = -dists
    order = np.argsort(score)
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes present to compute AUC")
    ranks = np.argsort(np.argsort(score)) + 1  # rank 1..N, ties handled approximately
    sum_ranks_pos = ranks[labels == 1.0].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def far_at_target(dists: np.ndarray, labels: np.ndarray, target_far: float) -> tuple[float, float]:
    """Return (threshold, frr) such that FAR(threshold) ~= target_far."""
    neg = np.sort(dists[labels == 0.0])
    t = np.percentile(neg, target_far * 100)
    _, far_arr, frr_arr = roc_points(dists, labels)
    frr = float(np.interp(t, np.unique(dists), frr_arr))
    far_actual = float(np.interp(t, np.unique(dists), far_arr))
    return t, frr, far_actual


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def load_model(exp_dir: Path, device: torch.device) -> SiameseNetwork:
    config_path = exp_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config.json not found in {exp_dir} - cannot recover embedding_dim. "
            "Verify this is the correct --exp-dir."
        )
    config = json.loads(config_path.read_text())
    embedding_dim = config.get("embedding_dim")
    if embedding_dim is None:
        raise KeyError("config.json does not contain 'embedding_dim' - check the file manually.")

    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    state_path = exp_dir / "best_model.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"best_model.pt not found in {exp_dir}")
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


@torch.no_grad()
def compute_distances(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    dists, labels = [], []
    for a, b, y in loader:
        a, b = a.to(device), b.to(device)
        ea, eb = model(a, b)
        d = torch.nn.functional.pairwise_distance(ea, eb)
        dists.append(d.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(dists), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Writer-level bootstrap
# ---------------------------------------------------------------------------

def writer_bootstrap(dists: np.ndarray, labels: np.ndarray, anchor_writers: np.ndarray,
                      n_iterations: int = 1000, seed: int = 42) -> dict:
    """Resample TEST WRITER IDENTITIES with replacement (not individual pairs).

    anchor_writers[i] is the writer_id that "generated" pair i (see note
    below on why this is well-defined for all pair kinds in this codebase).
    """
    rng = np.random.default_rng(seed)
    unique_writers = np.unique(anchor_writers)
    n_writers = len(unique_writers)

    eers, aucs = [], []
    for _ in range(n_iterations):
        sampled_writers = rng.choice(unique_writers, size=n_writers, replace=True)
        # gather all pairs belonging to each sampled writer (with duplicates
        # if a writer is drawn more than once, per standard writer-bootstrap)
        idx_list = []
        for w in sampled_writers:
            idx_list.append(np.where(anchor_writers == w)[0])
        idx = np.concatenate(idx_list)
        d_sample, y_sample = dists[idx], labels[idx]
        try:
            eer_i, _ = compute_eer(d_sample, y_sample)
            auc_i = compute_auc(d_sample, y_sample)
        except ValueError:
            continue  # skip degenerate resamples (e.g. all-one-class)
        eers.append(eer_i)
        aucs.append(auc_i)

    eers, aucs = np.array(eers), np.array(aucs)
    return {
        "n_valid_iterations": int(len(eers)),
        "n_test_writers": int(n_writers),
        "eer_ci": [float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))],
        "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "eer_all_samples": eers.tolist(),
        "auc_all_samples": aucs.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(dists, labels, out_path: Path):
    _, far, frr = roc_points(dists, labels)
    tpr = 1.0 - frr
    order = np.argsort(far)
    plt.figure(figsize=(6, 6))
    plt.plot(far[order], tpr[order], lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("ROC Curve - Institutional Test Set")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_det(dists, labels, out_path: Path):
    _, far, frr = roc_points(dists, labels)
    order = np.argsort(far)
    far_o, frr_o = far[order], frr[order]
    # avoid log(0)
    eps = 1e-4
    far_o = np.clip(far_o, eps, 1.0)
    frr_o = np.clip(frr_o, eps, 1.0)
    plt.figure(figsize=(6, 6))
    plt.plot(far_o, frr_o, lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Acceptance Rate (FAR, log scale)")
    plt.ylabel("False Rejection Rate (FRR, log scale)")
    plt.title("DET Curve - Institutional Test Set")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distance_distributions(dists, labels, kinds, out_path: Path):
    """Genuine-Genuine (kind='positive') vs Genuine-Forgery (kind='skilled').

    Note: 'random' pairs (genuine vs a different writer's genuine - the
    "random forgery" scenario) are excluded from this specific plot since
    the request was Genuine-Genuine vs Genuine-Forgery specifically. If
    you also want the random-negative distribution, ask and I'll add a
    third panel rather than silently conflating it with 'skilled'.
    """
    kinds = np.asarray(kinds)
    gg = dists[kinds == "positive"]
    gf = dists[kinds == "skilled"]
    plt.figure(figsize=(8, 5))
    plt.hist(gg, bins=50, alpha=0.6, label=f"Genuine-Genuine (n={len(gg)})", density=True)
    plt.hist(gf, bins=50, alpha=0.6, label=f"Genuine-Forgery/skilled (n={len(gf)})", density=True)
    plt.xlabel("Euclidean embedding distance")
    plt.ylabel("Density")
    plt.title("Distance Distributions - Institutional Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", required=True, help="training output dir, e.g. exp_institutional_v1")
    ap.add_argument("--dataset", default="institutional")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bootstrap-iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    print(f"[model] loading from {exp_dir} ...")
    model, config = load_model(exp_dir, device)
    margin = config.get("margin", 1.0)
    print(f"[model] embedding_dim={config.get('embedding_dim')}, margin={margin}")

    # ---- TEST split (deliberately not val - val was used for model selection) ----
    print(f"[data] loading {args.dataset} TEST split ...")
    test_samples = list_samples(args.dataset, "test", raw_root=raw_root)
    test_pairs = generate_pairs(test_samples, seed=args.seed + 2)  # distinct seed from train/val
    print(f"[data] test pairs: {pair_summary(test_pairs)}")

    test_ds = PairDataset(SignatureDataset(test_samples, cache_in_memory=False), test_pairs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print("[eval] computing embeddings + distances over test set ...")
    dists, labels = compute_distances(model, test_loader, device)
    kinds = np.array([p.kind for p in test_pairs])
    anchor_writers = np.array([test_samples[p.i].writer_id for p in test_pairs])

    # ---- 1. Metrics + curves ----
    eer, eer_threshold = compute_eer(dists, labels)
    auc = compute_auc(dists, labels)

    # average contrastive loss at reported margin, for reference only
    d_t = torch.from_numpy(dists).float()
    y_t = torch.from_numpy(labels).float()
    loss_similar = y_t * d_t.pow(2)
    loss_dissimilar = (1.0 - y_t) * torch.relu(margin - d_t).pow(2)
    avg_contrastive_loss = float((loss_similar + loss_dissimilar).mean())

    print(f"[metrics] EER={eer:.4f} (threshold={eer_threshold:.4f}) AUC={auc:.4f} "
          f"avg_contrastive_loss={avg_contrastive_loss:.4f}")

    plot_roc(dists, labels, plots_dir / "roc_curve.png")
    plot_det(dists, labels, plots_dir / "det_curve.png")
    print(f"[plots] saved roc_curve.png, det_curve.png to {plots_dir}")

    # ---- 2. FAR/FRR decision table ----
    rows = []
    for target_far in (0.01, 0.05, 0.10):
        t, frr_at, far_actual = far_at_target(dists, labels, target_far)
        rows.append((f"FAR @ {int(target_far*100)}%", t, far_actual, frr_at))
    rows.append(("EER threshold", eer_threshold, eer, eer))

    far_frr_md = ["| Operating Point | Threshold | FAR | FRR |",
                 "|---|---|---|---|"]
    for name, t, far_v, frr_v in rows:
        far_frr_md.append(f"| {name} | {t:.4f} | {far_v:.4f} | {frr_v:.4f} |")
    (exp_dir / "far_frr_table.md").write_text("\n".join(far_frr_md))
    print(f"[table] wrote far_frr_table.md")

    # ---- 3. Distance distributions ----
    plot_distance_distributions(dists, labels, kinds, plots_dir / "distance_distributions.png")
    print(f"[plots] saved distance_distributions.png")

    # ---- 4. Writer-level bootstrap ----
    print(f"[bootstrap] running {args.bootstrap_iterations} writer-resample iterations "
          f"over {len(np.unique(anchor_writers))} test writers ...")
    boot = writer_bootstrap(dists, labels, anchor_writers,
                            n_iterations=args.bootstrap_iterations, seed=args.seed)
    print(f"[bootstrap] EER 95% CI: [{boot['eer_ci'][0]:.4f}, {boot['eer_ci'][1]:.4f}]")
    print(f"[bootstrap] AUC 95% CI: [{boot['auc_ci'][0]:.4f}, {boot['auc_ci'][1]:.4f}]")

    (exp_dir / "bootstrap_raw.json").write_text(json.dumps({
        "n_valid_iterations": boot["n_valid_iterations"],
        "n_test_writers": boot["n_test_writers"],
        "eer_ci": boot["eer_ci"],
        "auc_ci": boot["auc_ci"],
    }, indent=2))
    # full per-iteration arrays kept separately (larger file) for audit trail
    (exp_dir / "bootstrap_full_samples.json").write_text(json.dumps({
        "eer_all_samples": boot["eer_all_samples"],
        "auc_all_samples": boot["auc_all_samples"],
    }))

    # ---- metrics_table.md (final summary) ----
    metrics_md = [
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Test EER (all pairs) | {eer:.4f} | [{boot['eer_ci'][0]:.4f}, {boot['eer_ci'][1]:.4f}] |",
        f"| Test AUC | {auc:.4f} | [{boot['auc_ci'][0]:.4f}, {boot['auc_ci'][1]:.4f}] |",
        f"| Avg contrastive loss (margin={margin}) | {avg_contrastive_loss:.4f} | - |",
        f"| Test writers (n) | {boot['n_test_writers']} | - |",
        f"| Valid bootstrap iterations | {boot['n_valid_iterations']} / {args.bootstrap_iterations} | - |",
    ]
    (exp_dir / "metrics_table.md").write_text("\n".join(metrics_md))
    print(f"[table] wrote metrics_table.md")

    print(f"\nDone. All outputs in {exp_dir} (plots in {plots_dir}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())