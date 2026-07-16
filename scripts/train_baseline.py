"""Train the baseline Siamese network with contrastive loss.

Writer-independent protocol: train pairs come only from train-split
writers; validation pairs only from val-split writers. The checkpoint
with the best validation EER is kept. The test split is NOT touched
here — final test evaluation is a separate, deliberate step.

Training-time augmentation: MorphAugment (random stroke erode/dilate)
is applied to TRAIN samples only, to decorrelate global stroke width
from the class label (CEDAR shortcut audit — see
report/dataset_checks/preproc_cedar_shortcut_eer.csv). Validation
stays deterministic.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\train_baseline.py --dataset cedar --epochs 10

Path/device overrides (defaults reproduce local behavior exactly):
    --raw-root <dir>   raw-data root (default data/raw under project root;
                       e.g. /kaggle/input/<dataset-name> on Kaggle)
    --out / --out-dir <dir>  output dir (default experiments/siamese_smallcnn_<dataset>)
    --device cpu|cuda  override autodetection (default: cuda if available)
    --resume <path>    resume from a last_checkpoint.pt (see below)

Outputs (under --out, default experiments/<name>):
    best_model.pt      state dict of the best-val-EER model
    best_info.json     which epoch the checkpoint is + its val metrics
    history.json       per-epoch train loss, val EER (all and skilled-only),
                       val AUC, best threshold
    config.json        run configuration snapshot (incl. augmentation)
    last_checkpoint.pt full resumable state (model, optimizer, epoch, best
                       EER so far, history, RNG states) written every epoch
                       -- distinct from best_model.pt so evaluate_checkpoint.py
                       keeps loading a bare state dict unchanged. Use
                       --resume last_checkpoint.pt to continue an
                       interrupted run (e.g. after a Kaggle 12h session cap)
                       with consistent epoch numbering, best-model
                       selection (min val_eer_all), and history.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigver.data.datasets import list_samples, SignatureDataset  # noqa: E402
from sigver.data.pairs import generate_pairs, pair_summary, PairDataset  # noqa: E402
from sigver.data.augmentation import MorphAugment  # noqa: E402
from sigver.models.siamese import SiameseNetwork  # noqa: E402
from sigver.losses import ContrastiveLoss  # noqa: E402
from sigver.evaluation.metrics import compute_eer, roc_auc  # noqa: E402


def _rng_state() -> dict:
    """Snapshot torch/numpy/python-random RNG state (+ CUDA if in use)."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def evaluate(model, loader, device):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cedar")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                    help="cache preprocessed images in RAM (fine for CEDAR/"
                         "BHSig; use --no-cache for GPDS/institutional)")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable training-time morphological augmentation")
    ap.add_argument("--out", "--out-dir", dest="out", default=None,
                    help="output dir (default experiments/siamese_smallcnn_<dataset>)")
    ap.add_argument("--raw-root", default=None,
                    help="override raw-data root (default data/raw under the "
                         "project root; e.g. /kaggle/input/<dataset-name> on Kaggle)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="override device autodetection (default: cuda if available)")
    ap.add_argument("--resume", default=None,
                    help="path to a last_checkpoint.pt to resume training from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")
    out_dir = Path(args.out) if args.out else (
        Path("experiments") / f"siamese_smallcnn_{args.dataset}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root) if args.raw_root else None

    # ---- data -------------------------------------------------------------
    print(f"[data] loading {args.dataset} (cache={args.cache}) ...")
    t0 = time.time()
    train_samples = list_samples(args.dataset, "train", raw_root=raw_root)
    val_samples = list_samples(args.dataset, "val", raw_root=raw_root)

    train_pairs = generate_pairs(train_samples, seed=args.seed)
    val_pairs = generate_pairs(val_samples, seed=args.seed + 1)
    print(f"[data] train pairs: {pair_summary(train_pairs)}")
    print(f"[data] val pairs:   {pair_summary(val_pairs)}")

    # train gets stroke-width augmentation; val stays deterministic
    augment = None if args.no_augment else MorphAugment(seed=args.seed)
    print(f"[data] augmentation: {augment}")

    train_ds = PairDataset(SignatureDataset(train_samples, cache_in_memory=args.cache,
                                            transform=augment),
                           train_pairs)
    val_ds = PairDataset(SignatureDataset(val_samples, cache_in_memory=args.cache),
                         val_pairs)
    print(f"[data] ready in {time.time() - t0:.1f}s")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- model ------------------------------------------------------------
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[model] device={device}, params="
          f"{sum(p.numel() for p in model.parameters()):,}")

    config = dict(vars(args))
    config["augment"] = repr(augment)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2,
                                                    default=str))

    # ---- skilled-only mask for validation reporting ------------------------
    val_kinds = np.array([p.kind for p in val_pairs])
    skilled_mask = (val_kinds == "positive") | (val_kinds == "skilled")

    # ---- resume (optional) -------------------------------------------------
    start_epoch = 1
    history = []
    best_eer = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        print(f"[resume] loading {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _restore_rng_state(ckpt["rng_state"])
        best_eer = ckpt["best_eer"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] resuming at epoch {start_epoch} "
              f"(best val EER so far: {best_eer:.4f})")

    # ---- training loop ------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t_epoch = time.time()
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            optimizer.zero_grad()
            ea, eb = model(a, b)
            loss = criterion(ea, eb, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        dists, labels = evaluate(model, val_loader, device)
        eer_all, thr = compute_eer(dists, labels)
        auc = roc_auc(dists, labels)
        eer_skilled, _ = compute_eer(dists[skilled_mask], labels[skilled_mask])

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_eer_all": round(eer_all, 4),
            "val_eer_skilled": round(eer_skilled, 4),
            "val_auc": round(auc, 4),
            "val_threshold": round(thr, 4),
            "seconds": round(time.time() - t_epoch, 1),
        })
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} "
              f"val_EER={eer_all:.4f} val_EER_skilled={eer_skilled:.4f} "
              f"AUC={auc:.4f} ({history[-1]['seconds']}s)")

        if eer_all < best_eer:
            best_eer = eer_all
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            # sidecar: makes the checkpoint self-documenting without
            # changing its format (evaluate_checkpoint.py keeps working)
            (out_dir / "best_info.json").write_text(json.dumps(
                history[-1] | {"selected_by": "min val_eer_all"}, indent=2))
            print(f"          new best (EER {best_eer:.4f}) -> saved")

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # last_checkpoint.pt: full resumable state, distinct from
        # best_model.pt (a bare state_dict so evaluate_checkpoint.py keeps
        # working unchanged). Written every epoch so a Kaggle session that
        # hits the 12h cap can resume with --resume without losing an epoch.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_eer": best_eer,
            "history": history,
            "rng_state": _rng_state(),
        }, out_dir / "last_checkpoint.pt")

    print(f"\nDone. Best val EER: {best_eer:.4f}. Artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
