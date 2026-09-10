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

    --dataset also accepts a comma-separated list, e.g.
    --dataset institutional,cedar, to train on the union of writers from
    several corpora (RQ3). Each dataset keeps its own split CSV under
    data/splits/; writer IDs are namespaced per dataset before pairs are
    generated so two datasets' raw writer_id ranges can never collide (see
    sigver.data.datasets.list_samples_multi).

Path/device overrides (defaults reproduce local behavior exactly):
    --raw-root <dir>   raw-data root (default data/raw under project root;
                       e.g. /kaggle/input/<dataset-name> on Kaggle)
    --out / --out-dir <dir>  output dir (default experiments/siamese_<backbone>_<dataset>)
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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigver.data.datasets import (  # noqa: E402
    list_samples_multi, writer_counts_by_dataset, SignatureDataset,
)
from sigver.data.pairs import generate_pairs, pair_summary, PairDataset  # noqa: E402
from sigver.data.augmentation import MorphAugment  # noqa: E402
from sigver.models.siamese import SiameseNetwork  # noqa: E402
from sigver.models.backbones import build_backbone, BACKBONE_NAMES  # noqa: E402
from sigver.data.preprocessing import PREPROCESSING_VERSION  # noqa: E402
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


def _git_provenance() -> dict:
    """Record which code produced this run. Never fails the run."""
    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def _dataset_list(value: str) -> list[str]:
    """argparse type for --dataset: comma-separated list, single name still works."""
    names = [v.strip() for v in value.split(",")]
    if any(not n for n in names):
        raise argparse.ArgumentTypeError(
            f"--dataset: empty dataset name in {value!r} (expected e.g. "
            f"'cedar' or 'institutional,cedar')")
    return names


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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cedar", type=_dataset_list,
                    help="dataset name, or comma-separated list to train on "
                         "the union (e.g. 'institutional,cedar')")
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
                    help="output dir (default experiments/siamese_<backbone>_<dataset>)")
    ap.add_argument("--raw-root", default=None,
                    help="override raw-data root (default data/raw under the "
                         "project root; e.g. /kaggle/input/<dataset-name> on Kaggle)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="override device autodetection (default: cuda if available)")
    ap.add_argument("--resume", default=None,
                    help="path to a last_checkpoint.pt to resume training from")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes (0=main thread only; "
                         "use >0 with --no-cache to avoid CPU-bound disk I/O "
                         "starving the GPU, e.g. on institutional/GPDS)")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop early if val_eer_all hasn't improved for this "
                         "many epochs (default: None = no early stopping, "
                         "run the full --epochs count). Recomputed from "
                         "history.json each epoch, so it stays correct "
                         "across --resume.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="L2 weight decay passed to Adam (default: 0.0, i.e. "
                         "off, matching all prior runs). A direct lever "
                         "against overfitting, distinct from --lr.")
    ap.add_argument("--backbone", default="smallcnn", choices=BACKBONE_NAMES,
                    help="embedding backbone (default: smallcnn, the PR2 baseline)")
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                    help="load ImageNet weights (ignored for smallcnn)")
    ap.add_argument("--l2-normalize", action="store_true",
                    help="L2-normalise embeddings; changes the distance scale, "
                         "so --margin must be revisited if set")
    ap.add_argument("--input-norm", default="none",
                    choices=["none", "symmetric", "imagenet"],
                    help="input standardisation (default: none, baseline-identical)")
    ap.add_argument("--max-positive-per-writer", type=int, default=None,
                    help="cap genuine-genuine combinations per writer when "
                         "building pairs (default: None = use all "
                         "combinations, i.e. today's behaviour unchanged). "
                         "Set this when combining corpora with different "
                         "genuine-samples-per-writer counts (RQ3) so every "
                         "writer contributes the same number of positive "
                         "pairs regardless of source dataset, instead of "
                         "corpora with more genuine samples per writer "
                         "dominating the positive-pair count.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")
    dataset_tag = "+".join(args.dataset)  # filesystem-safe even for a combined list
    out_dir = Path(args.out) if args.out else (
        Path("experiments") / f"siamese_{args.backbone}_{dataset_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root) if args.raw_root else None

    # ---- data -------------------------------------------------------------
    print(f"[data] loading {args.dataset} (cache={args.cache}) ...")
    t0 = time.time()
    train_samples, dataset_offsets = list_samples_multi(args.dataset, "train", raw_root=raw_root)
    val_samples, _ = list_samples_multi(args.dataset, "val", raw_root=raw_root)
    train_writer_counts = writer_counts_by_dataset(train_samples, dataset_offsets)
    val_writer_counts = writer_counts_by_dataset(val_samples, dataset_offsets)
    print(f"[data] train writers by dataset: {train_writer_counts}")
    print(f"[data] val writers by dataset:   {val_writer_counts}")

    print(f"[data] max_positive_per_writer: {args.max_positive_per_writer}")
    train_pairs = generate_pairs(train_samples, seed=args.seed,
                                 max_positive_per_writer=args.max_positive_per_writer)
    val_pairs = generate_pairs(val_samples, seed=args.seed + 1,
                               max_positive_per_writer=args.max_positive_per_writer)
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # ---- model ------------------------------------------------------------
    backbone = build_backbone(args.backbone,
                              embedding_dim=args.embedding_dim,
                              pretrained=args.pretrained and args.backbone != "smallcnn",
                              l2_normalize=args.l2_normalize,
                              input_norm=args.input_norm)
    model = SiameseNetwork(backbone=backbone).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    print(f"[model] device={device}, params="
          f"{sum(p.numel() for p in model.parameters()):,}")

    config = dict(vars(args))
    config["augment"] = repr(augment)
    config["git"] = _git_provenance()
    config["preprocessing_version"] = PREPROCESSING_VERSION
    config["dataset_offsets"] = dataset_offsets
    config["train_writer_counts"] = train_writer_counts
    config["val_writer_counts"] = val_writer_counts
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

        if args.patience is not None:
            best_epoch = min(history, key=lambda h: h["val_eer_all"])["epoch"]
            epochs_since_best = epoch - best_epoch
            print(f"          {epochs_since_best} epoch(s) since best "
                  f"(patience={args.patience})")
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no val_eer_all improvement in "
                      f"{epochs_since_best} epochs -> stopping at epoch {epoch}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eer": best_eer,
                    "history": history,
                    "rng_state": _rng_state(),
                }, out_dir / "last_checkpoint.pt")
                print(f"\nDone (early-stopped). Best val EER: {best_eer:.4f}. "
                      f"Artifacts in {out_dir}")
                return 0

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