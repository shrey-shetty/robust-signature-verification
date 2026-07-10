"""Standalone Section-4 check: CEDAR brightness-shortcut test.

Runs independently of the 02 notebook (safe to execute in a second
terminal while the full sweep is still running — read-only on raw data).

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\check_shortcut.py

Outputs:
    report/dataset_checks/preproc_cedar_shortcut_eer.csv
    report/figures/preproc/cedar_ink_intensity_by_class.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2  # noqa: E402
import matplotlib

matplotlib.use("Agg")  # no display needed; save figures only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigver.data.catalog import build_catalog  # noqa: E402
from sigver.data.preprocessing import preprocess_image  # noqa: E402

RNG_SEED = 42
np.random.seed(RNG_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "report" / "figures" / "preproc"
TABLE_DIR = PROJECT_ROOT / "report" / "dataset_checks"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def intensity_only_eer(stats: pd.DataFrame, n_pairs: int = 4000,
                       seed: int = RNG_SEED) -> float:
    """EER using |mean ink intensity difference| alone as the match score.

    Positive pairs: genuine-genuine (same writer).
    Negative pairs: genuine vs skilled forgery (same writer).
    Accept if score < threshold.
    """
    rng = np.random.default_rng(seed)
    by_writer_lbl = {
        k: g["mean_ink_intensity"].to_numpy()
        for k, g in stats.groupby(["writer_id", "label"], observed=True)
    }
    writers = sorted({w for (w, _) in by_writer_lbl})
    scores, labels = [], []
    for _ in range(n_pairs):
        w = writers[rng.integers(len(writers))]
        gen = by_writer_lbl.get((w, "genuine"))
        forg = by_writer_lbl.get((w, "forgery"))
        if gen is None or len(gen) < 2 or forg is None:
            continue
        i, j = rng.choice(len(gen), 2, replace=False)
        scores.append(abs(gen[i] - gen[j])); labels.append(1)
        scores.append(abs(gen[rng.integers(len(gen))]
                          - forg[rng.integers(len(forg))])); labels.append(0)
    scores, labels = np.asarray(scores), np.asarray(labels)
    thrs = np.unique(scores)
    best = 0.5
    for thr in thrs:
        far = float((scores[labels == 0] < thr).mean())
        frr = float((scores[labels == 1] >= thr).mean())
        if abs(far - frr) < 0.01:
            best = min(best, (far + frr) / 2)
    return best


def stat_eer(stats: pd.DataFrame, col: str) -> float:
    """Run intensity_only_eer using an arbitrary column as the score."""
    return intensity_only_eer(
        stats.rename(columns={col: "mean_ink_intensity"})
        [["writer_id", "label", "mean_ink_intensity"]]
    )


def main() -> int:
    catalog = build_catalog(RAW_ROOT)
    df = catalog.dropna(subset=["label"])
    ced = df[df["dataset"] == "cedar"]
    print(f"CEDAR files: {len(ced):,}")

    parts = [
        g.sample(min(len(g), 400), random_state=RNG_SEED)
        for _, g in ced.groupby("label", observed=True)
    ]  # pandas 3.0: keeps 'label' column (groupby().apply() would drop it)
    per_class = pd.concat(parts).reset_index(drop=True)

    records = []
    for _, row in per_class.iterrows():
        try:
            out = preprocess_image(row["path"])
        except ValueError:
            continue
        ink = out[out > 0]
        records.append({
            "label": row["label"],
            "writer_id": row["writer_id"],
            "mean_ink_intensity": float(ink.mean()),
            "ink_fraction": float((out > 0).mean()),
        })
    ink_stats = pd.DataFrame(records)
    print(f"Preprocessed {len(ink_stats)} images\n")

    print(ink_stats.groupby("label")[["mean_ink_intensity", "ink_fraction"]]
          .describe().round(4).T)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=ink_stats, x="mean_ink_intensity", hue="label",
                common_norm=False, fill=True,
                palette={"genuine": "#2a9d8f", "forgery": "#e76f51"}, ax=ax)
    ax.set_title("CEDAR preprocessed: mean ink intensity by class")
    fig.savefig(FIG_DIR / "cedar_ink_intensity_by_class.png",
                dpi=150, bbox_inches="tight")

    eer = intensity_only_eer(ink_stats)
    print(f"\nIntensity-only EER (CEDAR, genuine vs skilled forgery): {eer:.3f}")
    print("~0.5 => brightness carries no signal (good). Meaningfully below "
          "0.5 => shortcut survives preprocessing; consider per-image ink "
          "normalization or binarization before freezing.")

    pd.DataFrame([{"intensity_only_eer": eer,
                   "n_images": len(ink_stats)}]).to_csv(
        TABLE_DIR / "preproc_cedar_shortcut_eer.csv", index=False)
    print(f"\nSaved -> {TABLE_DIR / 'preproc_cedar_shortcut_eer.csv'}")

    # fragmentation: connected components in the final binary output
    n_comp = []
    for _, row in per_class.iterrows():
        try:
            out = preprocess_image(row["path"])
        except ValueError:
            continue
        n, _ = cv2.connectedComponents((out > 0.5).astype(np.uint8))
        n_comp.append(n - 1)  # minus background
    ink_stats["n_components"] = n_comp

    for col in ["mean_ink_intensity", "ink_fraction", "n_components"]:
        print(f"{col:22s} EER {stat_eer(ink_stats, col):.3f} | "
              f"forgery {ink_stats[ink_stats.label=='forgery'][col].mean():.3f} "
              f"genuine {ink_stats[ink_stats.label=='genuine'][col].mean():.3f}")

    # Attribution check: does raw ink darkness predict preprocessed
    # stroke width (mean intensity of the binary output)?
    from sigver.data.preprocessing import otsu_threshold
    from PIL import Image

    pairs_rd = []
    for _, row in per_class.iterrows():
        try:
            out = preprocess_image(row["path"])
        except ValueError:
            continue
        with Image.open(row["path"]) as img:
            g = np.asarray(img.convert("L"), dtype=np.uint8)
        t = otsu_threshold(g)
        ink_px = g[g <= t]
        if ink_px.size == 0:
            continue
        pairs_rd.append((float(ink_px.mean()),
                         float(out[out > 0].mean())))
    a = np.asarray(pairs_rd)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    print(f"\ncorr(raw ink darkness, preprocessed mean intensity): {r:.3f}  "
          f"(n={len(a)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())