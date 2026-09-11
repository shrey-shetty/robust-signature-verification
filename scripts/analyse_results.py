"""
ROC curves + negative-kind diagnosis, computed from saved per-pair scores.

No GPU, no re-evaluation. Walks a directory for test_scores.csv files and, for each:

  1. Splits the EER three ways -- positives vs ALL negatives, vs SKILLED only,
     vs RANDOM only. This is what localises the BHSig260-Hindi anomaly, where
     skilled EER (0.0841) came in far BELOW all-pairs EER (0.2035), implying
     random negatives are harder than skilled ones -- the opposite of the
     expected ordering.
  2. Reports distance distributions per pair kind, so you can see whether the
     inversion is a real property of the corpus or a pairing/labelling problem.
  3. Plots ROC curves (all-pairs and skilled-only), which the exposé commits to
     in section 4 and which nothing has generated yet.

Usage (from project root, after extracting the result zips somewhere):

    .\.venv\Scripts\python.exe analyse_results.py --results-root "C:\\path\\to\\extracted"

Writes:
    <out>/roc_<label>.png        one ROC figure per score file
    <out>/roc_combined.png       all arms overlaid, all-pairs
    <out>/negative_kind_diagnosis.csv
"""

import argparse
import csv
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
    print("matplotlib not installed -- diagnosis will run, plots will be skipped.")
    print("install with: pip install matplotlib")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def roc_points(dist, label):
    """ROC for a distance-based verifier: genuine = SMALL distance.

    Returns (fpr, tpr, thresholds) with thresholds ascending, where a pair is
    accepted when distance <= threshold.
    """
    order = np.argsort(dist, kind="mergesort")
    d, l = dist[order], label[order]
    n_gen = int((l == 1).sum())
    n_imp = int((l == 0).sum())
    if n_gen == 0 or n_imp == 0:
        return None
    tpr = np.cumsum(l == 1) / n_gen          # genuine accepted
    fpr = np.cumsum(l == 0) / n_imp          # impostor accepted
    # prepend the origin (accept nothing)
    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[d[0] - 1e-9, d]


def eer_from_roc(fpr, tpr):
    frr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - frr)))
    return float((fpr[i] + frr[i]) / 2.0), i


def auc_from_roc(fpr, tpr):
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_scores(path):
    d, l, k = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d.append(float(row["distance"]))
            l.append(int(row["label"]))
            k.append(row["kind"])
    return np.array(d), np.array(l), np.array(k, dtype=object)


def find_score_files(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        if "test_scores.csv" in files:
            s = os.path.join(dirpath, "test_scores.csv")
            m = os.path.join(dirpath, "test_metrics.json")
            meta = {}
            if os.path.exists(m):
                try:
                    meta = json.load(open(m))
                except Exception as e:
                    print(f"  (could not read {m}: {e})")
            label = os.path.basename(dirpath)
            out.append((label, s, meta))
    return sorted(out)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="directory to walk for test_scores.csv files")
    ap.add_argument("--out", default="report/figures",
                    help="output directory for figures and the diagnosis CSV")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    found = find_score_files(args.results_root)
    if not found:
        raise SystemExit(f"No test_scores.csv under {args.results_root}")

    print(f"found {len(found)} score file(s)\n")
    rows = []
    combined = []

    for label, path, meta in found:
        dist, lab, kind = load_scores(path)
        pos = lab == 1
        is_skilled = (lab == 0) & (kind == "skilled")
        is_random = (lab == 0) & (kind == "random")

        print("=" * 72)
        print(f"{label}   ({meta.get('dataset', '?')}, backbone={meta.get('backbone', '?')})")
        print("=" * 72)
        print(f"  pairs: {len(dist):,}  |  genuine {int(pos.sum()):,}  "
              f"skilled {int(is_skilled.sum()):,}  random {int(is_random.sum()):,}")

        entry = {"label": label,
                 "dataset": meta.get("dataset", ""),
                 "backbone": meta.get("backbone", ""),
                 "n_pairs": len(dist),
                 "n_genuine": int(pos.sum()),
                 "n_skilled": int(is_skilled.sum()),
                 "n_random": int(is_random.sum())}

        # --- three-way EER split --------------------------------------
        for name, negmask in (("all", lab == 0),
                              ("skilled", is_skilled),
                              ("random", is_random)):
            if negmask.sum() == 0:
                entry[f"eer_{name}"] = ""
                entry[f"auc_{name}"] = ""
                continue
            sub = pos | negmask
            r = roc_points(dist[sub], lab[sub])
            if r is None:
                continue
            fpr, tpr, _ = r
            e, _i = eer_from_roc(fpr, tpr)
            a = auc_from_roc(fpr, tpr)
            entry[f"eer_{name}"] = round(e, 4)
            entry[f"auc_{name}"] = round(a, 4)
            print(f"    EER vs {name:<8} {e:.4f}   AUC {a:.4f}")

        # --- the ordering check ---------------------------------------
        es, er = entry.get("eer_skilled"), entry.get("eer_random")
        if isinstance(es, float) and isinstance(er, float):
            if es < er:
                print(f"    >>> INVERTED: skilled ({es:.4f}) is EASIER than random ({er:.4f}).")
                print( "        Skilled forgeries are deliberate imitations and should be")
                print( "        HARDER. Investigate before reporting this corpus.")
                entry["ordering"] = "INVERTED"
            else:
                entry["ordering"] = "expected"

        # --- distance distributions by kind ---------------------------
        print("    distance distribution by pair kind:")
        print(f"      {'kind':<10}{'n':>8}{'mean':>9}{'std':>9}{'p05':>9}{'p50':>9}{'p95':>9}")
        for name, m in (("genuine", pos), ("skilled", is_skilled), ("random", is_random)):
            if m.sum() == 0:
                continue
            v = dist[m]
            print(f"      {name:<10}{int(m.sum()):>8,}{v.mean():>9.4f}{v.std():>9.4f}"
                  f"{np.percentile(v, 5):>9.4f}{np.percentile(v, 50):>9.4f}"
                  f"{np.percentile(v, 95):>9.4f}")
            entry[f"mean_{name}"] = round(float(v.mean()), 4)

        # a genuine/random overlap is the thing to look at when inverted
        if is_random.sum() and pos.sum():
            ov = float((dist[is_random] < np.percentile(dist[pos], 95)).mean())
            entry["random_below_genuine_p95"] = round(ov, 4)
            print(f"    random negatives below the genuine 95th pct: {ov:.1%}")

        rows.append(entry)

        # --- per-file ROC ---------------------------------------------
        if HAVE_PLT:
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            for name, negmask, style in (("all negatives", lab == 0, "-"),
                                         ("skilled only", is_skilled, "--"),
                                         ("random only", is_random, ":")):
                if negmask.sum() == 0:
                    continue
                sub = pos | negmask
                r = roc_points(dist[sub], lab[sub])
                if r is None:
                    continue
                fpr, tpr, _ = r
                e, _ = eer_from_roc(fpr, tpr)
                ax.plot(fpr, tpr, style, lw=1.6,
                        label=f"{name} (EER {e:.3f}, AUC {auc_from_roc(fpr, tpr):.3f})")
            ax.plot([0, 1], [1, 0], color="0.7", lw=0.8, label="EER line")
            ax.plot([0, 1], [0, 1], color="0.85", lw=0.8, ls="--")
            ax.set_xlabel("False Acceptance Rate")
            ax.set_ylabel("True Acceptance Rate (1 - FRR)")
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = os.path.join(args.out, f"roc_{label}.png")
            fig.savefig(p, dpi=200)
            plt.close(fig)
            print(f"    wrote {p}")

            r = roc_points(dist, lab)
            if r is not None:
                combined.append((label, r[0], r[1]))
        print()

    # --- overlay -------------------------------------------------------
    if HAVE_PLT and len(combined) > 1:
        fig, ax = plt.subplots(figsize=(6.0, 5.6))
        for label, fpr, tpr in combined:
            e, _ = eer_from_roc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"{label} (EER {e:.3f})")
        ax.plot([0, 1], [1, 0], color="0.7", lw=0.8)
        ax.set_xlabel("False Acceptance Rate")
        ax.set_ylabel("True Acceptance Rate (1 - FRR)")
        ax.set_title("ROC — all pairs", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.out, "roc_combined.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f"wrote {p}")

    # --- diagnosis CSV -------------------------------------------------
    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        p = os.path.join(args.out, "negative_kind_diagnosis.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {p}")

    inverted = [r["label"] for r in rows if r.get("ordering") == "INVERTED"]
    if inverted:
        print("\n" + "!" * 72)
        print("INVERTED skilled/random ordering in:", ", ".join(inverted))
        print("Do not tabulate those skilled EERs until the cause is understood.")
        print("!" * 72)


if __name__ == "__main__":
    main()
