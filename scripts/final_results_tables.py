"""
Final results tables: RQ2 backbone comparison, RQ3 paired effects, RQ4 metric table.

Runs on saved per-pair scores only -- no GPU, no re-evaluation.

Produces three CSVs ready to paste into the report, plus the paired writer-level
bootstrap CIs that were still missing for RQ3's cross-corpus half.

Method notes
------------
Paired bootstrap: pairs from one writer are not independent, so the resampling unit
is writer_a (the anchor writer), matching scripts/paired_bootstrap_rq2.py and
03_results_analysis.py. One resample per iteration is applied to BOTH arms and the
EER difference taken within it, which cancels the writer-difficulty component
common to both models. 1000 iterations, seed 42.

Row alignment: both arms' evaluations generate pairs via the same
generate_pairs(samples, seed=42+2) call on the same split, so the two score files
should agree row-for-row on label, kind and writer_a. The script verifies this and
REFUSES to compute a paired CI if they disagree -- an unpaired "paired" test is
worse than none.

Usage (from project root):

    .\\.venv\\Scripts\\python.exe scripts\\final_results_tables.py ^
        --results-root "C:\\Users\\Shreya\\Downloads\\all_results"

--results-root is walked recursively for test_scores.csv. Point it at a directory
containing every extracted result zip.

Outputs (under --out, default report/tables/):
    rq2_backbone_comparison.csv
    rq3_dataset_effect.csv
    rq4_metrics.csv
"""

import argparse
import csv
import json
import os

import numpy as np

N_BOOT = 1000
SEED = 42

# Which corpora each arm's validation split was drawn from. This decides whether
# an operating threshold is legitimate for the corpus being evaluated.
TRAINED_ON = {
    "smallcnn":        {"institutional"},
    "resnet18":        {"institutional"},
    "resnet34":        {"institutional"},
    "efficientnet_b0": {"institutional"},
    "rq3_combined":    {"institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"},
}


def threshold_provenance(arm, dataset):
    """own      -- threshold tuned on validation data from this same corpus
    global   -- tuned on a mixed validation set that included this corpus
    carried  -- tuned on a DIFFERENT corpus; threshold-dependent metrics are
                artifacts of threshold-transfer failure, not model quality
    """
    seen = TRAINED_ON.get(arm, set())
    if dataset not in seen:
        return "carried"
    return "own" if len(seen) == 1 else "global"


# ---------------------------------------------------------------- loading

def load_scores(path):
    d, l, k, wa = [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d.append(float(row["distance"]))
            l.append(int(row["label"]))
            k.append(row["kind"])
            wa.append(int(row["writer_a"]))
    return (np.array(d), np.array(l), np.array(k, dtype=object), np.array(wa))


DATASET_UNIVERSE = ["institutional", "cedar", "bhsig260_bengali",
                     "bhsig260_hindi", "gpds_synthetic_4000"]


def rq3_training_mix(root):
    """Read the actual training mix from exp_rq3_combined/config.json.

    Never assume this matches TRAINED_ON["rq3_combined"] above -- read it and
    compare, so a stale hardcoded set can't silently diverge from what was
    actually trained on.
    """
    cfg_path = os.path.join(root, "results_rq3", "exp_rq3_combined", "config.json")
    cfg = json.load(open(cfg_path))
    mix = cfg.get("dataset") or cfg.get("datasets")
    if set(mix) != TRAINED_ON["rq3_combined"]:
        raise RuntimeError(
            f"TRAINED_ON['rq3_combined']={TRAINED_ON['rq3_combined']} does not match "
            f"config.json dataset field {mix} -- update TRAINED_ON before trusting "
            f"threshold_provenance or in_training_mix"
        )
    held_out = [d for d in DATASET_UNIVERSE if d not in mix]
    print(f"RQ3 combined training mix: {mix}")
    print(f"Corpora IN the training mix (improvements are in-domain gains, NOT generalisation): {mix}")
    print(f"Corpora HELD OUT (the genuine unseen-writer generalisation test): {held_out}")
    return set(mix)


def discover(root):
    """Map every test_scores.csv to (arm, dataset, split) using its metrics file."""
    runs = {}
    for dp, _dirs, files in os.walk(root):
        if "test_scores.csv" not in files:
            continue
        sp = os.path.join(dp, "test_scores.csv")
        mp = os.path.join(dp, "test_metrics.json")
        meta = {}
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
            except Exception as e:
                print(f"  (unreadable {mp}: {e})")
        folder = os.path.basename(dp)
        dataset = meta.get("dataset", "?")
        split = meta.get("split", "test")
        backbone = meta.get("backbone", "?")

        # distinguish the two resnet18 arms by folder name: the RQ3 arm was
        # trained on combined data, the RQ2 arm on institutional only.
        if backbone == "resnet18" and "rq3" in folder.lower():
            arm = "rq3_combined"
        else:
            arm = backbone

        runs[(arm, dataset, split)] = {"path": sp, "meta": meta, "folder": folder}
    return runs


# ---------------------------------------------------------------- metrics

def eer(d, l):
    o = np.argsort(d, kind="mergesort")
    ls = l[o]
    ng, ni = int((ls == 1).sum()), int((ls == 0).sum())
    if ng == 0 or ni == 0:
        return float("nan")
    frr = 1.0 - np.cumsum(ls == 1) / ng
    far = np.cumsum(ls == 0) / ni
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0)


def auc(d, l):
    o = np.argsort(d, kind="mergesort")
    ls = l[o]
    ng, ni = int((ls == 1).sum()), int((ls == 0).sum())
    if ng == 0 or ni == 0:
        return float("nan")
    tpr = np.r_[0.0, np.cumsum(ls == 1) / ng]
    fpr = np.r_[0.0, np.cumsum(ls == 0) / ni]
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(tpr, fpr))


def eer_threshold(d, l):
    """The distance at which FAR and FRR cross on THIS data.

    Tuned on the evaluation set itself, so metrics computed here are an
    optimistic upper bound, not a deployable operating point. Reported only to
    separate "the model cannot do better" from "the threshold was wrong".
    """
    o = np.argsort(d, kind="mergesort")
    ds, ls = d[o], l[o]
    ng, ni = int((ls == 1).sum()), int((ls == 0).sum())
    if ng == 0 or ni == 0:
        return float("nan")
    frr = 1.0 - np.cumsum(ls == 1) / ng
    far = np.cumsum(ls == 0) / ni
    return float(ds[int(np.argmin(np.abs(far - frr)))])


def classification_metrics(d, l, thresh):
    acc = d <= thresh                      # genuine = small distance
    TP = int(((l == 1) & acc).sum())
    FN = int(((l == 1) & ~acc).sum())
    FP = int(((l == 0) & acc).sum())
    TN = int(((l == 0) & ~acc).sum())
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (TP + TN) / len(l),
            "far": FP / (FP + TN) if FP + TN else float("nan"),
            "frr": FN / (TP + FN) if TP + FN else float("nan")}


def paired_bootstrap(dA, lA, kA, waA, dB, lB, kB, waB, metric="all"):
    """EER(B) - EER(A) with a writer-level paired 95% CI.

    Returns None when the two score files are not row-aligned, rather than
    silently falling back to an unpaired comparison.
    """
    if not (len(dA) == len(dB)
            and np.array_equal(lA, lB)
            and np.array_equal(waA, waB)
            and np.array_equal(kA.astype(str), kB.astype(str))):
        return None

    if metric == "all":
        sel = np.ones(len(lA), bool)
    elif metric == "skilled":
        sel = (lA == 1) | ((lA == 0) & (kA == "skilled"))
    elif metric == "random":
        sel = (lA == 1) | ((lA == 0) & (kA == "random"))
    else:
        raise ValueError(metric)

    sel_idx = np.where(sel)[0]
    sel_set = np.zeros(len(lA), bool)
    sel_set[sel_idx] = True

    writers = np.unique(waA)
    idx_by_w = {w: np.where(waA == w)[0] for w in writers}
    rng = np.random.default_rng(SEED)

    obs = eer(dB[sel], lB[sel]) - eer(dA[sel], lA[sel])
    diffs = np.empty(N_BOOT)
    for it in range(N_BOOT):
        pick = rng.choice(writers, size=len(writers), replace=True)
        idx = np.concatenate([idx_by_w[w] for w in pick])
        idx = idx[sel_set[idx]]
        if idx.size == 0:
            diffs[it] = np.nan
            continue
        diffs[it] = eer(dB[idx], lB[idx]) - eer(dA[idx], lA[idx])

    good = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(good, [2.5, 97.5])
    return {"eer_a": eer(dA[sel], lA[sel]), "eer_b": eer(dB[sel], lB[sel]),
            "observed_diff": obs, "ci_lower": float(lo), "ci_upper": float(hi),
            "frac_b_worse": float((good > 0).mean()),
            "separates": bool(hi < 0 or lo > 0),
            "n_boot": len(good), "n_writers": len(writers),
            "n_pairs": int(sel.sum())}


def eer_point(d, l, k, wa, metric):
    """EER for one arm's own selection -- used only when two arms are not
    row-aligned and a paired comparison is invalid (see paired_bootstrap)."""
    if metric == "all":
        sel = np.ones(len(l), bool)
    elif metric == "skilled":
        sel = (l == 1) | ((l == 0) & (k == "skilled"))
    else:
        raise ValueError(metric)
    return eer(d[sel], l[sel])


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, float) and not np.isnan(x) else str(x)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", default="report/tables")
    ap.add_argument("--skip", nargs="*", default=[],
                     choices=["rq2", "rq3", "rq4"],
                     help="Skip regenerating these tables (e.g. --skip rq2 to "
                          "leave rq2_backbone_comparison.csv untouched while a "
                          "backbone's evaluations are still running elsewhere).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 78)
    mix = rq3_training_mix(args.results_root)
    print("=" * 78)
    print()

    runs = discover(args.results_root)
    print(f"discovered {len(runs)} evaluation(s):")
    for (arm, ds, sp), v in sorted(runs.items()):
        print(f"  {arm:<14} {ds:<20} {sp:<6} {v['folder']}")
    print()

    cache = {}

    def get(key):
        if key not in cache:
            if key not in runs:
                return None
            cache[key] = load_scores(runs[key]["path"])
        return cache[key]

    # ---------------- RQ2 -------------------------------------------
    if "rq2" in args.skip:
        print("=" * 78)
        print("RQ2 — SKIPPED (--skip rq2): rq2_backbone_comparison.csv left untouched")
        print("=" * 78)
    else:
        print("=" * 78)
        print("RQ2 — backbone comparison, all pairwise combinations "
              "(negative diff = arm_b BETTER, i.e. lower EER)")
        print("=" * 78)
        rq2 = []
        # Fixed order controls arm_a/arm_b assignment for every pair below:
        # for any two arms X before Y in this list, X is arm_a and Y is arm_b.
        # This matches the sanity-check table (smallcnn vs resnet18, resnet18
        # vs resnet34, resnet18 vs efficientnet_b0 all keep that arm as arm_a).
        BACKBONE_ARMS = ["smallcnn", "resnet18", "resnet34", "efficientnet_b0"]
        for ds in ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]:
            for i in range(len(BACKBONE_ARMS)):
                for j in range(i + 1, len(BACKBONE_ARMS)):
                    arm_a, arm_b = BACKBONE_ARMS[i], BACKBONE_ARMS[j]
                    A = get((arm_a, ds, "test"))
                    B = get((arm_b, ds, "test"))
                    if A is None or B is None:
                        missing = arm_a if A is None else arm_b
                        print(f"  {ds:<20} {arm_a} vs {arm_b:<16} SKIPPED (missing {missing})")
                        continue
                    for metric in ("all", "skilled"):
                        r = paired_bootstrap(*A, *B, metric=metric)
                        if r is not None:
                            print(f"  {ds:<20} {arm_a} vs {arm_b:<16} {metric:<8} "
                                  f"{fmt(r['eer_a'])} -> {fmt(r['eer_b'])}  "
                                  f"diff {r['observed_diff']:+.4f}  "
                                  f"CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
                                  f"{'  *' if r['separates'] else ''}")
                            rq2.append({
                                "dataset": ds, "metric": f"eer_{metric}",
                                "arm_a": arm_a, "arm_b": arm_b,
                                "eer_a": round(r["eer_a"], 4),
                                "eer_b": round(r["eer_b"], 4),
                                "diff_b_minus_a": round(r["observed_diff"], 4),
                                "ci_lower": round(r["ci_lower"], 4),
                                "ci_upper": round(r["ci_upper"], 4),
                                "separates_from_zero": r["separates"],
                                "n_writers": r["n_writers"], "n_pairs": r["n_pairs"],
                                "caveat": "",
                            })
                        else:
                            # Not row-aligned: paired bootstrap is invalid here.
                            # Fall back to each arm's own point EER (independent
                            # selections), leave CI columns empty, and flag it --
                            # never present an unpaired interval as paired.
                            eer_a_pt = eer_point(*A, metric)
                            eer_b_pt = eer_point(*B, metric)
                            print(f"  {ds:<20} {arm_a} vs {arm_b:<16} {metric:<8} "
                                  f"NOT ROW-ALIGNED — unpaired point diff only "
                                  f"({fmt(eer_a_pt)} -> {fmt(eer_b_pt)})")
                            rq2.append({
                                "dataset": ds, "metric": f"eer_{metric}",
                                "arm_a": arm_a, "arm_b": arm_b,
                                "eer_a": round(eer_a_pt, 4),
                                "eer_b": round(eer_b_pt, 4),
                                "diff_b_minus_a": round(eer_b_pt - eer_a_pt, 4),
                                "ci_lower": "", "ci_upper": "",
                                "separates_from_zero": "",
                                "n_writers": len(np.unique(A[3])), "n_pairs": len(A[1]),
                                "caveat": "not row-aligned; unpaired",
                            })
        write_csv(os.path.join(args.out, "rq2_backbone_comparison.csv"), rq2)

    # ---------------- RQ3 -------------------------------------------
    print()
    if "rq3" in args.skip:
        print("=" * 78)
        print("RQ3 — SKIPPED (--skip rq3): rq3_dataset_effect.csv left untouched")
        print("=" * 78)
        rq3 = None
    else:
      print("=" * 78)
      print("RQ3 — combined vs institutional-only training (negative = combined BETTER)")
      print("=" * 78)
      rq3 = []
      for ds in ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]:
        A = get(("resnet18", ds, "test"))          # Arm A: institutional only
        B = get(("rq3_combined", ds, "test"))      # Arm B: combined
        if A is None or B is None:
            print(f"  {ds:<20} SKIPPED (missing "
                  f"{'arm A' if A is None else 'arm B'})")
            continue
        for metric in ("all", "skilled"):
            r = paired_bootstrap(*A, *B, metric=metric)
            if r is None:
                print(f"  {ds:<20} {metric:<8} NOT ROW-ALIGNED — paired CI refused")
                continue
            flag = ""
            if ds == "bhsig260_hindi":
                flag = "  [shortcut-contaminated, see report]"
            print(f"  {ds:<20} {metric:<8} armA {fmt(r['eer_a'])}  "
                  f"armB {fmt(r['eer_b'])}  diff {r['observed_diff']:+.4f}  "
                  f"CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
                  f"{'  *' if r['separates'] else ''}{flag}")
            rq3.append({"dataset": ds, "metric": f"eer_{metric}",
                        "eer_institutional_only": round(r["eer_a"], 4),
                        "eer_combined": round(r["eer_b"], 4),
                        "diff_combined_minus_only": round(r["observed_diff"], 4),
                        "ci_lower": round(r["ci_lower"], 4),
                        "ci_upper": round(r["ci_upper"], 4),
                        "separates_from_zero": r["separates"],
                        "n_writers": r["n_writers"], "n_pairs": r["n_pairs"],
                        "caveat": "shortcut-contaminated" if ds == "bhsig260_hindi" else "",
                        "in_training_mix": ds in mix})
      write_csv(os.path.join(args.out, "rq3_dataset_effect.csv"), rq3)

    # ---------------- RQ4 -------------------------------------------
    print()
    if "rq4" in args.skip:
        print("=" * 78)
        print("RQ4 — SKIPPED (--skip rq4): rq4_metrics.csv left untouched")
        print("=" * 78)
        return
    print("=" * 78)
    print("RQ4 — metric table. thr = threshold provenance for this corpus:")
    print("      own = tuned on this corpus's own validation split")
    print("      glo = tuned on a mixed validation set that included this corpus")
    print("      CAR = CARRIED from a different corpus; the threshold-dependent")
    print("            columns below are threshold-transfer artifacts, NOT model quality")
    print("=" * 78)
    print(f"{'arm':<13}{'dataset':<19}{'thr':>5}{'EER':>8}{'skil':>8}{'AUC':>8}"
          f"{'FAR':>8}{'FRR':>8}{'prec':>7}{'rec':>7}{'F1':>7}{'acc':>7}")
    rq4 = []
    for (arm, ds, sp), v in sorted(runs.items()):
        if sp != "test":
            continue
        d, l, k, wa = get((arm, ds, sp))
        th = v["meta"].get("operating_threshold_from_val")
        if th is None:
            print(f"{arm:<13}{ds:<19} no operating threshold recorded — skipped")
            continue

        prov = threshold_provenance(arm, ds)
        tag = {"own": "own", "global": "glo", "carried": "CAR"}[prov]
        cm = classification_metrics(d, l, th)
        skl = (l == 1) | ((l == 0) & (k == "skilled"))

        # oracle operating point: best achievable on this corpus
        oth = eer_threshold(d, l)
        ocm = classification_metrics(d, l, oth)

        row = {"arm": arm, "dataset": ds,
               "threshold": th, "threshold_provenance": prov,
               "eer_all": round(eer(d, l), 4),
               "eer_skilled": round(eer(d[skl], l[skl]), 4),
               "auc": round(auc(d, l), 4),
               "far": round(cm["far"], 4), "frr": round(cm["frr"], 4),
               "precision": round(cm["precision"], 4),
               "recall": round(cm["recall"], 4),
               "f1": round(cm["f1"], 4),
               "accuracy": round(cm["accuracy"], 4),
               "TP": cm["TP"], "FP": cm["FP"], "TN": cm["TN"], "FN": cm["FN"],
               "oracle_threshold": round(oth, 4),
               "oracle_far": round(ocm["far"], 4),
               "oracle_frr": round(ocm["frr"], 4),
               "oracle_precision": round(ocm["precision"], 4),
               "oracle_recall": round(ocm["recall"], 4),
               "oracle_f1": round(ocm["f1"], 4),
               "oracle_accuracy": round(ocm["accuracy"], 4),
               "n_pairs": len(l)}
        rq4.append(row)
        print(f"{arm:<13}{ds:<19}{tag:>5}{row['eer_all']:>8.4f}{row['eer_skilled']:>8.4f}"
              f"{row['auc']:>8.4f}{row['far']:>8.4f}{row['frr']:>8.4f}"
              f"{row['precision']:>7.4f}{row['recall']:>7.4f}{row['f1']:>7.4f}"
              f"{row['accuracy']:>7.4f}")

    write_csv(os.path.join(args.out, "rq4_metrics.csv"), rq4)

    carried = [r for r in rq4 if r["threshold_provenance"] == "carried"]
    if carried:
        print()
        print("-" * 78)
        print("CARRIED-THRESHOLD ROWS — do not report these threshold-dependent columns")
        print("as verification performance. Each uses an operating point tuned on a")
        print("different corpus, which is the threshold-transfer failure already")
        print("documented as a finding. The oracle_* columns show what the SAME model")
        print("achieves with a correct operating point (tuned on the evaluation data")
        print("itself, so an optimistic upper bound, not deployable).")
        print("-" * 78)
        print(f"{'arm':<13}{'dataset':<19}{'FAR':>8}{'oracFAR':>9}{'F1':>8}{'oracF1':>8}")
        for r in carried:
            print(f"{r['arm']:<13}{r['dataset']:<19}{r['far']:>8.4f}"
                  f"{r['oracle_far']:>9.4f}{r['f1']:>8.4f}{r['oracle_f1']:>8.4f}")

    print()
    print("NOTE: EER and AUC are threshold-free and comparable across every row.")
    print("Everything else depends on the operating point and is only meaningful")
    print("where threshold_provenance is 'own' or 'global'. For cross-corpus")
    print("comparison in the report, use EER and AUC.")


if __name__ == "__main__":
    main()
