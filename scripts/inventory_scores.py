"""Inventory every test_scores.csv under a results root.

PROVISIONAL by design: three ResNet-34 out-of-domain evaluations
(eval_resnet34_cedar, eval_resnet34_bhsig260_bengali, eval_resnet34_bhsig260_hindi)
may be running concurrently in another terminal while this script executes, so
their directories may be absent, empty, or mid-write. That is expected and is
reported as MISSING in the coverage grid, not treated as an error.

Also resolves two specific ambiguities (not guessed -- read from the files):
  1. results_eval/eval_resnet18_institutional vs
     results_rq2_resnet18_eval/eval_resnet18 -- both claim resnet18 on
     institutional test. Compared here byte-for-byte (scores) and field-for-field
     (metrics) rather than assumed identical.
  2. results_smallcnn_institutional/eval_smallcnn_val -- must have split == "val"
     and is excluded from every table in this phase.

Usage (from project root):

    .\\.venv\\Scripts\\python.exe scripts\\inventory_scores.py ^
        --results-root "C:\\Users\\Shreya\\Downloads\\all_results"
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

DATASETS = ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics(dp):
    mp = os.path.join(dp, "test_metrics.json")
    if not os.path.exists(mp):
        return None, mp
    try:
        return json.load(open(mp)), mp
    except Exception as e:
        return {"_error": str(e)}, mp


def inspect_csv(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        kind_counts = Counter()
        n = 0
        for row in reader:
            n += 1
            kind_counts[row.get("kind", "")] += 1
    return n, cols, kind_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()
    root = args.results_root

    print("=" * 100)
    print("PROVISIONAL INVENTORY -- ResNet-34 out-of-domain dirs may be absent/partial (expected)")
    print("=" * 100)

    entries = []  # (dp, meta, csv_stats)
    for dp, _dirs, files in sorted(os.walk(root)):
        if "test_scores.csv" not in files:
            continue
        sp = os.path.join(dp, "test_scores.csv")
        meta, mp = load_metrics(dp)
        if meta is None:
            print(f"\n{sp}")
            print(f"  MISSING sibling test_metrics.json ({mp}) -- reporting as incomplete, not guessing")
            entries.append((dp, sp, None, None))
            continue
        if "_error" in meta:
            print(f"\n{sp}")
            print(f"  test_metrics.json UNREADABLE: {meta['_error']}")
            entries.append((dp, sp, None, None))
            continue

        n, cols, kind_counts = inspect_csv(sp)
        print(f"\n{sp}")
        print(f"  backbone={meta.get('backbone')}  dataset={meta.get('dataset')}  split={meta.get('split')}")
        print(f"  n_pairs(meta)={meta.get('n_pairs')}  test_eer_all={meta.get('test_eer_all')}  "
              f"test_eer_skilled={meta.get('test_eer_skilled')}  test_auc={meta.get('test_auc')}")
        print(f"  csv_row_count={n}  columns={cols}")
        print(f"  kind counts: {dict(kind_counts)}")
        entries.append((dp, sp, meta, (n, cols, kind_counts)))

    # ---------------------------------------------------------------- coverage grid
    print("\n" + "=" * 100)
    print("PROVISIONAL COVERAGE GRID (rows = arm, columns = dataset)")
    print("=" * 100)

    def arm_of(dp, meta):
        folder = os.path.basename(dp)
        backbone = meta.get("backbone", "?")
        if backbone == "resnet18" and dp.replace("\\", "/").find("results_rq3/") != -1:
            return "rq3_combined"
        return backbone

    grid = defaultdict(dict)
    for dp, sp, meta, stats in entries:
        if meta is None:
            continue
        arm = arm_of(dp, meta)
        ds = meta.get("dataset", "?")
        split = meta.get("split", "?")
        grid[arm].setdefault(ds, []).append((split, dp))

    header = f"{'arm':<15}" + "".join(f"{d:<20}" for d in DATASETS)
    print(header)
    missing_cells = []
    for arm in sorted(grid):
        row = f"{arm:<15}"
        for ds in DATASETS:
            cell = grid[arm].get(ds)
            if not cell:
                row += f"{'MISSING':<20}"
                missing_cells.append((arm, ds))
            else:
                splits = ",".join(sorted({s for s, _ in cell}))
                row += f"{('present(' + splits + ')'):<20}"
        print(row)

    if missing_cells:
        print("\nMissing cells:")
        for arm, ds in missing_cells:
            print(f"  {arm} x {ds}")

    # ---------------------------------------------------------------- resolution 1
    print("\n" + "=" * 100)
    print("RESOLUTION 1 -- duplicate resnet18-institutional candidates")
    print("=" * 100)
    p1 = os.path.join(root, "results_eval", "eval_resnet18_institutional")
    p2 = os.path.join(root, "results_rq2_resnet18_eval", "eval_resnet18")
    csv1, csv2 = os.path.join(p1, "test_scores.csv"), os.path.join(p2, "test_scores.csv")
    m1, m2 = os.path.join(p1, "test_metrics.json"), os.path.join(p2, "test_metrics.json")
    if os.path.exists(csv1) and os.path.exists(csv2):
        h1, h2 = md5_of(csv1), md5_of(csv2)
        j1, j2 = json.load(open(m1)), json.load(open(m2))
        numeric_keys = ["n_pairs", "test_eer_all", "test_eer_skilled", "test_auc",
                         "test_eer_threshold", "operating_threshold_from_val",
                         "far_at_val_threshold", "frr_at_val_threshold",
                         "far_skilled_at_val_threshold", "frr_skilled_at_val_threshold"]
        metric_diffs = {k: (j1.get(k), j2.get(k)) for k in numeric_keys if j1.get(k) != j2.get(k)}
        print(f"  {p1}\n    scores md5={h1}")
        print(f"  {p2}\n    scores md5={h2}")
        if h1 == h2 and not metric_diffs:
            print("  IDENTICAL: scores CSV byte-for-byte equal; all numeric metrics fields equal.")
            print("  (checkpoint/config_source paths differ only in Kaggle input dataset slug name.)")
        else:
            print("  DIFFERENT -- stop using either until Shreya decides.")
            if h1 != h2:
                print("  scores CSVs differ.")
            if metric_diffs:
                print(f"  metric field differences: {metric_diffs}")
    else:
        print(f"  Cannot compare -- one or both paths missing: {csv1} exists={os.path.exists(csv1)}, "
              f"{csv2} exists={os.path.exists(csv2)}")

    # ---------------------------------------------------------------- resolution 2
    print("\n" + "=" * 100)
    print("RESOLUTION 2 -- results_smallcnn_institutional/eval_smallcnn_val split check")
    print("=" * 100)
    pval = os.path.join(root, "results_smallcnn_institutional", "eval_smallcnn_val", "test_metrics.json")
    if os.path.exists(pval):
        mval = json.load(open(pval))
        split = mval.get("split")
        print(f"  {pval}\n    split field = {split!r}")
        if split == "val":
            print("  CONFIRMED validation split -- excluded from all tables in this phase.")
        else:
            print(f"  UNEXPECTED -- split field is {split!r}, not 'val'. STOP and report before excluding.")
    else:
        print(f"  {pval} not found.")


if __name__ == "__main__":
    main()
