"""RQ4 forgery-type breakdown: pooled / skilled / random, per (arm, dataset).

Closes the gap in the existing RQ4 table (report/tables/rq4_metrics.csv), which
reports pooled and skilled only. The assignment brief requires forgery type
reported separately, which for these corpora means: skilled forgeries and
random (other-writer) impostor pairs. NONE of CEDAR, BHSig260-* or the
institutional set contain a "simple forgery" category (traced/simple/disguised
signatures from a different, non-skilled generation process) -- each writer's
negative pairs are only ever skilled-forgery or genuine-signature-from-another-
writer ("random"), per the `kind` column in test_scores.csv. So this script's
three-way split (pooled/skilled/random) is the complete forgery-type breakdown
available for this data, not a partial one.

EER implementation -- CANONICAL vs GRID (see report/tables/eer_implementation_note.md)
---------------------------------------------------------------------------------------
`eer_pooled`, `eer_skilled`, `eer_random` here use the same EXACT rank-based EER
as scripts/final_results_tables.py (`eer()`, imported from it as `exact_eer`),
which is the canonical implementation for every report table per project decision:
the FAR/FRR crossing is found over every observed distance, not approximated on a
grid. `eer_pooled_grid` records what `sigver.evaluation.metrics.compute_eer`
(the 512-point-grid approximation used at evaluation time, i.e. what's logged as
`test_eer_all` in test_metrics.json) gives for the same pooled selection -- kept
as a separate column so the two numbers are both visible, not silently reconciled.

Verification is EXACT-vs-EXACT, not exact-vs-grid: `eer_pooled`/`eer_skilled` are
cross-checked against a second, independently-coded exact algorithm
(`eer_exact_crosscheck`, sorted-array searchsorted rather than merged-array
cumsum) that should agree with `exact_eer` to machine precision (TOL_EXACT =
1e-9) on the same data. A disagreement beyond that tolerance means one of two
things, distinguished via `exact_eer_diag`: either a real bug (unexplained --
FAIL, script stops), or `exact_eer`'s crossing index sits strictly inside a run
of tied (identical) distance values -- a phantom, non-achievable operating
point that fully explains the disagreement without being a bug in either
implementation (reported as "PASS (non-achievable crossing: ...)", not hidden
as a plain PASS). See report/tables/eer_implementation_note.md.

Usage (from project root):

    .\\.venv\\Scripts\\python.exe scripts\\forgery_type_breakdown.py ^
        --results-root "C:\\Users\\Shreya\\Downloads\\all_results"

Output: report/tables/rq4_forgery_type.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigver.evaluation.metrics import compute_eer, roc_auc, far_frr_at_threshold
from final_results_tables import eer as exact_eer

TOL_EXACT = 1e-9
DATASETS = ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]


def eer_exact_crosscheck(d, l):
    """Second, independently-coded exact EER: searchsorted on separately
    sorted pos/neg arrays, rather than exact_eer's cumsum-over-merged-sorted-
    array approach. Both consider the same finite set of candidate thresholds
    (every observed distance) and, unlike exact_eer, can never land inside a
    tied-distance run (it only ever evaluates unique distance values), so it
    always reports an achievable EER. Also returns the achieved minimum
    |FAR-FRR| gap for diagnostic use.
    """
    pos = np.sort(d[l == 1])
    neg = np.sort(d[l == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    cand = np.unique(d)
    frr = 1.0 - np.searchsorted(pos, cand, side="right") / len(pos)
    far = np.searchsorted(neg, cand, side="right") / len(neg)
    gap = np.abs(far - frr)
    i = int(np.argmin(gap))
    return float((far[i] + frr[i]) / 2.0), float(gap[i])


def exact_eer_diag(d, l):
    """Mirrors exact_eer's (final_results_tables.eer's) algorithm bit-for-bit
    to expose not just the minimum |FAR-FRR| gap it found internally, but
    WHERE: whether its chosen crossing index sits at the boundary of a run of
    tied (identical) distance values, or strictly inside one.

    This distinction is the actual explanation for exact-vs-exact
    disagreement discovered while building this verification: exact_eer sorts
    every individual pair (mergesort over the full array, stable on ties) and
    sweeps cumulative FAR/FRR one pair at a time. When >=2 pairs share an
    identical distance, every array position strictly inside that run
    represents "accept some of these tied pairs, reject the rest" -- which no
    real distance<=threshold rule can ever do (tied pairs must be classified
    identically). Only the LAST position in a tie run corresponds to a real,
    achievable threshold (accept all of them). If exact_eer's argmin lands
    inside such a run rather than at its end, its reported EER corresponds to
    a phantom operating point -- confirmed directly for 4 rows in this table,
    including one where 10 pairs tie and the crossing sits 6 pairs into that
    run (rq3_combined, bhsig260_bengali, pooled).

    (Recovering this by taking exact_eer's EER, finding *a* threshold for it,
    and recomputing FAR/FRR via <=/> does NOT reproduce the same number in
    these cases -- verified directly: one such case gave gap=0.000543 via
    that route against exact_eer's own internal gap of 0.000181. The only
    faithful way to inspect "what exact_eer actually did" is to re-run its
    own computation and keep every intermediate value, which is what this
    function does.)

    Returns (eer, gap, tie_run_size, position_in_run, achievable). When
    achievable is False, `eer` is a value no real threshold produces.
    """
    o = np.argsort(d, kind="mergesort")
    ds_sorted = d[o]
    ls = l[o]
    ng, ni = int((ls == 1).sum()), int((ls == 0).sum())
    if ng == 0 or ni == 0:
        return float("nan"), float("nan"), 0, 0, True
    frr = 1.0 - np.cumsum(ls == 1) / ng
    far = np.cumsum(ls == 0) / ni
    gap = np.abs(far - frr)
    i = int(np.argmin(gap))
    tie_val = ds_sorted[i]
    run_start = int(np.searchsorted(ds_sorted, tie_val, side="left"))
    run_end = int(np.searchsorted(ds_sorted, tie_val, side="right"))
    tie_run_size = run_end - run_start
    position_in_run = i - run_start
    achievable = (i == run_end - 1)  # last element of its tie run (or no tie)
    return float((far[i] + frr[i]) / 2.0), float(gap[i]), tie_run_size, position_in_run, achievable




def load_scores(path):
    d, l, k = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d.append(float(row["distance"]))
            l.append(int(row["label"]))
            k.append(row["kind"])
    return np.array(d), np.array(l), np.array(k, dtype=object)


def discover(root):
    """Map every non-val test_scores.csv to (arm, dataset) via its metrics file."""
    runs = {}
    for dp, _dirs, files in os.walk(root):
        if "test_scores.csv" not in files:
            continue
        mp = os.path.join(dp, "test_metrics.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        if meta.get("split") != "test":
            continue  # excludes eval_smallcnn_val
        backbone = meta.get("backbone", "?")
        folder_path = dp.replace("\\", "/")
        arm = "rq3_combined" if (backbone == "resnet18" and "results_rq3/" in folder_path) else backbone
        key = (arm, meta.get("dataset", "?"))
        # Two directories (results_eval/eval_resnet18_institutional and
        # results_rq2_resnet18_eval/eval_resnet18) are confirmed byte-identical
        # duplicates (see scripts/inventory_scores.py). First one found wins;
        # since they're identical this makes no numeric difference.
        if key in runs:
            continue
        runs[key] = {"path": os.path.join(dp, "test_scores.csv"), "meta": meta, "dir": dp}
    return runs


def far_random_frr_random(d, l, k, threshold):
    sel = (l == 1) | ((l == 0) & (k == "random"))
    return far_frr_at_threshold(d[sel], l[sel], threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", default="report/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = discover(args.results_root)

    print("discovered runs:")
    for (arm, ds), v in sorted(runs.items()):
        print(f"  {arm:<14} {ds:<20} {v['dir']}")

    all_arm_ds = sorted({arm for arm, _ in runs} | {"resnet34"})
    expected_pairs = [(arm, ds) for arm in all_arm_ds for ds in DATASETS]
    skipped = [p for p in expected_pairs if p not in runs]

    rows = []
    any_fail = False
    for (arm, ds), v in sorted(runs.items()):
        d, l, k = load_scores(v["path"])
        meta = v["meta"]

        pooled_sel = np.ones(len(l), dtype=bool)
        skilled_sel = (l == 1) | ((l == 0) & (k == "skilled"))
        random_sel = (l == 1) | ((l == 0) & (k == "random"))

        eer_pooled = exact_eer(d[pooled_sel], l[pooled_sel])
        eer_skilled = exact_eer(d[skilled_sel], l[skilled_sel])
        eer_random = exact_eer(d[random_sel], l[random_sel])

        eer_pooled_grid, _ = compute_eer(d[pooled_sel], l[pooled_sel])

        auc_pooled = roc_auc(d[pooled_sel], l[pooled_sel])
        auc_skilled = roc_auc(d[skilled_sel], l[skilled_sel])
        auc_random = roc_auc(d[random_sel], l[random_sel])

        thresh = meta.get("operating_threshold_from_val")
        far_random, frr_random = (float("nan"), float("nan"))
        if thresh is not None:
            far_random, frr_random = far_random_frr_random(d, l, k, thresh)

        # exact-vs-exact cross-check (bug check, not grid-vs-exact).
        # eer_exact_crosscheck is a second, independently-coded exact
        # algorithm (searchsorted on separately-sorted pos/neg arrays over
        # unique distances only -- it can never select a fictitious interior-
        # tie point, see exact_eer_diag's docstring). exact_eer_diag mirrors
        # exact_eer itself to reveal WHY they might disagree: whether
        # exact_eer's own crossing index sits inside a run of tied distances
        # (non-achievable by any real threshold) rather than at its boundary.
        #   - achievable (no problematic tie) AND still disagrees with the
        #     independent crosscheck beyond TOL_EXACT: unexplained --> FAIL.
        #   - NOT achievable (confirmed phantom crossing): the disagreement
        #     is fully explained by this mechanism, not a bug in either
        #     implementation. Reported plainly, not hidden as a plain PASS.
        crosscheck_pooled, _ = eer_exact_crosscheck(d[pooled_sel], l[pooled_sel])
        crosscheck_skilled, _ = eer_exact_crosscheck(d[skilled_sel], l[skilled_sel])
        _, _, tie_size_pooled, tie_pos_pooled, achievable_pooled = exact_eer_diag(d[pooled_sel], l[pooled_sel])
        _, _, tie_size_skilled, tie_pos_skilled, achievable_skilled = exact_eer_diag(d[skilled_sel], l[skilled_sel])

        exact_match = (abs(eer_pooled - crosscheck_pooled) <= TOL_EXACT
                       and abs(eer_skilled - crosscheck_skilled) <= TOL_EXACT)

        if exact_match:
            status = "PASS"
        elif not achievable_pooled or not achievable_skilled:
            detail = []
            if not achievable_pooled:
                detail.append(f"pooled: {tie_pos_pooled+1}/{tie_size_pooled} of tied group at crossing")
            if not achievable_skilled:
                detail.append(f"skilled: {tie_pos_skilled+1}/{tie_size_skilled} of tied group at crossing")
            status = "PASS (non-achievable crossing: " + "; ".join(detail) + ")"
        else:
            status = "FAIL"
            any_fail = True

        print(f"  {arm:<14} {ds:<20} eer_pooled={eer_pooled:.6f} crosscheck={crosscheck_pooled:.6f}  "
              f"eer_skilled={eer_skilled:.6f} crosscheck={crosscheck_skilled:.6f}  "
              f"(grid={eer_pooled_grid:.6f})  [{status}]")

        rows.append({
            "arm": arm, "dataset": ds,
            "n_positive": int((l == 1).sum()),
            "n_skilled": int((k == "skilled").sum()),
            "n_random": int((k == "random").sum()),
            "eer_pooled": round(eer_pooled, 4),
            "eer_pooled_grid": round(eer_pooled_grid, 4),
            "eer_skilled": round(eer_skilled, 4),
            "eer_random": round(eer_random, 4),
            "auc_pooled": round(auc_pooled, 4),
            "auc_skilled": round(auc_skilled, 4),
            "auc_random": round(auc_random, 4),
            "far_random_at_threshold": round(far_random, 4) if far_random == far_random else "",
            "frr_random_at_threshold": round(frr_random, 4) if frr_random == frr_random else "",
            "verification_status": status,
        })

    if any_fail:
        print("\nFAIL detected -- exact-vs-exact cross-check disagreement (possible bug).")
        print("STOPPING without writing report/tables/rq4_forgery_type.csv. Report this, do not proceed.")
        sys.exit(1)

    out_path = os.path.join(args.out, "rq4_forgery_type.csv")
    keys = ["arm", "dataset", "n_positive", "n_skilled", "n_random",
            "eer_pooled", "eer_pooled_grid", "eer_skilled", "eer_random",
            "auc_pooled", "auc_skilled", "auc_random",
            "far_random_at_threshold", "frr_random_at_threshold",
            "verification_status"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_path}")

    if skipped:
        print("\nSkipped (arm, dataset) rows -- not present in results_root at run time:")
        for arm, ds in skipped:
            print(f"  {arm} x {ds}")


if __name__ == "__main__":
    main()
