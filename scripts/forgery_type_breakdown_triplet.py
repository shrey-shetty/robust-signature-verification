"""RQ4 forgery-type breakdown for resnet18_triplet, kept separate from
forgery_type_breakdown.py's own --results-root run.

Why this can't just be `forgery_type_breakdown.py --results-root ...`
-----------------------------------------------------------------------
forgery_type_breakdown.py's `discover()` names each run by the `backbone` field
recorded in its test_metrics.json (with one hardcoded special case for
"rq3_combined"). Both the contrastive-loss and triplet-loss ResNet-18 runs record
`"backbone": "resnet18"` -- there is no metadata field distinguishing the loss
function -- so a --results-root walk containing both would silently collapse
resnet18_contrastive and resnet18_triplet onto the same ("resnet18", dataset) key,
with whichever directory os.walk visits last winning with no warning. This is the
same collision documented in scripts/loss_comparison_tables.py for RQ2. This script
sidesteps it by hardcoding the results/final/resnet18_triplet/<dataset>/ path
instead of walking a --results-root.

Threshold: rq4_metrics_with_triplet.csv already records resnet18_triplet's
operating threshold as 6.6311 (own, tuned on institutional validation), carried
unchanged to the three public corpora -- this script reuses that value directly
rather than recomputing it. Note for the report: 6.6311 is on a different scale
from the contrastive arms' thresholds (~0.52-0.73) because triplet loss without
L2-normalized embeddings leaves distances unbounded; thresholds are comparable
within a loss function, not across one. eer_pooled/eer_skilled/eer_random are
threshold-independent and are expected to match rq4_metrics_with_triplet.csv's
0.1524 / 0.2136 (institutional) exactly -- only far_random_at_threshold and
frr_random_at_threshold depend on the 6.6311 value.

Reuses, unmodified, from forgery_type_breakdown.py:
    - load_scores(), exact_eer (itself imported there from final_results_tables.eer)
    - compute_eer (sigver.evaluation.metrics, imported there)
    - roc_auc, far_frr_at_threshold
    - eer_exact_crosscheck(), exact_eer_diag()   (the exact-vs-exact verification pair)
Neither forgery_type_breakdown.py nor any file it imports is modified by this script.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\forgery_type_breakdown_triplet.py

Output (default): report/tables/rq4_forgery_type_triplet_regen.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import forgery_type_breakdown as ftb  # noqa: E402 -- reuse everything below unmodified

load_scores = ftb.load_scores
exact_eer = ftb.exact_eer
compute_eer = ftb.compute_eer
roc_auc = ftb.roc_auc
far_frr_at_threshold = ftb.far_frr_at_threshold
eer_exact_crosscheck = ftb.eer_exact_crosscheck
exact_eer_diag = ftb.exact_eer_diag
TOL_EXACT = ftb.TOL_EXACT

ARM = "resnet18_triplet"
DATASETS = ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]
THRESHOLD = 6.6311  # from report/tables/rq4_metrics_with_triplet.csv, resnet18_triplet rows

OUT_FIELDS = ["arm", "dataset", "n_positive", "n_skilled", "n_random",
              "eer_pooled", "eer_pooled_grid", "eer_skilled", "eer_random",
              "auc_pooled", "auc_skilled", "auc_random",
              "far_random_at_threshold", "frr_random_at_threshold",
              "verification_status"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-final", default=str(PROJECT_ROOT / "results" / "final"),
                     help="Directory containing one subfolder per arm (default: results/final)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "report" / "tables" / "rq4_forgery_type_triplet_regen.csv"))
    args = ap.parse_args()
    results_final = Path(args.results_final)

    rows = []
    any_fail = False
    for ds in DATASETS:
        path = results_final / ARM / ds / "test_scores.csv"
        if not path.exists():
            print(f"  {ARM:<18} {ds:<20} SKIPPED (missing {path})")
            continue
        d, l, k = load_scores(str(path))

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

        far_random, frr_random = far_frr_at_threshold(
            d[random_sel], l[random_sel], THRESHOLD)

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

        print(f"  {ARM:<18} {ds:<20} eer_pooled={eer_pooled:.6f} crosscheck={crosscheck_pooled:.6f}  "
              f"eer_skilled={eer_skilled:.6f} crosscheck={crosscheck_skilled:.6f}  "
              f"eer_random={eer_random:.6f}  (grid={eer_pooled_grid:.6f})  [{status}]")

        rows.append({
            "arm": ARM, "dataset": ds,
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
        print("STOPPING without writing output. Report this, do not proceed.")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
