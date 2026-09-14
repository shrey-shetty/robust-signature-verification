"""RQ2 extension: contrastive-loss vs triplet-loss ResNet-18 comparison.

Produces report/tables/rq2b_loss_comparison.csv (arm pairs resnet18-vs-resnet18_triplet
and smallcnn-vs-resnet18_triplet, all four datasets, eer_all + eer_skilled), using the
exact same paired writer-level bootstrap as every other RQ2/RQ3 table.

Why this can't just be `final_results_tables.py --results-root ...`
---------------------------------------------------------------------
final_results_tables.py's `discover()` names each run by the `backbone` field recorded
in its test_metrics.json, with one special case (folder name contains "rq3" -> arm
"rq3_combined"). Both the contrastive-loss and triplet-loss ResNet-18 runs record
`"backbone": "resnet18"` in their metadata -- there is no metadata field distinguishing
the loss function -- so `discover()` collapses `resnet18_contrastive` and
`resnet18_triplet` onto the same `("resnet18", dataset, "test")` key, and whichever run
`os.walk` happens to visit last silently wins. That collision is exactly why no existing
script invocation reproduces this comparison: it requires telling the loader which
directory is which arm explicitly, rather than inferring it from metadata.

This script sidesteps the collision by hardcoding the arm -> results/final/<dir> mapping
below instead of walking a --results-root. `arm_a="resnet18"` in the output matches the
bare-name convention already used in rq2_backbone_comparison.csv and rq4_metrics.csv (the
contrastive-loss run is what "resnet18" means everywhere else in report/tables/), sourced
here from results/final/resnet18_contrastive/.

Reuses, unmodified, from final_results_tables.py:
    - load_scores()      (CSV -> distance/label/kind/writer_a arrays)
    - paired_bootstrap()  (writer-level paired 95% CI on EER(B) - EER(A))
    - N_BOOT (1000), SEED (42)
Neither final_results_tables.py nor any file it imports is modified by this script.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\loss_comparison_tables.py

Output (default): report/tables/rq2b_loss_comparison_regen.csv
    Pass --out report/tables/rq2b_loss_comparison.csv to write the canonical filename
    once its contents have been reviewed and approved.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import final_results_tables as frt  # noqa: E402 -- reuse load_scores/paired_bootstrap/N_BOOT/SEED, unmodified

load_scores = frt.load_scores
paired_bootstrap = frt.paired_bootstrap
N_BOOT = frt.N_BOOT
SEED = frt.SEED

# arm label used in the output CSV -> source directory under results/final/
ARM_SOURCES = {
    "smallcnn": "smallcnn",
    "resnet18": "resnet18_contrastive",
    "resnet18_triplet": "resnet18_triplet",
}
DATASETS = ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]
PAIRS = [("resnet18", "resnet18_triplet"), ("smallcnn", "resnet18_triplet")]

OUT_FIELDS = ["dataset", "metric", "arm_a", "arm_b", "eer_a", "eer_b", "diff_b_minus_a",
              "ci_lower", "ci_upper", "separates_from_zero", "n_writers", "n_pairs", "caveat"]


def load(arm_label: str, dataset: str, results_final: Path):
    path = results_final / ARM_SOURCES[arm_label] / dataset / "test_scores.csv"
    if not path.exists():
        return None
    return load_scores(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-final", default=str(PROJECT_ROOT / "results" / "final"),
                     help="Directory containing one subfolder per arm (default: results/final)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "report" / "tables" / "rq2b_loss_comparison_regen.csv"),
                     help="Output CSV path (default writes the _regen filename, not the "
                          "canonical one, until the diff has been reviewed)")
    args = ap.parse_args()
    results_final = Path(args.results_final)

    print(f"N_BOOT={N_BOOT} SEED={SEED}")
    print(f"results_final={results_final}")

    rows = []
    for arm_a, arm_b in PAIRS:
        for ds in DATASETS:
            A = load(arm_a, ds, results_final)
            B = load(arm_b, ds, results_final)
            if A is None or B is None:
                missing = arm_a if A is None else arm_b
                print(f"  {ds:<20} {arm_a} vs {arm_b:<18} SKIPPED (missing {missing})")
                continue
            for metric in ("all", "skilled"):
                r = paired_bootstrap(*A, *B, metric=metric)
                if r is None:
                    print(f"  {ds:<20} {arm_a} vs {arm_b:<18} eer_{metric:<8} NOT ROW-ALIGNED")
                    rows.append({
                        "dataset": ds, "metric": f"eer_{metric}",
                        "arm_a": arm_a, "arm_b": arm_b,
                        "eer_a": "", "eer_b": "", "diff_b_minus_a": "",
                        "ci_lower": "", "ci_upper": "", "separates_from_zero": "",
                        "n_writers": "", "n_pairs": "",
                        "caveat": "not row-aligned; paired bootstrap refused",
                    })
                    continue
                print(f"  {ds:<20} {arm_a} vs {arm_b:<18} eer_{metric:<8} "
                      f"{r['eer_a']:.4f} -> {r['eer_b']:.4f}  diff {r['observed_diff']:+.4f}  "
                      f"CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
                      f"{'  *' if r['separates'] else ''}")
                rows.append({
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
