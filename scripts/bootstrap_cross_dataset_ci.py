"""Writer-level bootstrap 95% CIs on test EER for the RQ2 cross-dataset eval runs.

Reuses 03_results_analysis.py's own compute_eer/compute_auc/writer_bootstrap
UNMODIFIED (imported via importlib.util, since that filename starts with a
digit and cannot be `import`ed normally) rather than reimplementing them, so
these intervals are produced by the exact same method as Progress Report 1/2's
published CIs (resample TEST WRITER IDENTITIES with replacement, N=1000,
seed=42) and are directly comparable to them.

Does NOT recompute embeddings from raw images. Each of the seven Kaggle eval
runs already wrote a test_scores.csv (distance, label, kind, writer_a,
writer_b) and a test_metrics.json (point estimates from evaluate_checkpoint.py,
via the canonical sigver.evaluation.metrics.compute_eer). This script reads
those directly.

Resampling unit for `random` pairs (writer_a != writer_b by construction):
uses `writer_a`, matching 03_results_analysis.py's own existing convention
(`anchor_writers = [test_samples[p.i].writer_id for p in test_pairs]`, i.e.
the writer of the pair's first index -- confirmed identical to how
evaluate_checkpoint.py builds its own `writer_a` column).

Point-estimate policy (do not mix estimators): the point_estimate column is
computed with 03_results_analysis.py's own compute_eer, the SAME function
that produces the CI bounds, so estimate and interval are internally
consistent -- that is the number PR3 uses. The canonical test_eer_all /
test_eer_skilled value already stored in test_metrics.json (computed by
evaluate_checkpoint.py via the canonical sigver.evaluation.metrics.compute_eer)
is carried alongside as canonical_point_estimate, purely as a cross-check.
If any run's |point_estimate - canonical_point_estimate| exceeds 0.001, this
script stops without writing output -- see _check_estimator_agreement.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\bootstrap_cross_dataset_ci.py \\
        --results-root "C:\\Users\\nehas\\Downloads\\results_eval"

Output:
    report/dataset_checks/results_rq2_eer_bootstrap_ci.csv
    (columns follow report/dataset_checks/results_eer_bootstrap_ci.csv's
    existing PR2 naming: point_estimate, ci_lower_2.5pct, ci_upper_97.5pct,
    n_resamples -- with backbone/dataset/metric added since this file covers
    seven runs, not one.)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Import 03_results_analysis.py's compute_eer / compute_auc / writer_bootstrap
# UNMODIFIED. importlib is required because the filename starts with a digit.
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "results_analysis_03", PROJECT_ROOT / "scripts" / "03_results_analysis.py"
)
_results_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_results_analysis)
compute_eer = _results_analysis.compute_eer
compute_auc = _results_analysis.compute_auc
writer_bootstrap = _results_analysis.writer_bootstrap
plot_roc = _results_analysis.plot_roc
plot_det = _results_analysis.plot_det
plot_distance_distributions = _results_analysis.plot_distance_distributions

RUNS = [
    ("resnet18", "institutional", "eval_resnet18_institutional"),
    ("resnet18", "cedar", "eval_resnet18_cedar"),
    ("resnet18", "bhsig260_bengali", "eval_resnet18_bhsig260_bengali"),
    ("resnet18", "bhsig260_hindi", "eval_resnet18_bhsig260_hindi"),
    ("smallcnn", "cedar", "eval_smallcnn_cedar"),
    ("smallcnn", "bhsig260_bengali", "eval_smallcnn_bhsig260_bengali"),
    ("smallcnn", "bhsig260_hindi", "eval_smallcnn_bhsig260_hindi"),
]

DIFF_TOLERANCE = 0.001


def load_run(run_dir: Path):
    """Read test_scores.csv + test_metrics.json for one eval run."""
    with (run_dir / "test_scores.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    dists = np.array([float(r["distance"]) for r in rows])
    labels = np.array([float(r["label"]) for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    writer_a = np.array([int(r["writer_a"]) for r in rows])

    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    return dists, labels, kinds, writer_a, metrics


def _check_estimator_agreement(results: list[dict], tolerance: float = DIFF_TOLERANCE) -> None:
    """Abort (no CSV written) if any run's two point estimates disagree
    by more than `tolerance` -- see module docstring's point-estimate
    policy."""
    bad = [r for r in results if abs(r["point_estimate"] - r["canonical_point_estimate"]) > tolerance]
    if bad:
        print("\n[STOP] Point-estimate disagreement exceeds "
              f"{DIFF_TOLERANCE} on {len(bad)} run(s)/metric(s):")
        for r in bad:
            diff = abs(r["point_estimate"] - r["canonical_point_estimate"])
            print(f"  {r['backbone']}/{r['dataset']}/{r['metric']}: "
                  f"03_results_analysis={r['point_estimate']:.4f}  "
                  f"test_metrics.json={r['canonical_point_estimate']:.4f}  "
                  f"diff={diff:.4f}")
        print("\nNot writing output. This means test_scores.csv is being "
              "read/masked differently here than in evaluate_checkpoint.py, "
              "or the two compute_eer implementations disagree by more than "
              "expected -- investigate before trusting any interval below.")
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True,
                    help="directory containing the seven eval_<backbone>_<dataset> "
                         "subdirectories (extracted results_eval.zip)")
    ap.add_argument("--n-iterations", type=int, default=1000,
                    help="bootstrap resamples (default: 1000, matching PR1/PR2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="bootstrap RNG seed (default: 42, matching PR1/PR2)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: report/dataset_checks/"
                         "results_rq2_eer_bootstrap_ci.csv)")
    ap.add_argument("--diff-tolerance", type=float, default=DIFF_TOLERANCE,
                    help="max allowed |point_estimate - canonical_point_estimate| "
                         "before aborting without writing output "
                         f"(default: {DIFF_TOLERANCE}). Only raise this after "
                         "manually confirming a specific disagreement is the "
                         "known compute_eer estimator difference, not a "
                         "masking/parsing bug -- see module docstring.")
    ap.add_argument("--make-institutional-figures", action="store_true",
                    help="also render ROC/DET/distance-distribution plots for "
                         "eval_resnet18_institutional only, via "
                         "03_results_analysis.py's own plot_roc/plot_det/"
                         "plot_distance_distributions (unmodified), reusing "
                         "the same test_scores.csv arrays already loaded here")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "report" / "dataset_checks" / "results_rq2_eer_bootstrap_ci.csv"
    )

    rows: list[dict] = []

    for backbone, dataset, dirname in RUNS:
        run_dir = results_root / dirname
        print(f"\n[{backbone}/{dataset}] loading {run_dir} ...")
        dists, labels, kinds, writer_a, metrics = load_run(run_dir)

        if (args.make_institutional_figures
                and backbone == "resnet18" and dataset == "institutional"):
            figs_dir = PROJECT_ROOT / "report" / "figures" / "results"
            figs_dir.mkdir(parents=True, exist_ok=True)
            plot_roc(dists, labels, figs_dir / "rq2_resnet18_institutional_roc_curve.png")
            plot_det(dists, labels, figs_dir / "rq2_resnet18_institutional_det_curve.png")
            plot_distance_distributions(
                dists, labels, kinds,
                figs_dir / "rq2_resnet18_institutional_distance_distributions.png")
            print(f"  [figures] wrote roc_curve/det_curve/distance_distributions "
                  f"to {figs_dir}")

        skilled_mask = (kinds == "positive") | (kinds == "skilled")

        for metric_name, mask in (("eer_all", np.ones(len(labels), dtype=bool)),
                                  ("eer_skilled", skilled_mask)):
            d_sub, y_sub, w_sub = dists[mask], labels[mask], writer_a[mask]

            point_estimate, _ = compute_eer(d_sub, y_sub)
            canonical_key = "test_eer_all" if metric_name == "eer_all" else "test_eer_skilled"
            canonical_point_estimate = metrics[canonical_key]

            boot = writer_bootstrap(d_sub, y_sub, w_sub,
                                    n_iterations=args.n_iterations, seed=args.seed)

            n_pairs_key = "n_pairs" if metric_name == "eer_all" else "n_pairs_skilled_subset"
            rows.append({
                "backbone": backbone,
                "dataset": dataset,
                "metric": metric_name,
                "resample_unit": "writer",
                "point_estimate": point_estimate,
                "ci_lower_2.5pct": boot["eer_ci"][0],
                "ci_upper_97.5pct": boot["eer_ci"][1],
                "n_resamples": boot["n_valid_iterations"],
                "canonical_point_estimate": canonical_point_estimate,
                "n_test_writers": boot["n_test_writers"],
                "n_pairs": metrics[n_pairs_key],
            })
            print(f"  {metric_name}: point_estimate={point_estimate:.4f} "
                  f"(canonical={canonical_point_estimate:.4f}) "
                  f"CI=[{boot['eer_ci'][0]:.4f}, {boot['eer_ci'][1]:.4f}] "
                  f"n_writers={boot['n_test_writers']} n_pairs={metrics[n_pairs_key]}")

    _check_estimator_agreement(rows, tolerance=args.diff_tolerance)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backbone", "dataset", "metric", "resample_unit",
                 "point_estimate", "ci_lower_2.5pct", "ci_upper_97.5pct",
                 "n_resamples", "canonical_point_estimate",
                 "n_test_writers", "n_pairs"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[out] wrote {out_path}")

    # ---- readable stdout table -----------------------------------------
    header = (f"| {'backbone':9} | {'dataset':17} | {'metric':12} | "
             f"{'point_est':9} | {'95% CI':19} | {'n_writers':9} | {'n_pairs':7} |")
    print("\n" + header)
    print("|" + "-" * (len(header) - 2) + "|")
    for r in rows:
        ci = f"[{r['ci_lower_2.5pct']:.4f}, {r['ci_upper_97.5pct']:.4f}]"
        print(f"| {r['backbone']:9} | {r['dataset']:17} | {r['metric']:12} | "
              f"{r['point_estimate']:.4f}    | {ci:19} | {r['n_test_writers']:9} | "
              f"{r['n_pairs']:7} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
