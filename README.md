# Robust Biometric Signature Verification Against Skilled Forgeries

Writer-independent offline signature verification using deep metric learning
(Siamese networks). Trained across five datasets — CEDAR, BHSig260-Bengali,
BHSig260-Hindi, GPDS Synthetic 4000, and a private institutional dataset — to
compare backbones (SmallCNN, ResNet-18, ResNet-34, EfficientNet-B0, ViT-B/32)
and training objectives (contrastive loss, triplet loss) against skilled
forgeries.

**Primary reported outcome is performance on the private institutional
dataset.** The public corpora (CEDAR, BHSig260-*, GPDS) exist to improve
training diversity, not as the headline result.

## Repository structure

```
src/sigver/            installable package (models, data, losses, evaluation)
scripts/               entry-point scripts (training, evaluation, table generation)
configs/                YAML experiment configs (see "Configuration" below)
data/splits/           committed writer-independent train/val/test splits
notebooks/              EDA, preprocessing validation, results analysis
report/tables/          RQ2/RQ3/RQ4 result tables (CSV) + methodology notes
report/dataset_checks/  EDA and data-provenance evidence tables (CSV)
report/figures/         figures referenced by the report
tests/                  pytest regression tests
```

`results/` (per-run checkpoints, `test_scores.csv`, `test_metrics.json`) is
**gitignored and not part of this repository**. Every number in
`report/tables/` was computed from files under `results/final/` on the
machine that ran the evaluations; that directory is not committed, so a
fresh clone of this repo does not itself contain the score files behind the
tables — the tables are the artefact that gets committed, not the score
files that produced them. To regenerate the score files, retrain/evaluate
as described below.

## Installation

Requires Python 3.9+ (developed against 3.11) and, on GPU, a matching CUDA
build of PyTorch already installed in the environment.

```
python -m venv .venv
.venv\Scripts\activate            # Windows; venv\bin\activate on Linux/macOS
pip install -e .                  # installs the local `sigver` package only
pip install -r requirements.txt   # pulls in torch/torchvision + the rest
```

On a machine with a pre-installed, driver-matched PyTorch (e.g. Kaggle),
do **not** run `pip install -r requirements.txt` verbatim — it pins
`torch`/`torchvision` and reinstalling them can silently break GPU support.
See `kaggle_bootstrap.md` for the exact Kaggle-safe install sequence.

## Reproducibility notes

- **Seed 42 everywhere** — splits, training, and bootstrap resampling.
- **Writer-independent splits**: every writer's signatures land entirely in
  one split (`scripts/make_splits.py`); a test writer never appears in
  training. Splits are already generated and committed under
  `data/splits/` — re-running `make_splits.py` with the same seed reproduces
  them identically (it refuses to overwrite existing splits without
  `--force`).
- **Preprocessing is frozen** at `PREPROCESSING_VERSION = 2`
  (`src/sigver/data/preprocessing.py`), tagged `preproc-v2-freeze` in git
  history. Do not change preprocessing behaviour without deliberately
  bumping this version — anything cached or evaluated under a different
  version is not comparable.
- **Two EER implementations exist** in this codebase (an exact rank-based
  one used for every table in `report/tables/`, and a 512-point-grid
  approximation used at evaluation time and logged into `test_metrics.json`).
  They can differ by up to ~0.0018 EER. See
  `report/tables/eer_implementation_note.md` before citing a number from
  `test_metrics.json` directly instead of from `report/tables/`.
- **Cloud training runs from a fresh clone.** Kaggle sessions clone this
  repo at the top of each notebook run (`kaggle_bootstrap.md`) rather than
  reusing local state — a local fix must be committed and pushed before it
  takes effect on Kaggle; a local edit alone is not visible there.

## Kaggle Experiment Notebooks

The following Kaggle notebooks were used during the project's experiment and
evaluation workflow. They are linked here for reference and are not copied
into this repository.

| Experiment | Kaggle notebook |
|---|---|
| Institutional LR=1e-4 experiment | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/kaggle-institutional-lr1e4-experiment) |
| Institutional full training | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/kaggle-institutional-full-training) |
| ViT-B/32 recovery evaluation | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/vit-b-32-recovery-eval) |
| Triplet-loss run | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/kaggle-triplet-run) |
| RQ2 cells | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/kaggle-rq2-cells) |
| SmallCNN institutional evaluation | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/smallcnn-institutional-eval) |
| Results analysis / ROC | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/results-analysis-roc) |
| RQ3 combined datasets | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/rq3-combined-datasets) |
| RQ2 ResNet-18 evaluation | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/rq2-resnet18-eval) |
| Cross-dataset skilled-threshold evaluation | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/cross-dataset-skilled-threshold) |
| Weight-decay experiment | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/signature-verification-weight-decay) |
| Retraining | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/retraining) |
| Cross-evaluation | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/cross-eval) |
| RQ2 ResNet-18 | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/rq2-resnet18) |
| RQ2 cross-dataset evaluation | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/rq2-eval-crossdataset) |
| RQ2 ResNet-34 | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/rq2-resnet34) |
| ViT-B/32 1-epoch experiment | [Kaggle notebook](https://www.kaggle.com/code/shreyshettycoder/kaggle-vit-1epoch) |

### A known reproducibility trap: arm naming collision

Several table-generation scripts (`scripts/final_results_tables.py`,
`scripts/forgery_type_breakdown.py`) discover which run is which arm purely
from the `backbone` field recorded in each run's `test_metrics.json`, with
one hardcoded exception for the RQ3 combined-training arm. **The
contrastive-loss and triplet-loss ResNet-18 runs both record
`"backbone": "resnet18"`** — there is no metadata field distinguishing the
loss function — so pointing `--results-root` at a directory containing both
will silently collapse them onto the same key, with whichever directory
`os.walk` visits last winning and no warning printed. `rq2b_loss_comparison.csv`
and the `resnet18_triplet` rows of `rq4_forgery_type.csv` were therefore
generated by separate, explicit-path scripts
(`scripts/loss_comparison_tables.py`, `scripts/forgery_type_breakdown_triplet.py`)
that hardcode the two runs' directories instead of relying on that
discovery mechanism. If you add another same-backbone, different-loss (or
otherwise metadata-indistinguishable) arm, follow that pattern rather than
`--results-root` auto-discovery.

## Configuration

Training is driven entirely by CLI flags to `scripts/train_baseline.py`, not
by the YAML files under `configs/`. `configs/exp01_smoke.yaml` documents one
specific smoke-test's parameters for the record (see its own header comment
and `kaggle_bootstrap.md` §5) but is not read by any script — run the
equivalent CLI invocation shown there instead. `configs/base.yaml`,
`configs/experiments/*.yaml`, and `scripts/run_experiment.py` are an earlier,
unwired config-composition scaffold, kept for now rather than removed.

## Running the pipeline

All commands below are run from the project root. On Windows, invoke the
project's own interpreter directly (`.\.venv\Scripts\python.exe`) rather
than an activation script.

**1. Generate splits** (already committed under `data/splits/`; only needed
to reproduce them from scratch):
```
.\.venv\Scripts\python.exe scripts\make_splits.py
```

**2. Train a model:**
```
.\.venv\Scripts\python.exe scripts\train_baseline.py --dataset cedar --epochs 10 ^
    --backbone resnet18 --pretrained --loss contrastive
```
Key flags: `--backbone {smallcnn,resnet18,resnet34,efficientnet_b0,vit_b_16,vit_b_32}`,
`--loss {contrastive,triplet}`, `--dataset` (comma-separated for combined
training, e.g. `institutional,cedar`), `--raw-root`, `--resume
<last_checkpoint.pt>` for resuming an interrupted run. Full flag reference:
`scripts/train_baseline.py`'s module docstring and `--help`.

**3. Evaluate a checkpoint on its held-out test split:**
```
.\.venv\Scripts\python.exe scripts\evaluate_checkpoint.py --dataset cedar ^
    --checkpoint experiments\siamese_resnet18_cedar\best_model.pt
```
Writes `test_metrics.json` and `test_scores.csv` next to the checkpoint
(architecture flags are auto-resolved from the run's own `config.json`
unless overridden).

**4. Aggregate report tables** from a directory containing every extracted
evaluation output:
```
.\.venv\Scripts\python.exe scripts\final_results_tables.py --results-root <dir>
.\.venv\Scripts\python.exe scripts\forgery_type_breakdown.py --results-root <dir>
```
Writes `report/tables/{rq2_backbone_comparison,rq3_dataset_effect,rq4_metrics}.csv`
and `report/tables/rq4_forgery_type.csv` respectively. See the reproducibility
trap above before pointing either at a directory containing both ResNet-18
loss variants.

**5. Regenerate the RQ findings summary:**
```
.\.venv\Scripts\python.exe scripts\build_rq_findings.py
```

**Run the tests:**
```
.\.venv\Scripts\python.exe -m pytest tests\
```

## Dataset access

Datasets are not included in this repository (`data/raw/` is gitignored).
GPDS Synthetic 4000 is licensed via a signed ULPGC agreement; the
institutional dataset is private, supplied by the project supervisor, and is
not publicly redistributable
(including that the institutional dataset is confirmed to be based on GPDS
Synthetic with additional data layered on top, and why their splits are
deliberately mirrored to prevent leakage).

## Where results live

- `report/tables/` — the canonical, citable result tables (RQ2 backbone
  comparison, RQ3 dataset-combination effect, RQ4 full metrics and
  forgery-type breakdown), plus `eer_implementation_note.md` explaining the
  two EER estimators.
- `report/dataset_checks/` — EDA and data-provenance evidence (class counts,
  dimension summaries, near-duplicate checks between the institutional and
  GPDS corpora, preprocessing shortcut audits, bootstrap CIs).
- `report/figures/` — figures referenced by the written report.
- `report/RQ_FINDINGS.md` — a findings summary generated programmatically
  from the tables above (regenerate with `build_rq_findings.py`, never hand-edit).
