# Robust Biometric Signature Verification Against Skilled Forgeries

Case Study 2, M.Sc. Applied Data Science and Analytics, SRH University Heidelberg.
Supervisor: Prof. Dr.-Ing. Binh Vu (guides direction only — all technical decisions are made
independently; deviations from the exposé are documented in the final report, not the exposé).

## What this project is
Writer-independent offline signature verification using deep metric learning (Siamese networks).
Five datasets (~344,680 images): CEDAR, BHSig260-Bengali, BHSig260-Hindi, GPDS Synthetic 4000,
and an institutional dataset from Prof. Vu.

**Success criterion (confirmed by Prof. Vu, July 2026): performance on the private institutional
dataset is what counts.** CEDAR / BHSig260 / GPDS exist only to improve training diversity. Good
results on public datasets alone are not the goal — do not optimize for them at the expense of
private-dataset evaluation.

**Known dataset relationship:** the institutional dataset is confirmed (Prof. Vu, July 2026) to be
based on GPDS Synthetic with additional new data layered on top. Institutional/GPDS splits are
deliberately mirrored (same seed, same writer alignment) to prevent train/test leakage from the
shared content. Do not treat institutional and GPDS as independent samples anywhere in the pipeline
or analysis.

## Non-negotiable conventions
- **Seed = 42** everywhere (splits, training, bootstrap resampling).
- **Preprocessing is frozen at `PREPROCESSING_VERSION = 2`** (commit tag `preproc-v2-freeze`,
  `49444da4`). Do not change preprocessing behavior without explicit confirmation — a `v3` fix for
  the global-Otsu faint-pen issue is planned but not yet implemented.
- Writer-independent splits only. Never let a test writer appear in training.
- Any new baseline must pass the intensity-shortcut gate check before its results are trusted.
- Use writer-bootstrap (resample writer identities, not pairs) for confidence intervals.

## See also
- `.claude/rules/environment.md` — local vs. Kaggle execution, paths, cell conventions
- `.claude/rules/data-conventions.md` — dataset locations, splits, provenance caveats
- `.claude/rules/output-conventions.md` — where figures/evidence tables go and how they're named
- `.claude/skills/kaggle-checkpoint-resume/` — resuming a Kaggle GPU training session
- `.claude/skills/eda-evidence-output/` — producing dataset-check figures/CSVs correctly
