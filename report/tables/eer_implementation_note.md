# EER implementation note

## Two implementations exist in this codebase

**Exact (canonical for every report table).** `scripts/final_results_tables.py::eer()`
sorts all pairs by distance and sweeps cumulative FAR/FRR one pair at a time, finding the
threshold where the two curves cross. Every observed distance value participates as a
candidate threshold — nothing is interpolated or skipped.

**Grid (used at evaluation time; what `test_metrics.json` records as `test_eer_all` /
`test_eer_skilled`).** `sigver.evaluation.metrics.compute_eer()` sweeps a **fixed 512-point
grid** (`np.linspace(distances.min(), distances.max(), 512)`) and finds the crossing among
those 512 points. Its resolution is `(max − min) / 511`, which is coarser than the exact
method whenever a corpus has more than ~512 meaningfully distinct distance values — true for
every corpus in this project (6,072–105,600 pairs each).

**Decision:** the exact method is canonical for every table in `report/tables/`
(`rq2_backbone_comparison.csv`, `rq3_dataset_effect.csv`, `rq4_metrics.csv`,
`rq4_forgery_type.csv`). `test_metrics.json` values are grid approximations produced by the
evaluation pipeline and should not be treated as more authoritative than the report tables —
if anything, the reverse.

## Maximum observed discrepancy

Computed directly across all 5 arms × 4 datasets × {pooled, skilled} selections (40 values):

| arm | dataset | selection | exact | grid | diff |
|---|---|---|---|---|---|
| efficientnet_b0 | bhsig260_hindi | skilled | 0.335315 | 0.333560 | **0.001755** |
| resnet18 | cedar | skilled | 0.319499 | 0.318017 | 0.001482 |
| efficientnet_b0 | cedar | skilled | 0.329381 | 0.330698 | 0.001318 |
| resnet34 | cedar | skilled | 0.310935 | 0.309783 | 0.001153 |
| rq3_combined | bhsig260_bengali | pooled | 0.193297 | 0.192482 | 0.000815 |

**Maximum discrepancy: 0.001755** (0.18 percentage points of EER), on EfficientNet-B0 ×
BHSig260-Hindi, skilled-only selection. The pattern is consistent with grid resolution:
larger discrepancies cluster on smaller corpora (cedar: 6,072 pairs) and on the
skilled-only subset (fewer points than pooled), both cases where 512 grid points under-sample
the FAR/FRR curve more severely relative to the number of distinct observed distances.

## A second property of the exact method, found while building this note's verification

`rq4_forgery_type.csv`'s per-row verification cross-checks the canonical exact EER against a
second, independently-coded exact algorithm (unique-distance search rather than merged-array
cumulative sweep). In 4 of 20 rows the two disagreed by more than a numerical tolerance. Root
cause, confirmed directly on the data (not inferred): `eer()`'s cumulative sweep walks every
individual pair in sorted order: when 2 or more pairs share an identical distance value, every
array position **strictly inside** that tied run represents "accept some of these
identical-distance pairs, reject the rest" — a split no real `distance ≤ threshold` rule can
produce, since tied pairs must be classified identically. Only the *last* position in a tied
run is a real, achievable threshold.

This was confirmed to be the exact and complete explanation for all 4 disagreements (not a bug
in either implementation). All 4 affected cells, with the tie-group size, where in the group
`eer()`'s crossing landed, the reported (non-achievable) EER, and what EER would be reported if
the crossing were taken at the tie-group boundary instead (last position in the group — the
nearest point that IS a real, achievable threshold):

| arm | dataset | selection | tie-group size | position in group | reported EER | boundary EER | diff |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | bhsig260_bengali | pooled | 3 | 1st of 3 | 0.286051 | 0.285870 | 0.000181 |
| efficientnet_b0 | institutional | pooled | 2 | 1st of 2 | 0.179886 | 0.179896 | 0.000009 |
| resnet18 | bhsig260_bengali | skilled | 2 | 1st of 2 | 0.382518 | 0.382699 | 0.000181 |
| rq3_combined | bhsig260_bengali | pooled | 10 | 7th of 10 | 0.193297 | 0.193025 | 0.000272 |

These figures are reported for documentation only — `eer()` itself is unchanged, and no report
table has been altered to use the boundary value in place of the reported one.

**In plain terms:** when two or more pairs happen to land at the exact same computed distance,
there is no single similarity threshold that can accept one of them while rejecting another —
a threshold either lets all of those tied pairs through or none of them. `eer()`'s search
walks through the sorted pairs one at a time regardless of ties, so its reported crossing point
can fall *between* two pairs that share a distance, which corresponds to no such achievable
threshold. This is a genuine, bounded limitation, not a computation error: across every arm,
dataset, and selection checked (20 rows, 40 EER values), it affects **4 cells**, and the size
of the resulting discrepancy is **bounded at 0.000272** (about 0.03 percentage points of EER) —
an order of magnitude smaller than the exact-vs-grid gap documented above, and too small to
change any conclusion in this report, but worth naming explicitly rather than leaving implicit
in the `verification_status` column alone.

This is a property of `eer()` on data with repeated distance values, orthogonal to the
exact-vs-grid comparison above — noted here for completeness since it surfaced directly from
this exercise, not asserted as a reason to change the canonical implementation (that decision
stands as-is).

## SmallCNN institutional: the 0.1935 / 0.1936 pair

Checked directly against project notes that record SmallCNN's institutional EER as both
0.1935 and 0.1936, as if from two different preprocessing versions:

- Exact (`eer()`): **0.193485** → rounds to **0.1935**
- Grid (`compute_eer()`, same as logged `test_eer_all` in
  `results_smallcnn_institutional/eval_smallcnn_institutional/test_metrics.json`):
  **0.193627** → rounds to **0.1936**

Both numbers come from the **same** `test_scores.csv` (same preprocessing, same checkpoint,
same evaluation run) — the only variable between them is which EER algorithm computed them.
This exactly reproduces the two values in question. Whether this fully accounts for the
project notes' record of them as different preprocessing versions is not established here —
that would require confirming which of the two algorithms (or what else) produced whichever
number appears in that specific note. Reported as found; no cause asserted beyond what was
directly measured.
