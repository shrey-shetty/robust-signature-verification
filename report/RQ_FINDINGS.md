# RQ Findings

Input to the report, not the report itself. Every number below is read programmatically by `scripts/build_rq_findings.py` from the CSVs in `report/tables/` (plus the three explicitly-scoped exceptions noted in each subsection: ViT timing-probe figures, config.json confounds, and the literature search). Regenerate with:

    .\.venv\Scripts\python.exe scripts\build_rq_findings.py

## RQ1 — Can the framework reliably distinguish genuine from skilled forgeries?

**Best skilled-forgery EER on institutional:** arm `rq3_combined`, eer_skilled = 0.224 (rq4_metrics.csv, row 17)

- Corresponding pooled EER (eer_all): 0.1628 (rq4_metrics.csv, row 17)
- Corresponding AUC: 0.922 (rq4_metrics.csv, row 17)
- FAR at own operating threshold (pooled selection): 0.1591 (rq4_metrics.csv, row 17)
- FRR at own operating threshold (pooled selection): 0.1673 (rq4_metrics.csv, row 17)
- FAR at own operating threshold, skilled-only selection: NOT IN TABLES (rq4_metrics.csv has only the pooled far/frr columns; rq4_forgery_type.csv has far_random_at_threshold/frr_random_at_threshold for the random subset, but no skilled-subset equivalent)
- FRR at own operating threshold, skilled-only selection: NOT IN TABLES (same reason)

**Skilled-forgery EER on institutional, every arm** (rq4_metrics.csv, dataset == institutional, eer_skilled column):

| arm | eer_skilled |
|---|---|
| rq3_combined | 0.224 |
| resnet34 | 0.2301 |
| resnet18 | 0.2311 |
| efficientnet_b0 | 0.2399 |
| smallcnn | 0.2674 |

### Operational reading

At the equal-error operating point for `rq3_combined` on institutional, FAR and FRR for the skilled-only selection are equal by definition of EER ((rq4_forgery_type.csv, arm == rq3_combined, dataset == institutional, eer_skilled)):

- Proportion of skilled forgeries accepted at this point: 0.224 (22.40%)
- Proportion of genuine signatures rejected at this point: 0.224 (22.40%)

### Threshold transfer

NOT IN TABLES for arm `rq3_combined`: it has no `carried`-provenance rows in rq4_metrics.csv. Its threshold_provenance values across all datasets are ['global'] (rq4_metrics.csv, arm == rq3_combined, threshold_provenance column) — `rq3_combined` was trained on the combined corpus mix, so its operating threshold is tuned on a mixed validation set (`global`) everywhere, never carried specifically from institutional to a public corpus.

Shown instead for the best single-corpus-trained (institutional-only) arm, for illustration of what threshold transfer looks like in this framework:

**Arm `resnet34`** (best institutional-only-trained arm by skilled EER):

| dataset | carried FAR | oracle FAR (retuned per corpus) |
|---|---|---|
| bhsig260_bengali | 0.7125 (rq4_metrics.csv, row 10) | 0.3096 (rq4_metrics.csv, row 10) |
| bhsig260_hindi | 0.6701 (rq4_metrics.csv, row 11) | 0.3014 (rq4_metrics.csv, row 11) |
| cedar | 0.3976 (rq4_metrics.csv, row 12) | 0.247 (rq4_metrics.csv, row 12) |

A deployed system must fix its threshold in advance; it cannot retune per corpus at inference time the way the oracle columns do.

### Literature comparison — verify, do not assert

No SigNet paper or other offline-signature-verification paper file found under `literature/`, `docs/`, `papers/`, or the repo root (searched all three directory names and the root for filenames matching SigNet / signature-verification papers; none exist in this repository).

LITERATURE COMPARISON UNVERIFIED — Shreya must read the paper and supply the figure.

## RQ2 — Which backbone performs best?

**Pooled EER (eer_all), all cells from rq2_backbone_comparison.csv**

| arm | institutional | cedar | bhsig260_bengali | bhsig260_hindi |
|---|---|---|---|---|
| smallcnn | 0.1935 | 0.2105 | 0.2685 | 0.2519 |
| resnet18 | 0.1674 | 0.2408 | 0.3082 | 0.2682 |
| resnet34 | 0.1685 | 0.247 | 0.3096 | 0.3014 |
| efficientnet_b0 | 0.1799 | 0.2559 | 0.2861 | 0.2863 |

**Skilled EER (eer_skilled), all cells from rq2_backbone_comparison.csv**

| arm | institutional | cedar | bhsig260_bengali | bhsig260_hindi |
|---|---|---|---|---|
| smallcnn | 0.2674 | 0.2813 | 0.3312 | 0.2948 |
| resnet18 | 0.2311 | 0.3195 | 0.3825 | 0.3123 |
| resnet34 | 0.2301 | 0.3109 | 0.3891 | 0.3372 |
| efficientnet_b0 | 0.2399 | 0.3294 | 0.3594 | 0.3353 |

**Every pairwise comparison involving institutional** (rq2_backbone_comparison.csv, dataset == institutional):

| metric | arm_a | arm_b | diff_b_minus_a | ci_lower | ci_upper | separates_from_zero |
|---|---|---|---|---|---|---|
| eer_all | smallcnn | resnet18 | -0.0261 | -0.0319 | -0.0201 | True |
| eer_skilled | smallcnn | resnet18 | -0.0362 | -0.0445 | -0.0287 | True |
| eer_all | smallcnn | resnet34 | -0.025 | -0.0318 | -0.0186 | True |
| eer_skilled | smallcnn | resnet34 | -0.0373 | -0.0465 | -0.0299 | True |
| eer_all | smallcnn | efficientnet_b0 | -0.0136 | -0.0204 | -0.0068 | True |
| eer_skilled | smallcnn | efficientnet_b0 | -0.0275 | -0.036 | -0.0185 | True |
| eer_all | resnet18 | resnet34 | 0.0011 | -0.004 | 0.0058 | False |
| eer_skilled | resnet18 | resnet34 | -0.001 | -0.008 | 0.005 | False |
| eer_all | resnet18 | efficientnet_b0 | 0.0125 | 0.007 | 0.0175 | True |
| eer_skilled | resnet18 | efficientnet_b0 | 0.0088 | 0.0018 | 0.0159 | True |
| eer_all | resnet34 | efficientnet_b0 | 0.0114 | 0.0061 | 0.0172 | True |
| eer_skilled | resnet34 | efficientnet_b0 | 0.0098 | 0.0032 | 0.0177 | True |

**Every pairwise comparison against smallcnn on the three public corpora** (rq2_backbone_comparison.csv, dataset in public corpora, arm_a == smallcnn):

| dataset | metric | arm_b | diff_b_minus_a | ci_lower | ci_upper | separates_from_zero |
|---|---|---|---|---|---|---|
| cedar | eer_all | resnet18 | 0.0303 | 0.0059 | 0.0504 | True |
| cedar | eer_skilled | resnet18 | 0.0382 | -0.0025 | 0.0698 | False |
| cedar | eer_all | resnet34 | 0.0366 | 0.0013 | 0.0652 | True |
| cedar | eer_skilled | resnet34 | 0.0296 | -0.0173 | 0.0753 | False |
| cedar | eer_all | efficientnet_b0 | 0.0455 | 0.0198 | 0.0649 | True |
| cedar | eer_skilled | efficientnet_b0 | 0.0481 | 0.0087 | 0.0672 | True |
| bhsig260_bengali | eer_all | resnet18 | 0.0397 | 0.0132 | 0.0645 | True |
| bhsig260_bengali | eer_skilled | resnet18 | 0.0514 | 0.0129 | 0.0947 | True |
| bhsig260_bengali | eer_all | resnet34 | 0.0411 | 0.0027 | 0.079 | True |
| bhsig260_bengali | eer_skilled | resnet34 | 0.058 | 0.0087 | 0.1101 | True |
| bhsig260_bengali | eer_all | efficientnet_b0 | 0.0176 | -0.0225 | 0.0556 | False |
| bhsig260_bengali | eer_skilled | efficientnet_b0 | 0.0283 | -0.0307 | 0.0847 | False |
| bhsig260_hindi | eer_all | resnet18 | 0.0163 | -0.0058 | 0.0396 | False |
| bhsig260_hindi | eer_skilled | resnet18 | 0.0174 | -0.012 | 0.051 | False |
| bhsig260_hindi | eer_all | resnet34 | 0.0495 | 0.0241 | 0.0752 | True |
| bhsig260_hindi | eer_skilled | resnet34 | 0.0423 | 0.0056 | 0.0764 | True |
| bhsig260_hindi | eer_all | efficientnet_b0 | 0.0344 | 0.0044 | 0.0624 | True |
| bhsig260_hindi | eer_skilled | efficientnet_b0 | 0.0405 | 0.0011 | 0.0764 | True |

**Count:** smallcnn has the lower EER in **18/18** arm/dataset/metric cells across the public corpora (rq2_backbone_comparison.csv, dataset in public corpora, arm_a == smallcnn, diff_b_minus_a > 0).

### In-domain versus out-of-domain

**In-domain (institutional):** pretrained backbones have LOWER EER than smallcnn in all 6 smallcnn-vs-pretrained comparisons (rq2_backbone_comparison.csv, dataset == institutional, arm_a == smallcnn); 6/6 separate from zero.

**Out-of-domain (public corpora):** smallcnn has LOWER EER than every pretrained backbone in 18/18 comparisons (rq2_backbone_comparison.csv, dataset in public corpora, arm_a == smallcnn); 12/18 separate from zero.

Rows where the out-of-domain smallcnn advantage separates from zero:

| dataset | metric | arm_b | diff_b_minus_a | ci_lower | ci_upper |
|---|---|---|---|---|---|
| cedar | eer_all | resnet18 | 0.0303 | 0.0059 | 0.0504 |
| cedar | eer_all | resnet34 | 0.0366 | 0.0013 | 0.0652 |
| cedar | eer_all | efficientnet_b0 | 0.0455 | 0.0198 | 0.0649 |
| cedar | eer_skilled | efficientnet_b0 | 0.0481 | 0.0087 | 0.0672 |
| bhsig260_bengali | eer_all | resnet18 | 0.0397 | 0.0132 | 0.0645 |
| bhsig260_bengali | eer_skilled | resnet18 | 0.0514 | 0.0129 | 0.0947 |
| bhsig260_bengali | eer_all | resnet34 | 0.0411 | 0.0027 | 0.079 |
| bhsig260_bengali | eer_skilled | resnet34 | 0.058 | 0.0087 | 0.1101 |
| bhsig260_hindi | eer_all | resnet34 | 0.0495 | 0.0241 | 0.0752 |
| bhsig260_hindi | eer_skilled | resnet34 | 0.0423 | 0.0056 | 0.0764 |
| bhsig260_hindi | eer_all | efficientnet_b0 | 0.0344 | 0.0044 | 0.0624 |
| bhsig260_hindi | eer_skilled | efficientnet_b0 | 0.0405 | 0.0011 | 0.0764 |

Rows where it does not separate from zero:

| dataset | metric | arm_b | diff_b_minus_a | ci_lower | ci_upper |
|---|---|---|---|---|---|
| cedar | eer_skilled | resnet18 | 0.0382 | -0.0025 | 0.0698 |
| cedar | eer_skilled | resnet34 | 0.0296 | -0.0173 | 0.0753 |
| bhsig260_bengali | eer_all | efficientnet_b0 | 0.0176 | -0.0225 | 0.0556 |
| bhsig260_bengali | eer_skilled | efficientnet_b0 | 0.0283 | -0.0307 | 0.0847 |
| bhsig260_hindi | eer_all | resnet18 | 0.0163 | -0.0058 | 0.0396 |
| bhsig260_hindi | eer_skilled | resnet18 | 0.0174 | -0.012 | 0.051 |

### ViT-B/16 exclusion

(source: timing probe, Kaggle session 13 Sep 2026 — not a report/tables CSV; parameter counts independently re-verified below against the live model code)

- ViT-B/16: 2063.1 ms/step; ResNet-18: 127.9 ms/step; ratio 16.1x
- EfficientNet-B0: 209.8 ms/step, 1.64x ResNet-18
- Projected six-epoch time for ViT-B/16: 46.24 h; per epoch: approximately 7.7 h
- Kaggle session limit: 12 h, so not even one epoch fits within a single session
- Parameter counts (independently re-verified against `src/sigver/models/backbones.py` at build_rq_findings.py run time): ViT-B/16 85,503,872; ResNet-18 11,235,904; EfficientNet-B0 4,170,940
- Structural cause, confirmed directly in `src/sigver/models/backbones.py`: ViT-B/16 has a fixed 224x224 input and no adaptive pooling, so 150x220 inputs are resized (NEAREST interpolation, to avoid reintroducing grey values into the binary PREPROCESSING_VERSION 2 input); at patch size 16, 224/16 = 14, giving 14x14 = 196 patch tokens with quadratic attention cost. ResNet and EfficientNet both end in adaptive pooling and take 150x220 unchanged.

### Confounds

Read directly from config.json files under `C:\Users\Shreya\Downloads\all_results` (not report/tables — this subsection is explicitly scoped to those files):

| field | smallcnn | resnet18 | resnet34 | efficientnet_b0 |
|---|---|---|---|---|
| epochs | 15 | 6 | 6 | 6 |
| lr | 0.001 | 0.0001 | 0.0001 | 0.0001 |
| patience | 5 | None | None | None |
| pretrained | absent | True | True | True |
| backbone | absent | resnet18 | resnet34 | efficientnet_b0 |
| l2_normalize | absent | False | False | False |
| input_norm | absent | none | none | none |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 |
| batch_size | 32 | 32 | 32 | 32 |
| margin | 1.0 | 1.0 | 1.0 | 1.0 |
| embedding_dim | 128 | 128 | 128 | 128 |
| raw_root | /kaggle/working/data_raw | /kaggle/working/data_fixed | /kaggle/working/data_fixed | /kaggle/working/data_fixed |
| preprocessing_version | absent | 2 | 2 | 2 |
| git | absent | 08e70d6f4c | b34a67a3a2 | 8c06ee5a74 |

Source: `results_smallcnn_institutional/smallcnn_config.json`, `results_rq2/exp_resnet18_pretrained/config.json`, `results_rq2_resnet34/exp_resnet34_pretrained/config.json`, `results_rq2_efficientnet_b0/exp_efficientnet_b0_pretrained/config.json`, all under `C:\Users\Shreya\Downloads\all_results`.

Differing or absent fields between smallcnn and the pretrained arms (reported as found; no claim about effect size): epochs, lr, patience, pretrained, backbone, l2_normalize, input_norm, raw_root, preprocessing_version, git.

## RQ3 — Does combining datasets improve generalisation to unseen writers?

**Every row of rq3_dataset_effect.csv:**

| dataset | metric | eer_institutional_only | eer_combined | diff | ci_lower | ci_upper | separates_from_zero | in_training_mix | caveat |
|---|---|---|---|---|---|---|---|---|---|
| institutional | eer_all | 0.1674 | 0.1628 | -0.0046 | -0.0091 | -0.0008 | True | True |  |
| institutional | eer_skilled | 0.2311 | 0.224 | -0.0071 | -0.0131 | -0.0009 | True | True |  |
| cedar | eer_all | 0.2408 | 0.2289 | -0.0119 | -0.024 | 0.0079 | False | True |  |
| cedar | eer_skilled | 0.3195 | 0.2973 | -0.0222 | -0.0362 | 0.0033 | False | True |  |
| bhsig260_bengali | eer_all | 0.3082 | 0.1933 | -0.1149 | -0.1549 | -0.0734 | True | True |  |
| bhsig260_bengali | eer_skilled | 0.3825 | 0.1914 | -0.1911 | -0.254 | -0.129 | True | True |  |
| bhsig260_hindi | eer_all | 0.2682 | 0.2031 | -0.0651 | -0.1017 | -0.0293 | True | True | shortcut-contaminated |
| bhsig260_hindi | eer_skilled | 0.3123 | 0.0838 | -0.2285 | -0.2786 | -0.1807 | True | True | shortcut-contaminated |
(rq3_dataset_effect.csv, all rows)

**BHSig260-Hindi shortcut-contaminated caveat** — carried through prominently: both rows for this dataset are flagged `caveat = shortcut-contaminated` (rq3_dataset_effect.csv, dataset == bhsig260_hindi). Its combined-vs-only differences (-0.0651, -0.2285) should not be read as clean evidence of the combined-training effect.

### Scope of the claim

- Splits are writer-independent, so test writers are unseen in every case (project convention, not a table figure).
- `in_training_mix` is True for all 8 rows (rq3_dataset_effect.csv, in_training_mix column): all four evaluated corpora are in the combined training mix, so these results speak to unseen *writers* within represented *domains*, not to cross-corpus transfer.
- Cross-corpus transfer could not be tested: the only corpus outside the training mix is GPDS-Synthetic, and `eda_inst_gpds_near_duplicates.csv` records **41,220 near-duplicate pairs** spanning **4,000 institutional writers**, at Hamming distance **0-4**, including **10,665 exact matches** (distance 0), covering both genuine and forgery paths (eda_inst_gpds_near_duplicates.csv (report/dataset_checks/), all rows, computed directly).
- Therefore GPDS cannot serve as a held-out corpus.

**Near-duplicate methodology**, confirmed directly from `scripts/verify_inst_gpds_overlap.py`: perceptual hash (`imagehash.phash`, hash_size=HASH_SIZE) computed per image; within each writer ID shared between institutional and GPDS, pairwise Hamming distance is computed and any pair with distance <= 4 is flagged and written to the CSV. The file lists only flagged (below-cutoff) pairs, not an exhaustive distance table.

## RQ4 — What biometric performance does the framework achieve?

**Full metrics table** (rq4_metrics.csv, all rows):

| arm | dataset | threshold | threshold_provenance | eer_all | eer_skilled | auc | far | frr | precision | recall | f1 | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | bhsig260_bengali | 0.5738 | carried | 0.2861 | 0.3594 | 0.7834 | 0.781 | 0.033 | 0.5532 | 0.967 | 0.7038 | 0.593 |
| efficientnet_b0 | bhsig260_hindi | 0.5738 | carried | 0.2863 | 0.3353 | 0.7775 | 0.6754 | 0.1016 | 0.5709 | 0.8984 | 0.6981 | 0.6115 |
| efficientnet_b0 | cedar | 0.5738 | carried | 0.2559 | 0.3294 | 0.8309 | 0.4769 | 0.1074 | 0.6518 | 0.8926 | 0.7534 | 0.7078 |
| efficientnet_b0 | institutional | 0.5738 | own | 0.1799 | 0.2399 | 0.9083 | 0.1798 | 0.1801 | 0.8202 | 0.8199 | 0.82 | 0.8201 |
| resnet18 | bhsig260_bengali | 0.5833 | carried | 0.3082 | 0.3825 | 0.7602 | 0.8239 | 0.0368 | 0.539 | 0.9632 | 0.6912 | 0.5697 |
| resnet18 | bhsig260_hindi | 0.5833 | carried | 0.2682 | 0.3123 | 0.7901 | 0.6952 | 0.1033 | 0.5633 | 0.8967 | 0.6919 | 0.6008 |
| resnet18 | cedar | 0.5833 | carried | 0.2408 | 0.3195 | 0.8363 | 0.4783 | 0.0935 | 0.6546 | 0.9065 | 0.7602 | 0.7141 |
| resnet18 | institutional | 0.5833 | own | 0.1674 | 0.2311 | 0.9173 | 0.1694 | 0.1653 | 0.8313 | 0.8347 | 0.833 | 0.8327 |
| resnet34 | bhsig260_bengali | 0.7349 | carried | 0.3096 | 0.3891 | 0.7541 | 0.7125 | 0.063 | 0.568 | 0.937 | 0.7073 | 0.6122 |
| resnet34 | bhsig260_hindi | 0.7349 | carried | 0.3014 | 0.3372 | 0.7614 | 0.6701 | 0.1008 | 0.573 | 0.8992 | 0.7 | 0.6146 |
| resnet34 | cedar | 0.7349 | carried | 0.247 | 0.3109 | 0.8353 | 0.3976 | 0.1179 | 0.6893 | 0.8821 | 0.7739 | 0.7423 |
| resnet34 | institutional | 0.7349 | own | 0.1685 | 0.2301 | 0.9174 | 0.1752 | 0.1597 | 0.8275 | 0.8403 | 0.8338 | 0.8326 |
| rq3_combined | bhsig260_bengali | 0.7147 | global | 0.1933 | 0.1914 | 0.8881 | 0.3668 | 0.0629 | 0.7187 | 0.9371 | 0.8135 | 0.7851 |
| rq3_combined | bhsig260_hindi | 0.7147 | global | 0.2031 | 0.0838 | 0.8872 | 0.5265 | 0.0189 | 0.6508 | 0.9811 | 0.7825 | 0.7273 |
| rq3_combined | cedar | 0.7147 | global | 0.2289 | 0.2973 | 0.851 | 0.4341 | 0.0893 | 0.6772 | 0.9107 | 0.7768 | 0.7383 |
| rq3_combined | institutional | 0.7147 | global | 0.1628 | 0.224 | 0.922 | 0.1591 | 0.1673 | 0.8396 | 0.8327 | 0.8361 | 0.8368 |
| smallcnn | bhsig260_bengali | 0.5256 | carried | 0.2685 | 0.3312 | 0.8087 | 0.6473 | 0.0612 | 0.5919 | 0.9388 | 0.726 | 0.6457 |
| smallcnn | bhsig260_hindi | 0.5256 | carried | 0.2519 | 0.2948 | 0.8057 | 0.5923 | 0.1156 | 0.5989 | 0.8844 | 0.7142 | 0.6461 |
| smallcnn | cedar | 0.5256 | carried | 0.2105 | 0.2813 | 0.8642 | 0.3429 | 0.1222 | 0.7191 | 0.8778 | 0.7906 | 0.7675 |
| smallcnn | institutional | 0.5256 | own | 0.1935 | 0.2674 | 0.8938 | 0.1971 | 0.1886 | 0.8046 | 0.8114 | 0.8079 | 0.8071 |

**Forgery-type breakdown** (rq4_forgery_type.csv, all rows):

| arm | dataset | eer_random | eer_skilled | eer_pooled |
|---|---|---|---|---|
| efficientnet_b0 | bhsig260_bengali | 0.1979 | 0.3594 | 0.2861 |
| efficientnet_b0 | bhsig260_hindi | 0.2264 | 0.3353 | 0.2863 |
| efficientnet_b0 | cedar | 0.1509 | 0.3294 | 0.2559 |
| efficientnet_b0 | institutional | 0.0981 | 0.2399 | 0.1799 |
| resnet18 | bhsig260_bengali | 0.2199 | 0.3825 | 0.3082 |
| resnet18 | bhsig260_hindi | 0.2241 | 0.3123 | 0.2682 |
| resnet18 | cedar | 0.1515 | 0.3195 | 0.2408 |
| resnet18 | institutional | 0.0799 | 0.2311 | 0.1674 |
| resnet34 | bhsig260_bengali | 0.2141 | 0.3891 | 0.3096 |
| resnet34 | bhsig260_hindi | 0.2647 | 0.3372 | 0.3014 |
| resnet34 | cedar | 0.1606 | 0.3109 | 0.247 |
| resnet34 | institutional | 0.0816 | 0.2301 | 0.1685 |
| rq3_combined | bhsig260_bengali | 0.196 | 0.1914 | 0.1933 |
| rq3_combined | bhsig260_hindi | 0.274 | 0.0838 | 0.2031 |
| rq3_combined | cedar | 0.1337 | 0.2973 | 0.2289 |
| rq3_combined | institutional | 0.0757 | 0.224 | 0.1628 |
| smallcnn | bhsig260_bengali | 0.1829 | 0.3312 | 0.2685 |
| smallcnn | bhsig260_hindi | 0.2036 | 0.2948 | 0.2519 |
| smallcnn | cedar | 0.1145 | 0.2813 | 0.2105 |
| smallcnn | institutional | 0.0902 | 0.2674 | 0.1935 |

**Oracle vs. carried threshold columns** (rq4_metrics.csv, carried rows):

| arm | dataset | threshold | far | frr | oracle_threshold | oracle_far | oracle_frr |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | bhsig260_bengali | 0.5738 | 0.781 | 0.033 | 0.3312 | 0.2861 | 0.2857 |
| efficientnet_b0 | bhsig260_hindi | 0.5738 | 0.6754 | 0.1016 | 0.322 | 0.2863 | 0.2863 |
| efficientnet_b0 | cedar | 0.5738 | 0.4769 | 0.1074 | 0.4119 | 0.2559 | 0.2559 |
| resnet18 | bhsig260_bengali | 0.5833 | 0.8239 | 0.0368 | 0.3115 | 0.3082 | 0.3082 |
| resnet18 | bhsig260_hindi | 0.5833 | 0.6952 | 0.1033 | 0.3219 | 0.2682 | 0.2682 |
| resnet18 | cedar | 0.5833 | 0.4783 | 0.0935 | 0.4176 | 0.2408 | 0.2408 |
| resnet34 | bhsig260_bengali | 0.7349 | 0.7125 | 0.063 | 0.4481 | 0.3096 | 0.3096 |
| resnet34 | bhsig260_hindi | 0.7349 | 0.6701 | 0.1008 | 0.443 | 0.3014 | 0.3014 |
| resnet34 | cedar | 0.7349 | 0.3976 | 0.1179 | 0.5797 | 0.247 | 0.247 |
| smallcnn | bhsig260_bengali | 0.5256 | 0.6473 | 0.0612 | 0.3516 | 0.2685 | 0.2685 |
| smallcnn | bhsig260_hindi | 0.5256 | 0.5923 | 0.1156 | 0.3481 | 0.2519 | 0.2519 |
| smallcnn | cedar | 0.5256 | 0.3429 | 0.1222 | 0.4227 | 0.2105 | 0.2105 |

### Forgery types present

These corpora contain genuine, skilled, and random (other-writer) pairs only; there is no simple-forgery category in CEDAR, BHSig260, or the institutional set, so the assignment brief's three-way split is reported here as the two negative categories that actually exist (skilled, random).

Per-dataset pair counts by kind (rq4_forgery_type.csv, n_positive/n_skilled/n_random columns):

| dataset | n_positive (genuine) | n_skilled | n_random |
|---|---|---|---|
| bhsig260_bengali | 5520 | 2760 | 2760 |
| bhsig260_hindi | 8832 | 4416 | 4416 |
| cedar | 3036 | 1518 | 1518 |
| institutional | 52800 | 26400 | 26400 |

### Random versus skilled

Gap between eer_skilled and eer_random, institutional, every arm (rq4_forgery_type.csv, dataset == institutional):

| arm | eer_random | eer_skilled | gap (skilled - random) |
|---|---|---|---|
| efficientnet_b0 | 0.0981 | 0.2399 | 0.1418 |
| rq3_combined | 0.0757 | 0.224 | 0.1483 |
| resnet34 | 0.0816 | 0.2301 | 0.1485 |
| resnet18 | 0.0799 | 0.2311 | 0.1512 |
| smallcnn | 0.0902 | 0.2674 | 0.1772 |

## Methodological notes for the report

(all figures below parsed from `report/tables/eer_implementation_note.md`)

- **Two EER implementations exist.** Exact (`scripts/final_results_tables.py::eer()`) sweeps every observed distance as a candidate threshold; grid (`sigver.evaluation.metrics.compute_eer()`) sweeps a fixed 512-point linspace. The exact method is canonical for every table in `report/tables/`; `test_metrics.json` values are grid approximations from the evaluation pipeline and are not more authoritative than the report tables.
- **Maximum exact-vs-grid discrepancy: 0.001755**, occurring on EfficientNet-B0 × BHSig260-Hindi, skilled-only selection.
- **Tie-group limitation:** at a tie in distances, no single threshold can separate the tied pairs (a threshold either accepts all of them or none), so an interpolated crossing that falls inside a tie group is not an achievable operating point. This affects **4 of 20 rows** (40 EER values checked) in `rq4_forgery_type.csv`, bounded at a maximum magnitude of **0.000272**.
- **SmallCNN institutional 0.1935 / 0.1936 pair:** exact = 0.193485 (rounds to 0.1935), grid = 0.193627 (rounds to 0.1936), both from the identical `test_scores.csv`. This reproduces the two values recorded in project notes. **No claim is made here that this explains the discrepancy in the project notes** — the coincidence is stated; the conclusion is left to Shreya.

## Gaps and limitations

- **ViT-B/16 excluded** from the backbone comparison on timing grounds (16.1x slower per step than ResNet-18; projected 46.24 h for six epochs against a 12 h Kaggle session limit — see RQ2 ViT-B/16 exclusion).
- **SmallCNN vs. pretrained arms is confounded**, not a clean architecture-only comparison — see RQ2 Confounds for the specific differing config fields.
- **No cross-corpus generalisation test exists.** All four evaluated corpora are in the RQ3 combined training mix; GPDS-Synthetic (the only held-out corpus) cannot serve as an independent test set due to extensive near-duplication with the institutional set (see RQ3 Scope of the claim).
- **BHSig260-Hindi results are shortcut-contaminated** — flagged in `rq3_dataset_effect.csv` and carried through RQ3's findings.
- **Literature comparison is unverified** — no SigNet or other offline-signature-verification paper is present in the repository to check reported EER figures against (see RQ1 Literature comparison).
- **Skilled-selection FAR/FRR at the deployed operating threshold is not tabulated** for any arm (only the pooled and random-subset versions are) — see RQ1.

Every other item marked NOT IN TABLES or UNVERIFIED above:
- FAR at own operating threshold, skilled-only selection: NOT IN TABLES (rq4_metrics.csv has only the pooled far/frr columns; rq4_forgery_type.csv has far_random_at_threshold/frr_random_at_threshold for the random subset, but no skilled-subset equivalent)
- FRR at own operating threshold, skilled-only selection: NOT IN TABLES (same reason)
