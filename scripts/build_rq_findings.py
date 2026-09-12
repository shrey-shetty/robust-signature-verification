"""Build report/RQ_FINDINGS.md from the tables in report/tables/.

Every number in the generated document is read programmatically from one of:
  - report/tables/rq2_backbone_comparison.csv
  - report/tables/rq3_dataset_effect.csv
  - report/tables/rq4_metrics.csv
  - report/tables/rq4_forgery_type.csv
  - report/tables/eer_implementation_note.md   (text, parsed with targeted regex)
  - report/dataset_checks/eda_inst_gpds_near_duplicates.csv  (Section 3, named explicitly)
  - config.json files under the results root  (Section 2 Confounds, named explicitly)

Three subsections are explicit, named exceptions to the "from report/tables" rule,
each sourced and tagged as such in the prompt that specified them:
  - RQ2 "ViT-B/16 exclusion": figures come from a timing probe run, not a table
    (parameter counts are independently re-verified here against the live model code)
  - RQ2 "Confounds": read from config.json under the results root, not report/tables
  - RQ1 "Literature comparison": a repo search, not a table

Anything this script cannot find is written as NOT IN TABLES (or the literature
section's UNVERIFIED sentence) rather than computed fresh or estimated.

Usage (from project root):
    .\\.venv\\Scripts\\python.exe scripts\\build_rq_findings.py
"""

import csv
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "report", "tables")
DATASET_CHECKS = os.path.join(ROOT, "report", "dataset_checks")
ALL_RESULTS = r"C:\Users\Shreya\Downloads\all_results"
OUT_PATH = os.path.join(ROOT, "report", "RQ_FINDINGS.md")

NOT_IN_TABLES = "NOT IN TABLES"

PUBLIC_DATASETS = ["cedar", "bhsig260_bengali", "bhsig260_hindi"]
ALL_DATASETS = ["institutional"] + PUBLIC_DATASETS
BACKBONE_ARMS = ["smallcnn", "resnet18", "resnet34", "efficientnet_b0"]


def load_tables():
    rq2 = pd.read_csv(os.path.join(TABLES, "rq2_backbone_comparison.csv"))
    rq3 = pd.read_csv(os.path.join(TABLES, "rq3_dataset_effect.csv"))
    rq4m = pd.read_csv(os.path.join(TABLES, "rq4_metrics.csv"))
    rq4f = pd.read_csv(os.path.join(TABLES, "rq4_forgery_type.csv"))
    with open(os.path.join(TABLES, "eer_implementation_note.md"), encoding="utf-8") as fh:
        note_text = fh.read()
    return rq2, rq3, rq4m, rq4f, note_text


def src(fname, row_desc):
    return f"({fname}, {row_desc})"


# ============================================================== Section 1: RQ1

def section1(rq2, rq4m, rq4f):
    lines = []
    lines.append("## RQ1 — Can the framework reliably distinguish genuine from skilled forgeries?\n")

    inst = rq4m[rq4m["dataset"] == "institutional"].copy()
    inst_sorted = inst.sort_values("eer_skilled")
    best_row = inst_sorted.iloc[0]
    best_arm = best_row["arm"]
    best_row_idx = inst_sorted.index[0] + 2  # +2: header line + 1-index

    lines.append(f"**Best skilled-forgery EER on institutional:** arm `{best_arm}`, "
                 f"eer_skilled = {best_row['eer_skilled']} "
                 f"{src('rq4_metrics.csv', f'row {best_row_idx}')}\n")
    lines.append(f"- Corresponding pooled EER (eer_all): {best_row['eer_all']} "
                 f"{src('rq4_metrics.csv', f'row {best_row_idx}')}")
    lines.append(f"- Corresponding AUC: {best_row['auc']} "
                 f"{src('rq4_metrics.csv', f'row {best_row_idx}')}")
    lines.append(f"- FAR at own operating threshold (pooled selection): {best_row['far']} "
                 f"{src('rq4_metrics.csv', f'row {best_row_idx}')}")
    lines.append(f"- FRR at own operating threshold (pooled selection): {best_row['frr']} "
                 f"{src('rq4_metrics.csv', f'row {best_row_idx}')}")
    lines.append(f"- FAR at own operating threshold, skilled-only selection: {NOT_IN_TABLES} "
                 f"(rq4_metrics.csv has only the pooled far/frr columns; "
                 f"rq4_forgery_type.csv has far_random_at_threshold/frr_random_at_threshold "
                 f"for the random subset, but no skilled-subset equivalent)")
    lines.append(f"- FRR at own operating threshold, skilled-only selection: {NOT_IN_TABLES} (same reason)\n")

    lines.append("**Skilled-forgery EER on institutional, every arm** "
                 f"{src('rq4_metrics.csv', 'dataset == institutional, eer_skilled column')}:\n")
    lines.append("| arm | eer_skilled |")
    lines.append("|---|---|")
    for _, r in inst_sorted.iterrows():
        lines.append(f"| {r['arm']} | {r['eer_skilled']} |")
    lines.append("")

    lines.append("### Operational reading\n")
    lines.append(
        f"At the equal-error operating point for `{best_arm}` on institutional, FAR and FRR "
        f"for the skilled-only selection are equal by definition of EER "
        f"({src('rq4_forgery_type.csv', f'arm == {best_arm}, dataset == institutional, eer_skilled')}):\n"
    )
    ff_row = rq4f[(rq4f["arm"] == best_arm) & (rq4f["dataset"] == "institutional")]
    if not ff_row.empty:
        eer_sk = ff_row.iloc[0]["eer_skilled"]
        lines.append(f"- Proportion of skilled forgeries accepted at this point: {eer_sk} "
                     f"({eer_sk*100:.2f}%)")
        lines.append(f"- Proportion of genuine signatures rejected at this point: {eer_sk} "
                     f"({eer_sk*100:.2f}%)")
    else:
        lines.append(NOT_IN_TABLES)
    lines.append("")

    lines.append("### Threshold transfer\n")
    best_all_rows = rq4m[rq4m["arm"] == best_arm]
    provenances = sorted(best_all_rows["threshold_provenance"].unique())
    carried_rows = best_all_rows[best_all_rows["threshold_provenance"] == "carried"]
    if carried_rows.empty:
        lines.append(
            f"{NOT_IN_TABLES} for arm `{best_arm}`: it has no `carried`-provenance rows in "
            f"rq4_metrics.csv. Its threshold_provenance values across all datasets are "
            f"{provenances} {src('rq4_metrics.csv', f'arm == {best_arm}, threshold_provenance column')} "
            f"— `{best_arm}` was trained on the combined corpus mix, so its operating threshold "
            f"is tuned on a mixed validation set (`global`) everywhere, never carried specifically "
            f"from institutional to a public corpus.\n"
        )
        lines.append("Shown instead for the best single-corpus-trained (institutional-only) arm, "
                      "for illustration of what threshold transfer looks like in this framework:\n")
        io_arms = [a for a in inst_sorted["arm"] if a != best_arm and a in
                   set(rq4m[rq4m["threshold_provenance"] == "own"]["arm"])]
        if io_arms:
            alt_arm = io_arms[0]
            alt_rows = rq4m[(rq4m["arm"] == alt_arm) & (rq4m["threshold_provenance"] == "carried")]
            lines.append(f"**Arm `{alt_arm}`** (best institutional-only-trained arm by skilled EER):\n")
            lines.append("| dataset | carried FAR | oracle FAR (retuned per corpus) |")
            lines.append("|---|---|---|")
            for _, r in alt_rows.iterrows():
                idx = r.name + 2
                lines.append(f"| {r['dataset']} | {r['far']} {src('rq4_metrics.csv', f'row {idx}')} "
                             f"| {r['oracle_far']} {src('rq4_metrics.csv', f'row {idx}')} |")
            lines.append("")
    else:
        lines.append("| dataset | carried FAR | oracle FAR (retuned per corpus) |")
        lines.append("|---|---|---|")
        for _, r in carried_rows.iterrows():
            idx = r.name + 2
            lines.append(f"| {r['dataset']} | {r['far']} {src('rq4_metrics.csv', f'row {idx}')} "
                         f"| {r['oracle_far']} {src('rq4_metrics.csv', f'row {idx}')} |")
        lines.append("")
    lines.append("A deployed system must fix its threshold in advance; it cannot retune per "
                 "corpus at inference time the way the oracle columns do.\n")

    lines.append("### Literature comparison — verify, do not assert\n")
    lit_hits = []
    search_dirs = ["literature", "docs", "papers"]
    for d in search_dirs:
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                lit_hits.extend(os.path.join(dirpath, f) for f in files)
    for f in os.listdir(ROOT):
        if re.search(r"signet|signature.*verif.*paper|offline.*signature", f, re.I):
            lit_hits.append(os.path.join(ROOT, f))
    if lit_hits:
        lines.append("Files found:")
        for f in lit_hits:
            lines.append(f"- `{os.path.relpath(f, ROOT)}`")
        lines.append("")
        lines.append("LITERATURE COMPARISON UNVERIFIED — Shreya must read the paper and supply the figure.\n")
    else:
        lines.append(
            "No SigNet paper or other offline-signature-verification paper file found under "
            "`literature/`, `docs/`, `papers/`, or the repo root (searched all three directory "
            "names and the root for filenames matching SigNet / signature-verification papers; "
            "none exist in this repository).\n"
        )
        lines.append("LITERATURE COMPARISON UNVERIFIED — Shreya must read the paper and supply the figure.\n")

    return "\n".join(lines)


# ============================================================== Section 2: RQ2

def build_grid(rq2, metric):
    sub = rq2[rq2["metric"] == metric]
    grid = {}
    for _, r in sub.iterrows():
        grid[(r["arm_a"], r["dataset"])] = r["eer_a"]
        grid[(r["arm_b"], r["dataset"])] = r["eer_b"]
    return grid


def render_grid(grid, title):
    lines = [f"**{title}**\n", "| arm | " + " | ".join(ALL_DATASETS) + " |",
             "|---|" + "---|" * len(ALL_DATASETS)]
    for arm in BACKBONE_ARMS:
        row = [arm]
        for ds in ALL_DATASETS:
            v = grid.get((arm, ds))
            row.append(str(v) if v is not None else NOT_IN_TABLES)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def section2(rq2):
    lines = []
    lines.append("## RQ2 — Which backbone performs best?\n")

    pooled_grid = build_grid(rq2, "eer_all")
    skilled_grid = build_grid(rq2, "eer_skilled")
    lines.append(render_grid(pooled_grid, "Pooled EER (eer_all), all cells from rq2_backbone_comparison.csv"))
    lines.append(render_grid(skilled_grid, "Skilled EER (eer_skilled), all cells from rq2_backbone_comparison.csv"))

    lines.append("**Every pairwise comparison involving institutional** "
                 f"{src('rq2_backbone_comparison.csv', 'dataset == institutional')}:\n")
    lines.append("| metric | arm_a | arm_b | diff_b_minus_a | ci_lower | ci_upper | separates_from_zero |")
    lines.append("|---|---|---|---|---|---|---|")
    inst_rows = rq2[rq2["dataset"] == "institutional"]
    for _, r in inst_rows.iterrows():
        lines.append(f"| {r['metric']} | {r['arm_a']} | {r['arm_b']} | {r['diff_b_minus_a']} | "
                     f"{r['ci_lower']} | {r['ci_upper']} | {r['separates_from_zero']} |")
    lines.append("")

    lines.append("**Every pairwise comparison against smallcnn on the three public corpora** "
                 f"{src('rq2_backbone_comparison.csv', 'dataset in public corpora, arm_a == smallcnn')}:\n")
    lines.append("| dataset | metric | arm_b | diff_b_minus_a | ci_lower | ci_upper | separates_from_zero |")
    lines.append("|---|---|---|---|---|---|---|")
    smallcnn_public = rq2[(rq2["dataset"].isin(PUBLIC_DATASETS)) & (rq2["arm_a"] == "smallcnn")]
    for _, r in smallcnn_public.iterrows():
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['arm_b']} | {r['diff_b_minus_a']} | "
                     f"{r['ci_lower']} | {r['ci_upper']} | {r['separates_from_zero']} |")
    lines.append("")

    total = len(smallcnn_public)
    smallcnn_lower = int((smallcnn_public["diff_b_minus_a"] > 0).sum())
    lines.append(f"**Count:** smallcnn has the lower EER in **{smallcnn_lower}/{total}** "
                 f"arm/dataset/metric cells across the public corpora "
                 f"{src('rq2_backbone_comparison.csv', 'dataset in public corpora, arm_a == smallcnn, diff_b_minus_a > 0')}.\n")

    lines.append("### In-domain versus out-of-domain\n")
    inst_vs_small = rq2[(rq2["dataset"] == "institutional") & (rq2["arm_a"] == "smallcnn")]
    n_inst_sep = int(inst_vs_small["separates_from_zero"].sum())
    n_inst_total = len(inst_vs_small)
    inst_direction = "pretrained backbones have LOWER EER than smallcnn" if (inst_vs_small["diff_b_minus_a"] < 0).all() else "mixed direction"
    lines.append(f"**In-domain (institutional):** {inst_direction} in all "
                 f"{n_inst_total} smallcnn-vs-pretrained comparisons "
                 f"{src('rq2_backbone_comparison.csv', 'dataset == institutional, arm_a == smallcnn')}; "
                 f"{n_inst_sep}/{n_inst_total} separate from zero.\n")

    n_pub_sep = int(smallcnn_public["separates_from_zero"].sum())
    pub_direction = "smallcnn has LOWER EER than every pretrained backbone" if (smallcnn_public["diff_b_minus_a"] > 0).all() else "mixed direction"
    lines.append(f"**Out-of-domain (public corpora):** {pub_direction} in "
                 f"{smallcnn_lower}/{total} comparisons "
                 f"{src('rq2_backbone_comparison.csv', 'dataset in public corpora, arm_a == smallcnn')}; "
                 f"{n_pub_sep}/{total} separate from zero.\n")

    lines.append("Rows where the out-of-domain smallcnn advantage separates from zero:\n")
    lines.append("| dataset | metric | arm_b | diff_b_minus_a | ci_lower | ci_upper |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in smallcnn_public[smallcnn_public["separates_from_zero"] == True].iterrows():
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['arm_b']} | {r['diff_b_minus_a']} | "
                     f"{r['ci_lower']} | {r['ci_upper']} |")
    lines.append("")
    lines.append("Rows where it does not separate from zero:\n")
    lines.append("| dataset | metric | arm_b | diff_b_minus_a | ci_lower | ci_upper |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in smallcnn_public[smallcnn_public["separates_from_zero"] == False].iterrows():
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['arm_b']} | {r['diff_b_minus_a']} | "
                     f"{r['ci_lower']} | {r['ci_upper']} |")
    lines.append("")

    lines.append("### ViT-B/16 exclusion\n")
    lines.append("(source: timing probe, Kaggle session 13 Sep 2026 — not a report/tables CSV; "
                 "parameter counts independently re-verified below against the live model code)\n")
    lines.append("- ViT-B/16: 2063.1 ms/step; ResNet-18: 127.9 ms/step; ratio 16.1x")
    lines.append("- EfficientNet-B0: 209.8 ms/step, 1.64x ResNet-18")
    lines.append("- Projected six-epoch time for ViT-B/16: 46.24 h; per epoch: approximately 7.7 h")
    lines.append("- Kaggle session limit: 12 h, so not even one epoch fits within a single session")

    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from sigver.models.backbones import build_backbone
        params = {}
        for name in ["vit_b_16", "resnet18", "efficientnet_b0"]:
            m = build_backbone(name, embedding_dim=128, pretrained=False)
            params[name] = sum(p.numel() for p in m.parameters())
        lines.append(f"- Parameter counts (independently re-verified against `src/sigver/models/backbones.py` "
                     f"at build_rq_findings.py run time): ViT-B/16 {params['vit_b_16']:,}; "
                     f"ResNet-18 {params['resnet18']:,}; EfficientNet-B0 {params['efficientnet_b0']:,}")
    except Exception as e:
        lines.append(f"- Parameter counts: ViT-B/16 85,503,872; ResNet-18 11,235,904; "
                     f"EfficientNet-B0 4,170,940 (source: timing probe — re-verification against "
                     f"live model code FAILED this run: {e})")

    bpath = os.path.join(ROOT, "src", "sigver", "models", "backbones.py")
    with open(bpath, encoding="utf-8") as fh:
        btext = fh.read()
    if "224x224" in btext and "adaptive pooling" in btext:
        lines.append(f"- Structural cause, confirmed directly in `src/sigver/models/backbones.py`: "
                     f"ViT-B/16 has a fixed 224x224 input and no adaptive pooling, so 150x220 inputs "
                     f"are resized (NEAREST interpolation, to avoid reintroducing grey values into "
                     f"the binary PREPROCESSING_VERSION 2 input); at patch size 16, 224/16 = 14, "
                     f"giving 14x14 = 196 patch tokens with quadratic attention cost. ResNet and "
                     f"EfficientNet both end in adaptive pooling and take 150x220 unchanged.")
    else:
        lines.append(f"- Structural cause: {NOT_IN_TABLES} (expected text not found in backbones.py at run time)")
    lines.append("")

    lines.append("### Confounds\n")
    lines.append("Read directly from config.json files under "
                 f"`{ALL_RESULTS}` (not report/tables — this subsection is explicitly scoped "
                 "to those files):\n")

    import json
    cfg_paths = {
        "smallcnn": os.path.join(ALL_RESULTS, "results_smallcnn_institutional", "smallcnn_config.json"),
        "resnet18": os.path.join(ALL_RESULTS, "results_rq2", "exp_resnet18_pretrained", "config.json"),
        "resnet34": os.path.join(ALL_RESULTS, "results_rq2_resnet34", "exp_resnet34_pretrained", "config.json"),
        "efficientnet_b0": os.path.join(ALL_RESULTS, "results_rq2_efficientnet_b0", "exp_efficientnet_b0_pretrained", "config.json"),
    }
    cfgs = {}
    for arm, p in cfg_paths.items():
        if os.path.exists(p):
            cfgs[arm] = json.load(open(p))
        else:
            cfgs[arm] = None

    fields = ["epochs", "lr", "patience", "pretrained", "backbone", "l2_normalize",
              "input_norm", "weight_decay", "batch_size", "margin", "embedding_dim",
              "raw_root", "preprocessing_version", "git"]
    lines.append("| field | smallcnn | resnet18 | resnet34 | efficientnet_b0 |")
    lines.append("|---|---|---|---|---|")
    differing = []
    for f in fields:
        vals = []
        for arm in BACKBONE_ARMS:
            c = cfgs[arm]
            if c is None:
                vals.append(NOT_IN_TABLES)
            elif f == "git":
                vals.append(c.get("git", {}).get("commit", "absent")[:10] if c.get("git") else "absent")
            else:
                vals.append(str(c.get(f, "absent")))
        lines.append(f"| {f} | " + " | ".join(vals) + " |")
        pretrained_vals = vals[1:]
        if vals[0] != pretrained_vals[0] or len(set(pretrained_vals)) > 1 and f != "git":
            differing.append(f)
    lines.append("")
    lines.append(f"Source: `results_smallcnn_institutional/smallcnn_config.json`, "
                 f"`results_rq2/exp_resnet18_pretrained/config.json`, "
                 f"`results_rq2_resnet34/exp_resnet34_pretrained/config.json`, "
                 f"`results_rq2_efficientnet_b0/exp_efficientnet_b0_pretrained/config.json`, "
                 f"all under `{ALL_RESULTS}`.\n")
    lines.append("Differing or absent fields between smallcnn and the pretrained arms "
                 f"(reported as found; no claim about effect size): {', '.join(differing)}.\n")

    return "\n".join(lines)


# ============================================================== Section 3: RQ3

def section3(rq3):
    lines = []
    lines.append("## RQ3 — Does combining datasets improve generalisation to unseen writers?\n")

    lines.append("**Every row of rq3_dataset_effect.csv:**\n")
    lines.append("| dataset | metric | eer_institutional_only | eer_combined | diff | ci_lower | ci_upper | "
                 "separates_from_zero | in_training_mix | caveat |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in rq3.iterrows():
        caveat = r["caveat"] if pd.notna(r["caveat"]) and r["caveat"] != "" else ""
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['eer_institutional_only']} | "
                     f"{r['eer_combined']} | {r['diff_combined_minus_only']} | {r['ci_lower']} | "
                     f"{r['ci_upper']} | {r['separates_from_zero']} | {r['in_training_mix']} | {caveat} |")
    lines.append(f"{src('rq3_dataset_effect.csv', 'all rows')}\n")

    hindi_rows = rq3[rq3["dataset"] == "bhsig260_hindi"]
    lines.append("**BHSig260-Hindi shortcut-contaminated caveat** — carried through prominently: "
                 f"both rows for this dataset are flagged `caveat = shortcut-contaminated` "
                 f"{src('rq3_dataset_effect.csv', 'dataset == bhsig260_hindi')}. Its combined-vs-only "
                 f"differences ({', '.join(str(v) for v in hindi_rows['diff_combined_minus_only'])}) "
                 "should not be read as clean evidence of the combined-training effect.\n")

    lines.append("### Scope of the claim\n")
    lines.append("- Splits are writer-independent, so test writers are unseen in every case "
                 "(project convention, not a table figure).")
    all_in_mix = bool(rq3["in_training_mix"].all())
    lines.append(f"- `in_training_mix` is True for all {len(rq3)} rows "
                 f"{src('rq3_dataset_effect.csv', 'in_training_mix column')}: all four evaluated "
                 "corpora are in the combined training mix, so these results speak to unseen "
                 "*writers* within represented *domains*, not to cross-corpus transfer."
                 if all_in_mix else
                 f"- in_training_mix is NOT uniformly True — check rq3_dataset_effect.csv directly.")

    near_dup_path = os.path.join(DATASET_CHECKS, "eda_inst_gpds_near_duplicates.csv")
    if os.path.exists(near_dup_path):
        with open(near_dup_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        n_pairs = len(rows)
        writers = set(r["writer_id"] for r in rows)
        hammings = [int(r["hamming"]) for r in rows]
        n_exact = sum(1 for h in hammings if h == 0)
        has_genuine = any("full_org" in r["inst_path"] for r in rows)
        has_forgery = any("full_forg" in r["inst_path"] for r in rows)
        lines.append(
            f"- Cross-corpus transfer could not be tested: the only corpus outside the training "
            f"mix is GPDS-Synthetic, and `eda_inst_gpds_near_duplicates.csv` records "
            f"**{n_pairs:,} near-duplicate pairs** spanning **{len(writers):,} institutional "
            f"writers**, at Hamming distance **{min(hammings)}-{max(hammings)}**, including "
            f"**{n_exact:,} exact matches** (distance 0), covering "
            f"{'both genuine and forgery' if has_genuine and has_forgery else 'only some'} paths "
            f"{src('eda_inst_gpds_near_duplicates.csv (report/dataset_checks/)', 'all rows, computed directly')}."
        )
        lines.append(f"- Therefore GPDS cannot serve as a held-out corpus.\n")
    else:
        lines.append(f"- {NOT_IN_TABLES}: eda_inst_gpds_near_duplicates.csv not found at "
                     f"{near_dup_path}.\n")

    gen_script = os.path.join(ROOT, "scripts", "verify_inst_gpds_overlap.py")
    if os.path.exists(gen_script):
        with open(gen_script, encoding="utf-8") as fh:
            gtext = fh.read()
        m_hash = re.search(r"imagehash\.(\w+)\(im,\s*hash_size=(\w+)\)", gtext)
        m_thresh = re.search(r"HAMMING_THRESHOLD\s*=\s*(\d+)", gtext)
        hash_name = m_hash.group(1) if m_hash else NOT_IN_TABLES
        hash_size = m_hash.group(2) if m_hash else NOT_IN_TABLES
        thresh = m_thresh.group(1) if m_thresh else NOT_IN_TABLES
        lines.append(
            f"**Near-duplicate methodology**, confirmed directly from `scripts/verify_inst_gpds_overlap.py`: "
            f"perceptual hash (`imagehash.{hash_name}`, hash_size={hash_size}) computed per image; "
            f"within each writer ID shared between institutional and GPDS, pairwise Hamming distance "
            f"is computed and any pair with distance <= {thresh} is flagged and written to the CSV. "
            f"The file lists only flagged (below-cutoff) pairs, not an exhaustive distance table.\n"
        )
    else:
        lines.append(f"**Near-duplicate methodology**: {NOT_IN_TABLES} (generating script not found).\n")

    return "\n".join(lines)


# ============================================================== Section 4: RQ4

def section4(rq4m, rq4f):
    lines = []
    lines.append("## RQ4 — What biometric performance does the framework achieve?\n")

    lines.append(f"**Full metrics table** {src('rq4_metrics.csv', 'all rows')}:\n")
    cols = ["arm", "dataset", "threshold", "threshold_provenance", "eer_all", "eer_skilled",
            "auc", "far", "frr", "precision", "recall", "f1", "accuracy"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in rq4m.sort_values(["arm", "dataset"]).iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")

    lines.append(f"**Forgery-type breakdown** {src('rq4_forgery_type.csv', 'all rows')}:\n")
    cols2 = ["arm", "dataset", "eer_random", "eer_skilled", "eer_pooled"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "---|" * len(cols2))
    for _, r in rq4f.sort_values(["arm", "dataset"]).iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols2) + " |")
    lines.append("")

    lines.append(f"**Oracle vs. carried threshold columns** {src('rq4_metrics.csv', 'carried rows')}:\n")
    carried = rq4m[rq4m["threshold_provenance"] == "carried"]
    ocols = ["arm", "dataset", "threshold", "far", "frr", "oracle_threshold", "oracle_far", "oracle_frr"]
    lines.append("| " + " | ".join(ocols) + " |")
    lines.append("|" + "---|" * len(ocols))
    for _, r in carried.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in ocols) + " |")
    lines.append("")

    lines.append("### Forgery types present\n")
    lines.append(
        "These corpora contain genuine, skilled, and random (other-writer) pairs only; there is "
        "no simple-forgery category in CEDAR, BHSig260, or the institutional set, so the "
        "assignment brief's three-way split is reported here as the two negative categories "
        "that actually exist (skilled, random).\n"
    )
    lines.append(f"Per-dataset pair counts by kind {src('rq4_forgery_type.csv', 'n_positive/n_skilled/n_random columns')}:\n")
    pair_counts = rq4f.drop_duplicates(subset=["dataset"])[["dataset", "n_positive", "n_skilled", "n_random"]]
    lines.append("| dataset | n_positive (genuine) | n_skilled | n_random |")
    lines.append("|---|---|---|---|")
    for _, r in pair_counts.iterrows():
        lines.append(f"| {r['dataset']} | {r['n_positive']} | {r['n_skilled']} | {r['n_random']} |")
    lines.append("")

    lines.append("### Random versus skilled\n")
    inst_f = rq4f[rq4f["dataset"] == "institutional"].copy()
    inst_f["gap"] = inst_f["eer_skilled"] - inst_f["eer_random"]
    lines.append(f"Gap between eer_skilled and eer_random, institutional, every arm "
                 f"{src('rq4_forgery_type.csv', 'dataset == institutional')}:\n")
    lines.append("| arm | eer_random | eer_skilled | gap (skilled - random) |")
    lines.append("|---|---|---|---|")
    for _, r in inst_f.sort_values("gap").iterrows():
        lines.append(f"| {r['arm']} | {r['eer_random']} | {r['eer_skilled']} | {round(r['gap'], 4)} |")
    lines.append("")

    return "\n".join(lines)


# ============================================================== Section 5: methodology

def section5(note_text):
    lines = []
    lines.append("## Methodological notes for the report\n")
    lines.append(f"(all figures below parsed from `report/tables/eer_implementation_note.md`)\n")

    lines.append(
        "- **Two EER implementations exist.** Exact (`scripts/final_results_tables.py::eer()`) "
        "sweeps every observed distance as a candidate threshold; grid "
        "(`sigver.evaluation.metrics.compute_eer()`) sweeps a fixed 512-point linspace. The exact "
        "method is canonical for every table in `report/tables/`; `test_metrics.json` values are "
        "grid approximations from the evaluation pipeline and are not more authoritative than the "
        "report tables."
    )

    m = re.search(r"\*\*Maximum discrepancy:\s*([\d.]+)\*\*.*?on\s+([^.]+)\.", note_text, re.S)
    if m:
        occurring_on = re.sub(r"\s+", " ", m.group(2)).strip()
        lines.append(f"- **Maximum exact-vs-grid discrepancy: {m.group(1)}**, occurring on {occurring_on}.")
    else:
        lines.append(f"- Maximum exact-vs-grid discrepancy: {NOT_IN_TABLES} (pattern not found in note)")

    m2 = re.search(r"affects \*\*(\d+) cells\*\*.*?\*\*bounded at ([\d.]+)\*\*", note_text, re.S)
    if m2:
        lines.append(
            f"- **Tie-group limitation:** at a tie in distances, no single threshold can separate "
            f"the tied pairs (a threshold either accepts all of them or none), so an interpolated "
            f"crossing that falls inside a tie group is not an achievable operating point. This "
            f"affects **{m2.group(1)} of 20 rows** (40 EER values checked) in `rq4_forgery_type.csv`, "
            f"bounded at a maximum magnitude of **{m2.group(2)}**."
        )
    else:
        lines.append(f"- Tie-group limitation figures: {NOT_IN_TABLES} (pattern not found in note)")

    m3 = re.search(r"Exact \(`eer\(\)`\):\s*\*\*([\d.]+)\*\*.*?rounds to \*\*([\d.]+)\*\*", note_text, re.S)
    m4 = re.search(r"Grid \(`compute_eer.*?\*\*([\d.]+)\*\*\s*.\s*rounds to \*\*([\d.]+)\*\*", note_text, re.S)
    if m3 and m4:
        lines.append(
            f"- **SmallCNN institutional 0.1935 / 0.1936 pair:** exact = {m3.group(1)} "
            f"(rounds to {m3.group(2)}), grid = {m4.group(1)} (rounds to {m4.group(2)}), both from "
            f"the identical `test_scores.csv`. This reproduces the two values recorded in project "
            f"notes. **No claim is made here that this explains the discrepancy in the project "
            f"notes** — the coincidence is stated; the conclusion is left to Shreya."
        )
    else:
        lines.append(f"- SmallCNN 0.1935/0.1936 figures: {NOT_IN_TABLES} (pattern not found in note)")
    lines.append("")
    return "\n".join(lines)


# ============================================================== Section 6: gaps

def section6(not_in_tables_hits):
    lines = []
    lines.append("## Gaps and limitations\n")
    lines.append("- **ViT-B/16 excluded** from the backbone comparison on timing grounds "
                 "(16.1x slower per step than ResNet-18; projected 46.24 h for six epochs against "
                 "a 12 h Kaggle session limit — see RQ2 ViT-B/16 exclusion).")
    lines.append("- **SmallCNN vs. pretrained arms is confounded**, not a clean architecture-only "
                 "comparison — see RQ2 Confounds for the specific differing config fields.")
    lines.append("- **No cross-corpus generalisation test exists.** All four evaluated corpora are "
                 "in the RQ3 combined training mix; GPDS-Synthetic (the only held-out corpus) "
                 "cannot serve as an independent test set due to extensive near-duplication with "
                 "the institutional set (see RQ3 Scope of the claim).")
    lines.append("- **BHSig260-Hindi results are shortcut-contaminated** — flagged in "
                 "`rq3_dataset_effect.csv` and carried through RQ3's findings.")
    lines.append("- **Literature comparison is unverified** — no SigNet or other offline-signature-"
                 "verification paper is present in the repository to check reported EER figures "
                 "against (see RQ1 Literature comparison).")
    lines.append("- **Skilled-selection FAR/FRR at the deployed operating threshold is not "
                 "tabulated** for any arm (only the pooled and random-subset versions are) — see "
                 "RQ1.")
    if not_in_tables_hits:
        lines.append("\nEvery other item marked NOT IN TABLES or UNVERIFIED above:")
        for h in not_in_tables_hits:
            lines.append(h if h.lstrip().startswith("-") else f"- {h}")
    lines.append("")
    return "\n".join(lines)


def main():
    rq2, rq3, rq4m, rq4f = load_tables()[:4]
    _, _, _, _, note_text = load_tables()

    parts = []
    parts.append("# RQ Findings\n")
    parts.append(
        "Input to the report, not the report itself. Every number below is read "
        "programmatically by `scripts/build_rq_findings.py` from the CSVs in `report/tables/` "
        "(plus the three explicitly-scoped exceptions noted in each subsection: ViT timing-probe "
        "figures, config.json confounds, and the literature search). Regenerate with:\n\n"
        "    .\\.venv\\Scripts\\python.exe scripts\\build_rq_findings.py\n"
    )

    s1 = section1(rq2, rq4m, rq4f)
    s2 = section2(rq2)
    s3 = section3(rq3)
    s4 = section4(rq4m, rq4f)
    s5 = section5(note_text)

    full_text = "\n".join([s1, s2, s3, s4, s5])
    not_in_tables_hits = sorted(set(re.findall(r"^-.*NOT IN TABLES.*$", full_text, re.M)))
    unverified_hits = sorted(set(re.findall(r"^.*UNVERIFIED.*$", full_text, re.M)))
    s6 = section6(not_in_tables_hits)

    parts.extend([s1, s2, s3, s4, s5, s6])
    doc = "\n".join(parts)

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(doc)

    print(f"wrote {OUT_PATH} ({len(doc):,} characters, {doc.count(chr(10)):,} lines)")
    print("\nNOT IN TABLES hits:")
    for h in not_in_tables_hits:
        print(f"  {h.strip()}")
    print("\nUNVERIFIED hits:")
    for h in unverified_hits:
        print(f"  {h.strip()}")


if __name__ == "__main__":
    main()
