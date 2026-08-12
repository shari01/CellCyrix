# Single-cell 10x Consensus-Annotation Pipeline — Master Documentation

**Package** `single_cell_pipeline_agent_standalone`
**Entry point** `python main.py` (configuration only — there is no CLI)
**Owner** Sheryar Malik · Ayass Bioscience
**Document date** 2026-08-06
**Scope** This document describes the pipeline **as it exists in this package today**. Every
threshold, model name, file name and formula below was read out of the current source. Nothing
is aspirational and nothing describes a version that used to exist.

![Pipeline architecture](pipeline_architecture.png)

---

## Table of contents

| § | Section |
|---|---|
| 1 | [What this pipeline does](#1-what-this-pipeline-does) |
| 2 | [How to run it](#2-how-to-run-it) |
| 3 | [Inputs](#3-inputs) |
| 4 | [Stage 0 — pre-flight data validation](#4-stage-0--pre-flight-data-validation) |
| 5 | [Stage 1 — QC filtering and doublet detection](#5-stage-1--qc-filtering-and-doublet-detection) |
| 6 | [Stage 2 — normalization and the three expression matrices](#6-stage-2--normalization-and-the-three-expression-matrices) |
| 7 | [Stage 3 — highly variable genes](#7-stage-3--highly-variable-genes) |
| 8 | [Stage 4 — scaling and PCA](#8-stage-4--scaling-and-pca) |
| 9 | [Stage 5 — batch integration and embeddings](#9-stage-5--batch-integration-and-embeddings) |
| 10 | [Stage 6 — Leiden clustering](#10-stage-6--leiden-clustering) |
| 11 | [Stage 7 — diffusion pseudotime](#11-stage-7--diffusion-pseudotime) |
| 12 | [Stage 8 — cell-type identity: the consensus system](#12-stage-8--cell-type-identity-the-consensus-system) |
| 12.1 | [Marker evidence — what every voter reads](#121-marker-evidence--what-every-voter-reads) |
| 12.2 | [Voter A — CellTypist (machine-learning)](#122-voter-a--celltypist-machine-learning) |
| 12.3 | [Voter B — SingleR (reference correlation)](#123-voter-b--singler-reference-correlation) |
| 12.4 | [Voter C — knowledge-based LLM marker reasoning](#124-voter-c--knowledge-based-llm-marker-reasoning) |
| 12.5 | [Voter D — PubMed literature RAG](#125-voter-d--pubmed-literature-rag) |
| 12.6 | [The knowledge base — 404-node cell hierarchy](#126-the-knowledge-base--404-node-cell-hierarchy) |
| 12.7 | [Harmonization to a controlled vocabulary](#127-harmonization-to-a-controlled-vocabulary) |
| 12.8 | [The lineage gate — an independent biology check](#128-the-lineage-gate--an-independent-biology-check) |
| 12.9 | [Vote counting and tie-breaking](#129-vote-counting-and-tie-breaking) |
| 12.10 | [Adjudication and out-of-domain deference](#1210-adjudication-and-out-of-domain-deference) |
| 12.11 | [Confidence tiers](#1211-confidence-tiers) |
| 12.12 | [Cell state versus cell identity](#1212-cell-state-versus-cell-identity) |
| 12.13 | [The subtype layer and its marker-evidence gate](#1213-the-subtype-layer-and-its-marker-evidence-gate) |
| 13 | [Stage 9 — identity markers per cell type](#13-stage-9--identity-markers-per-cell-type) |
| 14 | [Stage 10 — differential expression](#14-stage-10--differential-expression) |
| 15 | [Stage 11 — composition analysis](#15-stage-11--composition-analysis) |
| 16 | [Stage 12 — pathway enrichment](#16-stage-12--pathway-enrichment) |
| 17 | [Stage 13 — report, manifest, deconvolution export](#17-stage-13--report-manifest-deconvolution-export) |
| 18 | [Complete output inventory](#18-complete-output-inventory) |
| 19 | [Complete configuration reference](#19-complete-configuration-reference) |
| 20 | [Reproducibility](#20-reproducibility) |
| 21 | [Worked example — a real run, end to end](#21-worked-example--a-real-run-end-to-end) |
| 22 | [Known limitations and what the pipeline refuses to claim](#22-known-limitations-and-what-the-pipeline-refuses-to-claim) |
| 23 | [Module map](#23-module-map) |
| 24 | [Environment and dependencies](#24-environment-and-dependencies) |

---

## 1. What this pipeline does

Given raw 10x Genomics feature-barcode matrices — one sample or a whole `<group>/<sample>/`
cohort tree — the pipeline produces:

1. a QC'd, normalized, clustered, embedded `.h5ad` object;
2. a **cell-type identity per cell**, decided by up to four independent annotators voting,
   with the disagreement, the evidence and a confidence tier all written down;
3. **donor-level differential expression** between experimental arms, overall and per cell type;
4. cell-type composition comparisons with the donor as the unit of replication;
5. pathway enrichment;
6. a branded HTML/PDF report and a provenance manifest that makes the run re-runnable.

Two design commitments shape everything below.

**The structural partition is always Leiden.** Cells are grouped by Scanpy's Leiden algorithm on
the neighbour graph. No annotator is ever allowed to define the clusters — CellTypist's internal
over-clustering is not used, SingleR's per-cell mode is not used. Annotation is a *labelling* of
a partition that was decided before any annotator ran, so the partition is independent of the
labels and every cluster can be audited against its own markers.

**Identity is disease-agnostic.** No annotator is ever told the disease. Tissue/organ context is
supplied because it is cell biology (a keratinocyte is a keratinocyte), but "psoriasis",
"tumour" or "Alzheimer" never enters a model selection or a prompt. A cell type must be
recognisable from what the cell expresses, not from what the study is about — otherwise the
annotation confirms the hypothesis it is supposed to test. The PubMed voter is allowed to use
disease as *soft retrieval context* only (it changes which abstracts are fetched, never which
label is permitted).

---

## 2. How to run it

```bash
# 1. environment  (Python >= 3.11; the pipeline is verified on 3.12)
python -m pip install -r requirements.txt
python -m playwright install chromium        # only needed for PDF report export

# 2. credentials (only for the LLM voter, the PubMed voter and the report narratives)
cp .env.example .env
#   set OPENROUTER_API_KEY and OPENROUTER_MODEL

# 3. edit config.yaml  — mode, input paths, thresholds, which voters are on

# 4. run
python main.py
```

`main.py` reads `config.yaml`, flattens the nested sections onto the driver's keyword
arguments, and calls `run_pipeline` (single) or `run_pipeline_multi` (multi). Unknown options
are **reported and ignored**, never silently dropped — a typo in a config key prints a line
rather than quietly changing nothing.

Everything is controlled from YAML. To run a second configuration, copy the YAML and repoint
`CONFIG_FILE` in `main.py`. There is deliberately no console script and no `python -m` CLI.

### Test it

```bash
pytest                        # unit + integration tests
ruff format . && ruff check . # the formatting/lint gate
```

---

## 3. Inputs

> Full input-format reference — every accepted filename pattern, the raw-count gate, the
> `metadata.csv` contract and the failure modes — is in
> [INPUT_FORMATS.md](INPUT_FORMATS.md).

### 3.1 Expression matrices (required)

Standard Cell Ranger trio, compressed or not, bare or sample-prefixed filenames:

```
matrix.mtx[.gz]        barcodes.tsv[.gz]        features.tsv[.gz]  (or genes.tsv[.gz])
```

The Matrix Market matrix is read and **transposed to cells × genes**; barcodes populate `.obs`,
gene IDs/symbols populate `.var`, and `var_names` are made unique. A missing file raises
`FileNotFoundError`; a corrupt one raises `ValueError` naming the offending path.

**Single mode** — one folder:

```
single:
  single_10x_dir: 'input_data/demo_cohort_GSE283500/Healthy_skin/GSM8664023_NST8_CD8skin'
```

**Multi mode** — a two-level tree, where the first level is the experimental arm:

```
<multi_base_dir>/
├── Healthy_skin/
│   ├── GSM8664023_NST8_CD8skin/    matrix.mtx.gz  barcodes.tsv.gz  features.tsv.gz
│   └── GSM8664024_.../
└── Psoriasis_skin/
    ├── GSM8664031_.../
    └── GSM8664032_.../
```

If `metadata.csv` / `group_map.csv` (`sample`, `group` columns) sits at `multi_base_dir`, it is
auto-detected and becomes the **source of truth** for group assignment, overriding the
folder-derived group. Samples whose group is a sentinel — `_EXCLUDED`, `_REVIEW`,
`_NON_EXPRESSION`, or any leading-underscore value — are skipped.

### 3.2 Study metadata (optional but recommended)

`GSE*_metadata.json`. Used for two things and nothing else:

- **species** from the `taxon` field;
- **tissue** inferred by the LLM from the title / summary / sample titles.

Species and tissue then select the CellTypist model and the SingleR reference, and the group
names are searched for a baseline arm to root pseudotime at. Without this file, set `tissue:`
and `species:` explicitly in the config.

### 3.3 Bundled reference data

`shared_reference/` ships inside the package so a run needs no downloads:

| Bucket | Size | What reads it |
|---|---|---|
| `celltypist_models/` — 41 `.pkl` models | ~54 MB | the CellTypist voter, loaded by **full forward-slash path**. Passing a bare model name makes CellTypist try to download its whole repertoire. |
| `celltype_markers_references/TIS_CELL_markers_v3/master_celltype_markers_long.csv` | 1.2 MB | `lineage_panels.py` builds the lineage-gate panels from it — 7,135 curated cell-type/marker rows carrying `specificity_human` and `marker_score`. |

Override the location with `SHARED_REFERENCE_ROOT` (absolute paths only — a relative path
resolves against the working directory, which is a silent-wrong-answer trap).

### 3.4 Credentials

`.env` (never committed; `.env.example` holds the placeholders):

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | knowledge-based voter, PubMed voter's adjudication, adjudicator, report narratives, tissue/model auto-selection |
| `OPENROUTER_MODEL` | the model slug, e.g. `anthropic/claude-sonnet-4.6`. **No default model is hardcoded** — if the LLM layer is on and this is unset, construction raises immediately. |
| `REPORT_LLM_MODEL` | optional override for the report's model only |
| `RSCRIPT_EXE` | optional path to `Rscript` for the SingleR bridge |
| `LLM_TEMPERATURE`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_MAX_TOKENS` | optional; defaults 0 / 60 s / 3 / 800 |

---

## 4. Stage 0 — pre-flight data validation

Runs on the freshly loaded matrix, **before** any processing, so a fatal defect costs seconds
instead of hours. Four severities, and the split between them is the whole point: silently
"repairing" a scientific problem is how a pipeline produces plausible-but-wrong results.

| Severity | Meaning | Examples |
|---|---|---|
| **FAIL** | would corrupt or mislead every downstream result → raises `DataValidationError`, run stops | matrix is already normalized/scaled rather than raw counts; NaN/Inf present; negative values; empty matrix; **integration batch identical to the biological condition**; one sample carrying two group labels |
| **FIXED** | mechanical, information-preserving defect, repaired in place and logged | non-unique barcodes; duplicate gene names; all-zero genes or cells; float dtype holding integers; Ensembl IDs when a symbol column exists |
| **WARN** | a real scientific limitation the pipeline already handles safely; a human should know, but it blocks nothing | fewer than 2 samples in a group (pseudobulk DE will correctly skip); no detectable mitochondrial genes; `sample` column absent |
| **PASS** | check passed | — |

**Why raw counts are mandatory.** DESeq2's negative-binomial model and its size-factor
normalization are defined on counts. Scrublet's simulated-doublet construction is defined on
counts. Handing either a normalized matrix produces numbers that look fine and mean nothing —
so the pipeline refuses rather than proceeds.

Report: `00_data_validation/*_data_validation.csv` and `.txt`.

Also usable standalone:

```python
from ...data_validation import validate_and_fix, validate_path
adata, report = validate_path("path/to/10x_dir")
report.write("out/00_data_validation")
```

---

## 5. Stage 1 — QC filtering and doublet detection

### 5.1 Cell filters

A cell is **kept** only when all three hold:

| Rule | Default | Config key |
|---|---|---|
| `n_genes_by_counts > min_genes` | 200 | `qc.min_genes` |
| `n_genes_by_counts < max_genes` | 6000 | `qc.max_genes` |
| `pct_counts_mt < max_mito_percent` | 15.0 | `qc.max_mito_percent` |

**Biology.** A barcode with very few detected genes is usually an empty droplet or a dying
cell whose transcriptome has already degraded. A barcode with an unusually high gene count is
often two cells in one droplet. A high mitochondrial fraction means the cell membrane ruptured
and cytoplasmic mRNA leaked out while the mitochondria stayed put, so what remains is not a
representative transcriptome.

**These numbers are tissue-dependent and belong in the config.** 15% mitochondrial reads is
lenient for epithelium and strict for cardiomyocytes; a flat 6000-gene ceiling discards
legitimately large cells in some tissues. The defaults are the historical values so an
unchanged config filters exactly the cells it always did.

Per-rule rejection counts are recorded in the summary and the manifest. **The rules overlap** —
one cell can fail two — so the per-rule counts generally sum to more than the total removed,
and the manifest says so explicitly rather than leaving the reader to assume they are disjoint.

### 5.2 Doublet detection

`sc.pp.scrublet(adata, batch_key=<sample>, random_state=seed)` — run on **raw counts, per
sample**. Per sample matters: Scrublet simulates doublets by adding pairs of observed
transcriptomes, and pooling libraries with different depths distorts the simulated
distribution.

- `do_doublet_detection: true` — run it
- `remove_doublets: true` — drop predicted doublets; `false` — flag only, keep the cells

---

## 6. Stage 2 — normalization and the three expression matrices

```python
sc.pp.normalize_total(adata, target_sum=1e4)   # counts per 10,000 (CP10k)
sc.pp.log1p(adata)                             # log(x+1)
adata.raw = adata.copy()                       # a COPY, deliberately
```

**Why CP10k then log.** Sequencing depth varies several-fold between cells for purely technical
reasons, so raw counts are not comparable cell to cell — library-size scaling fixes that.
`log1p` then stabilises the variance: without it, a handful of very highly expressed genes
dominate the PCA and the neighbour graph reflects depth rather than biology.

**Why `.raw` is a copy.** The later `sc.pp.scale()` is in-place, and `adata.raw.X` and
`adata.X` can share one buffer. Assigning `adata.raw = adata` without copying means scaling
silently corrupts `.raw`, and every downstream step that reads `.raw` for log-normalized
expression — markers, DE, annotation — is then reading z-scores. The copy costs memory and
prevents an entire class of invisible wrongness.

The object therefore carries **three matrices**, each with one job:

| Location | Content | Read by |
|---|---|---|
| `layers['counts']` | raw integer counts, never mutated | pseudobulk DESeq2, Scrublet, the Bisque export |
| `.raw` | log1p(CP10k), unscaled | identity markers, cell-level DE, and the annotation working copy |
| `.X` | scaled (z-scored, clipped) after Stage 4 | PCA → neighbours → UMAP → Leiden only |

Annotation never trusts `.X` or `.raw` blindly: `get_lognorm()` re-derives a clean
log1p(CP10k) matrix from the `counts` layer, because the counts layer is the one thing scaling
cannot touch. It falls back to `.raw` (only if it still looks log-normed) then `.X`, with an
explicit warning.

---

## 7. Stage 3 — highly variable genes

`seurat_v3` flavour, computed on the `counts` layer, `batch_key = "sample"`.

| Cell count | `n_top_genes` |
|---|---|
| < 4,000 | 2,000 |
| ≥ 4,000 | 4,000 |

**Why HVGs at all.** Most genes are not informative for distinguishing cell types — they are
either off everywhere or on everywhere. Restricting PCA to the genes whose variance exceeds
what their mean expression predicts concentrates the signal and cuts the noise.

**Why batch-aware.** `seurat_v3` with `batch_key` ranks a gene by how variable it is *within*
each sample, so a gene that is high in one donor purely because of a batch effect does not win
a slot.

**The four-rung ladder.** `seurat_v3` fits a separate LOESS mean-variance curve **per batch**
(`span=0.3, degree=2`). A local quadratic needs enough distinct x-values in every neighbourhood
it visits, and a very small library does not have them — one 353-nucleus sample out of 21 aborted
a 158,084-cell run *after* 15 minutes of loading, QC and Scrublet, with
`ValueError: There are other near singularities as well. 0.090619`. The variance model is fine;
the fit is simply undefined at that sample size. So `select_hvgs()` walks a ladder and stops at
the first rung that fits:

1. `seurat_v3` + `batch_key` over every batch — **unchanged**, so any cohort that works today
   selects precisely the genes it selected before;
2. `seurat_v3` + `batch_key` with unfittable batches excluded **from the gene ranking only**.
   Membership is established by actually fitting each batch, not by guessing a cell-count
   cutoff. Excluded cells stay in the object and are still clustered, annotated and DE-tested —
   they only lose their vote on which genes are variable;
3. `seurat_v3` pooled over all cells (batch-awareness lost, curve well-conditioned);
4. `seurat` (dispersion binning on log-normalized data). No LOESS anywhere, so this rung cannot
   raise — it exists so the pipeline can never die here.

The rung that ran and any excluded batch go into the report and the manifest. A silently
different gene set is worse than a crash.

---

## 8. Stage 4 — scaling and PCA

```python
sc.pp.scale(adata, max_value=10)          # z-score per gene, clipped at 10 SD
sc.tl.pca(adata, n_comps=50, random_state=seed)
```

**Why scale.** PCA maximises variance, so without z-scoring, highly expressed genes dominate
purely because their absolute variance is larger. Clipping at 10 standard deviations stops a
single extreme outlier cell from defining a component.

**Why 50 components.** Enough to capture the biological structure of a typical scRNA-seq
dataset while discarding the long tail of noise components. The downstream neighbour graph uses
the first **30** (see Stage 5) — the extra 20 exist so the variance-ratio plot shows where the
elbow actually is instead of cutting the curve off at the point being justified.

Output: `03_dimensionality_reduction_and_embeddings/pca_variance_ratio_*.png`.

---

## 9. Stage 5 — batch integration and embeddings

### 9.1 Integration

| `integration_method` | Call |
|---|---|
| `bbknn` | `bbknn` on `batch_key`, which forces every cell's neighbourhood to be drawn from several batches |
| `null` | `sc.pp.neighbors(n_neighbors=15, n_pcs=30, random_state=seed)` |

If BBKNN fails or is unavailable, the pipeline degrades to plain neighbours **with a logged
reason** and records which method actually ran (`integration_method_used` in the manifest).

### 9.2 The confounding problem, stated plainly

In a cohort study each donor belongs to exactly one arm, so `sample` is **nested inside**
`group`. That is the standard case-control design, not a metadata error, and nothing aborts
because of it. But it means batch correction cannot be selective: BBKNN mixes neighbourhoods
across samples, and because samples do not straddle arms, between-condition differences get
mixed away along with the technical ones. **Over-integration is a real risk in exactly the
designs this pipeline is built for.**

The manifest records the design verdict for every run:

```
params.integration.batch_design.verdict                  = nested
params.integration.batch_design.confounded_with_condition = True
params.integration.batch_design.interpretation           = every one of the 4 'sample' levels
                                                           belongs to exactly one 'group' …
```

### 9.3 The policy: integration is annotation-only

Integration writes the graph, and the graph feeds how cells are **grouped and named**. It never
touches the numbers a statistical test reads:

```
integration writes → obsp['connectivities'], obsp['distances'], uns['neighbors']
        and thence → obsm['X_umap'], obs['leiden'], obsm['X_diffmap']

DE reads           → layers['counts'] (raw), .raw (log-norm),
                     obs['group'], obs['sample'], obs[celltype]
```

Pseudobulk DE sums raw counts per donor; cell-level DE reads log-normalized `.raw`. Neither
consults a graph, a corrected embedding, or a corrected expression matrix — BBKNN does not
produce one. `check_de_inputs_uncorrected()` **verifies that boundary at run time** rather than
trusting the comment, and writes `de_boundary.ok` plus any `boundary_violations` to the manifest.

**The one indirect path, stated rather than implied.** Per-cell-type DE is stratified by
`celltype`, which comes from clustering, which comes from the corrected graph. Integration
therefore cannot bias a *test statistic*, but it can change *which cells are grouped together*
before the test runs. That dependency is recorded verbatim in the manifest so a reviewer sees
it without reading the source.

### 9.4 Embeddings

- `sc.tl.umap(random_state=seed)` — always
- `sc.tl.tsne(n_pcs=30, use_rep="X_pca", random_state=seed)` — only if `skip_tsne: false`, and
  automatically skipped above 50,000 cells (cost, not correctness)

---

## 10. Stage 6 — Leiden clustering

```python
sc.tl.leiden(adata, resolution=0.5, random_state=seed)   # → obs['leiden']
```

Leiden optimises modularity on the neighbour graph and, unlike Louvain, guarantees
well-connected communities. Resolution 0.5 is the pipeline's historical value; changing it
changes cluster IDs and therefore every per-cluster annotation, so it is a deliberate config
decision rather than something the pipeline tunes for you.

`clustering.min_cluster_cells: 20` — a smaller cluster is *reported* as too small to annotate
reliably. It is not deleted.

### Optional resolution audit

With `evaluate_resolutions: true`, each candidate in `resolution_candidates`
(`[0.2, 0.4, 0.5, 0.6, 0.8, 1.0]`) is additionally clustered **on the same neighbour graph**
into its own `obs['leiden_res_<r>']` column, and a diagnostics table is written to
`04_clustering_and_cell_states/leiden_resolution_evaluation.csv`:

- `n_clusters`, min / median / max cluster size, clusters below `min_cluster_cells`
- **ARI and NMI against the adjacent resolution** — how much the partition actually changes
- **PCA silhouette** — computed on a seeded random subsample above
  `resolution_silhouette_max_cells: 5000`, because silhouette is O(n²) in memory

`obs['leiden']` is **never** overwritten and nothing is auto-selected. The configured
resolution stays primary; the CSV records `selection_rule = user_configured (no automatic
re-selection)` so that fact is auditable. The audit costs one extra Leiden run per candidate,
and the primary clustering is bit-wise unaffected by whether it ran.

---

## 11. Stage 7 — diffusion pseudotime

Optional (`do_dpt`). `sc.tl.diffmap()` → `sc.tl.dpt()`.

**Pseudotime needs a biological starting point.** DPT measures diffusion distance *from a root
cell*, so the root determines the direction of the entire ordering. Rooting arbitrarily
produces a trajectory that looks exactly as convincing as a correct one and means nothing.

Root resolution (`dpt_root_group`):

- an explicit group name → rooted at that group's diffusion tip;
- `auto` → the LLM (with a name heuristic as backstop) picks the **baseline/normal** arm;
- **no identifiable baseline → DPT is skipped**, with a logged reason telling the user to set
  `dpt_root_group`. It is not rooted arbitrarily and it is not silently omitted.

Outputs: `umap_*_dpt_pseudotime.png`, `diffmap_*_dpt_pseudotime.png`, and
`obs['dpt_pseudotime']`.

---

## 12. Stage 8 — cell-type identity: the consensus system

This is the scientific core. A Leiden cluster is a group of transcriptionally similar cells; it
has no name. Assigning the name is where single-cell analysis usually goes wrong, because a
single annotator is confidently wrong in ways nothing downstream can detect.

The design: **up to four independent annotators, drawing on different kinds of evidence, vote
per cluster.** Their disagreement is not hidden — it is the confidence signal.

```
   Leiden cluster
        │
        ├─── ranked marker genes ──┬──► CellTypist       (ML, closed vocabulary)
        │                          ├──► SingleR          (reference correlation, closed)
        │                          ├──► Knowledge LLM    (marker reasoning, open)
        │                          └──► PubMed RAG       (literature, open)
        │                                    │
        │                          harmonize to the 404-node hierarchy
        │                                    │
        └─── lineage gate (5 marker panels, independent of all voters)
                                             │
                              tally → adjudicate → tier → subtype
                                             │
                              obs['celltype'], obs['consensus_tier'], …
```

Each voter is switchable (`enable_celltypist`, `enable_singler`, `enable_knowledge_based`,
`enable_pubmed`); the consensus uses whoever is on, and at least one must be.

### 12.1 Marker evidence — what every voter reads

`rank_genes_groups` per Leiden cluster, one-vs-rest, then filtered and ranked:

| Setting | Value | Why |
|---|---|---|
| positive `logfoldchanges` only | — | a marker is something the cluster *has* |
| `min_detection_fraction` | 0.10 | the gene must be detected in ≥10% of the cluster's cells. Filters "significant but barely detected" genes, which read as convincing in a prompt and are not |
| sort order | `logfoldchanges` ↓, `scores` ↓, `pvals_adj` ↑ (stable) | **effect size first** |
| `top_n_markers` | 50 | handed to the reasoning voters |

**Why not sort by adjusted p-value.** That was the previous behaviour and it collapses in any
well-powered dataset: thousands of genes underflow to `padj == 0.0`, so the "top" markers handed
to the reasoning voters were effectively arbitrary among the tied set rather than the strongest.
Effect size does not tie like that. The rule is recorded in the manifest as
`marker_ranking_method`, so a run's marker evidence is attributable to a specific selection rule.

### 12.2 Voter A — CellTypist (machine-learning)

**Method.** CellTypist is a logistic-regression classifier trained on annotated reference
atlases. It predicts **per cell**, and the pipeline then takes the per-cluster **majority** of
those per-cell labels.

**Confidence** = the fraction of the cluster's cells carrying the winning label.

**Heterogeneity metrics** (advisory — they never change a vote):

| Metric | Meaning |
|---|---|
| `celltypist_dominant_fraction` | fraction on the winning label |
| `celltypist_second_label` / `_second_fraction` | the runner-up |
| `celltypist_label_entropy` | Shannon entropy of the label distribution |
| `celltypist_unique_label_count` | how many distinct labels appear |
| `mixed_cluster_flag` | **true** when dominant < 0.70 **or** runner-up ≥ 0.20 |

A mixed flag means the cluster is heterogeneous — either the clustering under-split it or the
classifier is unsure. It is surfaced, not acted on, except that it is one of the four conditions
for out-of-domain deference (§12.10).

#### The 41 bundled models (sub-models)

All 41 ship in `shared_reference/celltypist_models/`, so a run needs no network. With
`celltypist_model: auto`, the LLM picks a **tissue-appropriate** model from this catalog;
if no organ matches, it falls back to `Immune_All_Low.pkl`. Selection is tissue-driven,
**never disease-driven**, and the selector can only choose a name that exists in the catalog.

| Group | Models |
|---|---|
| **General / immune (pan-tissue)** | `Immune_All_Low.pkl` (high-resolution immune subtypes from 20 tissues — the default fallback), `Immune_All_High.pkl` (broad immune types), `Pan_Fetal_Human.pkl` |
| **Epithelial-rich solid tissue** | `Human_Endometrium_Atlas.pkl`, `Human_Placenta_Decidua.pkl`, `Cells_Adult_Breast.pkl`, `Cells_Intestinal_Tract.pkl`, `Human_Colorectal_Cancer.pkl`, `Adult_Human_Skin.pkl`, `Fetal_Human_Skin.pkl`, `Cells_Human_Tonsil.pkl` |
| **Lung / airway** | `Human_Lung_Atlas.pkl`, `Cells_Lung_Airway.pkl`, `Nuclei_Lung_Airway.pkl`, `Human_IPF_Lung.pkl`, `Human_PF_Lung.pkl`, `Cells_Fetal_Lung.pkl` |
| **Abdominal / endocrine** | `Healthy_Human_Liver.pkl`, `Adult_Human_PancreaticIslet.pkl`, `Fetal_Human_Pancreas.pkl`, `Fetal_Human_AdrenalGlands.pkl`, `Fetal_Human_Pituitary.pkl` |
| **Cardiovascular** | `Healthy_Adult_Heart.pkl`, `Adult_Human_Vascular.pkl` |
| **Nervous system / eye / ear** | `Adult_Human_MTG.pkl`, `Adult_Human_PrefrontalCortex.pkl`, `Human_AdultAged_Hippocampus.pkl`, `Human_Longitudinal_Hippocampus.pkl`, `Developing_Human_Brain.pkl`, `Developing_Human_Hippocampus.pkl`, `Fetal_Human_Retina.pkl`, `Human_Developmental_Retina.pkl`, `Nuclei_Human_InnerEar.pkl` |
| **Developmental / other** | `Developing_Human_Organs.pkl`, `Developing_Human_Gonads.pkl`, `Developing_Human_Thymus.pkl`, `Human_Embryonic_YolkSac.pkl` |
| **Blood / PBMC** | `Healthy_COVID19_PBMC.pkl`, `Adult_COVID19_PBMC.pkl`, `COVID19_HumanChallenge_Blood.pkl`, `COVID19_Immune_Landscape.pkl` |

#### The failure mode this voter has, and how it is handled

**A classifier cannot name a cell type its model has no class for.** It cannot abstain either —
it emits its nearest class with a plausible-looking score.

Measured on a psoriasis run: tissue resolved to `skin`, so `Adult_Human_Skin.pkl` was loaded —
but all 11 clusters were immune, with zero keratinocyte, fibroblast or endothelial clusters.
Skin contains few B cells, so that model's B-cell coverage is thin, and cluster 10 (a
`TCL1A+ CD79A+ IGHM+` B-cell population) was called `DC1`.

Two mechanisms address this:

1. **Observed-lineage model refinement** (`observed_lineage_profile` →
   `refine_celltypist_model_for_observed_lineage`). The pipeline measures which coarse lineages
   the *data actually contains* and, when ≥90% of clusters are one lineage that has a pan-tissue
   specialist, offers the swap (`{"Immune": "Immune_All_Low.pkl"}` is the only specialist in the
   catalog; the other lineages deliberately have no entry, so no swap is offered where no
   specialist exists).

   **Why detection fraction and not marker overlap or `score_genes`:**
   - *marker overlap fails* — `rank_genes_groups` returns cluster-**discriminating** genes, so in
     an all-T-cell dataset CD3D is nobody's marker. Measured: 4 of 11 clusters produced no
     lineage-panel hit at all.
   - *`score_genes` fails* for the same reason the lineage gate abstains — it subtracts a
     background of similarly-expressed genes, so in a homogeneous dataset the dominant lineage
     scores ≈0 **by construction**. Measured: the gate abstained on 6 of 11 clusters (72.9% of
     cells) in pure immune tissue.
   - *panel detection fraction is absolute*, so it is unaffected by either. Measured: 6/6
     shipped labels correctly identified as Immune. Floor `LINEAGE_PANEL_MIN_DETECTION = 0.02`
     (observed range 0.087–0.140).

2. **Vote suppression.** An immune-only model on a non-immune cluster has its vote dropped, and
   the provenance string says `celltypist vote dropped (immune-only model on non-immune
   cluster)`.

### 12.3 Voter B — SingleR (reference correlation)

**Method.** SingleR correlates each query profile against a labelled reference and assigns the
best-matching label — Spearman correlation on the shared variable genes, with a fine-tuning
step. It runs through **rpy2 in-process**, and the pipeline aggregates to **per-cluster
pseudobulk** (mean log-norm expression) in Python before handing anything to R.

**Confidence** = the maximum Spearman ρ, typically 0.1–0.5 for scRNA-seq against a bulk
reference. That native scale is much lower than CellTypist's — which is exactly why
confidences are rank-normalized before they are ever compared (§12.9).

**Why pseudobulk, and why in-process.** This is what SingleR's own `clusters=` mode computes
internally. Aggregating in Python means embedded R only receives a **genes × n_clusters** matrix
(kilobytes) instead of the full genes × n_cells matrix — which, dumped to a MatrixMarket text
file, reached 2.88 GB and made R hang inside `readMM`, blocking both the `.h5ad` write and
pseudobulk DE. R console output is streamed live into the module logger. **There is no silent
fallback:** any R error propagates as `RuntimeError` and the caller decides whether SingleR was
required.

#### The 5 references (sub-models)

`singler_reference: auto` → the LLM picks by **species + tissue**, never disease; no organ match
falls back to `BlueprintEncodeData`. Anything outside this list is rejected up front rather than
handed to R, so a typo cannot quietly annotate the data.

| celldex reference | Tissue scope | Use it for |
|---|---|---|
| `HumanPrimaryCellAtlasData` | broad / solid tissue, all lineages | mixed solid tissue — the widest human reference (immune, epithelial, stromal, endothelial, neural) |
| `BlueprintEncodeData` | broad: immune + stromal + endothelial | the solid-tissue-safe general default |
| `MonacoImmuneData` | blood / immune | PBMC or immune-only data; fine-grained immune subsets |
| `DatabaseImmuneCellExpressionData` (DICE) | blood / immune | immune-focused blood data |
| `NovershternHematopoieticData` | bone marrow / hematopoietic | HSPC / marrow data |

**Human only, on purpose.** `celldex::MouseRNAseqData` is deliberately *not* listed. SingleR is
the only voter with a mouse reference available, while CellTypist has human models only and the
lineage-gate panels are human gene symbols (`PTPRC`, `EPCAM` — mouse writes `Ptprc`, `Epcam`,
which the gate's exact match does not find, so it would abstain on every cluster). Offering the
mouse reference would make SingleR right and the rest of the annotation stack wrong on the same
run. Proper mouse support needs four things together: mouse CellTypist models, mouse (or
ortholog-mapped) gate panels, a mouse marker table for `lineage_panels.py`, and species passed
to the mygene lookup.

#### The failure mode: saturation

Measured on GSE157827 (Alzheimer prefrontal cortex), SingleR against
`HumanPrimaryCellAtlasData` returned **`Astrocyte` for 18 of 20 clusters** at ρ 0.459–0.566 —
including both oligodendrocyte clusters (`PLP1/MAG/MOBP/OPALIN`) and every neuronal cluster.
Only one of those 18 is really an astrocyte, so it was ~6% accurate. Consensus outvoted it every
time, but its **permanent dissent** set `voters_disagree` on 17 of 20 clusters and dragged 83.5%
of cells to Low/Review with **zero** clusters reaching High — making the tier column useless for
triage.

`degenerate_voters()` therefore reports any voter that produced a usable label for ≥5 clusters
where ≥80% of them are the **same** label, with its modal label and fraction. The threshold is
deliberately high: a blood cohort that genuinely is 75% T cells must not have its T-cell voter
suppressed. This targets a voter with no discriminative power at all, not a skewed but real
biological distribution. Nothing is discarded — the raw calls are still reported.

### 12.4 Voter C — knowledge-based LLM marker reasoning

**Method.** The LLM receives the cluster's **top 50 marker genes plus tissue/organ context** and
returns a cell-type label with a confidence and a written rationale. It reasons the way a
domain expert reads a marker list: `CD3D/CD3E/TRAC` → T cell; `LYZ/CD68/CD14` → myeloid;
`TPSAB1/CPA3/MS4A2` → mast cell.

**Open vocabulary.** This is the property the two classifiers lack: it can name a cell type no
reference contains, and it can hedge. That is also why it must not be trusted unconditionally.

| Property | Value |
|---|---|
| Transport | OpenRouter, `OPENROUTER_MODEL` (no hardcoded default) |
| Temperature | 0 |
| `max_tokens` | 800 — pinned so the JSON cannot silently truncate |
| Retries | `LLM_MAX_RETRIES` (3) on transport error, then raise |
| JSON parse failure | one repair retry, then a documented `Unassigned (LLM parse fail)` label |
| Disease | the word never enters a prompt |

`agent.py` is the **only** module that talks to OpenRouter. `tools.py` holds an explicit
zero-LLM invariant, verifiable with
`grep -Ei "openrouter|chat/completions|requests\.post|openai" tools.py` → no hits. That
separation is what makes the deterministic logic testable without a network.

The same agent layer also powers the `auto` selections: tissue/species inference from the study
metadata, CellTypist model choice, SingleR reference choice, and the DPT root group.

### 12.5 Voter D — PubMed literature RAG

**Method.** Retrieval-augmented, so the annotation is **grounded in citable literature** rather
than in model weights, and every call returns the PMIDs it used.

```
top marker genes (low-information genes dropped)
   → build a disease- and biosample-aware PubMed query
   → esearch → PMIDs → efetch → abstracts        (rate-limited + disk-cached)
   → LLM adjudication over ONLY the retrieved abstracts + markers  (few-shot, strict)
   → {cell_type, lineage, state, supporting/contradicting markers, PMIDs, confidence}
   → confidence score
   → evidence table (.csv) + confidence graph (.png)
```

Design rules:

- **Identity is disease-agnostic**; disease and biosample are *soft retrieval context* only —
  they change which abstracts are fetched, never which label is allowed.
- **Identity is kept separate from state** (activated / exhausted / cycling / …).
- **Low-information genes are removed from queries** — ribosomal, mitochondrial, `MALAT1`,
  haemoglobin — so they never drive identity.
- **The model may abstain** (`Unknown`). It is never forced to guess.
- **Only PMIDs actually retrieved may be cited**, and markers may not be invented. Citations are
  verified against the retrieved set before scoring.
- **Retrieval is cached on disk** (`_pubmed_cache/`) and PMIDs are recorded, so a run is
  reproducible and auditable.

**Confidence** blends the LLM's self-report with objective evidence signals:

```
base            = 0.9 (high) | 0.6 (medium) | 0.3 (low)   ← LLM self-report
+0.05  if ≥3 distinct supporting markers
+0.05  if ≥2 distinct PMIDs
−0.20  if 0 PMIDs                      (uncited → weak)
−0.10 × min(n_contradicting, 2)
cap 0.25   if nothing was retrieved
cap 0.60   if 0 PMIDs — a literature voter citing no literature can never be "high"
cap 0.20   if the label is Unknown / unclear / none
band: high ≥ 0.75 | medium ≥ 0.50 | low otherwise
review_required if the LLM asked for it, or unknown, or band=low, or 0 PMIDs
```

Outputs `pubmed_annotation_table.csv` and `pubmed_annotation_confidence.png`, and the PMIDs land
in the main consensus table's `pubmed_pmids` column — so any label this voter influenced can be
traced to the abstracts behind it.

### 12.6 The knowledge base — 404-node cell hierarchy

A Python-defined, disease-agnostic cell-type ontology. It reads **no files** — the tree is built
from the spec modules by `CellHierarchy.from_spec()`.

**Five levels:** `lineage → class → main_cell_type → subtype → fine_subtype`

| Level | Name | Nodes |
|---|---|---|
| 0 | lineage | 8 |
| 1 | class | 40 |
| 2 | main cell type | 210 |
| 3 | subtype | 117 |
| 4 | fine subtype | 29 |
| | **total** | **404** |

**The 8 lineages:** Haematopoietic cell · Epithelial cell · Stromal / mesenchymal cell ·
Endothelial cell · Muscle cell · Neural cell · Germ-cell and placental lineage ·
Unassigned / not resolvable to a lineage

**Cross-vocabulary crosswalk** — 971 exact aliases, 778 partial-match aliases, 1,090 fuzzy keys
at a 0.88 similarity threshold. This is what lets CellTypist's `CD16+ NK cells`, SingleR's
`NK cells` and Azimuth's `CD8 TEM` be compared at all.

**Also carried:** 400 of 404 nodes carry marker gene lists; 202 carry a **Cell Ontology (CL)
identifier**, so labels are mappable to a public ontology rather than being local strings.

**Spec modules:** `hematopoietic.py`, `epithelial.py`, `mesenchymal.py`,
`neural_germ_other.py`, `aliases.py` (the crosswalk), `states.py` (the separate state axis).

**Design invariants** (enforced, not aspirational):

1. No disease string appears anywhere in the package, in data or in logic.
2. Tissue is metadata only. It can break a tie among valid candidates; it can never gate or
   override a confident match.
3. **Cell state is a separate axis from cell identity.** States (cycling, malignant, exhausted,
   doublet, …) live in `spec/states.py` and never enter the tree. A cycling T cell is a T cell.
4. Everything in `resolver.py` is deterministic pure logic — no LLM, no network. LLM
   verification belongs in the agent layer, consuming `Resolution` objects.
5. **`total_in == total_out`.** Batch calls return one result per input; unresolvable labels are
   flagged, never dropped. Violations raise `LabelConservationError`.

**API:**

```python
h = CellHierarchy.from_spec()

r = h.resolve("CD16+ NK cells", source="celltypist_immune")
r.node_id          # 'cd56_dim_nk_cell'
r.lineage          # 'Haematopoietic cell'
r.main_cell_type   # 'Natural killer cell'
r.confidence       # 0..1

c = h.consensus({"celltypist_immune": "CD16+ NK cells",
                 "singler_blueprint": "NK cells",
                 "azimuth_pbmc":      "CD8 TEM"})
c.consensus_label  # lowest common ancestor across voters
c.agreement_score  # depth-weighted, abstention-penalised
```

The `states.py` axis covers ~22 state ids across the groups `cell_cycle`, `activation`,
`signalling`, `stress`, `malignancy`, `localisation` and `quality` — cycling, quiescent,
activated, naive, memory, exhausted, senescent, anergic, interferon-stimulated, inflammatory,
hypoxic, stressed, apoptotic, malignant, premalignant, EMT, tissue-resident, circulating,
infiltrating, doublet, low-quality.

### 12.7 Harmonization to a controlled vocabulary

Four annotators produce four vocabularies. `CD16+ NK cells`, `NK cells`, `NK cell` and
`CD56-dim NK` are one cell type, and unless they are reconciled first, **a real 3-vote agreement
is counted as three single votes** and loses its majority.

Two routes, in order:

1. **The hierarchy resolver** — accepted only at confidence ≥ **0.95**
   (`MIN_RESOLVE_CONFIDENCE`). A fuzzy match must not define a consensus label.
2. **Keyword fallback** — 23 canonical nodes with whole-word aliases:

| Canonical node | Coarse lineage | Aliases |
|---|---|---:|
| T cell | Immune | 19 |
| NK cell | Immune | 4 |
| B cell | Immune | 6 |
| Microglia | Immune | 2 |
| Monocyte | Immune | 1 |
| Macrophage | Immune | 4 |
| Dendritic cell | Immune | 4 |
| Granulocyte | Immune | 5 |
| Hematopoietic progenitor | Immune | 6 |
| Epithelial cell | Epithelial | 15 |
| Fibroblast | Fibroblast | 7 |
| Endothelial cell | Endothelial | 3 |
| Mural cell | Mural | 4 |
| Erythrocyte | Other | 3 |
| Melanocyte | Other | 1 |
| Excitatory (glutamatergic) neuron | Other | 3 |
| Inhibitory (GABAergic) neuron | Other | 3 |
| Astrocyte | Other | 2 |
| Oligodendrocyte precursor cell | Other | 2 |
| Oligodendrocyte | Other | 1 |
| Schwann cell | Other | 1 |
| Neuron | Other | 2 |
| Glial cell | Other | 2 |

**The biological decisions encoded in that table, and why:**

- **Microglia are checked before macrophage and kept as their own node.** They are
  yolk-sac-derived brain-resident cells, not infiltrating monocyte-derived macrophages, and in a
  neuro cohort *the distinction is the finding*. Measured on GSE157827 cluster 8 (6,785 cells,
  `CSF1R/P2RY12/APBB1IP/TLR2/CD86`): CellTypist, the knowledge voter and PubMed all said
  microglia, yet the cluster shipped as "Macrophage" because every route collapsed it.
- **Monocyte and macrophage are separate nodes**, so the keyword route emits the same strings the
  hierarchy does. They previously shared one `Monocyte/Macrophage` node while the hierarchy
  returned plain `Monocyte` / `Macrophage`, so the two routes produced different labels for one
  cell type. Measured on GSE337706 cluster 7: a real 3-vote agreement was counted as
  2-vs-1-vs-1, lost its majority, and was adjudicated to "Dendritic cell" on a single dissent.
  They remain reconcilable at `Myeloid cell` in the hierarchy.
- **`carcinoma` / `adenocarcinoma` are epithelial by definition; generic `malignant` /
  `tumor cell` are not** — sarcomas, lymphomas and melanomas are malignant but non-epithelial.
  So those generic terms are deliberately absent and an unqualified malignant label falls through
  to `Other` (lineage unknown) rather than inventing an epithelial compartment.
- **`myocyte` is not a mural cell.** Only pericytes and vascular smooth muscle are, so
  cardiomyocyte/skeletal myocyte are deliberately omitted from the Mural row.
- **Neural classes are named exactly as the hierarchy names them.** A single `Neuron/Glia`
  catch-all lumped neurons, astrocytes, oligodendrocytes and Schwann cells while the hierarchy
  returned each separately — the same split-vote failure as Monocyte/Macrophage. Note
  `interneuron` needs its own alias: `\bneuron` does **not** match "interneuron" (no word
  boundary inside the word), which left GSE157827 clusters 6/9/11/13 spread across
  `Other: GABAergic interneuron` and `Other: Interneuron` instead of one inhibitory-neuron
  population, fragmenting the per-cell-type DE.

### 12.8 The lineage gate — an independent biology check

The gate is not a voter. It is a check on the voters, built from marker genes rather than from
any model, and it answers one coarse question per cluster: *is this cluster immune, epithelial,
fibroblast, endothelial or mural?*

**How the panels are built** (`lineage_panels.py`), from two references each used for what it is
authoritative about:

- `TIS_CELL_markers_v3/master_celltype_markers_long.csv` — 7,135 curated cell-type/marker rows
  carrying `specificity_human` and `marker_score` → supplies the **genes**;
- the 404-node hierarchy → supplies the **lineage** each of those cell types belongs to, so the
  mapping is a lookup rather than a guess.

Construction, in order:

1. Resolve every TIS_CELL `cell_type` through the hierarchy; keep only
   `confidence ≥ MIN_MAPPING_CONFIDENCE = 0.95`.
2. Map the hierarchy lineage onto the gate's five coarse lineages. **Pericytes and vascular
   smooth muscle sit under *Stromal / mesenchymal* in the hierarchy but are Mural to the gate**,
   so they are split out by `main_cell_type`. The hierarchy's *Muscle cell* lineage
   (cardiomyocyte, skeletal myocyte) maps to **nothing** — a myocyte is not a mural cell.
3. **Drop any gene claimed by more than one gate lineage.** A lineage panel is only meaningful
   if its genes discriminate *between* lineages.
4. Keep genes with `specificity_human < 0.05` (that field is the fraction of *other* cell types
   also expressing the gene, so lower is more specific), rank by `marker_score`, take the top
   `PANEL_SIZE = 40`.
5. **Union with the original hand-written panel**, so nothing the gate could previously detect is
   ever lost.

Resulting panels (recorded in the manifest as `lineage_panel_sizes`):

| Lineage | Genes | Hand-written floor |
|---|--:|---|
| Immune | 51 | PTPRC, CD3D, CD3E, CD8A, CD4, MS4A1, CD79A, NKG7, GNLY, LYZ, CD68, CD14, FCGR3A, ITGAM |
| Epithelial | 46 | EPCAM, KRT8, KRT18, KRT19, CDH1, KRT7 |
| Fibroblast | 45 | COL1A1, COL1A2, DCN, LUM, PDGFRA, PDGFRB |
| Endothelial | 42 | PECAM1, VWF, CLDN5, CDH5, FLT1 |
| Mural | 44 | RGS5, ACTA2, MYH11, TAGLN |

81 cell types map to a gate lineage. If the reference data is missing or unreadable the builder
returns the hand-written panels unchanged and **says so in the provenance** — the gate degrades
to its previous behaviour rather than failing.

**Why derive them at all.** The original five hand-written panels of 5–14 genes had **no
mast-cell and no dendritic-cell coverage whatsoever**, so those clusters scored ≈0 on every panel
and a bare `idxmax` handed them whichever panel was *least negative* — in practice "Epithelial",
producing a phantom epithelial compartment with 0% EPCAM. Widening panels by hand does not scale
and is not auditable.

**How the gate scores** (`lineage_gate_per_cluster`):

1. `sc.tl.score_genes` per lineage panel on the log-normalized working copy;
2. per-cell `idxmax` → a provisional lineage;
3. **PTPRC (CD45) gate**: an Immune call requires PTPRC > 0 — *but* PTPRC is dropout-prone
   (it reads zero in roughly half of real monocytes/macrophages and most mast cells), so a cell
   is only ejected from Immune when an **alternative lineage has positive support** above the
   floor. Ejecting on the dropout alone pushed genuine immune cells onto whichever non-immune
   panel was least negative;
4. per-cluster **majority** of the per-cell calls;
5. **abstention.** `MIN_LINEAGE_SCORE = 0.1`, applied to the **cluster mean**.

**Why abstention, and why 0.1.** `score_genes` subtracts a background of similarly-expressed
genes, so 0.0 means "indistinguishable from random genes" — but sampling noise alone can push a
panel slightly positive, so a bare `> 0` test is not enough. Measured on synthetic controls:
clusters with **no** matching panel (mast cell, dendritic cell, pure background) peak at
**+0.040**, while genuine lineage calls start at **+0.85** — a >20× gap. 0.1 sits an order of
magnitude clear of both sides. It is applied to the cluster mean because per-cell scores are
noisy and the gate's output is a per-cluster call, so the cluster mean is the stable statistic.
When the gate abstains it returns `Other`, which is already understood downstream as "no
opinion", and the voters decide.

**The gate never overturns a unanimous vote.** A gate built from a handful of pan-lineage markers
must not overrule every independent annotator at once — when the voters are unanimous the gate is
the more likely thing to be wrong (it has no panel for e.g. mast cells), so the vote stands and
the disagreement is logged. Split votes still route to the adjudicator.

### 12.9 Vote counting and tie-breaking

`tally_votes()` counts harmonized labels among the voters that produced a **usable**
(non-`Unassigned`) label, and returns `majority_label`, `majority_count`, `n_methods`,
`unanimous`, `has_majority`, `tied`, `top_labels`, `pattern`.

| Rule | Definition |
|---|---|
| **unanimous** | one distinct label **and** `n ≥ 2` |
| **has_majority** | `n ≥ 2` **and** `top_count × 2 > n` **and** not unanimous |
| **tied** | more than one label shares the top count |

**A single voter is never a majority.** There is nothing to corroborate it, so a one-voter
cluster falls through to the review tier rather than being promoted to a Medium-confidence
consensus.

**Confidence normalization is mandatory before any comparison.** The voters report on
incompatible native scales:

| Voter | Native confidence | Typical range |
|---|---|---|
| CellTypist | fraction of cells backing the label | 0.3 – 1.0 |
| SingleR | max Spearman ρ vs a bulk reference | 0.1 – 0.5 |
| Knowledge LLM | self-reported 0–1 | clusters near ~0.9 |
| PubMed | evidence score (formula in §12.5) | 0 – 1 |

Summing those raw numbers to break a tie silently under-weights the voter whose native scale is
smallest — in practice SingleR — even when it is right. So each voter's per-cluster confidences
are **rank-normalized into [0, 1] within that voter** first.

**Tie-break order:** highest summed supporter confidence → alphabetical. Fully deterministic.
This replaces `Counter.most_common`, which broke ties by dict-insertion order and so silently
favoured whichever voter happened to be added first (CellTypist).

### 12.10 Adjudication and out-of-domain deference

**Adjudication.** A split vote, or a majority contradicted by the gate, goes to the LLM
adjudicator, which receives the candidate labels with their normalized confidences, the cluster's
marker genes, the gate's lineage and the tissue — and re-reasons from the markers. Its written
reasoning is stored in `adjudicator_reasoning_full`. On transport failure it falls back to the
majority (or the gate) with the failure recorded, never silently.

Without an adjudicator (LLM disabled), the fallbacks are explicit:

- gate contradicts the vote and gate ≠ `Other` → **trust the gate**. This is what corrects
  non-immune clusters (e.g. an `EPCAM+` cluster voted "T cell") when the LLM voter is off;
- otherwise prefer a usable majority, else the gate;
- a tie is broken by preferring the tied label whose coarse lineage matches the gate — biology
  over an arbitrary pick.

**Out-of-domain deference** (`out_of_domain_deference`) — the one place where one confident voter
may overturn a majority. All four conditions must hold:

1. there **is** a majority or unanimous call (a split already routes to the adjudicator);
2. **every** voter supporting it has a **closed** vocabulary
   (`CLOSED_VOCABULARY_VOTERS = {celltypist, singler}`). If any open-vocabulary voter backs the
   majority, the markers already agree and nothing is forced;
3. CellTypist's own call is **unreliable** — its per-cell labels are scattered below
   `mixed_cluster_min_dominant_fraction`;
4. an open-vocabulary voter (`{knowledge_based, pubmed}`) is confident at
   ≥ `OPEN_VOCAB_MIN_CONFIDENCE = 0.80` and names something else.

**The measured case.** GSE157827 cluster 15 (`PLP1 +15.1 / CRYAB +15.0 / APLP1 +14.2 /
MAG +12.2`) is an unambiguous myelinating-glia programme with zero T-cell markers. A blood
model has no Schwann-cell class, so CellTypist returned `T CD4 Naive` spread over 17 labels with
only 0.536 of cells on top, and SingleR returned `CD4+ T cells` at 0.553. The knowledge voter
read the markers and said **Schwann cell at 0.82**. Plain vote counting made it 2-to-1 and the
cluster shipped as "T cell", tier Medium. The *brain* model calls the same marker set
Oligodendrocyte at 0.997 — which proves the failure is reference scope, not biology.

The 0.80 threshold sits above the LLM's hedging band and below its assertive calls (Schwann cell
0.82, Microglia 0.97). **The result is always tiered Low/Review, never promoted** — preferring
the marker-driven label is the better guess, not a validated identity.

### 12.11 Confidence tiers

| Tier | Awarded when |
|---|---|
| **High** | unanimous **and** lineage-consistent |
| **Medium** | strict majority **and** lineage-consistent |
| **Low/Review** | split vote, gate contradiction, adjudicated, out-of-domain deference, a single voter, or a state-dominated cluster |

The tier is a **triage instruction**, not decoration. `Low/Review` means *this label is a
hypothesis*; `downstream.exclude_low_confidence_de: true` keeps those cells out of DE and
composition tests (opt-in, because turning it on legitimately changes results).

Every cluster also carries a `provenance` string that reconstructs the decision:

```
votes[celltypist=Macrophage/knowledge_based=Macrophage/pubmed=Macrophage/singler=T cell]
 | gate=Other | pattern=majority
 | majority, lineage-consistent | tier capped: 67% of top 15 markers are
   stress_heat_shock state genes, not identity markers
```

### 12.12 Cell state versus cell identity

**The problem.** A cluster can be defined by a *state* rather than by an identity — dissociation
heat shock, cell cycle, an interferon response, ambient haemoglobin. Such a cluster still gets
marker genes, and every voter still names a cell type from them, so it ships with the same
confidence as a cluster carrying real lineage evidence. Identity and state are different axes: a
cycling T cell is a T cell, and "cycling" is not an identity.

**Measured on the psoriasis run** (11 clusters, 97,108 cells):

- cluster 1, 16,933 cells — top markers `HSPA1B/HSPA1A/HSPA6/DNAJB1/HSPH1/HSP90AA1` plus
  `NR4A1/NR4A2/JUND`: **10 of the top 15 are stress / immediate-early**. It shipped as "T cell",
  tier **High**, subtype "CD4-positive T cell" — with no CD4, IL7R or CD40LG anywhere in its
  markers.
- cluster 10, 1,271 cells — 10 of the top 15 are cell cycle
  (`MKI67/BIRC5/CCNB2/CCNA2/AURKB/UBE2C/RRM2/CDCA5/GTSE1/DLGAP5`).

The nine identity-driven clusters score 0.00–0.13 on the same statistic, so the two populations
separate by ~5×.

**The five state panels:**

| Programme | Genes | Biology |
|---|--:|---|
| `cell_cycle` | 51 | proliferation: MKI67, TOP2A, BIRC5, CCNB1/2, CCNA2, CDK1, AURKA/B, UBE2C, TYMS, RRM2, PCNA, … |
| `stress_heat_shock` | 25 | dissociation stress: HSPA1A/1B, HSPA6, DNAJB1, HSPH1, HSP90AA1, … |
| `immediate_early` | 24 | handling/activation artefact: FOS, JUN, JUNB, JUND, EGR1, NR4A1/2/3, … |
| `interferon` | 28 | IFN response: ISG15, IFI6, IFIT1/3, MX1, OAS1, STAT1, … |
| `hemoglobin_ambient` | 8 | ambient RNA from lysed erythrocytes: HBB, HBA1/2, … |

Plus a technical regex for ribosomal/mitochondrial genes
(`^(MT-|MTRNR|RPL\d|RPS\d|MRPL\d|MRPS\d)`) — not a biological programme, but a cluster defined by
them is just as unusable for identity.

**Scoring** (`state_programme_profile`) — pure list logic, directly unit-testable:

- only the leading `STATE_PROFILE_TOP_N = 15` markers are scored, so the statistic does not
  drift when `top_n_markers` changes. State genes concentrate at the top of a state-driven
  cluster.
- `state_fraction` is the **union** of the panels, not the sum, so a gene in two panels is not
  double-counted.
- `state_dominated` when `state_fraction ≥ STATE_DOMINANCE_THRESHOLD = 0.40`. **Measured, not
  assumed:** identity clusters 0.00–0.13, the two state clusters both 0.67. 0.40 sits an order
  of magnitude clear of both sides.
- an **empty** marker list is *not* reported as state-dominated — absent evidence is handled by
  the separate `markers_empty` flag, and conflating the two would hide it.

**Effect:** a state-dominated cluster is capped at **Low/Review** and gets **no subtype at all**.
The label is kept; only the confidence claim is withdrawn, with the reason spelled out in the
provenance.

### 12.13 The subtype layer and its marker-evidence gate

The consensus produces a coarse identity. The subtype layer adds the **finest label offered by
any single annotator whose coarse identity equals the consensus**, so a subtype can never
contradict the coarse call.

Priority: PubMed literature subtype → knowledge-LLM label → CellTypist fine label → PubMed cell
type; else the coarse consensus. The producing annotator and its native confidence are recorded
(`celltype_subtype_source`, `celltype_subtype_confidence`) so a subtype is never mistaken for a
consensus-validated identity. `annotation.use_subtypes_for_downstream` defaults to **false** for
exactly that reason.

**The problem this gate fixes.** Coarse-identity agreement does not check that the cluster
actually shows the marker the subtype name asserts. Measured on the psoriasis run, where all 11
subtypes came from one voter:

- cluster 4 — subtype "CD8-positive T cell" with **no CD8A and no CD8B** in its markers
  (`CXCL13/CXCR6/ADGRG1`; CXCL13 is classically a CD4 Tph programme). CellTypist said `Th`
  (CD4) and was overridden.
- cluster 9 — "CCR4-positive T cell (likely regulatory or Th2 …)" with no `FOXP3/IL2RA/CTLA4`
  and no `GATA3/IL4/IL13`.
- cluster 10 — "cycling B cell / plasmablast" with no `MZB1/JCHAIN/XBP1`; it is a
  `TCL1A+ IGHM+` naive B population that happens to be cycling.

**Corroboration by a second voter was considered and rejected as the gate:** the two
open-vocabulary voters share one model and agree 89–100% of the time (measured), so their
agreement is not independent evidence. **Marker presence in the cluster is.** Each of the
**32 rules** therefore asks the data, not another voter.

Rule form: `(claim tokens, requires_any, contradicted_by, note)`.

- `requires_any` where the transcript is reliably detected in 10x;
- `contradicted_by` where it is **not**: CD4 mRNA drops out in a large fraction of real CD4 T
  cells, so demanding it would reject correct calls. A CD4 claim therefore fails only when
  positive CD8 evidence is present.

| Claim | Requires any of | Contradicted by |
|---|---|---|
| cd8 / cytotoxic t | CD8A, CD8B | — |
| cd4 / helper t / th1/th2/th17 | *(nothing — dropout)* | CD8A, CD8B |
| treg / regulatory t | FOXP3, IL2RA, IKZF2, CTLA4, TIGIT | — |
| th17 | RORC, IL17A, IL17F, CCR6, IL23R | — |
| th2 | GATA3, IL4, IL5, IL13, IL1RL1 | — |
| th1 | TBX21, IFNG, CXCR3 | — |
| naive | TCF7, LEF1, SELL, CCR7, MAL | — |
| central memory / tcm | TCF7, LEF1, SELL, CCR7, IL7R | — |
| effector memory / tem / temra | GZMK, GZMH, GZMA, NKG7, KLRG1, CCL5 | — |
| cytotoxic / effector | GZMA, GZMB, GZMH, GZMK, PRF1, NKG7, GNLY | — |
| exhausted | PDCD1, HAVCR2, LAG3, TOX, TIGIT, CTLA4 | — |
| mait | SLC4A10, KLRB1, TRAV1-2, NCR3 | — |
| gamma delta | TRDC, TRGC1, TRGC2, TRDV1, TRDV2 | — |
| cd16 | FCGR3A | — |
| cd56 | NCAM1 | — |
| plasma cell / plasmablast | MZB1, JCHAIN, XBP1, PRDM1, SDC1, DERL3, TNFRSF17 | — |
| memory b | CD27, TNFRSF13B, AIM2 | — |
| mregDC / LAMP3 | LAMP3, CCL22, CCL19, IDO1, FSCN1, EBI3 | — |
| pDC / plasmacytoid | LILRA4, CLEC4C, IL3RA, GZMB, IRF7 | — |
| cDC1 | CLEC9A, XCR1, CADM1, BATF3 | — |
| cDC2 | CD1C, FCER1A, CLEC10A | — |
| cycling / proliferating | the 51-gene cell-cycle panel | — |
| interferon / ISG | the 28-gene interferon panel | — |
| AT1 | AGER, PDPN, CAV1, CLIC5 | — |
| AT2 | SFTPC, SFTPB, SFTPA1, NAPSA, LAMP3 | — |
| club / secretory | SCGB1A1, SCGB3A2, SCGB3A1, MGP | — |
| ciliated | FOXJ1, TPPP3, PIFO, CAPS, TUBA1A | — |
| goblet | MUC5AC, MUC5B, TFF3, SPDEF | — |
| basal | KRT5, KRT14, TP63, KRT15 | — |
| myofibroblast / CAF | ACTA2, TAGLN, POSTN, FAP, MYL9 | — |
| pericyte | RGS5, PDGFRB, NOTCH3, KCNJ8 | — |
| lymphatic | PROX1, LYVE1, PDPN, CCL21, FLT4 | — |

Semantics:

- **All** recognised claims in the string must hold: "cycling B cell / plasmablast" asserts both,
  so absent plasma-cell evidence rejects it even though the cycling half is supported. A hedged
  subtype naming several alternatives is rejected if **any** named alternative is unsupported —
  a subtype is an assertion, not a shortlist.
- A subtype making no recognised claim passes. This function must never reject a label merely
  for being unfamiliar.
- `markers=None` means *not checked* and passes everything (for callers with no marker context);
  `markers=[]` means the cluster **has** no markers, which is evidence of absence, so a
  claim-bearing subtype is rejected. The production path passes
  `markers_by_cluster.get(cl, [])` — it fails closed.

Rejections are logged and recorded in `celltype_subtype_rejected`, so a withdrawn subtype leaves
a trace instead of vanishing.

### 12.14 Broadcast and conservation

Cluster-level decisions are broadcast to cells, then validated: **every cell in must be a cell
out, and every cell must carry a consensus label.** A violation raises
`tools.CellConservationError`.

This was previously an `assert`, which is stripped under `python -O` — exactly the deployment
where it matters most. A violation means a downstream table would silently describe a different
cell population than the one analysed, so it raises rather than warns.

Columns written to `.obs`: `celltype`, `consensus_tier`, `celltype_subtype`,
`celltype_subtype_source`, `celltype_subtype_confidence`, per-voter labels and confidences,
`lineage_gate`, `mixed_cluster_flag`, `state_dominated`, `state_programme`,
`state_marker_fraction`, `voters_disagree`, `include_in_downstream_analysis`, and more — 52
columns in the exported evidence table.

---

## 13. Stage 9 — identity markers per cell type

`sc.tl.rank_genes_groups` grouped by the consensus cell-type label: the genes that distinguish
each cell type from all the rest. One-vs-rest and naturally up-regulation biased — these describe
**what a cluster is**, not condition DE.

**No p-values are exported.** The groups being compared were **defined by the same expression
matrix the test then reads**, so the null hypothesis "this gene does not differ between these
groups" was already falsified by the act of forming the groups. The reported p-values and FDRs
are anti-conservative by construction — classic double dipping / selection bias.

The **ranking is still perfectly useful**; only the significance claim is unsupportable. So
`marker_stats.py` strips the p-value columns on the way out and keeps `rank` (explicit, 1-based),
`scores` and `logfoldchanges`. Both marker writers — cell-type markers and Leiden cluster markers
— make the same claim, and a `_README_marker_statistics.txt` ships beside every marker table
saying so. `marker_stats.py` is deliberately dependency-free (stdlib + pandas) because
`rank_genes_subprocess.py` runs as a standalone script where relative imports do not resolve.

---

## 14. Stage 10 — differential expression

### 14.1 Contrast design — the single place that decides direction

Every group comparison used to use `sorted(groups)` + `combinations`, making the
**alphabetically first** group the reference. That is a coin flip against biology:

```
["Control", "Tumor"]    → ref = Control    correct by luck
["Disease", "Healthy"]  → ref = Disease    INVERTED  (log2FC > 0 means up in Healthy)
["AD", "Control"]       → ref = AD         INVERTED
["Post", "Pre"]         → ref = Post       INVERTED
```

The p-values and gene lists were right; the **sign** was arbitrary. A silent sign flip is the
worst class of bug in a DE pipeline, because nothing downstream looks wrong.

`contrasts.py` resolves **one** reference arm per run — explicitly configured
(`de.reference_group`), else matched from a baseline vocabulary (normal / healthy / control /
baseline / untreated / pre-treatment / …) — and orients every contrast as
`focus_vs_reference`. So **a positive log2FoldChange always means "higher in the non-reference
(case) arm."**

The literal contrast is **stamped onto every output row** (`stamp_contrast`), so a result's
direction is auditable from the CSV alone and never has to be re-derived from a filename or
column order. When no baseline can be identified the orientation falls back to alphabetical —
but `reference_selection` records that the direction is arbitrary instead of leaving it implicit,
and the fix is to name the arm in the config.

### 14.2 Pseudobulk DESeq2 — the primary, statistically correct test

`do_pseudobulk_de: true`. This is the test that supports a cohort claim.

**Aggregation:** a **SUM of raw integer counts** per `(sample × cell type)`. Never a mean of
normalized values — DESeq2's negative-binomial model and its size-factor normalization are
defined on counts, and averaging CP10k values destroys both the count scale and the mean-variance
relationship the dispersion estimate depends on.

**Why donor-level.** Cell-level tests treat thousands of cells from one donor as independent
observations. They are not: cells from one donor share that donor's genetics, batch, treatment
and handling. Pseudoreplication inflates significance enormously — n becomes the cell count
instead of the donor count. The donor is the unit of replication because the donor is what was
randomised.

Three properties this module guarantees:

| Property | Implementation |
|---|---|
| **Direction is biological, not alphabetical** | reference from `contrasts`; `focus_vs_reference` plus a plain-language direction note stamped on every row |
| **Effect size is TESTED, not filtered** | `lfc_threshold` is passed to DESeq2 as a formal null — `H0: |log2FC| ≤ threshold`, Wald test — instead of testing against zero and then filtering `|log2FC| > 1` post hoc, which inflates the false-positive rate because the filter is not part of the test. Set `0` for the classic against-zero test. |
| **Fold changes are shrunk** | apeGLM (`lfc_shrink`) pulls the estimate toward zero where the evidence is thin, so low-count genes stop producing enormous fold changes at n = 2–3 donors per arm. p-values are unaffected by shrinkage; both the shrunken and the raw MLE estimate are reported. |

Defaults: `de.lfc_threshold: 1.0`, `de.alpha: 0.05`.

**Replication floor:** a contrast needs **≥ 2 samples per group**. Contrasts with fewer are
skipped and reported, never silently faked. If pyDESeq2 is not installed the whole step is
skipped with a clear log line (non-fatal).

Outputs: `06_groupwise_deg/pseudobulk_deg/pseudobulk_overall_DE.csv`,
`per_celltype/<CellType>_pseudobulk_DE.csv`, `pseudobulk_contrast_design.csv`, and
`volcano/volcano_<CellType>_<focus>_vs_<reference>.png`.

### 14.3 Cell-level DE — exploratory only

Wilcoxon rank-sum per cell type, deliberately retained as a descriptive view. It pseudoreplicates
donors, so it **cannot** support a cohort claim, and the pipeline makes that impossible to
overlook:

- the filename carries it: `<CellType>_<focus>_vs_<reference>_CELLLEVEL_EXPLORATORY.csv`
- a `_README_cell_level_DE.txt` ships in the folder
- a `validity` stamp goes into the rows
- the run is deterministic and stamped

### 14.4 Which cells may carry an inferential claim

`downstream_gating.py`. A `Low/Review` cluster is one where the voters disagreed or the gate
contradicted them — its label is a hypothesis. Feeding those cells into condition-level DE
presents that hypothesis as a result.

The module **marks, never deletes**. It writes boolean `obs['include_in_downstream_analysis']`
and hands the caller a view-derived copy for inferential steps only. The full object — every
cell, every label, every tier — is what gets written to the `.h5ad`, the UMAPs and the audit
tables, so nothing becomes unauditable.

Default is **off** (`exclude_low_confidence_de: false`): every cell is included, matching
behaviour before this module existed. Turning it on legitimately **changes** DE and composition
results, which is why it is an explicit opt-in. If excluding cells would drop a contrast below
2 samples per group, filtering is skipped with a logged reason.

---

## 15. Stage 11 — composition analysis

Two views ship, because they answer different questions.

**Pooled per-group proportions** (`celltype_proportions_by_group.csv`) — one fraction per group.
Descriptive. Pooling treats every cell as an independent observation, so a cohort of 21 donors is
reported as if it were 158,084 samples, and ordinary between-donor variation reads as a group
difference.

**Per-donor proportions** (`celltype_proportions_per_donor.csv` + `_stats.csv`) — the donor is
the unit of replication, with a Mann-Whitney test between arms. **This is what a reviewer will
ask for.**

**Measured on GSE157827:** pooled fractions put excitatory neurons at 30.04% control vs 27.19%
AD, which looks like the textbook finding. Re-tested per donor that is 28.47% vs 23.91% with
**p = 0.337** — control donors alone span 13%–47%, so the shift is inside the noise. The pooled
number was not wrong, it was *untestable*.

Two annotation-audit figures ship alongside:

- **annotation marker dotplot** — canonical markers against final cell types. The fastest way to
  see a wrong label: a T-cell row with an empty `CD3D/CD3E/TRAC` column is a mislabel you can
  spot in one glance. On GSE337706, cluster 0 (24,272 cells, 27.8% of the run) shipped as
  "T cell" with zero T-cell markers — this plot would have shown it immediately.
- **voter agreement** — which annotator disagreed, on which cluster, and whether the final call
  followed the majority. Makes a saturated voter obvious as a solid band.

Every function here is defensive: missing columns, absent groups or too few donors produce a
logged skip and `None`, never an exception. A figure is never worth failing a run over.

---

## 16. Stage 12 — pathway enrichment

Optional (`do_pathway_clustering`). Over-representation analysis via gseapy/Enrichr on ranked
gene lists (cluster markers or DE hits), across six libraries:

`GO_Biological_Process_2021` · `GO_Molecular_Function_2021` · `GO_Cellular_Component_2021` ·
`KEGG_2021_Human` · `Reactome_2022` · `WikiPathways_2019_Human`

Enrichr returns many near-identical terms ("T cell activation", "regulation of T cell
activation", "positive regulation of T cell activation"), so results are **semantically
de-duplicated** and the report shows distinct biology rather than the same finding six times. The
raw pre-dedup CSVs and the dedup log can be kept or cleaned up (`cleanup_raw_pathway_csvs`,
`cleanup_dedup_logs`).

Network access is required for the Enrichr call. The step degrades gracefully when gseapy is
unavailable or the request fails.

Output: `07_pathway_enrichment/`.

---

## 17. Stage 13 — report, manifest, deconvolution export

### 17.1 HTML/PDF report

`generate_report: true`. Collects every artifact the run produced — QC tables, embeddings,
cluster and cell-type figures, marker and DE tables, pathway results — and renders one branded,
self-contained report through a Jinja2 template, **embedding images as base64** so the HTML is
portable as a single file. PDF export goes through Playwright/Chromium.

An optional LLM pass writes short narrative summaries per section. All calls are deterministic
(temperature 0) and degrade to templated text when no API key is configured. Model resolution
order:

```
REPORT_LLM_MODEL → OPENROUTER_MODEL → LLM_MODEL_FAST → OPENAI_MODEL → llm_settings.LLM_MODEL_CHAT
```

and the resolved model is logged as `[REPORT] LLM sections via OpenRouter, model=…` so the
report can never quietly be written by a different model than the annotation.

Output: `singlecell_report/index.html` and a dated PDF.

### 17.2 Provenance manifest

`provenance/manifest.json` — written every run:

- `seed`, `timestamp_utc`, `git_commit`
- `dataset` (n_obs, n_vars), `initial_cells`, `initial_genes`
- **every resolved parameter**, including the ones that were `auto`: resolved CellTypist model,
  resolved SingleR reference, resolved tissue/species, resolved reference group
- QC per-rule removal counts, HVG rung and excluded batches, cluster count
- the integration design verdict and the DE-boundary check result
- lineage panel source, sizes and any fallback reason
- `tier_counts`, `n_mixed_clusters`, `clusters_with_empty_markers`
- `package_versions` for python, scanpy, anndata, numpy, pandas, scipy, scikit-learn, leidenalg,
  igraph, bbknn, celltypist, umap-learn, pydeseq2, matplotlib, seaborn

### 17.3 Bisque-ready export

`prepare_for_bisque: true` → `bisque_ready_*.h5ad`.

Bisque bulk-deconvolution needs a single-cell reference that is small and clean: raw counts in
`.X`, a `celltype` label and a `sample` id per cell, and nothing else. The processed object
carries `.raw`, a counts layer, PCA/UMAP embeddings, neighbour graphs and large `.uns` blobs —
all of which bloat the file and confuse deconvolution. `prepare_for_bisque()` returns a **copy**
stripped to exactly what Bisque needs, preferring the untouched `counts` layer for `.X`.

---

## 18. Complete output inventory

From a real multi-mode run (`outputs/demo_multi/`, 4 samples, 2 groups, 9,820 cells):

```
00_analysis_summary/
    *_analysis_summary.txt                     cell/gene counts, QC accounting, cluster and tier summary
00_data_validation/
    *_data_validation.csv / .txt               every check, its severity and what was auto-fixed
01_qc_and_filtering/
    *_qc_metric_histograms.png
    violin_*_qc_violin.png
    scatter_*_qc_total_vs_genes.png
    scatter_*_qc_total_vs_mito.png
02_highly_variable_genes/
    *_highly_variable_genes_table.csv          the full .var table (index = gene, so index IS data)
    filter_genes_dispersion_*.png
03_dimensionality_reduction_and_embeddings/
    pca_variance_ratio_*_PCA_variance_explained.png
    umap_*_UMAP_samples_groups.png
    umap_*_UMAP_total_counts.png  /  _n_genes_by_counts.png  /  _pct_counts_mt.png
    umap_*_UMAP_dpt_pseudotime.png             (if DPT ran)
    diffmap_*_DIFFMAP_dpt_pseudotime.png
    groupwise_embeddings/umap_group_<GROUP>_celltype.png
04_clustering_and_cell_states/
    umap_*_UMAP_leiden.png
    *_cluster_cell_counts_leiden.csv
    leiden_resolution_evaluation.csv           (only if evaluate_resolutions: true)
    Intercluster_analysis_deg/
        intercluster_cluster_markers.csv
        dotplot_ / heatmap_ / rank_genes_groups_leiden_*.png
        _README_marker_statistics.txt          why no p-values
05_celltype_analysis/
    celltype_annotation/
        *_consensus_annotation.csv             ← 52-column evidence table, the audit trail
        *_consensus_run.log                    every consensus decision, in order
        *_method_agreement.csv / .png
        *_method_celltypist_calls.png  _singler_  _knowledge_based_  _pubmed_
        *_voter_agreement.png
        pubmed_annotation_table.csv            per-cluster PMIDs + cited markers
        pubmed_annotation_confidence.png
        _pubmed_cache/esearch_*.json, efetch_*.json    cached retrieval → reproducible
        umap_*_UMAP_celltypes.png
        umap_*_UMAP_leiden_vs_celltype.png
        *_annotation_marker_dotplot.png        the mislabel-spotting figure
        *_celltype_composition_barplot.png / _by_sample.png
        *_celltype_by_group_counts.csv / _by_sample_counts.csv / _by_sample_proportions.csv
        celltype_proportions_by_group.csv / .png            pooled  (descriptive)
        celltype_proportions_per_donor.csv / .png           per donor (testable)
        celltype_proportions_per_donor_stats.csv            Mann-Whitney
    celltype_specific_markers/
        celltype_marker_genes_celltype_ALL.csv
        dotplot_ / heatmap_ / rank_genes_groups_celltype_*.png
        _README_marker_statistics.txt
06_groupwise_deg/
    pseudobulk_deg/                            ← the statistically primary results
        pseudobulk_overall_DE.csv
        pseudobulk_contrast_design.csv         which arm is the reference, and why
        per_celltype/<CellType>_pseudobulk_DE.csv
        volcano/volcano_<CellType>_<focus>_vs_<reference>.png
    celltype_specific_deg/                     ← exploratory only
        <CellType>/<CellType>_<focus>_vs_<ref>_CELLLEVEL_EXPLORATORY.csv
        _README_cell_level_DE.txt
07_pathway_enrichment/                         (if do_pathway_clustering: true)
08_reference_summary/
    celltype_DEG_marker_genelevel_summary.csv
    celltype_DEG_marker_pathway_overview.csv
    celltype_DEG_marker_pathway_summary.txt
    *_celltype_markers_celltype_ALL.csv
    *_cluster_to_celltype_map.csv
provenance/manifest.json
singlecell_report/index.html  +  <title> <analysis> <date>.pdf
*_processed_scanpy_output.h5ad
bisque_ready_*_processed_scanpy_output.h5ad    (if prepare_for_bisque: true)
pipeline.log
```

### The consensus evidence table — all 52 columns

| Group | Columns |
|---|---|
| identity | `cluster`, `leiden`, `n_cells`, `consensus`, `final_celltype`, `celltype_subtype` |
| confidence | `tier`, `consensus_tier`, `n_voters`, `voters_disagree`, `voters_withheld`, `harmonized_agreement` |
| per-voter calls | `celltypist`, `singler`, `knowledge_based`, `pubmed` (+ `_label` duplicates) |
| per-voter confidence | `celltypist_conf`, `singler_conf`, `knowledge_based_conf`, `pubmed_conf` (+ `_confidence` duplicates) |
| CellTypist diagnostics | `mixed_cluster_flag`, `celltypist_dominant_fraction`, `celltypist_second_label`, `celltypist_second_fraction`, `celltypist_label_entropy`, `celltypist_unique_label_count` |
| PubMed evidence | `pubmed_supporting_markers`, `pubmed_pmids`, `pubmed_cell_state` |
| lineage | `lineage_gate`, `lineage_coarse`, `observed_lineage` |
| state | `state_dominated`, `state_programme`, `state_marker_fraction` |
| subtype audit | `celltype_subtype_source`, `celltype_subtype_confidence`, `celltype_subtype_rejected` |
| evidence + audit | `top_markers`, `markers_empty`, `provenance`, `decision_reason`, `adjudicator_reasoning_full`, `included_in_downstream_analysis` |

---

## 19. Complete configuration reference

Every key in `config.yaml`, its default, and what it does.

### Top level

| Key | Default | Effect |
|---|---|---|
| `mode` | — | `single` or `multi`. Required. |

### `common:`

| Key | Default | Effect |
|---|---|---|
| `do_pathway_clustering` | false | run pathway enrichment |
| `do_dpt` | true | compute diffusion pseudotime |
| `dpt_root_group` | auto | baseline group to root DPT at; `auto` = LLM/heuristic; no baseline → DPT skipped |
| `geo_json_path` | null | null = auto-detect `GSE*_metadata.json` in the input dir |
| `logos_dir` | null | report branding |
| `generate_report` | true | build the HTML/PDF report |
| `prepare_for_bisque` | true | write the deconvolution-ready h5ad |
| `enable_celltypist` | true | CellTypist ML voter |
| `enable_knowledge_based` | true | LLM marker-reasoning voter — needs `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`; also powers all `auto` selections |
| `enable_singler` | true | SingleR voter — needs R + SingleR + celldex |
| `enable_pubmed` | true | PubMed literature voter — needs internet + OpenRouter; ~1 extra LLM call per cluster |
| `tissue` | auto | organ context (cell biology, **not** disease); `auto` = infer from metadata JSON |
| `species` | auto | `auto` = from the metadata `taxon` |
| `celltypist_model` | auto | `auto` = LLM picks from the 41-model catalog by tissue; fallback `Immune_All_Low.pkl` |
| `singler_reference` | auto | `auto` = LLM picks by species+tissue; fallback `BlueprintEncodeData` |
| `skip_tsne` | true | skip t-SNE (UMAP is usually enough) |
| `skip_pca_cluster_plots` | true | skip PCA-based cluster/celltype overlays |
| `skip_per_celltype_plots` | true | skip per-celltype dot/rank plots |
| `skip_per_celltype_csvs` | true | skip per-celltype marker CSVs |
| `skip_per_cluster_marker_csvs` | true | skip per-cluster marker CSVs |
| `cleanup_raw_pathway_csvs` | true | delete `*_combined_pathways_RAW.csv` |
| `cleanup_dedup_logs` | true | delete `*_pathway_dedup_log.txt` |
| `cleanup_pipeline_log` | false | keep `pipeline.log` (retained execution record) |
| `cleanup_per_cluster_marker_csvs` | true | delete per-cluster marker CSVs |

### `qc:`

| Key | Default | Effect |
|---|---|---|
| `min_genes` | 200 | keep cells with `n_genes_by_counts >` this |
| `max_genes` | 6000 | keep cells with `n_genes_by_counts <` this |
| `max_mito_percent` | 15.0 | keep cells with `pct_counts_mt <` this |
| `do_doublet_detection` | true | run Scrublet on raw counts, per sample |
| `remove_doublets` | true | drop predicted doublets (false = flag only) |

### `clustering:`

| Key | Default | Effect |
|---|---|---|
| `leiden_resolution` | 0.5 | resolution for `obs['leiden']`. Must be positive. |
| `evaluate_resolutions` | false | non-destructive resolution audit |
| `resolution_candidates` | [0.2, 0.4, 0.5, 0.6, 0.8, 1.0] | audited resolutions |
| `min_cluster_cells` | 20 | below this a cluster is reported as too small to annotate reliably |
| `resolution_silhouette_max_cells` | 5000 | above this, silhouette uses a seeded subsample |

### `annotation:`

| Key | Default | Effect |
|---|---|---|
| `mixed_cluster_min_dominant_fraction` | 0.70 | flag mixed when the dominant CellTypist label covers less |
| `mixed_cluster_second_label_fraction` | 0.20 | flag mixed when the runner-up covers at least this |
| `use_subtypes_for_downstream` | false | may DE/composition key off `celltype_subtype`? Subtypes are single-annotator calls |
| `reuse_existing_final_annotation` | false | when false the consensus is **always** recomputed and a leftover single-voter column can never be promoted to final |
| `final_annotation_column` | null | the explicit column to reuse when the above is true |

### `downstream:`

| Key | Default | Effect |
|---|---|---|
| `exclude_low_confidence_de` | false | hold Low/Review clusters out of DE and composition. Recommended `true` once you have reviewed the tiers |
| `excluded_consensus_tiers` | [Low/Review] | which tiers that means |
| `do_pseudobulk_de` | true | donor-level DESeq2 |

### `de:`

| Key | Default | Effect |
|---|---|---|
| `reference_group` | null | the CONTROL arm. null = detect from baseline vocabulary; no match → alphabetical **and logged as arbitrary** |
| `lfc_threshold` | 1.0 | DESeq2 formal null `H0: |log2FC| ≤ this`. 0 = classic against-zero Wald test |
| `alpha` | 0.05 | padj cutoff for up/down calls |

### `single:` / `multi:`

| Key | Mode | Effect |
|---|---|---|
| `single_10x_dir` | single | the 10x matrix folder (required) |
| `multi_base_dir` | multi | the `<group>/<sample>/` tree (required) |
| `out_name` | both | output directory, resolved relative to the working directory |
| `sample_label` / `group_label` | single | null = derived from the folder name |
| `group_map_path` | multi | null = auto-detect `metadata.csv` / `group_map.csv` |
| `do_groupwise_de` | both | needs ≥2 groups |
| `batch_key` | both | e.g. `sample`; null = no batch correction |
| `integration_method` | both | `bbknn` or null |
| `run_per_sample` | multi | also run each sample individually (multiplies runtime) |

---

## 20. Reproducibility

| Mechanism | Implementation |
|---|---|
| **Seeding** | `set_global_seed(seed)` seeds Python, NumPy and `PYTHONHASHSEED`; `random_state=seed` is threaded through Scrublet, PCA, neighbours, UMAP, t-SNE and Leiden. Default `seed = 0`. |
| **Manifest** | `provenance/manifest.json` — see §17.2 |
| **Environment capture** | 15 package versions plus the Python version, recorded per run |
| **Git commit** | recorded per run |
| **Deterministic ordering** | numeric-aware cluster sort (`'2'` before `'10'`); tie-breaks resolve to alphabetical as the final fallback, never to dict-insertion order |
| **Cached retrieval** | PubMed `esearch`/`efetch` responses are cached on disk, and the cited PMIDs are recorded — the literature evidence for a label does not drift between runs |
| **No Excel round-trips** | every intermediate is CSV / h5ad / JSON |
| **Config-driven** | the run is fully described by `config.yaml` + `.env` + the input tree |

Two honest caveats: the LLM voters call a hosted model, so exact reproduction depends on that
model remaining available at the pinned slug (temperature is 0 and `max_tokens` is pinned, but a
provider-side model update is outside the package's control). And Enrichr is a live service, so
pathway results can shift as its libraries are updated — which is why the library versions are
in the library names (`KEGG_2021_Human`, `Reactome_2022`).

---

## 21. Worked example — a real run, end to end

`outputs/demo_multi/` — bundled demo cohort GSE283500, **2 groups × 2 samples**, skin.

**Input** 4 × 10x folders under `Healthy_skin/` and `Psoriasis_skin/`, plus
`GSE283500_metadata.json`.

**Resolution of the `auto` settings** (from the manifest):

```
requested tissue = auto        → resolved: skin
requested species = auto       → resolved: human
requested celltypist_model= auto → resolved: Immune_All_Low.pkl
requested singler_reference=auto → resolved: HumanPrimaryCellAtlasData
llm_model                        = anthropic/claude-sonnet-4.6
lineage panels    = TIS_CELL_markers_v3 + cell_hierarchy
                    {Immune: 51, Epithelial: 46, Fibroblast: 45, Endothelial: 42, Mural: 44}
```

**QC accounting:**

```
33,538 genes × 9,993 cells loaded
  failed min_genes (>200)          60
  failed max_genes (<6000)          3
  failed max_mito_percent (<15%)  154
  removed_total                   164     ← less than 60+3+154: the rules overlap
  after QC                      9,829
  predicted doublets                9  → removed
  final                         9,820 cells × 18,254 genes
HVG: seurat_v3 (per-batch), n_top_genes = 4,000, no batch excluded
Leiden 0.5 → 7 clusters
```

**Contrast design:**

```
groups                = [Healthy_skin, Psoriasis_skin]
reference_group       = Healthy_skin
reference_selection   = baseline vocabulary matched 'Healthy_skin'
contrast              = Psoriasis_skin_vs_Healthy_skin
direction             = positive log2FoldChange = higher in Psoriasis_skin
effect_size_rule      = DESeq2 formal null H0:|log2FC| <= 1 (no post-hoc filter)
shrinkage             = apeGLM; p-values unaffected
```

**Integration design verdict:** `nested`, `confounded_with_condition = True`, 4 batches / 2
groups, 0 batches spanning groups → the confounding warning is recorded, and
`de_boundary.ok = True` confirms no corrected matrix reached a test.

**Annotation result — all 7 clusters, with the disagreement visible:**

| cl | cells | consensus | tier | subtype | gate | CellTypist | SingleR | Knowledge | PubMed |
|---:|---:|---|---|---|---|---|---|---|---|
| 0 | 2,984 | T cell | **High** | CD4-positive T cell | Other | Tem/Effector helper T cells | T_cells | T cell | T cell |
| 1 | 1,920 | T cell | **High** | Regulatory T cells | Other | Regulatory T cells | T_cells | T cell | T cell |
| 2 | 1,532 | Dendritic cell | Low/Review | Dendritic cell | Other | Intermediate macrophages | DC | Dendritic cell | macrophage / mo-DC |
| 3 | 1,132 | T cell | **High** | cytotoxic effector CD8+ T cell | Immune | Tem/Trm cytotoxic T cells | T_cells | Cytotoxic T cell | CD8-positive T cell |
| 4 | 1,078 | Mast cell | **High** | Mast cells | Immune | Mast cells | T_cells | Mast cell | mast cell |
| 5 | 983 | Macrophage | Low/Review | Macrophage | Other | Tem/Effector helper T cells | T_cells | Macrophage | macrophage |
| 6 | 191 | Innate lymphoid cell | Low/Review | ILC3 | Other | ILC3 | T_cells | NK cell | T cell |

Tier counts: **High 4 · Medium 0 · Low/Review 3**. Five clusters carry `mixed_cluster_flag`.

**Reading the three Low/Review calls — this is the system working, not failing:**

- **cluster 2** — CellTypist said "Intermediate macrophages", the knowledge voter said dendritic
  cell, PubMed hedged "macrophage / monocyte-derived dendritic cell". Split → adjudicated:
  *"LAMP3 is a well-established marker of mature/migratory dendritic cells…"*. Cited PMIDs
  34279540, 33483337, 32344053. The mononuclear-phagocyte boundary is genuinely hard, and the
  tier says so.
- **cluster 5** — majority Macrophage and lineage-consistent, which would normally be Medium.
  Capped to Low/Review because **67% of its top 15 markers are `stress_heat_shock` state genes,
  not identity markers** (§12.12). The label is kept; the confidence claim is withdrawn.
- **cluster 6** — 191 cells; CellTypist `ILC3`, knowledge voter `NK cell`, PubMed `T cell`. Three
  voters, three answers. Adjudicated to Innate lymphoid cell on `XCL1`/`XCL2`.

**Also note cluster 4.** Every voter and the gate say mast cell — except SingleR, which says
`T_cells`. SingleR returns `T_cells` for **all seven** clusters, which is exactly the saturation
pattern of §12.3: a voter with no discriminative power on this dataset. Its permanent dissent is
recorded and reported, and it did not prevent four clusters from reaching High.

**Lineage gate behaviour.** It abstained (`Other`) on 5 of 7 clusters and fired `Immune` on 2.
That is the designed behaviour in a sorted-immune dataset — `score_genes` subtracts a background
of similarly-expressed genes, so in a homogeneous population the dominant lineage scores ≈0 by
construction (§12.2). Abstaining is correct; crowning the least-negative panel is what produced
phantom compartments before.

**Downstream:** pseudobulk DESeq2 ran for 4 cell types (Dendritic cell, Macrophage, Mast cell,
T cell) — the ILC cluster has too few cells per donor. Cell-level exploratory DE ran for 5.
`exclude_low_confidence_de` was false, so all 9,820 cells were included and the manifest records
`n_cells_excluded = 0` with the reason.

---

## 22. Known limitations and what the pipeline refuses to claim

**Refusals — the pipeline will not produce these silently:**

| It will not | Instead |
|---|---|
| present a cohort claim from cell-level p-values | pseudobulk is primary; cell-level files are named `*_CELLLEVEL_EXPLORATORY.csv` with a README |
| pick a fold-change sign arbitrarily | one reference arm resolved per run and stamped into every row; an arbitrary fallback is *recorded as arbitrary* |
| export p-values for one-vs-rest markers | ranking only (`rank`, `scores`, `logfoldchanges`) — the groups were defined by that same matrix |
| call a state-driven cluster High confidence | tier capped at Low/Review with the state fraction in the reason |
| crown a lineage with no marker support | the gate abstains below 0.1 |
| assert a subtype with no marker backing | 32 evidence rules reject it and record the rejection |
| lose or duplicate cells during annotation | `CellConservationError` |
| root pseudotime arbitrarily | DPT skipped when no baseline group exists |
| run DE on 1 sample per arm | the contrast is skipped and reported |
| proceed on a normalized matrix | `DataValidationError` |
| promote a leftover single-voter column to the final label | `reuse_existing_final_annotation` defaults to false and the consensus is always recomputed |

**Real limitations:**

1. **Human only.** The CellTypist catalog, the SingleR reference list and the lineage-gate panels
   are all human. Mouse support needs four coordinated changes (§12.3), not one config value.
2. **The two open-vocabulary voters are not independent.** The knowledge and PubMed voters share
   one model and agree 89–100% of the time (measured). Four voters is not four independent
   opinions — which is exactly why the subtype gate asks the *data* rather than a second voter,
   and why out-of-domain deference requires a closed-vocabulary-only majority.
3. **Integration is confounded by design** in cohort studies (§9.2). It is contained to
   annotation and verified at run time, but it cannot be made non-confounded.
4. **The tissue inference is LLM-based** and reads free text from GEO metadata. It resolved
   `skin` correctly on the worked example, but a mis-inferred tissue selects a mis-scoped
   CellTypist model — which is why observed-lineage refinement and vote suppression exist.
5. **Pathway enrichment depends on a live Enrichr**, so those results can shift over time.
6. **No benchmarking against a gold-standard annotated dataset ships with the package.** The
   thresholds are measured against specific real datasets (documented inline at each constant)
   rather than tuned on a held-out benchmark.
7. **No formal calibration.** The tiers are ordinal and useful for triage, but they are not
   calibrated probabilities — "High" is not a claim about an error rate.
8. **`run_scanpy_pipeline` is one 1,505-line function with 47 parameters.** It works and it is
   tested, but it is the package's main architectural debt.

---

## 23. Module map

| Module | Responsibility |
|---|---|
| `main.py` | the only entry point: read YAML, flatten sections, dispatch to a driver |
| `config.yaml` | the entire run configuration |
| `main_single.py` / `main_multi.py` | load one dataset / discover and combine a cohort tree |
| `loader_10x.py` | read a 10x feature-barcode matrix into AnnData |
| `data_validation.py` | pre-flight validation with safe auto-repair |
| `qc_filters.py` | configurable QC thresholds with per-rule accounting |
| `hvg_selection.py` | HVG selection that survives a degenerate sample (4-rung ladder) |
| `integration.py` | what batch integration may and may not influence; run-time boundary check |
| `clustering.py` | Leiden config, execution, and the optional resolution audit |
| `pipeline.py` | the core Scanpy orchestration (`run_scanpy_pipeline`) |
| `gene_names.py` | Ensembl → HGNC symbol normalization (mygene optional) |
| **`celltype_consensus/`** | |
| `consensus.py` | the consensus orchestrator: stages 2–9 |
| `tools.py` | all deterministic annotation logic — **zero LLM calls, by invariant** |
| `agent.py` | the **only** module that talks to OpenRouter |
| `config.py` | `.env`-driven consensus configuration; fails fast on a missing key |
| `celltypist_catalog.py` | the 41-model CellTypist catalog + the 5 SingleR references |
| `lineage_panels.py` | builds the lineage-gate panels from TIS_CELL + the hierarchy |
| `consensus_plots.py` | voter agreement and per-method call figures |
| `cell_hierarchy/` | the 404-node ontology, alias crosswalk and pure-logic resolver |
| `cell_hierarchy/spec/` | `hematopoietic` · `epithelial` · `mesenchymal` · `neural_germ_other` · `aliases` · `states` |
| **downstream** | |
| `pubmed_annotation.py` | the self-contained PubMed RAG voter |
| `markers.py` | per-cell-type identity markers |
| `marker_stats.py` | what a one-vs-rest table may claim (strips p-values) |
| `rank_genes_subprocess.py` | cluster markers in a subprocess (memory isolation) |
| `contrasts.py` | the single place that decides which arm is the reference |
| `pseudobulk_de.py` | donor-level DESeq2 — the primary group test |
| `group_de.py` | cell-level exploratory DE + composition views |
| `downstream_gating.py` | which cells may carry an inferential claim |
| `pathway_enrichment.py` | GO/KEGG/Reactome/WikiPathways over-representation + dedup |
| `summary_ct_deg.py` | joins pseudobulk DE + markers + pathways per cell type |
| `celltype_qc_plots.py` | per-donor proportions, marker dotplot, voter agreement |
| `sc_to_bisq.py` | the Bisque-ready deconvolution export |
| `singlecell_sc_report_generation.py` | the HTML/PDF report |
| `reproducibility.py` | seeds, manifest, environment capture |
| `figure_style.py` | one place for DPI, panel geometry, point size |
| `safe_names.py` | filesystem-safe names for labels used as paths |

---

## 24. Environment and dependencies

**Python ≥ 3.11** (verified on 3.12.10). Python 3.10 will not work — several dependencies
require ≥3.11.

Install: `python -m pip install -r requirements.txt`, then
`python -m playwright install chromium` if you want the PDF report.

Optional, feature-gated:

| Extra | Enables | Without it |
|---|---|---|
| R + `SingleR` + `celldex` + `scrapper` | the SingleR voter | voter unavailable; error is loud, not silent |
| `pydeseq2` | pseudobulk DESeq2 | the whole step skips with a clear log line |
| `gseapy` + network | pathway enrichment | step skips |
| `mygene` | Ensembl → symbol mapping | `var_names` kept as-is, logged |
| network + OpenRouter | knowledge/PubMed voters, adjudicator, report narratives, `auto` selections | voters off; report falls back to templated text |
| Playwright/Chromium | PDF report | HTML report still builds |

`requirements.txt` pins exact verified versions. `pyproject.toml` carries the same set with
lower **and upper** bounds (upper at the next major, so a breaking release cannot silently land
in a rebuild), and `uv.lock` pins the fully resolved graph.

**Note on `python-decouple`.** The import name is `decouple`, and an unrelated PyPI package
called `decouple` claims the same import name. If you see
`ImportError: cannot import name 'config' from 'decouple'`, you have the wrong package (or the
wrong virtual environment):

```bash
python -m pip uninstall -y decouple
python -m pip install python-decouple==3.8
```

**Resource requirements** are in `RESOURCE_REQUIREMENTS.xlsx` (measured runtimes and peak memory
across six real runs from 900 to 87,343 cells, plus a sizing calculator). The short version: the
`.h5ad` and the peak RAM both scale with `cells × genes`, and `h5ad` lands at roughly 2.4× the
dense matrix size.

---

*Generated from the current codebase — Ayass Bioscience · Sheryar Malik · 2026-08-06.*
