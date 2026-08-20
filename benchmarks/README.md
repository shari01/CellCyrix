# Annotation benchmark

The numbers a paper needs, from a pipeline output that carries a ground-truth column.

Distinct from [`tests/benchmarks/`](../tests/benchmarks/), which measures runtime and
peak memory. This measures whether the annotation is **correct**.

---

## What it produces

One directory per dataset under `benchmarks/results/<name>/`:

| File | Contents |
| --- | --- |
| `00_resolution_report.csv` | Per-column label resolution rates. **Read this first** — see the confound warning below |
| `00_label_mapping.csv` | Every distinct raw label and where it mapped. Publish as a supplement |
| `01_method_comparison.csv` | Macro-F1 + 95% bootstrap CI: each single voter vs the consensus |
| `02_ablation.csv` | Every voter subset, so "why four voters?" has an answer |
| `03_per_class_f1.csv` | Precision/recall/F1/support per cell type, per method |
| `04_risk_coverage.csv` | The headline curve: error vs coverage, per method |
| `05_risk_coverage_summary.csv` | AURC + error at 100/95/90/80/70% coverage |
| `06_calibration.csv` | Reliability curve per method |
| `07_calibration_summary.csv` | ECE, MCE, overconfidence |
| `08_disagreement.csv` | Per-cell voter entropy, plus correlations against doublet score, pseudotime, QC metrics |
| `09_confusion_<method>.csv` | Row-normalised confusion matrix per method |
| `manifest.json` | Inputs, columns used, seed, package versions, cells dropped |

## Running it

```bash
python benchmarks/run_annotation_benchmark.py \
    --h5ad outputs/tabula_sapiens/ts_processed_scanpy_output.h5ad \
    --truth-column cell_type \
    --name tabula_sapiens
```

Options that matter:

- `--level` — hierarchy level to compare at. Default `main_cell_type`. Coarser is
  fairer across methods with different granularity; `subtype` measures whose
  granularity happens to match the truth set, which is not accuracy.
- `--extra-method OBS_COLUMN` — repeatable. Score an externally-computed baseline
  (Azimuth, GPTCelltype) that you have already joined into `obs`.
- `--n-bootstrap` — resamples for the CI. `0` skips it, but then you cannot claim one
  method beats another.
- `--seed` — bootstrap and tie-breaking. Fixed so results are reproducible.

## Single-voter baselines are free

The pipeline already writes every voter's own call into `obs`:

| Column | Baseline it provides |
| --- | --- |
| `celltype_celltypist` | CellTypist alone |
| `celltype_singler` | SingleR alone |
| `celltype_knowledge_based` | LLM marker-reasoning voter alone |
| `celltype_consensus` | The pipeline's consensus |

So **one pipeline run per dataset gives the whole comparison**, and the ablations are
arithmetic over those columns rather than five more runs.

## The two external baselines

Azimuth and GPTCelltype need R and are **not** computed here. Run them separately, join
their per-cell labels onto the same `obs` index, and pass them with `--extra-method`.
They then flow through the identical harmonisation and scoring — no special-casing.

```r
# Azimuth
library(Azimuth); library(Seurat)
obj <- RunAzimuth(seurat_obj, reference = "pbmcref")
write.csv(data.frame(cell = colnames(obj),
                     celltype_azimuth = obj$predicted.celltype.l2),
          "azimuth_labels.csv", row.names = FALSE)
```

```r
# GPTCelltype — takes your Leiden marker table from 04_clustering_and_cell_states/
library(GPTCelltype)
labels <- gptcelltype(markers_df, tissuename = "lung", model = "gpt-4")
```

Then join in Python and re-write the `.h5ad` before running the benchmark.

## Datasets to use

Ground truth must not be someone's opinion:

| Dataset | Why | Where |
| --- | --- | --- |
| **Hao et al. 2021 PBMC CITE-seq** (~161k cells) | Protein-based labels — the field's reference standard | GSE164378 |
| **Zheng et al. 2017 purified PBMCs** | FACS-sorted, physically unarguable | 10x Genomics public datasets |
| **Tabula Sapiens** | Multi-tissue expert annotation — this is what tests the *disease-agnostic* claim | CELLxGENE Census |
| **Human Lung Cell Atlas** (Sikkema 2023) | Cross-dataset generalisation | CELLxGENE Census |

Pulling a labelled dataset from CELLxGENE Census:

```python
import cellxgene_census

with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter="dataset_id == '<dataset_id>'",
    )
```

`obs["cell_type"]` arrives as expert-curated, Cell Ontology-mapped labels — use it as
`--truth-column`. The matrix must be **raw integer counts**, or `00_data_validation/`
rejects the run.

## The confound this harness reports on itself

Harmonising ground-truth labels with the *pipeline's own* resolver can inflate the
pipeline's score: if the resolver knows the truth vocabulary better than a competitor's,
the competitor is penalised for a mapping failure rather than a biological error.

Three guards, all in [`harmonise.py`](harmonise.py):

1. **One function, no per-method branch.** `harmonise()` takes no source argument. Truth
   and every prediction go through the identical call.
2. **Resolution rates are reported per column.** Any method resolving more than 10
   percentage points worse than the truth column is logged as a warning and recorded in
   `manifest.json` under `confounded_by_resolution_gap`.
3. **Coarse comparison by default.**

State this explicitly in the methods section, and publish `00_label_mapping.csv`. A
reviewer who can check every label's destination has no reason to suspect the mapping.

## What the harness will not do for you

- **A method's own failures never shrink the denominator.** Unresolved predictions score
  as errors; only unresolvable *ground truth* is dropped. Otherwise a method could
  improve its score by abstaining.
- **A method with no confidence column gets a flat risk-coverage curve.** That is the
  honest depiction of a method that cannot rank its own calls, not a bug.
- **Freeze your dataset and metric choices before you look at results.** The seed is
  fixed and the manifest records everything, but nothing here prevents dataset
  cherry-picking. That is on you, and as a first author you will not win that argument
  with a reviewer.

## Verifying the harness itself

The metrics are tested against inputs whose correct answers are known by construction —
a perfect classifier, a constant classifier, a perfectly calibrated one, an
oracle-ranked one — plus an end-to-end run on a synthetic `.h5ad`:

```bash
pytest tests/test_benchmarks.py -v
```

Run this before trusting any number the harness produces.
