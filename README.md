# CellCyrix

Single- and multi-sample Scanpy 10x pipeline with multi-voter consensus cell-type
annotation and donor-level pseudobulk differential expression.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## Purpose

Takes raw 10x Genomics feature-barcode matrices and produces an auditable, end-to-end
single-cell analysis: QC and filtering, doublet detection, HVG selection, PCA/UMAP,
Leiden clustering, cell-type annotation by consensus vote, differential expression at
the donor level, pathway enrichment, and an HTML/PDF report.

Two things distinguish it from a plain Scanpy script:

- **Annotation is a consensus, not one call.** Four independent voters — CellTypist
  (ML), SingleR (reference-based, via rpy2), a knowledge-based LLM marker-reasoning
  voter, and a PubMed literature voter — are harmonised onto a 404-node cell-type
  hierarchy so that spelling and granularity differences stop reading as disagreement.
  A coarse lineage gate, built from curated marker panels, can veto a call outright
  rather than accept the least-bad option.
- **Differential expression uses the donor as the unit of replication.** Group
  contrasts run as pseudobulk DESeq2 over per-sample summed raw counts, not
  cell-level Wilcoxon, so *n* is the number of donors. Cell-level DE is still written,
  but is stamped as exploratory.

Every run writes a provenance manifest recording the seed, package versions, the
reference table actually used, the resolved parameters, and two things that make the
annotation numbers auditable rather than merely reproducible-looking:

- **Which scikit-learn pickled each CellTypist model, against the one that ran it.**
  30 of the 41 bundled models were written by scikit-learn 0.24.1, 8 by 1.2.2, 3 by
  1.0.1 — upstream's own model zoo — so a modern runtime warns that results "might be
  invalid". Rather than silence that, every model is checked at load: `predict_proba` is
  recomputed from the persisted `coef_`/`intercept_`/`mean_`/`scale_` arrays with NumPy
  alone and must match exactly (observed difference: 0.0). A model that ever fails
  aborts the run instead of quietly annotating. The models are **not** re-pickled, which
  would invalidate `SHA256SUMS.txt` and destroy upstream provenance while changing no
  number. See `celltype_consensus/model_provenance.py`.
- **How much of the LLM output was replayed from cache.** `fully_cached: true` means the
  run made no LLM network call and can be regenerated exactly, offline, from the cache
  directory alone.

## Inputs

| Input | Required | Notes |
| --- | --- | --- |
| 10x feature-barcode directory | yes | `matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, and `features.tsv[.gz]` (or `genes.tsv[.gz]`). Gzipped or plain; prefixed filenames tolerated. |
| `config.yaml` | yes | Drives the entire run. Sections: `common`, `clustering`, `qc`, `annotation`, `downstream`, `de`, plus `single:` / `multi:`. |
| `group_map.csv` or `metadata.csv` | multi-sample only | Maps sample → experimental group. `.xlsx` also accepted. Without it, the top-level folder names under the cohort directory are used as groups. |
| `<GSE>_metadata.json` | optional | GEO payload; supplies species (taxon) and lets tissue be inferred. |
| `.env` | optional | Credentials. **With no keys at all the pipeline still runs**: QC, doublets, HVG, PCA/UMAP, Leiden, the CellTypist voter, markers, pseudobulk DESeq2 and composition all work offline. See `.env.example`. |

**Input data is not committed.** `input_data/` is git-ignored (research data does not
belong in a source repository, and the previously-tracked copy was missing the
`matrix.mtx.gz` files anyway, so it could not run). Download the accessions you want
from GEO into that folder — the demos in `config.yaml` expect `GSE212966` (PDAC vs
adjacent normal) and `demo_cohort_GSE283500` (psoriasis vs healthy skin).

Cohort layout for a multi-sample run:

```
input_data/<cohort>/
├── group_map.csv
├── <GSE>_metadata.json          # optional
├── <Group_A>/<SAMPLE_ID>/       # barcodes.tsv.gz · features.tsv.gz · matrix.mtx.gz
└── <Group_B>/<SAMPLE_ID>/
```

### Static reference data

`shared_reference/` (55 MB) holds everything the code reads that is not run input:

| Bucket | Read when | Purpose |
| --- | --- | --- |
| `celltypist_models/data/models/` | every run | 41 CellTypist `.pkl` models, so the CellTypist voter needs no network. **Fetched, not committed** — see below. |
| `.../TIS_CELL_markers_v3/master_celltype_markers_long.csv` | every run | builds the lineage gate's marker panels |

The `.pkl` models are 54 MB of binary pickles and are git-ignored; the two small text
files that describe them — `models.json` (upstream URLs) and `SHA256SUMS.txt` (expected
digest and size per model) — are tracked. **No manual step is needed:** the pipeline
fetches whichever model a run actually selects on first use and caches it in place, so
every run after the first is fully offline.

To pre-warm all 41 — before going offline, or when building an image:

```bash
fetch-celltypist-models                              # after pip install
python scripts/fetch_celltypist_models.py            # or, from a clone
fetch-celltypist-models --force                      # re-download all 41
```

Every download is checked against `SHA256SUMS.txt` and **deleted** if it does not
match, and every model is verified again at load time — a CellTypist model is a pickle,
so loading one executes whatever it contains. A corrupt, truncated or swapped file
raises instead of being unpickled.

If either bucket is missing the run still completes and says so — CellTypist falls back
to downloading, and the lineage gate falls back to built-in panels while recording
`lineage_panel_fallback_reason` in the provenance.

Leave `SCPIPE_SHARED_REFERENCE_ROOT` **unset** to use the package-relative default. If
you do set it, use an absolute path: a relative value resolves against the working
directory, so running from elsewhere silently loses the reference data. The unprefixed
`SHARED_REFERENCE_ROOT` / `AGENTIC_REFERENCE_DATA_ROOT` still work and log a
deprecation notice; other pipeline-owned variables are likewise prefixed
(`SCPIPE_RSCRIPT_EXE`, `SCPIPE_ENABLE_SINGLER`, `SCPIPE_ENABLE_PUBMED`,
`SCPIPE_REPORT_LLM_MODEL`). Third-party names — `OPENROUTER_API_KEY`, `NCBI_API_KEY`,
`NCBI_EMAIL`, `R_HOME` — keep the spelling their own tools document.

## Outputs

Everything lands under the caller-supplied output directory, in numbered stages. Both
drivers take `out_name` plus the `output_root` it is resolved against; a relative
`out_name` with no root raises rather than writing wherever the process happens to be
running.

Every filename is `lower_snake_case` and every table shares one column vocabulary
(`gene`, `log2_fold_change`, `p_value`, `p_value_adj`, `base_mean`, `lfc_se`,
`pathway`, `combined_score`, ...) regardless of which library produced it — DESeq2's
`log2FoldChange`, scanpy's `logfoldchanges` and gseapy's `Adjusted P-value` are mapped
on write. Tables written by earlier versions are still readable: the readers normalise
headers on input too. Writes are atomic (temp file, then rename), so a crash never
leaves a truncated table that parses as a complete one.

| Directory | Contents |
| --- | --- |
| `00_data_validation/` | raw-count validation report — fails loudly if the matrix is not integer counts |
| `00_analysis_summary/` | plain-text run summary |
| `01_qc_and_filtering/` | QC violins, scatters, histograms; pre/post filter counts |
| `02_highly_variable_genes/` | HVG table + figure |
| `03_dimensionality_reduction_and_embeddings/` | PCA variance, UMAP/diffmap embeddings, group-wise embeddings |
| `04_clustering_and_cell_states/` | Leiden clusters, cluster marker table, inter-cluster DEG |
| `05_celltype_analysis/` | consensus annotation table, per-voter columns, confidence tiers, per-cell-type markers |
| `06_groupwise_deg/` | `pseudobulk_deg/` (donor-level DESeq2 + contrast design) and `celltype_specific_deg/` (exploratory, cell-level) |
| `08_reference_summary/` | integrated per-cell-type DEG / marker / pathway summaries |
| `provenance/` | `manifest.json` — seed, package versions, dataset shape, resolved parameters, reference tables used |
| `singlecell_report/` | HTML / PDF report |
| `*_processed_scanpy_output.h5ad` | processed AnnData: `layers["counts"]` keeps raw counts, `obsm` carries embeddings, `obs` carries design + annotation + gating columns |
| `bisque_ready_*.h5ad` | deconvolution-ready export |

## How to run

Install first:

```bash
pip install -e .            # or: uv sync
```

The config drives everything, including whether the run is single-sample or a cohort.
Set `mode:` to `single` or `multi` at its top. Three equivalent ways in:

```bash
cellcyrix --config config.yaml                                  # installed console script
cellcyrix --config my_config.yaml --output-root /data/runs
cellcyrix --config config.yaml --print-config                   # resolve and exit, run nothing
```

```bash
python main.py              # from a clone; uses ./config.yaml and ./outputs
```

```python
from pathlib import Path
import main

main.main(config_path=Path("my_config.yaml"))  # outputs under ./outputs/
main.main(config_path=Path("my_config.yaml"), output_root=Path("/data/runs"))
```

All three land on the same `cellcyrix.runner.run_from_config`, so they cannot disagree
about what a config means. `main.py` holds only the repo-relative defaults; the console
script defaults to `./config.yaml` and `./outputs` in the working directory, because an
installed package has no repository root to be relative to.

`--print-config` is worth knowing about: it resolves the config — including nested
sections flattened onto driver arguments — prints it as JSON and exits, so you can check
what a run will actually do before spending the compute.

### In Docker

```bash
docker build -t cellcyrix:1.0.0 .
docker run --rm \
    -v "$PWD/input_data:/data/input:ro" \
    -v "$PWD/outputs:/data/outputs" \
    cellcyrix:1.0.0 --config /data/input/config.yaml --output-root /data/outputs
```

The image pins R and its CRAN snapshot via `rocker/r-ver`, installs SingleR + celldex,
fetches and checksum-verifies all 41 CellTypist models at build time, and runs the full
test suite as a build step — so a build that succeeds is an environment that produces
correct output. It needs no network at run time.

The CellTypist models download on first use and are verified on every load; run
`fetch-celltypist-models` if you want all 41 in place up front.

Requires Python **>= 3.11**. Exact resolved versions are pinned in `uv.lock`.

Optional external tooling: **R >= 4.x** with `SingleR`, `celldex`,
`SummarizedExperiment`, `Matrix` and `scrapper` for the SingleR voter; an
OpenRouter key for the knowledge-based voter, PubMed adjudication, automatic
tissue/model selection, and the report's narrative sections.

## How to test

```bash
pip install -e ".[dev]"

ruff format .    # formatter
ruff check .     # linter
pytest           # 260 passed, 8 skipped, 24 deselected, ~25 s
```

`pytest` excludes the `slow` marker so the gate stays a ~25-second command. The two slow
suites are the ones that verify the claims a paper rests on, and CI runs them on every
push:

```bash
pytest -m slow tests/test_pipeline_determinism.py   # two full runs must agree exactly
pytest -m slow tests/test_pseudobulk_fdr.py -s      # DE false-discovery behaviour
pytest -m ""                                        # everything, no marker filter
```

Two further suites are not collected by `pytest` because they are executables
rather than test modules:

```bash
python tests/smoke_test.py            # 53-check offline end-to-end run, ~60 s
python tests/run_resolver_tests.py    # 59 hierarchy-resolver tests, no pytest needed
```

`smoke_test.py` is the one to run before shipping: it builds a synthetic cohort in a
temp workspace, runs both the single and multi drivers through the real `main.py`
entry point, and asserts on every artifact downstream consumers read — including a
direction positive control that the genes simulated as up in CASE come out with a
positive `log2FoldChange`. It needs no network, no credentials and no R.

## Benchmarking the annotation

[`benchmarks/`](benchmarks/) scores the consensus against ground truth alongside every
single-voter baseline and voter-subset ablation, with bootstrap confidence intervals,
risk–coverage curves, calibration/ECE, and per-cell voter-disagreement entropy:

```bash
python benchmarks/run_annotation_benchmark.py \
    --h5ad outputs/<run>/<name>_processed_scanpy_output.h5ad \
    --truth-column cell_type
```

Single-voter baselines are free — the pipeline already writes each voter's own call into
`obs`, so one run per dataset gives the whole comparison and the ablations are arithmetic
over those columns rather than five more pipelines. Azimuth and GPTCelltype need R; run
them separately, join their labels into `obs`, and pass `--extra-method`.

The harness reports on its own weakest point: harmonising labels through the pipeline's
own hierarchy could flatter the pipeline, so resolution rates are reported per method
before any accuracy number, and any method resolving >10pp worse than the truth column is
flagged in the manifest. See [`benchmarks/README.md`](benchmarks/README.md).

The held-out cell-type experiment — the one that tests the disease-agnostic claim — is
separate, because it asks a different question: when a type is absent from the reference,
does the method abstain or confidently mislabel it?

```bash
python benchmarks/run_heldout_celltype.py \
    --h5ad outputs/<run>/<name>_processed_scanpy_output.h5ad \
    --truth-column cell_type --n-types 5
```

### Known limitation: do not claim "FDR controlled at 5%"

`tests/test_pseudobulk_fdr.py` establishes what is safe to state — the null p-values are
uniform, `p_value_adj` is exactly Benjamini-Hochberg on `p_value` (agreement to 4e-14), a
cohort with no true differences yields ~0.075% discoveries, power and direction are
correct, and discoveries do not scale with cells per donor, so the donor really is the
unit of replication.

What it does **not** establish is nominal FDR control. On its simulator the mean false
discovery proportion is ~0.17 against a nominal 0.05, and that does not converge with
more genes (600 → 5000) or more donors (4 → 12). The adjustment step is provably correct
and the null is uniform, so the residual sits in the far tail: null `p < 0.01` fires at
~1.5× its nominal rate. Two causes remain live and the test cannot separate them —
DESeq2's Wald test using an asymptotic approximation on shrunken dispersions, and the
simulator drawing a constant dispersion that DESeq2's `a/mean + b` trend cannot represent.
Resolve this against a real dataset with matched bulk RNA-seq before making any FDR claim
in print.

## Citing

See [`CITATION.cff`](CITATION.cff). Licensed under the [MIT License](LICENSE).

## Owner

**Ayass Bioscience — Computational Biology / Engineering**
Maintainer: Sheryar Malik · <sheryar.malik@ayassbioscience.com>

## Documentation

Four documents, all generated from the current codebase (2026-08-06). Start with the
diagram to see the shape of a run, then the master document for anything specific.

| Document | Covers |
| --- | --- |
| [`pipeline_architecture.png`](docs/pipeline_architecture.png) | **one-page diagram** — every stage from input to output, the four voters, the guard rails |
| [`PIPELINE_MASTER_DOCUMENTATION.md`](docs/PIPELINE_MASTER_DOCUMENTATION.md) | **the full manual** — 24 sections: every stage, every method, every threshold with the measurement behind it, the complete config and output reference, a worked example on a real run, and the known limitations |
| [`PIPELINE_REFERENCE.xlsx`](docs/PIPELINE_REFERENCE.xlsx) | **the lookup tables** — 17 sheets: stages, all config keys, the 4 voters, all 41 CellTypist models, the 5 SingleR references, the 404-node hierarchy, the 5 lineage panels, the 5 state programmes, all 32 subtype rules, the 23-node vocabulary, every threshold, the full output inventory, all 52 consensus columns, dependencies, guard rails |
| [`Single_cell_pipeline_overview.pptx`](docs/Single_cell_pipeline_overview.pptx) | **22-slide deck** for presenting the pipeline and its design decisions |
| [`RESOURCE_REQUIREMENTS.xlsx`](docs/RESOURCE_REQUIREMENTS.xlsx) | measured runtime and peak RAM across six real runs (900 → 87,343 cells), plus a sizing calculator |

**If a run fails**, the master document's §24 covers the failures that actually happen —
most often the wrong `decouple` package (the import name is claimed by an unrelated
PyPI project) or the wrong virtual environment (Python 3.10 will not work; the pipeline
needs ≥ 3.11).
