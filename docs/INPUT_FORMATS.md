# Input Format Support

**Scope** Everything the pipeline accepts as input, exactly as the current source reads it.
Every pattern, column name and rule below was taken from `loader_10x.py`, `main_single.py`,
`main_multi.py` and `data_validation.py` — nothing here is aspirational.

**Short answer** The pipeline takes **10x Genomics Cell Ranger feature-barcode matrix folders**
(the Matrix Market trio). That is the only accepted entry format. `.h5ad` is accepted only by
side utilities and by re-entry steps, never by `run_pipeline` / `run_pipeline_multi`.

![Input formats](figures/input_formats.png)

---

## 1. Supported vs not supported

| Input | Supported | Read by |
|---|---|---|
| 10x Cell Ranger folder — `matrix.mtx` + `barcodes.tsv` + `features.tsv`/`genes.tsv`, plain or `.gz` | **Yes — the entry format** | `load_10x_feature_barcode_matrix()` |
| `<GROUP>/<SAMPLE>/` tree of such folders | **Yes — multi mode** | `_discover_multi_samples()` |
| `metadata.csv` / `group_map.csv` (`sample`, `group`) | Yes, optional | `_load_group_map()` |
| `metadata.xlsx` / `group_map.xlsx` | Yes, fallback only (logs a warning — Excel mangles identifiers) | `_load_group_map()` |
| `GSE*_metadata.json` study metadata | Yes, optional | `_find_geo_json()` |
| `.h5ad` | **Not a pipeline entry point.** Accepted only by `validate_path()`, `sc_to_bisq.process_h5ad_file()`, `rank_genes_subprocess.py`, and the `consensus.py` / `pubmed_annotation.py` standalone CLIs | — |
| 10x `.h5` (`filtered_feature_bc_matrix.h5`) | No | — |
| Loom, Seurat `.rds`, CSV/TSV count matrix, AnnData `.zarr` | No | — |
| Normalized / log / scaled / imputed matrices in any format | No — rejected by the raw-count gate | `_validate_raw_counts()` |

---

## 2. The file trio

`load_10x_feature_barcode_matrix()` globs each slot in order and takes the first match, so both
bare and sample-prefixed Cell Ranger / GEO filenames work
([loader_10x.py:114-160](../cellcyrix/single_cell_pipeline_agent/singlecell_10x/loader_10x.py#L114-L160)).

| Slot | Accepted filename patterns (in priority order) |
|---|---|
| matrix | `matrix.mtx`, `matrix.mtx.gz`, `*.matrix.mtx`, `*.matrix.mtx.gz`, `*.mtx`, `*.mtx.gz` |
| barcodes | `barcodes.tsv`, `barcodes.tsv.gz`, `*barcodes.tsv`, `*barcodes.tsv.gz`, `barcode.tsv`, `barcode.tsv.gz`, `*barcode.tsv`, `*barcode.tsv.gz` |
| features | `features.tsv`, `features.tsv.gz`, `*features.tsv`, `*features.tsv.gz`, `genes.tsv`, `genes.tsv.gz`, `*genes.tsv`, `*genes.tsv.gz` |

All three are required. A missing slot raises `FileNotFoundError`; an unreadable or corrupt file
raises `ValueError` naming the offending path.

### 2.1 `matrix.mtx[.gz]`

Matrix Market coordinate format as written by Cell Ranger — **genes × cells**. The loader
transposes it to **cells × genes** on read. Gzip is detected from the `.gz` suffix.

```
%%MatrixMarket matrix coordinate integer general
%
32738 2700 2286884
32709 1 4
32707 1 1
...
```

### 2.2 `barcodes.tsv[.gz]`

Headerless TSV, one row per cell, **column 1 = barcode**. Extra columns are ignored. Row order
must match the matrix column order.

```
AAACATACAACCAC-1
AAACATTGAGCTAC-1
AAACATTGATCAGC-1
```

### 2.3 `features.tsv[.gz]` / `genes.tsv[.gz]`

Headerless TSV, one row per gene, row order matching the matrix row order. Columns are assigned
positionally, and 1-, 2- and 3-column files are all valid:

| Column | Name assigned | Used as |
|---|---|---|
| 1 | `feature_id` | `var['feature_id']`; also `var_names` if there is no column 2 |
| 2 | `feature_name` | `var_names` and `var['gene_symbol']` |
| 3 | `feature_type` | `var['feature_type']` |
| 4+ | `extra_3`, `extra_4`, … | ignored |

```
ENSG00000243485	MIR1302-10	Gene Expression
ENSG00000237613	FAM138A	Gene Expression
ENSG00000186092	OR4F5	Gene Expression
```

`var_names_make_unique()` runs after load, so duplicate gene symbols become `SYMBOL-1`,
`SYMBOL-2`.

---

## 3. Raw counts are mandatory

Before the matrix becomes `layers['counts']`, `_validate_raw_counts()` rejects anything that is
not a genuine count matrix
([loader_10x.py:50-90](../cellcyrix/single_cell_pipeline_agent/singlecell_10x/loader_10x.py#L50-L90)):

| Check | Failure message |
|---|---|
| all values finite | `contains non-finite values (NaN/Inf)` |
| minimum ≥ 0 | `contains negative values (min=…). This looks like normalized/scaled data` |
| values integer-valued (tolerance `1e-8`) | `N% of stored values are non-integer … looks like normalized/log/scaled data` |

Nothing is silently repaired — the run stops with `ValueError`. The reason is that Scrublet,
Seurat-v3 HVG selection, DESeq2 pseudobulk and the Bisque export all assume raw counts, so a
normalized matrix would produce confidently wrong results rather than an error. An all-zero /
empty matrix passes (nothing to validate).

---

## 4. Single mode layout

One folder, holding one sample's trio.

```
input_data/demo_cohort_GSE283500/Healthy_skin/GSM8664023_NST8_CD8skin/
├── matrix.mtx.gz
├── barcodes.tsv.gz
└── features.tsv.gz
```

```yaml
mode: single
single:
  single_10x_dir: 'input_data/demo_cohort_GSE283500/Healthy_skin/GSM8664023_NST8_CD8skin'
  geo_json_path: 'input_data/demo_cohort_GSE283500/GSE283500_metadata.json'
  out_name: outputs/demo_single
  sample_label: null      # null -> the folder name
  group_label: null       # null, or CASE / CONTROL / TUMOR / NORMAL
```

`sample_label` defaults to the folder's own name and becomes `obs['sample']` and the analysis
name that prefixes every output file
([main_single.py:167-171](../cellcyrix/single_cell_pipeline_agent/singlecell_10x/main_single.py#L167-L171)).
Single mode has one group, so group DE is normally off.

---

## 5. Multi mode layout

A two-level tree: level 1 is the experimental arm, level 2 is the sample
([main_multi.py:134-164](../cellcyrix/single_cell_pipeline_agent/singlecell_10x/main_multi.py#L134-L164)).

```
input_data/GSE212966/                     <- multi_base_dir
├── metadata.csv                          optional, sample + group columns
├── GSE212966_metadata.json               optional study metadata
├── PDAC/                                 <- group
│   ├── GSM6567157_PDAC1/                 <- sample: matrix + barcodes + features
│   ├── GSM6567159_PDAC2/
│   └── GSM6567160_PDAC3/
└── adjacent_normal/
    ├── GSM6567169_N1/
    └── GSM6567170_N2/
```

```yaml
mode: multi
multi:
  multi_base_dir: 'input_data/GSE212966'
  out_name: outputs/gse212966
  group_map_path: null      # null -> auto-detect metadata.csv / group_map.csv
  do_groupwise_de: true
  batch_key: sample
  integration_method: bbknn
  run_per_sample: false
```

Rules the discovery step applies:

- Every immediate sub-directory of `multi_base_dir` is a **group**; every sub-directory of a
  group is a **sample**.
- A group folder with **no** sub-directories is treated as a single sample, with
  `group name == sample name`.
- Each sample is loaded, tagged with `obs['sample']` (folder name) and `obs['group']`, then all
  samples are concatenated into one AnnData and analysed jointly.
- A sample that fails to load is **skipped with a warning**, not fatal — the cohort run
  continues.

---

## 6. `metadata.csv` / `group_map.csv` (optional)

Placed at `multi_base_dir`, auto-detected in this order: `metadata.csv`, `group_map.csv`,
`metadata.xlsx`, `group_map.xlsx`. Override with `group_map_path`
([main_multi.py:54-131](../cellcyrix/single_cell_pipeline_agent/singlecell_10x/main_multi.py#L54-L131)).

Required columns — matched case-insensitively, extra columns ignored:

| Column | Meaning |
|---|---|
| `sample` | must equal the sample **folder name** |
| `group` | the arm the sample belongs to; **overrides** the folder-derived group |

```csv
sample,group,gse,gsm,title,organism,condition,raw_file
GSM6567157_PDAC1,PDAC,GSE212966,GSM6567157,"PDAC1, scRNAseq",Homo sapiens,PDAC,GSM6567157_PDAC1_matrix.mtx.gz
GSM6567169_N1,adjacent_normal,GSE212966,GSM6567169,"Normal1, scRNAseq",Homo sapiens,normal,GSM6567169_N1_matrix.mtx.gz
```

Exclusion sentinels — a sample is **skipped** when its group is empty, `nan`, or starts with an
underscore. The named markers are `_excluded`, `_review`, `_non_expression`.

A file missing `sample`/`group` is ignored with a warning and the folder-derived group is used.
`.xlsx` is read only as a fallback and logs a warning, because Excel silently rewrites
identifiers (`SEPT9` → `2-Sep`, long GSM strings → scientific notation) and a mangled sample id
assigns a donor to the wrong arm.

---

## 7. `GSE*_metadata.json` (optional)

Auto-detected by glob, in order: `GSE*_metadata.json`, `*_metadata.json`, `GSE*.json` —
inside `single_10x_dir` for single mode, inside `multi_base_dir` for multi mode. Set
`geo_json_path` to point elsewhere (single mode needs this when the JSON sits at a cohort root
above the sample folder).

Fields actually read: `taxon` → species; `title` / `summary` / sample titles → tissue inferred by
the LLM. Species and tissue then select the CellTypist model and SingleR reference. Without this
file, set `tissue:` and `species:` explicitly in `config.yaml`.

---

## 8. What the loader produces

```
AnnData
  X      cells × genes, raw counts (CSR)
  obs    barcode      from barcodes.tsv
         sample       sample_label / folder name
         group        group_label / folder name / metadata.csv
  var    feature_id   features.tsv column 1
         gene_symbol  features.tsv column 2 (falls back to column 1)
         feature_type features.tsv column 3, when present
  var_names  gene symbols, made unique
```

Stage 0 pre-flight validation then runs on this object before any processing, and writes
`<analysis_name>_data_validation.txt`.

---

## 9. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: No matrix.mtx[.gz] file found in …` | trio incomplete, or files sit one level deeper | point at the folder that directly contains the three files |
| `FileNotFoundError: No group folders found in: …` | `multi_base_dir` has no sub-directories | use the `<GROUP>/<SAMPLE>/` layout |
| `ValueError: … contains negative values` / `non-integer` | normalized, log-transformed or scaled matrix | export raw counts from Cell Ranger (or the counts layer of your object) |
| `ValueError: Failed to read matrix file …` | truncated or non-Matrix-Market file | re-download the sample |
| Sample silently absent from results | its `group` in `metadata.csv` is blank or starts with `_` | correct the group label |
| All samples land in one group | group column missing from `metadata.csv`, so folder names were used | add `sample`,`group` columns |

---

## 10. Regenerating the figure

```bash
python scripts/make_input_format_figure.py     # -> docs/figures/input_formats.png
```
