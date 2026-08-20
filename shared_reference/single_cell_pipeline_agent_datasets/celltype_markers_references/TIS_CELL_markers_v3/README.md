# Cell-type marker reference for annotation (human)

Disease-agnostic canonical cell-type marker panels for marker-based single-cell
annotation (e.g. `sc.tl.score_genes`, Seurat `AddModuleScore`, AUCell, or as a
prior/verifier layer in a consensus annotator). Markers are **ranked best-first**
within each cell type, so you can take the top-N you need.

## Source
All markers are from **PanglaoDB** (expert-curated), 27-Mar-2020 release, human-relevant
entries only (`species` contains `Hs`).
Franzen O, Gan L-M, Bjorkegren JLM. *PanglaoDB: a web server for exploration of mouse and
human single-cell RNA sequencing data.* Database (2019). doi:10.1093/database/baz046

Only PanglaoDB is used, so every row is expert-curated with human specificity/sensitivity
metrics. A `source` column is included so you can merge in additional databases
(CellMarker 2.0, Azimuth) later and still filter back to this curated core.

## The "100 per cell type" caveat (important)
Most cell types do **not** have 100 robustly validated markers — the biological reality is
that padding a list to a fixed 100 injects weak, non-specific genes that *reduce* annotation
specificity. So this reference is **"up to 100 markers per cell type, ranked by quality."**

Coverage across 178 human cell types / 30 tissue groups:
- Median markers per cell type: **32**
- Cell types reaching the 100 cap: **15** (e.g. Neurons, Interneurons, Endothelial cells,
  Fibroblasts, Macrophages, Hepatocytes, Enterocytes, Dendritic cells, Cardiomyocytes)
- Total ranked marker rows: **7,135**

Per-cell-type counts are in `celltype_marker_counts.csv`.

## Ranking logic (`rank_in_celltype`)
Within each cell type, markers are sorted:
1. **Canonical markers first** (`canonical_marker = Y`)
2. then **higher `sensitivity_human`** (fraction of cells of that type expressing the gene)
3. then **lower `specificity_human`** (fraction of *other* cell types expressing it — lower = more specific)

`marker_score = sensitivity_human * (1 - specificity_human)`, rounded to 4 dp, is provided as a
single convenience score (higher = better). Rank uses the ordered keys above, not the score,
so canonical status is always respected.

## Files

| File | What it is |
|---|---|
| `master_celltype_markers_long.csv` | Single source of truth — every (cell_type, gene) row with all metadata + rank. |
| `celltype_markers/` (178 CSVs) | One file per cell type, ranked markers. Filenames slugged; original name is in the `cell_type`/content. |
| `tissue_markers/` (30 CSVs) | One file per tissue/organ, all its cell types + markers. |
| `celltype_marker_counts.csv` | Markers-per-cell-type coverage table (with file names). |
| `celltype_markers_dict.json` | `{cell_type: [genes ranked best-first]}` for direct loading in Python. |
| `tissue_celltype_markers_dict.json` | `{tissue: {cell_type: [genes]}}` nested version. |

### Column definitions (master)
- `organ` — PanglaoDB organ/tissue group (`Unspecified` where PanglaoDB had none)
- `germ_layer` — developmental germ layer
- `cell_type` — cell type name (annotation label)
- `rank_in_celltype` — 1 = best marker (see ranking logic)
- `gene_symbol` — HGNC-style uppercase human symbol
- `canonical_marker` — `Y` if a classical defining marker, else `N`
- `sensitivity_human` — within-type detection rate (0-1, higher better)
- `specificity_human` — fraction of other types expressing it (0-1, lower better)
- `ubiquitousness_index` — how broadly expressed across clusters (lower = more restricted)
- `marker_score` — `sensitivity_human * (1 - specificity_human)`
- `gene_aliases`, `product_description`, `source`

## Usage

**Python / scanpy** — score every cell type from the dict:
```python
import json, scanpy as sc
markers = json.load(open("celltype_markers_dict.json"))
for ct, genes in markers.items():
    genes = [g for g in genes[:50] if g in adata.var_names]   # top-50, present in data
    if genes:
        sc.tl.score_genes(adata, genes, score_name=f"score_{ct}")
```

**Canonical-only or top-N from the master table:**
```python
import pandas as pd
m = pd.read_csv("master_celltype_markers_long.csv")
canonical = m[m.canonical_marker == "Y"]                       # strict set
top20 = m[m.rank_in_celltype <= 20]                            # top-20 per type
brain  = m[m.organ == "Brain"]                                 # tissue-restricted
```

**Tissue-restricted annotation** (only score cell types plausible for the tissue): use
`tissue_celltype_markers_dict.json[<tissue>]` or a `tissue_markers/<Tissue>.csv` file.

## Extending toward broader body coverage / higher counts
To push thin cell types toward 100 or add finer tissues (PanglaoDB has 30 organ groups):
merge **CellMarker 2.0** (656 tissues, 2,578 cell types) or **Azimuth** references, tag them in
the `source` column, and re-rank. Keep PanglaoDB canonical markers at the top so quality
ordering is preserved.
