# Multi-Method Consensus Cell-Type Annotation

Extends the single-cell pipeline from **CellTypist-only** to a **consensus** of
three independent annotators — **CellTypist**, **SingleR**, and an **LLM
annotator** — reconciled into one label per Leiden cluster with a confidence
tier and full provenance. Fully **disease-agnostic**.

## Why
CellTypist alone is reliable for **immune** cells but forces **non-immune** cells
(epithelial / stromal / endothelial) into immune classes, because immune-focused
models lack non-immune categories. Adding a broad reference method (SingleR) and
a knowledge method (LLM over marker genes), then voting, produces correct labels
across every lineage.

## Architecture (tools vs agent — enforced)
| Module | Layer | Contents |
|---|---|---|
| `config.py` | config | `.env` loader; fail-fast validation |
| `tools.py` | **logic only** | markers, CellTypist, SingleR bridge, harmonize, lineage gate, vote counting. **Zero LLM/HTTP calls.** |
| `agent.py` | **agent only** | LLM annotator + LLM adjudicator (the only OpenRouter calls in the package) |
| `consensus.py` | orchestrator | `run_consensus_annotation(...)` — the single entry point |
| `singler_bridge.R` | R subprocess | SingleR + celldex reference |

Prove the separation:
```bash
grep -Ein "openrouter|chat/completions|requests\.post|import openai" tools.py   # -> no hits
```

## Invariants
1. **Disease-agnostic.** No function or prompt receives a disease name. Only
   `tissue` (organ context — cell biology) is passed to annotators. There is no
   `disease` field anywhere in `ConsensusConfig`.
2. **Conservation of cells.** `broadcast_and_validate` asserts
   `adata.n_obs` in == out and that every cell has a `celltype_consensus` label
   (may be `Unassigned`, but always counted).
3. **No silent failures.** CellTypist abstains-with-log on failure; the LLM
   retries then degrades with an explicit log; SingleR raises with captured
   `stderr` on non-zero exit. No bare `except`.
4. **LLM confined to the agent layer** (see grep above).

## Pipeline stages
`markers (S2)` → `CellTypist (S3)` → `SingleR (S4, optional)` → `LLM annotator
(S5)` → `harmonize to controlled vocabulary (S6)` → `lineage sanity gate (S7)` →
`reconcile (S8)` → `broadcast / validate / export (S9)`.
Stages 0–1 (load, QC, normalization, clustering) are provided by the host
pipeline; this package plugs in at annotation time using the existing Leiden
clusters.

## Reconciliation rules (Stage 8)
Per cluster, over the methods that produced a usable harmonized label:
- **All agree** and not contradicted by the lineage gate → that label, tier **High**.
- **Strict majority** and not contradicted → majority label, tier **Medium**.
- **Disagreement, OR majority contradicted by the lineage gate** → routed to the
  **LLM adjudicator**; tier **Low/Review**. If the LLM is disabled/unavailable, it
  falls back to the majority label (or the lineage flag) and is marked Low/Review.

The **lineage gate** (canonical pan-lineage markers: `PTPRC`, `EPCAM`, `COL1A1`,
`PECAM1`, `RGS5`, …) is universal cell biology, so it preserves disease-agnosticism
while catching contradictions (e.g. an EPCAM-high cluster voted "T cell").

## Output
Written under `05_celltype_analysis/celltype_annotation/`:
- `<name>_consensus_annotation.csv` — one row per cluster:
  `cluster, celltypist, singler, llm, harmonized_agreement, lineage_gate,
  consensus, tier, markers_empty, provenance, top_markers`.
- `<name>_consensus_run.log` — every failure/fallback that occurred.

New `adata.obs` columns: `lineage_coarse`, `celltype_celltypist`,
`celltype_singler` (if enabled), `celltype_llm`, `celltype_consensus`,
`consensus_tier`, `annotation_provenance`. Downstream steps read `celltype`
(mapped from `celltype_consensus`) unchanged.

## Configuration (`.env`)
```
OPENROUTER_API_KEY=sk-or-...        # required when the LLM voter is enabled
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet   # no model is hardcoded — you set it
OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1/chat/completions   # optional
LLM_MAX_RETRIES=3                   # optional
LLM_TIMEOUT_S=60                    # optional
LLM_TEMPERATURE=0                   # optional
CONSENSUS_ENABLE_SINGLER=false      # set true to enable the SingleR voter (needs R)
RSCRIPT_EXE=Rscript                 # optional path to Rscript
```
If `enable_llm=True` but the key/model is missing, `load_config` raises
`ConsensusConfigError`. In the pipeline this is caught and the run continues with
the LLM voter disabled (logged); when called directly it fails fast, per spec.

## Dependencies
- Python: `scanpy`, `celltypist`, `scipy`, `pandas`, `numpy`, `requests`
  (+ `python-dotenv` optional for `.env`).
- SingleR voter (optional): R ≥ 4.x with `SingleR`, `celldex`, `SummarizedExperiment`, `Matrix`
  (`BiocManager::install(c("SingleR","celldex","SummarizedExperiment"))`).

## How to run
**Inside the pipeline** — automatic. When no curated cell-type labels are found,
`run_scanpy_pipeline` calls the consensus annotator (with `.env` configured, all
enabled voters run; without a key, CellTypist runs alone and it's logged).

**Standalone** on an existing clustered `.h5ad`:
```bash
python -m agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.consensus \
    --h5ad path/to/processed.h5ad --tissue "lung" --out-dir ./consensus_out
# add --enable-singler to run SingleR, --no-llm to skip the LLM voter
```

## Disease-agnostic proof
Changing/removing the disease or `group` label does not alter any annotation
output: the disease is never read by markers, models, references, prompts, voting,
or tie-breaking. Tissue may change model/reference selection — that is cell
biology, not disease.

## Note for later
`scGPT` can be added as a **4th voter** with no architecture change: implement a
`tools.annotate_scgpt(...)` returning `{cluster: (label, score)}`, add it to the
orchestrator's method dict before harmonization, and it automatically feeds
Stage 6 (harmonize) and Stage 8 (vote). Not included in this pass.
