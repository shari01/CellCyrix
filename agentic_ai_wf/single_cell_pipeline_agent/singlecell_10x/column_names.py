"""
column_names.py — one lower_snake_case vocabulary for the columns of every table.

Why this exists
---------------
The pipeline wrote the SAME quantity under different names depending on which library
produced the frame:

    pseudobulk_de.py (pydeseq2)  log2FoldChange   lfcSE    pvalue   padj    baseMean
    group_de.py      (scanpy)    logfoldchanges   —        pvals    pvals_adj
    pathway_enrichment.py (gseapy) "Adjusted P-value"  "Combined Score"  "Odds Ratio"

So a downstream reader had to know which file it was holding before it could ask for
the fold change, and one of the three spellings has a SPACE in it, which breaks a
column reference in most query tools. Standards Rule 5.4: column names inside data
files are lower_snake_case and consistent across every file a module writes.

How it is applied
-----------------
Renaming is done at the WRITE boundary, not throughout the code. In-memory frames keep
whatever their producing library called things — pydeseq2 hands back `results_df` with
its own names, and rewriting those in place would mean touching every arithmetic
expression in the DE modules for no behavioural gain. `atomic_io.write_table` maps the
names on the way out, so:

  * no computation changes, and no risk of a rename breaking a filter or a sort;
  * every emitted CSV carries the same vocabulary;
  * this mapping is the single place the vocabulary is defined.

The values are untouched. Only headers change.
"""

from __future__ import annotations

import pandas as pd

# Source spelling -> canonical output name. Several sources map to one canonical name
# on purpose: `padj` and `pvals_adj` are the same statistic from two libraries.
CANONICAL_COLUMNS: dict[str, str] = {
    # --- pydeseq2 / DESeq2 ---------------------------------------------------
    "baseMean": "base_mean",
    "log2FoldChange": "log2_fold_change",
    "log2FoldChange_MLE": "log2_fold_change_mle",
    "lfcSE": "lfc_se",
    "lfcSE_MLE": "lfc_se_mle",
    "stat": "wald_stat",
    "pvalue": "p_value",
    "padj": "p_value_adj",
    # --- scanpy rank_genes_groups -------------------------------------------
    "logfoldchanges": "log2_fold_change",
    "pvals": "p_value",
    "pvals_adj": "p_value_adj",
    "names": "gene",
    "scores": "test_statistic",
    # --- gseapy / Enrichr (note the spaces in the source names) -------------
    "Adjusted P-value": "p_value_adj",
    "P-value": "p_value",
    "Old P-value": "p_value_unadjusted_old",
    "Old Adjusted P-value": "p_value_adj_old",
    "Combined Score": "combined_score",
    "Odds Ratio": "odds_ratio",
    "Overlap": "overlap",
    "Genes": "genes",
    "Term": "pathway",
    "Pathways": "pathway",
    "Biological_Database": "biological_database",
    # --- pipeline-internal spellings that drifted ---------------------------
    "log2FC": "log2_fold_change",
    "FDR": "p_value_adj",
    "adj_pval": "p_value_adj",
    "pval_adj": "p_value_adj",
    "gene_symbol": "gene",
}

# Accepted spellings for the quantities readers look up, canonical first. Reader
# candidate lists are built from these so a table written before this rename is still
# readable.
GENE_COLUMNS: tuple[str, ...] = ("gene", "names", "gene_symbol")
LOG2FC_COLUMNS: tuple[str, ...] = (
    "log2_fold_change",
    "log2FoldChange",
    "logfoldchanges",
    "log2FC",
)
PADJ_COLUMNS: tuple[str, ...] = (
    "p_value_adj",
    "padj",
    "pvals_adj",
    "FDR",
    "adj_pval",
    "pval_adj",
)


def to_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename a frame's columns to the canonical output vocabulary.

    Only headers listed in `CANONICAL_COLUMNS` are touched; anything else (including
    the pipeline's own provenance columns, which are already snake_case) is left
    alone. A rename that would collide with a column already present is skipped, so a
    frame carrying both `padj` and `p_value_adj` keeps both rather than losing one.

    Args:
        frame: The table about to be written.

    Returns:
        A frame with canonical headers. The input is not modified.
    """
    mapping = {
        source: target
        for source, target in CANONICAL_COLUMNS.items()
        if source in frame.columns and target not in frame.columns
    }
    if not mapping:
        return frame
    return frame.rename(columns=mapping)


__all__ = [
    "CANONICAL_COLUMNS",
    "GENE_COLUMNS",
    "LOG2FC_COLUMNS",
    "PADJ_COLUMNS",
    "to_canonical_columns",
]
