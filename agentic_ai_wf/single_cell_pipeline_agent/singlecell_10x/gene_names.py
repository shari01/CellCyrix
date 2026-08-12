"""
gene_names.py — gene identifier normalization for the AnnData ``.var`` axis.

Single-cell matrices arrive with either Ensembl gene IDs (``ENSG...``) or HGNC
symbols in ``var_names``. Downstream steps (marker detection, annotation, DE)
expect human-readable symbols, so this module normalizes ``var_names`` to gene
symbols where possible:

  1. prefer an existing symbol column in ``.var`` (``gene_symbol`` / ``symbol`` / ...);
  2. otherwise, if ``var_names`` look like Ensembl IDs and ``mygene`` is installed,
     map Ensembl -> symbol via mygene (chunked, failure-tolerant);
  3. otherwise keep the original ``var_names`` unchanged.

``mygene`` is an optional dependency: if it is not installed the Ensembl mapping
path is skipped with a log line rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from .config_cli import logger

if TYPE_CHECKING:  # import only for type checkers; not needed at runtime
    from anndata import AnnData

# Optional: mygene for Ensembl → gene symbol mapping
try:
    import mygene

    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False


def looks_like_ensembl(ids: Sequence, prefix: str = "ENSG") -> bool:
    """
    Heuristic: check if a list/array of IDs mostly look like Ensembl IDs.
    """
    ids = list(ids)
    if len(ids) == 0:
        return False
    sample = ids[: min(50, len(ids))]
    flags = [str(x).upper().startswith(prefix) for x in sample]
    return np.mean(flags) > 0.6


def update_gene_names(adata_in: "AnnData") -> "AnnData":
    """
    Update adata.var_names to use nicer gene symbols where possible.

    Strategy:
      1. Try known symbol/name columns in .var.
      2. If var_names look like Ensembl IDs and mygene is available,
         map Ensembl → symbol using mygene.
      3. Otherwise, keep original var_names.
    """
    candidate_cols = [
        "gene_symbol",
        "symbol",
        "GeneSymbol",
        "SYMBOL",
        "gene_name",
        "GeneName",
        "name",
    ]
    for col in candidate_cols:
        if col in adata_in.var.columns:
            col_vals = adata_in.var[col].astype(str)
            non_na_ratio = (col_vals != "nan").mean()
            if non_na_ratio > 0.5:
                logger.info("Using adata.var['%s'] as gene names.", col)
                new_names = []
                for old, new in zip(adata_in.var_names, col_vals, strict=True):
                    if new != "nan" and new is not None and len(new) > 0:
                        new_names.append(new)
                    else:
                        new_names.append(old)
                adata_in.var_names = new_names
                adata_in.var_names_make_unique()
                return adata_in

    if MYGENE_AVAILABLE and looks_like_ensembl(adata_in.var_names, prefix="ENSG"):
        logger.info("var_names look like Ensembl – mapping via mygene.")
        mg = mygene.MyGeneInfo()
        gene_ids = adata_in.var_names.tolist()
        chunk_size = 1000
        all_results = []

        for i in range(0, len(gene_ids), chunk_size):
            try:
                chunk = gene_ids[i : i + chunk_size]
                res = mg.querymany(
                    chunk,
                    scopes="ensembl.gene",
                    fields="symbol,name",
                    species="human",
                    as_dataframe=True,
                    df_index=True,
                )
                all_results.append(res)
            except Exception as e:
                logger.warning("mygene chunk %s failed: %s", i, e, exc_info=True)
                continue

        if len(all_results) > 0:
            mapping_df = pd.concat(all_results, axis=0)
            mapping_df = mapping_df[~mapping_df.index.duplicated(keep="first")]
            symbols = []
            names = []
            for gid in adata_in.var_names:
                if gid in mapping_df.index:
                    row = mapping_df.loc[gid]
                    symbols.append(row.get("symbol", np.nan))
                    names.append(row.get("name", np.nan))
                else:
                    symbols.append(np.nan)
                    names.append(np.nan)
            adata_in.var["mapped_symbol"] = pd.Series(
                symbols,
                index=adata_in.var.index,
                dtype="string",
            )
            adata_in.var["mapped_name"] = pd.Series(
                names,
                index=adata_in.var.index,
                dtype="string",
            )
            mapped_sym = adata_in.var["mapped_symbol"]
            valid_mask = mapped_sym.notna() & (mapped_sym != "") & (mapped_sym != "nan")
            mapped_count = int(valid_mask.sum())
            logger.info("Mapped Ensembl → symbol for ~%s genes.", mapped_count)
            new_names = []
            for old, sym, ok in zip(
                adata_in.var_names, mapped_sym, valid_mask, strict=True
            ):
                if ok:
                    new_names.append(str(sym))
                else:
                    new_names.append(old)
            adata_in.var_names = new_names
            adata_in.var_names_make_unique()
        else:
            logger.warning("No successful mygene mapping. Keeping original names.")
    else:
        if not MYGENE_AVAILABLE:
            logger.info("mygene not installed; skipping Ensembl mapping.")
        else:
            logger.info("var_names do not look like Ensembl; skipping mapping.")
    return adata_in
