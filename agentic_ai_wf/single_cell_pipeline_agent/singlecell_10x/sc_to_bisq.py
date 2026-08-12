"""
sc_to_bisq.py — export a processed AnnData into a Bisque-ready ``.h5ad``.

Bisque bulk-deconvolution needs a *single-cell reference* that is small and clean:
raw counts in ``.X``, a ``celltype`` label and a ``sample`` id per cell, and nothing
else. The processed pipeline object, by contrast, carries ``.raw``, a ``counts``
layer, PCA/UMAP embeddings, neighbor graphs, and large ``.uns`` blobs — all of which
bloat the file and confuse deconvolution.

``prepare_for_bisque`` returns a *copy* stripped down to exactly what Bisque needs
(preferring the untouched ``counts`` layer for ``.X``), and ``process_h5ad_file``
wraps it as read -> prepare -> write. Both are importable; the module is also a small
CLI (``python -m ...sc_to_bisq --input file.h5ad --run``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import scanpy as sc
import scipy.sparse as sp

from .config_cli import logger

if TYPE_CHECKING:  # for type checkers only; not needed at runtime
    from anndata import AnnData


# ----------------------------------------------------------
# 1) Argument parser
# ----------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the Bisque-export CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare a Scanpy h5ad file for Bisque deconvolution."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input h5ad file."
    )
    parser.add_argument(
        "--run", action="store_true", help="Run the Bisque preparation pipeline."
    )
    return parser.parse_args()


# ----------------------------------------------------------
# 2) Function: Prepare AnnData for Bisque
# ----------------------------------------------------------
def prepare_for_bisque(adata_in: "AnnData") -> "AnnData":
    """
    Create a *copy* of the AnnData object with only what Bisque needs:
      - adata.X: raw counts (preferred) or normalized matrix
      - adata.obs['celltype']
      - adata.obs['sample']
      - adata.var_names
    """
    adata = adata_in.copy()

    # ---- Use RAW COUNTS for Bisque (deconvolution expects counts, not log-norm) ----
    # Prefer the untouched 'counts' layer. adata.raw holds log-normalized data
    # (and can be scale-corrupted), so it is only a last-resort fallback.
    if "counts" in adata.layers:
        logger.info(
            "Using adata.layers['counts'] (raw counts) as main matrix for Bisque."
        )
        adata.X = adata.layers["counts"].copy()
    elif getattr(adata, "raw", None) is not None:
        logger.warning(
            "No 'counts' layer; falling back to adata.raw.X (log-normalized, NOT counts)."
        )
        adata.X = adata.raw.X.copy()
    else:
        logger.warning("No 'counts' layer and adata.raw is None — using adata.X as-is.")

    # ---- Required columns ----
    for col in ["celltype", "sample"]:
        if col not in adata.obs.columns:
            logger.warning("obs['%s'] missing!", col)
        else:
            logger.info(
                "obs['%s'] present, %s unique values", col, adata.obs[col].nunique()
            )

    # ---- Remove raw layer ----
    if getattr(adata, "raw", None) is not None:
        logger.info("Removing adata.raw...")
        adata.raw = None

    # ---- Remove embeddings ----
    if adata.obsm:
        logger.info("Clearing obsm keys: %s", list(adata.obsm.keys()))
        adata.obsm.clear()

    if adata.varm:
        logger.info("Clearing varm keys: %s", list(adata.varm.keys()))
        adata.varm.clear()

    # ---- Remove graphs ----
    if hasattr(adata, "obsp"):
        for key in ["connectivities", "distances"]:
            if key in adata.obsp:
                logger.info("Removing obsp['%s']", key)
                del adata.obsp[key]

    # ---- Remove unnecessary var columns ----
    if "highly_variable" in adata.var.columns:
        logger.info("Dropping var['highly_variable']")
        del adata.var["highly_variable"]

    # ---- Clean obs ----
    for col in ["leiden", "pct_counts_mt", "total_counts"]:
        if col in adata.obs.columns:
            logger.info("Dropping obs['%s']", col)
            del adata.obs[col]

    # ---- Clear uns ----
    if adata.uns:
        logger.info("Clearing uns keys: %s", list(adata.uns.keys()))
        adata.uns.clear()

    # ---- Clear layers (counts already copied into .X) ----
    if adata.layers:
        logger.info("Clearing layers keys: %s", list(adata.layers.keys()))
        adata.layers.clear()

    # ---- Ensure CSR format ----
    if sp.issparse(adata.X):
        logger.info("Converting X to CSR format...")
        adata.X = adata.X.tocsr()

    logger.info(
        "[FINAL] Bisque-ready shape: %s cells × %s genes", adata.n_obs, adata.n_vars
    )
    return adata


# ----------------------------------------------------------
# 3) Main processing function (can be called directly)
# ----------------------------------------------------------
def process_h5ad_file(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    return_adata: bool = False,
) -> Optional["AnnData"]:
    """
    Process an h5ad file to prepare it for Bisque deconvolution.

    Parameters
    ----------
    input_file : str or Path
        Path to input h5ad file.
    output_file : str or Path, optional
        Path to output h5ad file. If None, will be generated automatically
        as "bisque_ready_<input_filename>".
    return_adata : bool, default False
        If True, return the processed AnnData object instead of saving to file.

    Returns
    -------
    AnnData or None
        If return_adata=True, returns the processed AnnData object.
        Otherwise, returns None and saves to file.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading input AnnData from: %s", input_path)
    try:
        adata_full = sc.read_h5ad(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read h5ad file {input_path}: {e}") from e
    logger.info("Loaded %s cells × %s genes", adata_full.n_obs, adata_full.n_vars)
    # Generate output filename if not provided
    if output_file is None:
        output_path = input_path.with_name("bisque_ready_" + input_path.name)
    else:
        output_path = Path(output_file)

    if not return_adata:
        logger.info("Output file will be saved as: %s", output_path)

    # Prepare copy
    adata_bisque = prepare_for_bisque(adata_full)

    if return_adata:
        logger.info("Done. Returning AnnData object.")
        return adata_bisque

    # Save output
    logger.info("Saving Bisque-ready AnnData to: %s", output_path)
    adata_bisque.write_h5ad(output_path, compression="gzip")

    # Show file sizes
    full_size = os.path.getsize(input_path) / (1024**3)
    bisque_size = os.path.getsize(output_path) / (1024**3)

    logger.info(
        "Done. Original file size: %s GB | Bisque-ready size: %s GB",
        format(full_size, ".2f"),
        format(bisque_size, ".2f"),
    )

    return None


# ----------------------------------------------------------
# 4) CLI entry point
# ----------------------------------------------------------
def main() -> None:
    """CLI: read ``--input`` h5ad and, when ``--run`` is given, write the Bisque-ready copy."""
    args = parse_args()
    if not args.run:
        logger.info("Nothing to do: pass --run to produce the Bisque-ready h5ad.")
        return
    process_h5ad_file(input_file=args.input)


if __name__ == "__main__":
    main()
