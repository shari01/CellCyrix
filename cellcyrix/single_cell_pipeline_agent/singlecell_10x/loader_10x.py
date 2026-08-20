"""
loader_10x.py — read a 10x Genomics feature-barcode matrix into an AnnData.

Handles the standard Cell Ranger trio — ``matrix.mtx``, ``barcodes.tsv``,
``features.tsv``/``genes.tsv`` — with or without ``.gz`` compression and with the
common filename variants (bare or sample-prefixed). The Matrix Market matrix is
read (transposed to cells x genes), barcodes populate ``.obs`` and gene ids/symbols
populate ``.var`` (with ``var_names`` made unique). Missing files raise a clear
``FileNotFoundError``; unreadable/corrupt files raise a ``ValueError`` naming the
offending path, so loading failures are diagnosable rather than opaque.
"""

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from scipy import io as spio
from scipy import sparse as sp_sparse

from .config_cli import logger

if TYPE_CHECKING:  # anndata is imported lazily in the function body, not at module load
    import anndata


def _find_first_matching(dir_path: Path, patterns) -> Path | None:
    """
    Return the first file in dir_path matching any of the glob patterns in `patterns`.
    """
    for pat in patterns:
        matches = sorted(dir_path.glob(pat))
        if matches:
            return matches[0]
    return None


def _mmread_auto(path: Path):
    """
    Read a Matrix Market file, supporting optional .gz compression.
    """
    path = Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return spio.mmread(f)
    else:
        return spio.mmread(str(path))


def _validate_raw_counts(M, source_path: Path) -> None:
    """
    Fail fast if the loaded matrix is not a valid raw count matrix.

    A 10x feature-barcode matrix is expected to be finite, non-negative and
    integer-valued. Silently accepting a normalized / scaled / imputed / corrupt
    matrix here would let it propagate into ``layers['counts']`` and then into
    Scrublet, Seurat-v3 HVG, DESeq2 pseudobulk and the Bisque export — all of
    which assume genuine counts. We raise a clear ``ValueError`` instead of
    "repairing" the data.
    """
    import numpy as np

    data = getattr(M, "data", None)
    if data is None:
        data = np.asarray(M).ravel()
    data = np.asarray(data)
    if data.size == 0:
        return  # all-zero / empty matrix — nothing to validate

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"{source_path} is not a valid raw-count matrix: it contains "
            f"non-finite values (NaN/Inf). Expected finite, non-negative, "
            f"integer counts from Cell Ranger (or equivalent)."
        )
    if float(data.min()) < 0:
        raise ValueError(
            f"{source_path} is not a valid raw-count matrix: it contains "
            f"negative values (min={float(data.min()):.4g}). This looks like "
            f"normalized/scaled data, not raw counts."
        )
    # integer-like: values may be stored as float but must round-trip to ints.
    if not np.allclose(data, np.rint(data), rtol=0, atol=1e-8):
        frac = float(np.mean(~np.isclose(data, np.rint(data), rtol=0, atol=1e-8)))
        raise ValueError(
            f"{source_path} is not a valid raw-count matrix: {frac:.1%} of "
            f"stored values are non-integer. This looks like normalized/log/"
            f"scaled data, not raw counts. Load counts, or route processed data "
            f"through a different entry point."
        )


def load_10x_feature_barcode_matrix(tenx_dir: Path) -> "anndata.AnnData":
    """
    Load a single 10x feature-barcode matrix from a folder containing:
      - matrix.mtx[.gz]
      - barcodes.tsv[.gz]
      - features.tsv/genes.tsv[.gz]

    Returns
    -------
    AnnData
        Cells x genes matrix with barcodes in .obs and gene info in .var.
    """
    # local import to avoid circular dependencies
    import anndata as ad

    tenx_dir = Path(tenx_dir)
    if not tenx_dir.exists():
        raise FileNotFoundError(f"10X folder not found: {tenx_dir}")

    logger.info("Loading 10X feature-barcode matrix from: %s", tenx_dir)

    matrix_path = _find_first_matching(
        tenx_dir,
        [
            "matrix.mtx",
            "matrix.mtx.gz",
            "*.matrix.mtx",
            "*.matrix.mtx.gz",
            "*.mtx",
            "*.mtx.gz",
        ],
    )
    if matrix_path is None:
        raise FileNotFoundError(f"No matrix.mtx[.gz] file found in {tenx_dir}.")

    barcodes_path = _find_first_matching(
        tenx_dir,
        [
            "barcodes.tsv",
            "barcodes.tsv.gz",
            "*barcodes.tsv",
            "*barcodes.tsv.gz",
            "barcode.tsv",
            "barcode.tsv.gz",
            "*barcode.tsv",
            "*barcode.tsv.gz",
        ],
    )
    if barcodes_path is None:
        raise FileNotFoundError(f"No barcodes.tsv[.gz] file found in {tenx_dir}.")

    features_path = _find_first_matching(
        tenx_dir,
        [
            "features.tsv",
            "features.tsv.gz",
            "*features.tsv",
            "*features.tsv.gz",
            "genes.tsv",
            "genes.tsv.gz",
            "*genes.tsv",
            "*genes.tsv.gz",
        ],
    )
    if features_path is None:
        raise FileNotFoundError(
            f"No features.tsv[.gz] or genes.tsv[.gz] file found in {tenx_dir}."
        )

    logger.info("matrix:   %s", matrix_path.name)
    logger.info("barcodes: %s", barcodes_path.name)
    logger.info("features: %s", features_path.name)

    # Read matrix (guard corrupt/unreadable files with a path-specific error)
    try:
        M = _mmread_auto(matrix_path)
    except Exception as e:
        raise ValueError(f"Failed to read matrix file {matrix_path}: {e}") from e
    if not sp_sparse.issparse(M):
        M = sp_sparse.coo_matrix(M)
    M = M.tocsr()
    # Enforce raw-count integrity before this becomes layers['counts'] downstream.
    _validate_raw_counts(M, matrix_path)
    X = M.T  # cells x genes

    # Read barcodes
    try:
        barcodes_df = pd.read_csv(
            barcodes_path, sep="\t", header=None, compression="infer"
        )
    except Exception as e:
        raise ValueError(f"Failed to read barcodes file {barcodes_path}: {e}") from e
    barcodes = barcodes_df.iloc[:, 0].astype(str).values

    # Read features / genes
    try:
        feat_df = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
    except Exception as e:
        raise ValueError(f"Failed to read features file {features_path}: {e}") from e
    ncols = feat_df.shape[1]
    colnames = []
    if ncols >= 1:
        colnames.append("feature_id")
    if ncols >= 2:
        colnames.append("feature_name")
    if ncols >= 3:
        colnames.append("feature_type")
    while len(colnames) < ncols:
        colnames.append(f"extra_{len(colnames)}")
    feat_df.columns = colnames

    gene_ids = feat_df["feature_id"].astype(str).values
    if "feature_name" in feat_df.columns:
        gene_names = feat_df["feature_name"].astype(str).values
    else:
        gene_names = gene_ids

    # Build AnnData
    adata_ = ad.AnnData(X=X)
    adata_.obs_names = barcodes
    adata_.obs["barcode"] = barcodes
    adata_.var_names = gene_names
    adata_.var["feature_id"] = gene_ids
    adata_.var["gene_symbol"] = gene_names
    if "feature_type" in feat_df.columns:
        adata_.var["feature_type"] = feat_df["feature_type"].astype(str).values
    adata_.var_names_make_unique()

    logger.info("Loaded AnnData from 10x: %s", adata_)
    return adata_
