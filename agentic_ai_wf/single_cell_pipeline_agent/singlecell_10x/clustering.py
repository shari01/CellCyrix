"""
clustering.py — Leiden clustering: configuration, execution, and resolution audit.

The pipeline's structural partition is ALWAYS Scanpy Leiden (never CellTypist's
internal over-clustering, never SingleR). This module owns three concerns that
used to be a single hard-coded line in ``pipeline.py``:

  * ``resolve_leiden_resolution``  — validate a config value into a positive float
    (missing/None -> the historical default 0.5, so old configs keep working).
  * ``run_leiden``                 — the primary clustering call. Deliberately
    argument-identical to the historical
    ``sc.tl.leiden(adata, resolution=..., random_state=seed)`` so cluster IDs do
    not shift for an unchanged resolution.
  * ``evaluate_leiden_resolutions`` — an OPTIONAL, non-destructive sweep over
    candidate resolutions on the SAME neighbor graph, writing a diagnostics CSV.

Ordering guarantee: the caller runs the primary resolution FIRST and the sweep
afterwards into separate ``leiden_res_<r>`` columns. ``adata.obs['leiden']`` is
therefore never overwritten by the audit, and the primary clustering is bit-wise
unaffected by whether the audit ran at all. No automatic re-selection happens:
the user-configured resolution stays primary unless a future, explicit selection
rule is added (the CSV records ``selection_rule`` so this is auditable).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc

from .atomic_io import atomic_to_csv
from .exceptions import PipelineInputError
from .scanpy_params import NEIGHBORS_N_PCS

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

# Historical pipeline default — preserved so configs without a clustering block
# produce byte-identical clusterings.
DEFAULT_LEIDEN_RESOLUTION = 0.5

# Candidate grid used when `evaluate_resolutions` is on but no list is given.
DEFAULT_RESOLUTION_CANDIDATES: tuple[float, ...] = (0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

# A Leiden cluster smaller than this is counted as "too small to annotate".
DEFAULT_MIN_CLUSTER_CELLS = 20

# Silhouette is O(n^2) in memory; above this many cells we subsample (seeded).
DEFAULT_SILHOUETTE_MAX_CELLS = 5000

# Number of PCs scored by the silhouette metric. Aliased to the neighbours setting
# rather than restated: scoring a different number of PCs than the graph was built on
# would silently measure a different embedding than the one being clustered.
SILHOUETTE_N_PCS = NEIGHBORS_N_PCS

PRIMARY_CLUSTER_COL = "leiden"

EVALUATION_CSV_NAME = "leiden_resolution_evaluation.csv"


class ClusteringConfigError(PipelineInputError):
    """Raised when a clustering configuration value is not usable."""


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def resolve_leiden_resolution(
    value: object = None,
    *,
    name: str = "leiden_resolution",
    default: float = DEFAULT_LEIDEN_RESOLUTION,
) -> float:
    """Validate a Leiden resolution from config into a positive finite float.

    ``None`` / missing / empty-string -> ``default`` (so a config file written
    before the clustering block existed keeps working unchanged). Anything that
    is present but NOT a positive finite number raises ``ClusteringConfigError``
    rather than being silently coerced — a typo'd resolution would otherwise
    change every downstream cluster without a trace.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default)
    if isinstance(value, bool):  # bool is an int subclass; almost surely a mistake
        raise ClusteringConfigError(
            f"{name} must be a positive number, got boolean {value!r}."
        )
    try:
        res = float(value)
    except (TypeError, ValueError) as e:
        raise ClusteringConfigError(
            f"{name} must be a positive number, got {value!r}."
        ) from e
    if not np.isfinite(res) or res <= 0:
        raise ClusteringConfigError(
            f"{name} must be a positive finite number, got {res!r}."
        )
    return res


def resolution_column(resolution: float) -> str:
    """Stable obs-column name for a swept resolution: 0.2 -> ``leiden_res_0_2``."""
    s = f"{float(resolution):.4f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return f"leiden_res_{s.replace('.', '_')}"


def resolve_resolution_candidates(
    candidates: Optional[Iterable] = None,
    *,
    primary: float = DEFAULT_LEIDEN_RESOLUTION,
) -> List[float]:
    """Validate/normalize the candidate grid: sorted, de-duplicated, primary included.

    Each entry goes through ``resolve_leiden_resolution`` so an invalid candidate
    fails loudly. The primary resolution is always appended (an audit that omits
    the resolution actually in use would be useless for comparison).
    """
    raw: Sequence = (
        list(candidates) if candidates else list(DEFAULT_RESOLUTION_CANDIDATES)
    )
    out: List[float] = []
    for i, c in enumerate(raw):
        out.append(resolve_leiden_resolution(c, name=f"resolution_candidates[{i}]"))
    out.append(float(primary))
    # de-duplicate on the column name so 0.5 and 0.50 collapse to one run
    seen, uniq = set(), []
    for r in sorted(out):
        key = resolution_column(r)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def resolve_positive_int(value: object = None, *, name: str, default: int) -> int:
    """Validate a positive-integer config value (missing -> ``default``)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return int(default)
    if isinstance(value, bool):
        raise ClusteringConfigError(
            f"{name} must be a positive integer, got boolean {value!r}."
        )
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ClusteringConfigError(
            f"{name} must be a positive integer, got {value!r}."
        ) from e
    if n <= 0:
        raise ClusteringConfigError(f"{name} must be > 0, got {n}.")
    return n


# ---------------------------------------------------------------------------
# Primary clustering
# ---------------------------------------------------------------------------
def run_leiden(
    adata: AnnData,
    *,
    resolution: float,
    seed: int,
    key_added: str = PRIMARY_CLUSTER_COL,
) -> str:
    """Run Leiden on the existing neighbor graph; return the obs column written.

    Kept argument-for-argument identical to the historical call so that an
    unchanged resolution reproduces the previous cluster IDs exactly. ``key_added``
    is only passed when it differs from Scanpy's default, because passing it
    explicitly is not guaranteed to be a no-op across Scanpy versions.
    """
    if key_added == PRIMARY_CLUSTER_COL:
        sc.tl.leiden(adata, resolution=resolution, random_state=seed)
    else:
        sc.tl.leiden(
            adata, resolution=resolution, random_state=seed, key_added=key_added
        )
    n = int(adata.obs[key_added].nunique())
    logger.info(
        "[CLUSTERING] leiden(resolution=%s) -> %s clusters in obs['%s'].",
        resolution,
        n,
        key_added,
    )
    return key_added


# ---------------------------------------------------------------------------
# Optional resolution audit
# ---------------------------------------------------------------------------
def _cluster_size_stats(
    labels: pd.Series, min_cluster_cells: int
) -> dict[str, float | int]:
    """Cluster-count and size distribution for one clustering."""
    sizes = labels.value_counts()
    return {
        "n_clusters": int(sizes.size),
        "min_cluster_size": int(sizes.min()) if sizes.size else 0,
        "median_cluster_size": float(sizes.median()) if sizes.size else 0.0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "n_clusters_below_min_cells": int((sizes < min_cluster_cells).sum()),
        "n_cells_in_small_clusters": int(sizes[sizes < min_cluster_cells].sum()),
    }


def _agreement(prev: Optional[pd.Series], cur: pd.Series) -> dict[str, float | None]:
    """ARI/NMI of a clustering against the previous (adjacent) resolution."""
    if prev is None:
        return {"ari_vs_previous": None, "nmi_vs_previous": None}
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError as e:
        logger.warning(
            "[CLUSTERING] scikit-learn unavailable (%s); ARI/NMI skipped.",
            e,
            exc_info=True,
        )
        return {"ari_vs_previous": None, "nmi_vs_previous": None}
    a = prev.astype(str).to_numpy()
    b = cur.astype(str).to_numpy()
    return {
        "ari_vs_previous": float(adjusted_rand_score(a, b)),
        "nmi_vs_previous": float(normalized_mutual_info_score(a, b)),
    }


def _silhouette(
    adata, labels: pd.Series, *, seed: int, max_cells: int
) -> Optional[float]:
    """Silhouette score in PCA space, seeded-subsampled when the data is large.

    Returns None (with a logged reason) when it cannot be computed — a missing
    score is reported as missing, never as 0.
    """
    if "X_pca" not in adata.obsm:
        logger.info("[CLUSTERING] no X_pca; silhouette skipped.")
        return None
    if labels.nunique() < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        logger.warning(
            "[CLUSTERING] scikit-learn unavailable (%s); silhouette skipped.",
            e,
            exc_info=True,
        )
        return None

    X = np.asarray(adata.obsm["X_pca"])[:, :SILHOUETTE_N_PCS]
    y = labels.astype(str).to_numpy()
    if X.shape[0] > max_cells:
        rng = np.random.default_rng(seed)  # seeded -> reproducible subsample
        idx = rng.choice(X.shape[0], size=max_cells, replace=False)
        X, y = X[idx], y[idx]
        if len(set(y.tolist())) < 2:
            return None
    try:
        return float(silhouette_score(X, y))
    except ValueError as e:  # e.g. a single label survived the subsample
        logger.info("[CLUSTERING] silhouette not computable (%s).", e)
        return None


def evaluate_leiden_resolutions(
    adata: AnnData,
    *,
    primary_resolution: float,
    candidates: Optional[Iterable] = None,
    seed: int = 0,
    out_dir: Optional[Path] = None,
    analysis_name: str = "analysis",
    min_cluster_cells: int = DEFAULT_MIN_CLUSTER_CELLS,
    silhouette_max_cells: int = DEFAULT_SILHOUETTE_MAX_CELLS,
    primary_cluster_col: str = PRIMARY_CLUSTER_COL,
) -> pd.DataFrame:
    """Sweep candidate Leiden resolutions on the existing graph; write a CSV.

    NON-DESTRUCTIVE: each candidate is written to its own ``leiden_res_<r>``
    column and ``adata.obs[primary_cluster_col]`` is never touched. The primary
    resolution is expected to have been clustered already by ``run_leiden``.

    Returns the diagnostics DataFrame (also written to
    ``<out_dir>/leiden_resolution_evaluation.csv`` when ``out_dir`` is given).
    """
    grid = resolve_resolution_candidates(candidates, primary=primary_resolution)
    logger.info(
        "[CLUSTERING] resolution audit over %s (primary=%s, non-destructive; '%s' untouched).",
        grid,
        primary_resolution,
        primary_cluster_col,
    )

    rows: List[dict] = []
    prev_labels: Optional[pd.Series] = None
    prev_res: Optional[float] = None

    for res in grid:
        col = resolution_column(res)
        is_primary = resolution_column(primary_resolution) == col
        # Reuse the primary labels for the primary resolution instead of
        # re-clustering: identical result, and it keeps the audit honest about
        # what the pipeline actually used.
        if is_primary and primary_cluster_col in adata.obs.columns:
            labels = adata.obs[primary_cluster_col].astype(str)
            adata.obs[col] = pd.Categorical(labels)
        else:
            run_leiden(adata, resolution=res, seed=seed, key_added=col)
            labels = adata.obs[col].astype(str)

        row = {"resolution": float(res), "obs_column": col}
        row.update(_cluster_size_stats(labels, min_cluster_cells))
        row.update(_agreement(prev_labels, labels))
        row["compared_to_resolution"] = prev_res
        row["silhouette_pca"] = _silhouette(
            adata, labels, seed=seed, max_cells=silhouette_max_cells
        )
        row["is_primary_resolution"] = bool(is_primary)
        row["min_cluster_cells_threshold"] = int(min_cluster_cells)
        rows.append(row)

        prev_labels, prev_res = labels, float(res)

    df = pd.DataFrame(rows)
    df["selected_primary_resolution"] = float(primary_resolution)
    # Explicit and auditable: nothing is auto-selected.
    df["selection_rule"] = "user_configured (no automatic re-selection)"

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / EVALUATION_CSV_NAME
        atomic_to_csv(df, path, index=False)
        logger.info("[CLUSTERING] wrote resolution audit: %s", path)

    for r in rows:
        logger.info(
            "[CLUSTERING] res=%s: %s clusters (min=%s, median=%s, max=%s, <%s cells: %s), ARI vs prev=%s, silhouette=%s",
            r["resolution"],
            r["n_clusters"],
            r["min_cluster_size"],
            r["median_cluster_size"],
            r["max_cluster_size"],
            min_cluster_cells,
            r["n_clusters_below_min_cells"],
            r["ari_vs_previous"],
            r["silhouette_pca"],
        )
    return df
