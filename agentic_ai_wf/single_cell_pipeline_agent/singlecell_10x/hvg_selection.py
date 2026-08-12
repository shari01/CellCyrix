"""
hvg_selection.py — highly-variable-gene selection that survives a degenerate sample.

Why
---
``sc.pp.highly_variable_genes(flavor="seurat_v3", batch_key=...)`` fits a **separate**
LOESS mean-variance curve for **each batch**, with ``span=0.3, degree=2``. A local
quadratic needs enough *distinct* x-values inside every neighbourhood it visits. A
very small library does not have them: in GSE157827 (snRNA-seq brain) sample
``GSM4775565_AD6`` retains only 353 nuclei after QC, so its 15,657 expressed genes
occupy just 452 distinct ``log10(mean)`` values and 3,069 genes land on a single one.
skmisc then aborts the whole call with::

    ValueError: b'There are other near singularities as well. 0.090619'

One 353-cell library out of 21 killed a 158,084-cell run *after* ~15 minutes of
loading, QC and Scrublet. The variance model is fine; the fit is simply undefined at
that sample size.

How
---
:func:`select_hvgs` walks a ladder and stops at the first rung that fits, exactly the
way the pipeline already degrades BBKNN to plain neighbours:

1. ``seurat_v3`` + ``batch_key`` over every batch — **unchanged**, so any cohort that
   works today takes this rung and selects precisely the genes it selected before.
2. ``seurat_v3`` + ``batch_key`` with the batches that cannot be fitted excluded from
   the *gene ranking*. Membership is established by actually fitting each batch, not
   by guessing a cell-count cutoff, so no healthy sample is ever dropped. Excluded
   cells stay in the object and are still clustered, annotated and tested for DE —
   they only lose their vote on which genes are variable.
3. ``seurat_v3`` pooled over all cells. Batch-awareness is lost, but the curve over
   the full cohort is well conditioned.
4. ``seurat`` (dispersion binning on log-normalised data). No LOESS anywhere, so this
   rung cannot raise; it exists so the pipeline can never die here.

The rung that ran, and any batch that was excluded, are returned in the report and
recorded in the provenance manifest — a silently different gene set is worse than a
crash.

Nothing here changes the variance model, the gene count, or the downstream use of
``adata.var['highly_variable']``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import scanpy as sc

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

# scanpy's seurat_v3 LOESS parameters, mirrored here so the per-batch probe in
# rung 2 fails under exactly the same conditions as the real call.
_LOESS_SPAN = 0.3
_LOESS_DEGREE = 2

_HVG_VAR_COLS = (
    "highly_variable",
    "highly_variable_rank",
    "highly_variable_nbatches",
    "highly_variable_intersection",
    "means",
    "variances",
    "variances_norm",
    "dispersions",
    "dispersions_norm",
)


def _batch_fits(adata, batch: str, batch_key: str, layer: Optional[str]) -> bool:
    """Can seurat_v3's LOESS be fitted on this batch alone?

    Replicates scanpy's per-batch computation (counts -> per-gene mean/variance ->
    drop zero-variance genes -> LOESS on log10) and reports whether it converges.
    Any failure to even attempt the probe is reported as "fits", so a missing
    ``skmisc`` degrades to the next rung instead of excluding every batch.
    """
    try:
        from skmisc.loess import loess
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001 - optional dependency, handled by the ladder
        logger.debug("%s: falling back after %r", __name__, exc)
        return True

    sub = adata[adata.obs[batch_key] == batch]
    X = sub.layers[layer] if layer else sub.X
    n = X.shape[0]
    if n < 3:
        return False

    if hasattr(X, "multiply"):  # sparse
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = np.multiply(X, X).mean(axis=0)
    var = (mean_sq - mean**2) * (n / (n - 1))

    nonconst = var > 0
    if nonconst.sum() < 10:
        return False

    try:
        model = loess(
            np.log10(mean[nonconst]),
            np.log10(var[nonconst]),
            span=_LOESS_SPAN,
            degree=_LOESS_DEGREE,
        )
        model.fit()
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001 - this is the condition being probed for
        logger.debug("%s: falling back after %r", __name__, exc)
        return False


def _degenerate_batches(adata, batch_key: str, layer: Optional[str]) -> List[str]:
    """Batches whose own LOESS will not converge, in ``obs`` category order."""
    return [
        str(b)
        for b in adata.obs[batch_key].astype(str).unique()
        if not _batch_fits(adata, str(b), batch_key, layer)
    ]


def _copy_hvg_results(src, dst) -> None:
    """Move the HVG results computed on a subset back onto the full object.

    The subset shares ``var_names`` with the full object (only cells were removed),
    so a plain reindex is exact. Genes absent from the subset cannot be highly
    variable, hence ``highly_variable`` fills with ``False``.

    ``uns['hvg']`` travels too: ``sc.pl.highly_variable_genes`` reads
    ``uns['hvg']['flavor']`` and raises ``KeyError`` without it, so leaving it on the
    discarded subset breaks the plot two lines later in the pipeline.
    """
    for col in _HVG_VAR_COLS:
        if col not in src.var.columns:
            continue
        fill = False if col == "highly_variable" else np.nan
        dst.var[col] = src.var[col].reindex(dst.var_names).fillna(fill)
    dst.var["highly_variable"] = dst.var["highly_variable"].astype(bool)
    if "hvg" in src.uns:
        dst.uns["hvg"] = dict(src.uns["hvg"])


def _stamp_uns(adata, flavor: str) -> None:
    """Guarantee ``uns['hvg']`` matches the rung that actually ran.

    A rung that raises part-way can leave ``uns['hvg']`` behind from an earlier
    attempt, which would mislabel the plot. Every return path stamps it.
    """
    adata.uns["hvg"] = {"flavor": flavor}


def select_hvgs(
    adata: AnnData,
    *,
    n_top_genes: int,
    batch_key: Optional[str] = None,
    layer: Optional[str] = "counts",
    analysis_name: str = "",
) -> Dict[str, object]:
    """Flag highly variable genes on ``adata`` in place; return what actually ran.

    Report keys: ``method`` (the rung), ``flavor``, ``batch_key`` (the one that took
    effect, ``None`` if batch-awareness was given up), ``n_top_genes``,
    ``excluded_batches``, ``n_hvg`` and ``fallback_reason`` (empty on rung 1).
    """
    tag = f"[{analysis_name}] " if analysis_name else ""
    report: Dict[str, object] = {
        "method": "seurat_v3 (per-batch)" if batch_key else "seurat_v3",
        "flavor": "seurat_v3",
        "batch_key": batch_key,
        "n_top_genes": int(n_top_genes),
        "excluded_batches": [],
        "fallback_reason": "",
    }

    # --- Rung 1: unchanged behaviour -------------------------------------------
    try:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            layer=layer,
            **({"batch_key": batch_key} if batch_key else {}),
        )
        report["n_hvg"] = int(adata.var["highly_variable"].sum())
        _stamp_uns(adata, "seurat_v3")
        return report
    except Exception as e:  # noqa: BLE001 - the LOESS singularity, and anything like it
        reason = f"{type(e).__name__}: {e}"
        report["fallback_reason"] = reason
        logger.warning("%s[HVG] seurat_v3 failed (%s).", tag, reason)

    # --- Rung 2: drop only the batches that genuinely cannot be fitted ----------
    if batch_key and batch_key in adata.obs.columns:
        bad = _degenerate_batches(adata, batch_key, layer)
        keep = ~adata.obs[batch_key].astype(str).isin(bad)
        if bad and keep.sum() > 0 and len(bad) < adata.obs[batch_key].nunique():
            sizes = adata.obs[batch_key].astype(str).value_counts()
            logger.warning(
                "%s[HVG] %s batch(es) too small for a per-batch LOESS fit and excluded from HVG ranking only: %s. Their cells remain in the analysis.",
                tag,
                len(bad),
                ", ".join((f"{b} ({int(sizes.get(b, 0))} cells)" for b in bad)),
            )
            try:
                sub = adata[keep].copy()
                sc.pp.highly_variable_genes(
                    sub,
                    flavor="seurat_v3",
                    n_top_genes=n_top_genes,
                    layer=layer,
                    batch_key=batch_key,
                )
                _copy_hvg_results(sub, adata)
                del sub
                _stamp_uns(adata, "seurat_v3")
                report.update(
                    method="seurat_v3 (per-batch; degenerate batches excluded)",
                    excluded_batches=bad,
                    n_hvg=int(adata.var["highly_variable"].sum()),
                )
                logger.info(
                    "%s[HVG] Selected %s genes batch-aware over %s batch(es).",
                    tag,
                    report["n_hvg"],
                    int(adata.obs.loc[keep, batch_key].nunique()),
                )
                return report
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "%s[HVG] Batch-aware retry failed (%s).", tag, e, exc_info=True
                )

    # --- Rung 3: pooled seurat_v3 ----------------------------------------------
    try:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer=layer
        )
        _stamp_uns(adata, "seurat_v3")
        report.update(
            method="seurat_v3 (pooled; batch-aware selection unavailable)",
            batch_key=None,
            n_hvg=int(adata.var["highly_variable"].sum()),
        )
        logger.warning(
            "%s[HVG] Fell back to pooled seurat_v3 over all cells; HVG selection is not batch-aware for this run.",
            tag,
        )
        return report
    except Exception as e:  # noqa: BLE001
        logger.warning("%s[HVG] Pooled seurat_v3 failed (%s).", tag, e, exc_info=True)

    # --- Rung 4: dispersion binning, no LOESS anywhere --------------------------
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    _stamp_uns(adata, "seurat")
    report.update(
        method="seurat (dispersion binning; seurat_v3 unavailable)",
        flavor="seurat",
        batch_key=None,
        n_hvg=int(adata.var["highly_variable"].sum()),
    )
    logger.warning(
        "%s[HVG] Fell back to flavor='seurat' dispersion binning on log-normalised data.",
        tag,
    )
    return report
