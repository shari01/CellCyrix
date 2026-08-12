"""
qc_filters.py — configurable cell-level QC thresholds with per-rule accounting.

The pipeline used to hard-code ``MIN_GENES=200 / MAX_GENES=6000 / MAX_MT_PCT=15.0``
inline. Those numbers are tissue-dependent (15% mitochondrial reads is lenient for
epithelium, a flat 6000-gene ceiling discards legitimately large cells), so they
belong in the config — but the *defaults here are exactly the historical values*,
and the comparison operators are preserved verbatim (``> min``, ``< max``,
``< max_mito``) so an unchanged config filters exactly the same cells as before.

``apply_qc_filters`` also returns how many cells each individual rule rejected,
which the summary/report and the provenance manifest record. Rules overlap (a cell
can fail two at once), so per-rule counts are reported as "cells failing this rule"
and will generally sum to more than ``removed_total``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from .exceptions import PipelineInputError

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

# Historical hard-coded values — the defaults, so old configs are unchanged.
DEFAULT_MIN_GENES = 200
DEFAULT_MAX_GENES = 6000
DEFAULT_MAX_MITO_PERCENT = 15.0

_N_GENES_COL = "n_genes_by_counts"
_MITO_PCT_COL = "pct_counts_mt"


class QCConfigError(PipelineInputError):
    """Raised when a QC threshold in the configuration is not usable."""


@dataclass(frozen=True)
class QCThresholds:
    """Validated cell-level QC thresholds actually applied to a run."""

    min_genes: int = DEFAULT_MIN_GENES
    max_genes: int = DEFAULT_MAX_GENES
    max_mito_percent: float = DEFAULT_MAX_MITO_PERCENT

    def as_dict(self) -> dict[str, float | int | None]:
        """Return the thresholds as a plain dict, for the provenance manifest."""
        return asdict(self)


def _num(value, *, name: str, default, cast):
    """Cast a config value, treating None/'' as 'not set' -> default."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return cast(default)
    if isinstance(value, bool):
        raise QCConfigError(f"qc.{name} must be numeric, got boolean {value!r}.")
    try:
        return cast(value)
    except (TypeError, ValueError) as e:
        raise QCConfigError(f"qc.{name} must be numeric, got {value!r}.") from e


def resolve_qc_thresholds(
    min_genes: int | None = None,
    max_genes: int | None = None,
    max_mito_percent: float | None = None,
) -> QCThresholds:
    """Validate QC thresholds from config; missing values fall back to the defaults.

    Raises ``QCConfigError`` on values that are present but nonsensical (negative
    gene counts, ``max_genes <= min_genes``, a mito percentage outside 0-100) so a
    typo cannot silently discard most of the dataset.
    """
    mn = _num(min_genes, name="min_genes", default=DEFAULT_MIN_GENES, cast=int)
    mx = _num(max_genes, name="max_genes", default=DEFAULT_MAX_GENES, cast=int)
    mt = _num(
        max_mito_percent,
        name="max_mito_percent",
        default=DEFAULT_MAX_MITO_PERCENT,
        cast=float,
    )

    if mn < 0:
        raise QCConfigError(f"qc.min_genes must be >= 0, got {mn}.")
    if mx <= mn:
        raise QCConfigError(f"qc.max_genes ({mx}) must be > qc.min_genes ({mn}).")
    if not np.isfinite(mt) or not (0 < mt <= 100):
        raise QCConfigError(f"qc.max_mito_percent must be in (0, 100], got {mt}.")
    return QCThresholds(min_genes=mn, max_genes=mx, max_mito_percent=mt)


def apply_qc_filters(
    adata: AnnData,
    thresholds: QCThresholds,
    *,
    analysis_name: str = "analysis",
) -> Tuple["object", dict]:
    """Filter cells on ``thresholds``; return ``(filtered_adata, report)``.

    Requires ``sc.pp.calculate_qc_metrics`` to have run (needs
    ``n_genes_by_counts`` and ``pct_counts_mt`` in ``.obs``). The boolean mask is
    identical to the previous inline implementation.
    """
    missing = [c for c in (_N_GENES_COL, _MITO_PCT_COL) if c not in adata.obs.columns]
    if missing:
        raise QCConfigError(
            f"apply_qc_filters requires obs columns {missing}; "
            "run sc.pp.calculate_qc_metrics first."
        )

    n_genes = adata.obs[_N_GENES_COL]
    mito = adata.obs[_MITO_PCT_COL]

    fail_min = ~(n_genes > thresholds.min_genes)
    fail_max = ~(n_genes < thresholds.max_genes)
    fail_mito = ~(mito < thresholds.max_mito_percent)
    keep = (~fail_min) & (~fail_max) & (~fail_mito)

    report = {
        "thresholds_applied": thresholds.as_dict(),
        "n_cells_before": int(adata.n_obs),
        "removed_min_genes": int(fail_min.sum()),
        "removed_max_genes": int(fail_max.sum()),
        "removed_max_mito_percent": int(fail_mito.sum()),
        "removed_total": int((~keep).sum()),
        "n_cells_after": int(keep.sum()),
        "note": "per-rule counts overlap (a cell can fail several rules).",
    }

    if report["n_cells_after"] == 0:
        raise QCConfigError(
            f"[{analysis_name}] QC thresholds {thresholds.as_dict()} removed every "
            f"cell ({report['n_cells_before']} in). Loosen qc.* in the config."
        )

    logger.info(
        "[%s] QC filter (min_genes>%s, max_genes<%s, pct_mt<%s): %s -> %s cells (removed %s; by rule: min_genes=%s, max_genes=%s, mito=%s).",
        analysis_name,
        thresholds.min_genes,
        thresholds.max_genes,
        thresholds.max_mito_percent,
        report["n_cells_before"],
        report["n_cells_after"],
        report["removed_total"],
        report["removed_min_genes"],
        report["removed_max_genes"],
        report["removed_max_mito_percent"],
    )
    return adata[keep].copy(), report
