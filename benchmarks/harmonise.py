"""
harmonise.py — map any label vocabulary onto one comparable level, identically.

Why this module is the most important part of the benchmark
-----------------------------------------------------------
Annotation methods disagree about spelling and granularity far more often than they
disagree about biology. "CD16+ NK cells", "NK cells", and "Natural killer cell" are the
same call written three ways; scoring them as strings measures vocabulary, not accuracy.
So every label — ground truth and every method's prediction alike — is resolved through
the pipeline's own hierarchy and compared at one fixed level.

The obvious hazard, and the reason this is a separate reviewable module: normalising
ground-truth labels with the *pipeline's own* resolver can inflate the pipeline's score.
If the resolver knows the truth vocabulary better than it knows a competitor's, the
competitor is penalised for a mapping failure rather than a biological error. Three
things guard against that:

* **One function, applied identically.** :func:`harmonise` takes no method argument and
  has no per-source branch. Truth and predictions go through the same call.
* **Unresolved rates are reported per source.** :func:`resolution_report` returns the
  fraction of each source's labels the resolver could not place. If a baseline resolves
  at a much lower rate than the truth column, the comparison is confounded and the
  benchmark says so instead of quietly scoring it.
* **Coarse evaluation by default.** ``main_cell_type`` (with a ``lineage`` fallback for
  labels that only resolve that far) is the level at which methods can be meaningfully
  compared; ``subtype`` rewards whoever happens to share the truth set's granularity.

Publish the mapping table from :func:`mapping_table` as a supplement. A reader can then
check every label's destination without rerunning anything.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: Level the benchmark scores at. Coarser than the pipeline's own output on purpose:
#: comparing at ``subtype`` measures granularity agreement, not correctness.
DEFAULT_LEVEL = "main_cell_type"

#: Value used when the resolver cannot place a label at all. Kept as an explicit
#: category rather than dropped — a method that emits unplaceable labels should be
#: visible in the confusion matrix, not silently excluded from the denominator.
UNRESOLVED = "Unresolved"

#: Labels that mean "no call" rather than a cell type. Treated as abstentions.
ABSTENTION_TOKENS = frozenset(
    {
        "",
        "na",
        "nan",
        "none",
        "unknown",
        "unassigned",
        "unclassified",
        "unknown cell",
        "unassigned (llm parse fail)",
        "doublet",
        "doublets",
    }
)


@lru_cache(maxsize=1)
def _hierarchy():
    """The pipeline's hierarchy, built once per process."""
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.cell_hierarchy import (  # noqa: E501
        CellHierarchy,
    )

    return CellHierarchy.from_spec()


def is_abstention(label: object) -> bool:
    """Whether a raw label means "no call" rather than a cell type."""
    if label is None:
        return True
    text = str(label).strip().lower()
    return text in ABSTENTION_TOKENS


@lru_cache(maxsize=100_000)
def _harmonise_one(label: str, level: str) -> str:
    """Resolve one raw label to `level`, cached because label sets repeat heavily.

    Args:
        label: Raw label as written by whichever method produced it.
        level: Hierarchy level to report, e.g. ``main_cell_type``.

    Returns:
        The canonical label at `level`, falling back to the coarser ``lineage`` when a
        label only resolves that far, or :data:`UNRESOLVED` when it does not resolve.
    """
    if is_abstention(label):
        return UNRESOLVED

    resolution = _hierarchy().resolve(label, source="benchmark")
    if not resolution.resolved:
        return UNRESOLVED

    value = str(getattr(resolution, level, "") or "").strip()
    if value:
        return value

    # A label like "Epithelial cell" resolves only to a lineage, so main_cell_type is
    # empty. Falling back keeps it comparable instead of throwing it away as
    # unresolved — it IS a valid, if coarse, call.
    fallback = str(getattr(resolution, "lineage", "") or "").strip()
    return fallback or UNRESOLVED


def harmonise(labels: Iterable[object], *, level: str = DEFAULT_LEVEL) -> pd.Series:
    """Map a label column onto the comparison level.

    Takes no method or source argument by design: the identical transformation must be
    applied to ground truth and to every method's predictions, or the comparison is not
    a comparison.

    Args:
        labels: Raw labels, e.g. ``adata.obs["celltype_consensus"]``.
        level: Hierarchy level to compare at. One of ``lineage``, ``class``,
            ``main_cell_type``, ``subtype``, ``fine_subtype``.

    Returns:
        Harmonised labels, index preserved when the input was a Series.
    """
    series = labels if isinstance(labels, pd.Series) else pd.Series(list(labels))
    # astype(str) before mapping: AnnData stores obs columns as Categorical, and mapping
    # a Categorical returns a Categorical whose categories are the mapped values — which
    # then refuses ordinary reductions like .sum(). Casting first keeps the result a
    # plain object Series that behaves like every other column here.
    return series.astype(str).map(lambda value: _harmonise_one(value, level))


def resolution_report(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    level: str = DEFAULT_LEVEL,
) -> pd.DataFrame:
    """Per-column resolution rates, so mapping failure cannot masquerade as error.

    Read this before the accuracy table. If the truth column resolves at 99% and a
    baseline at 70%, that baseline's macro-F1 is measuring the resolver's coverage of
    its vocabulary as much as its biology, and the gap must be reported rather than
    presented as an accuracy difference.

    Args:
        frame: Table holding the label columns.
        columns: Column names to report on.
        level: Comparison level.

    Returns:
        One row per column: ``n``, ``n_unresolved``, ``unresolved_fraction``,
        ``n_abstained``, ``abstained_fraction``, ``n_distinct_raw``,
        ``n_distinct_harmonised``.
    """
    rows = []
    for column in columns:
        if column not in frame.columns:
            logger.warning(
                "[BENCH] column %r absent; skipping in resolution report", column
            )
            continue
        # Same Categorical hazard as in `harmonise`: cast before mapping so the
        # boolean result supports .sum().
        raw = frame[column].astype(str)
        harmonised = harmonise(raw, level=level)
        n = int(len(raw))
        n_unresolved = int((harmonised == UNRESOLVED).sum())
        n_abstained = int(raw.map(is_abstention).astype(bool).sum())
        rows.append(
            {
                "column": column,
                "n": n,
                "n_unresolved": n_unresolved,
                "unresolved_fraction": (n_unresolved / n) if n else 0.0,
                "n_abstained": n_abstained,
                "abstained_fraction": (n_abstained / n) if n else 0.0,
                "n_distinct_raw": int(raw.astype(str).nunique()),
                "n_distinct_harmonised": int(harmonised.nunique()),
            }
        )
    return pd.DataFrame(rows)


def mapping_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    level: str = DEFAULT_LEVEL,
) -> pd.DataFrame:
    """Every distinct raw label and where it mapped, for publication as a supplement.

    Args:
        frame: Table holding the label columns.
        columns: Column names to enumerate.
        level: Comparison level.

    Returns:
        Columns ``source_column``, ``raw_label``, ``harmonised_label``, ``n_cells``,
        ``resolved``, sorted by column then descending cell count.
    """
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        counts = frame[column].astype(str).value_counts()
        for raw_label, n_cells in counts.items():
            harmonised = _harmonise_one(raw_label, level)
            rows.append(
                {
                    "source_column": column,
                    "raw_label": raw_label,
                    "harmonised_label": harmonised,
                    "n_cells": int(n_cells),
                    "resolved": harmonised != UNRESOLVED,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        ["source_column", "n_cells"], ascending=[True, False]
    ).reset_index(drop=True)


def evaluable_mask(
    truth: pd.Series, predictions: Optional[pd.Series] = None
) -> pd.Series:
    """Cells whose GROUND TRUTH is usable, independent of any prediction.

    The mask deliberately ignores `predictions`: dropping cells because a *method*
    failed on them would let a method improve its own score by abstaining. A prediction
    of :data:`UNRESOLVED` is scored as a wrong answer, not excluded.

    Args:
        truth: Harmonised ground-truth labels.
        predictions: Accepted and ignored, so callers cannot pass it by accident and
            silently change the denominator.

    Returns:
        Boolean mask of cells with a resolvable ground-truth label.
    """
    if predictions is not None:
        logger.debug(
            "[BENCH] evaluable_mask ignores predictions by design; "
            "unresolved predictions are scored as errors, not excluded."
        )
    return truth != UNRESOLVED


__all__ = [
    "DEFAULT_LEVEL",
    "UNRESOLVED",
    "ABSTENTION_TOKENS",
    "is_abstention",
    "harmonise",
    "resolution_report",
    "mapping_table",
    "evaluable_mask",
]
