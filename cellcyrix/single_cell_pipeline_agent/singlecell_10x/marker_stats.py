"""
marker_stats.py — what a one-vs-rest marker table may and may not claim.

Both marker tables in this pipeline (cell-type markers in :mod:`markers`, Leiden
cluster markers in :mod:`rank_genes_subprocess`) compare groups that were DERIVED
FROM THE SAME EXPRESSION MATRIX the test then reads. The null hypothesis "this gene
does not differ between these groups" was already falsified by the act of forming
the groups, so the reported p-values and FDRs are anti-conservative by construction
— selection bias, commonly called double dipping.

The ranking is still perfectly useful; only the significance claim is unsupportable.
So this module strips the p-value columns on the way out and keeps ``rank``,
``scores`` and ``logfoldchanges``.

Deliberately dependency-free (stdlib + pandas, no intra-package imports):
``rank_genes_subprocess`` is executed as a standalone script by ``subprocess.run``,
where relative imports do not resolve, and both writers must make the same claim.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must not leave a one-vs-rest marker table.
PVALUE_COLUMNS = ("pvals", "pvals_adj")

RANKING_ONLY_NOTE = (
    "ranking only — p-values omitted (groups defined from the same data)"
)

MARKER_STATS_NOTE = (
    "Marker tables in this folder deliberately carry NO p-values.\n"
    "\n"
    "These markers come from sc.tl.rank_genes_groups run one-vs-rest on cell-type\n"
    "labels or on Leiden clusters. Those groups were DERIVED FROM THE SAME\n"
    "EXPRESSION MATRIX the test is then applied to, so the null hypothesis 'this\n"
    "gene is not different between groups' was already falsified by the act of\n"
    "forming the groups. The resulting p-values and FDRs are anti-conservative by\n"
    "construction (selection bias, 'double dipping') and cannot be read as evidence.\n"
    "\n"
    "What IS usable here:\n"
    "  * rank            — 1-based position of the gene within its cluster/cell type\n"
    "  * scores          — the test statistic, a monotone ranking score\n"
    "  * logfoldchanges  — effect size versus the rest of the cells\n"
    "\n"
    "Use these tables to answer 'what characterises this population'.\n"
    "For condition (case-vs-control) significance use the donor-level pseudobulk\n"
    "DESeq2 output in 06_groupwise_deg/pseudobulk_deg/, where the groups come from\n"
    "the experimental design and not from the data.\n"
)


def drop_selection_biased_pvalues(
    df: pd.DataFrame,
    group_col: str = "group",
) -> pd.DataFrame:
    """Return ``df`` without p-value columns and with a 1-based ``rank`` per group.

    ``rank`` reflects the order ``rank_genes_groups`` emitted, which is the ordering
    the table is meant to convey. Adding it before the p-values disappear means no
    downstream consumer loses the ability to take "the top N markers".
    """
    out = df.copy()
    dropped = [c for c in PVALUE_COLUMNS if c in out.columns]
    if dropped:
        out = out.drop(columns=dropped)
    if "rank" not in out.columns and group_col in out.columns:
        out["rank"] = out.groupby(group_col, sort=False).cumcount() + 1
    out["statistic_note"] = RANKING_ONLY_NOTE
    return out


def write_marker_stats_note(out_dir: str | Path) -> bool:
    """Drop :data:`MARKER_STATS_NOTE` next to the marker CSVs. Never raises."""
    try:
        (Path(out_dir) / "readme_marker_statistics.txt").write_text(
            MARKER_STATS_NOTE, encoding="utf-8"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001 - a missing note must not fail a run
        logger.debug("%s: falling back after %r", __name__, exc)
        return False
