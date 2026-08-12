"""
downstream_gating.py — which cells may carry an inferential claim.

Consensus annotation grades every Leiden cluster ``High`` / ``Medium`` /
``Low/Review`` (see ``celltype_consensus.consensus``). A ``Low/Review`` cluster is
one where the voters disagreed or the lineage gate contradicted them — its label is
a hypothesis, not an identity. Feeding those cells into condition-level DE and
composition tests presents that hypothesis as a result.

This module marks, never deletes. It writes a boolean ``include_in_downstream_analysis``
column into ``.obs`` and hands the caller a *view-derived copy* for inferential
steps only. The full object — every cell, every label, every tier — is what gets
written to the ``.h5ad``, the UMAPs, and the audit tables, so nothing becomes
unauditable.

Default is OFF (``exclude_low_confidence_de=False``): every cell is included and
behaviour is identical to before this module existed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .atomic_io import atomic_to_csv

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

INCLUDE_COL = "include_in_downstream_analysis"
TIER_COL = "consensus_tier"

# Tiers dropped when filtering is switched on. Matches consensus.TIER_LOW.
DEFAULT_EXCLUDED_TIERS: tuple[str, ...] = ("Low/Review",)

# Mirrors pseudobulk_de.MIN_SAMPLES_PER_GROUP — the point below which a
# pseudobulk contrast has no real biological replication.
MIN_SAMPLES_PER_GROUP = 2


def resolve_excluded_tiers(tiers: Optional[Iterable] = None) -> List[str]:
    """Normalize the configured tier list (missing -> the default Low/Review)."""
    if tiers is None:
        return list(DEFAULT_EXCLUDED_TIERS)
    if isinstance(tiers, str):
        tiers = [tiers]
    out = [str(t).strip() for t in tiers if str(t).strip()]
    return out or list(DEFAULT_EXCLUDED_TIERS)


def annotate_downstream_inclusion(
    adata: AnnData,
    *,
    exclude_low_confidence: bool = False,
    excluded_tiers: Optional[Sequence[str]] = None,
    tier_col: str = TIER_COL,
    cluster_col: str = "leiden",
) -> dict[str, Any]:
    """Write ``include_in_downstream_analysis`` into ``adata.obs``; return a report.

    The column is ALWAYS written (all-True when filtering is off) so downstream
    code and the exported ``.h5ad`` have one unambiguous field to read instead of
    re-deriving the rule. Cells are never dropped here.
    """
    tiers = resolve_excluded_tiers(excluded_tiers)
    n_obs = int(adata.n_obs)

    report = {
        "exclude_low_confidence_de": bool(exclude_low_confidence),
        "excluded_consensus_tiers": tiers,
        "tier_column_present": tier_col in adata.obs.columns,
        "n_cells_total": n_obs,
        "n_cells_included": n_obs,
        "n_cells_excluded": 0,
        "n_clusters_total": 0,
        "n_clusters_excluded": 0,
        "excluded_clusters": [],
        "reason": "",
    }
    if cluster_col in adata.obs.columns:
        report["n_clusters_total"] = int(adata.obs[cluster_col].astype(str).nunique())

    if not exclude_low_confidence:
        adata.obs[INCLUDE_COL] = True
        report["reason"] = (
            "filtering disabled (exclude_low_confidence_de=False); all cells included."
        )
        logger.info("[GATING] %s", report["reason"])
        return report

    if tier_col not in adata.obs.columns:
        adata.obs[INCLUDE_COL] = True
        report["reason"] = (
            f"exclude_low_confidence_de=True but obs['{tier_col}'] is absent "
            "(annotation did not run or produced no tiers); all cells included."
        )
        logger.warning("[GATING] %s", report["reason"])
        return report

    tier_str = adata.obs[tier_col].astype(str)
    excluded_mask = tier_str.isin(tiers)
    include = ~excluded_mask

    n_excl = int(excluded_mask.sum())
    if n_excl == n_obs:
        # Excluding everything is never the right answer; report it and include all.
        adata.obs[INCLUDE_COL] = True
        report["reason"] = (
            f"every cell is in an excluded tier {tiers}; filtering would leave 0 cells, "
            "so all cells were included instead (annotation quality needs review)."
        )
        logger.warning("[GATING] %s", report["reason"])
        return report

    adata.obs[INCLUDE_COL] = include.to_numpy()

    excluded_clusters: List[str] = []
    if cluster_col in adata.obs.columns:
        cl = adata.obs[cluster_col].astype(str)
        excluded_clusters = sorted(cl[excluded_mask.to_numpy()].unique().tolist())

    report.update(
        {
            "n_cells_included": int(include.sum()),
            "n_cells_excluded": n_excl,
            "n_clusters_excluded": len(excluded_clusters),
            "excluded_clusters": excluded_clusters,
            "reason": (
                f"excluded tiers {tiers}: {n_excl}/{n_obs} cells in "
                f"{len(excluded_clusters)} cluster(s) {excluded_clusters} flagged "
                "out of inferential analyses (retained in the h5ad and all audit tables)."
            ),
        }
    )
    logger.info("[GATING] %s", report["reason"])
    return report


def stamp_consensus_table_inclusion(
    csv_path: str | Path,
    adata: AnnData,
    *,
    cluster_col: str = "leiden",
) -> bool:
    """Fill ``included_in_downstream_analysis`` in the consensus CSV from ``.obs``.

    The authoritative gating decision is ``obs[INCLUDE_COL]``; the consensus table is
    written before gating runs, so this back-fills the column from the single source
    of truth rather than re-deriving the rule (which could drift). Returns True when
    the CSV was updated.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.info("[GATING] consensus table %s absent; inclusion not stamped.", path)
        return False
    if INCLUDE_COL not in adata.obs.columns or cluster_col not in adata.obs.columns:
        logger.info(
            "[GATING] obs['%s'] / obs['%s'] missing; inclusion not stamped.",
            INCLUDE_COL,
            cluster_col,
        )
        return False

    # A cluster is included when any of its cells are (inclusion is per-cluster).
    per_cluster = (
        adata.obs.groupby(adata.obs[cluster_col].astype(str), observed=True)[
            INCLUDE_COL
        ]
        .max()
        .astype(bool)
    )
    df = pd.read_csv(path)
    key = "cluster" if "cluster" in df.columns else cluster_col
    if key not in df.columns:
        logger.warning(
            "[GATING] consensus table has no '%s' column; inclusion not stamped.", key
        )
        return False
    df["included_in_downstream_analysis"] = (
        df[key].astype(str).map(per_cluster).fillna(True).astype(bool)
    )
    atomic_to_csv(df, path, index=False)
    logger.info(
        "[GATING] stamped included_in_downstream_analysis into %s (%s/%s clusters included).",
        path.name,
        int(df["included_in_downstream_analysis"].sum()),
        len(df),
    )
    return True


def check_replication(
    adata: AnnData,
    *,
    group_col: str = "group",
    sample_col: str = "sample",
    min_samples_per_group: int = MIN_SAMPLES_PER_GROUP,
) -> Tuple[bool, str]:
    """Is there still real biological replication? Returns ``(ok, reason)``.

    Used to catch the case where tier filtering removes so many cells that a
    pseudobulk contrast would no longer have ``>= min_samples_per_group`` donors
    per arm. The caller reports the reason instead of proceeding on a broken design.
    """
    if group_col not in adata.obs.columns or sample_col not in adata.obs.columns:
        return False, f"obs['{group_col}'] and/or obs['{sample_col}'] absent."

    df = pd.DataFrame(
        {
            "g": adata.obs[group_col].astype(str).to_numpy(),
            "s": adata.obs[sample_col].astype(str).to_numpy(),
        }
    )
    per_group = df.groupby("g")["s"].nunique()
    if per_group.size < 2:
        return False, f"only {per_group.size} group(s) remain ({per_group.to_dict()})."
    weak = per_group[per_group < min_samples_per_group]
    if len(weak) > 0:
        return False, (
            f"group(s) {weak.to_dict()} have < {min_samples_per_group} samples "
            f"(all groups: {per_group.to_dict()})."
        )
    return True, f"ok ({per_group.to_dict()})."


def subset_for_downstream(
    adata: AnnData,
    *,
    group_col: str = "group",
    sample_col: str = "sample",
    require_replication: bool = True,
    analysis_name: str = "analysis",
) -> Tuple["object", dict]:
    """Return ``(adata_for_inference, report)`` using ``include_in_downstream_analysis``.

    Falls back to the FULL object — with an explicit warning and a recorded
    reason — when the column is absent, excludes nothing, or when filtering would
    destroy the replication a pseudobulk contrast needs. Falling back rather than
    silently analysing an under-replicated subset keeps the statistics honest;
    ``pseudobulk_de.compute_pseudobulk_de`` still applies its own replication gate
    and skips-with-reason from there.
    """
    report = {"filtered": False, "n_cells_used": int(adata.n_obs), "reason": ""}

    if INCLUDE_COL not in adata.obs.columns:
        report["reason"] = f"obs['{INCLUDE_COL}'] absent; using all cells."
        return adata, report

    include = adata.obs[INCLUDE_COL].astype(bool)
    if bool(include.all()):
        report["reason"] = "nothing flagged for exclusion; using all cells."
        return adata, report

    if require_replication:
        candidate = adata[include.to_numpy()]
        ok, why = check_replication(
            candidate, group_col=group_col, sample_col=sample_col
        )
        if not ok:
            report["reason"] = (
                f"SKIPPED tier filtering for inferential steps: after excluding "
                f"low-confidence clusters, {why} Using all cells so the contrast "
                f"keeps its replication; low-confidence labels remain flagged in "
                f"obs['{INCLUDE_COL}']."
            )
            logger.warning("[%s] [GATING] %s", analysis_name, report["reason"])
            return adata, report

    out = adata[include.to_numpy()].copy()
    report.update(
        {
            "filtered": True,
            "n_cells_used": int(out.n_obs),
            "reason": (
                f"inferential steps run on {out.n_obs}/{adata.n_obs} cells flagged "
                f"obs['{INCLUDE_COL}']=True."
            ),
        }
    )
    logger.info("[%s] [GATING] %s", analysis_name, report["reason"])
    return out, report
