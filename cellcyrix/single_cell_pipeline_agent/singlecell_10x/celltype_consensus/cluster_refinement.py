"""
cluster_refinement.py — split heterogeneous clusters so the voters can see inside them.

The problem this solves
-----------------------
The consensus assigns ONE label per cluster. That is the right design — a cluster is the
unit the voters can reason about — but it caps accuracy at whatever the clustering
resolved. If a cluster holds three cell types, the pipeline cannot express that: it emits
one label and the other two types are silently absorbed.

Measured on a real lung cohort: **8 of 12 clusters** were flagged
``mixed_cluster_flag = True``, and cluster 5 alone carried 2,852 smooth-muscle cells,
650 myofibroblasts and 532 neurons/Schwann cells — all labelled "Airway smooth muscle
cell". The neurons were not misannotated so much as never asked about. Twelve clusters
for a tissue with 77 annotated cell types is the whole story.

The pipeline already detected this. ``summarize_celltypist_by_cluster`` computes
dominant-label fraction, label entropy and ``mixed_cluster_flag`` per cluster, and the
consensus logged them as "advisory only". Nothing consumed the signal.

What this module does
---------------------
Re-runs Leiden **only on the cells of the flagged clusters**, via scanpy's
``restrict_to``, and writes a refined cluster column. Unflagged clusters keep their exact
ids, so nothing that was already coherent is disturbed. The voters then run on the
refined column and get one question per homogeneous group instead of one question per
merged group.

Why ``restrict_to`` rather than raising the global resolution: raising it globally would
re-partition the clusters that were already clean, changing labels that were already
right and making runs incomparable. Splitting only what is flagged is a strictly local
edit, and it is auditable — the report says which cluster became which children.

Cost note: each new sub-cluster is one more question for the per-cluster voters, so an
LLM- or PubMed-enabled run pays for the extra sub-clusters. ``max_new_clusters`` bounds
that, and the bound is reported rather than applied silently.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Leiden resolution used INSIDE a flagged cluster. Lower than a typical global
#: resolution on purpose: the goal is to separate the two or three populations that were
#: merged, not to shatter the cluster into singletons.
DEFAULT_REFINE_RESOLUTION = 0.30

#: A sub-cluster smaller than this is merged back into its largest sibling. Below roughly
#: this size a per-cluster voter has too few marker genes to reason from, and the LLM and
#: PubMed voters are being asked to name noise.
DEFAULT_MIN_SUBCLUSTER_CELLS = 30

#: Hard ceiling on how many NEW clusters refinement may create across the whole run.
#: Every new cluster is an extra LLM/PubMed call, so an unbounded split is a real cost.
DEFAULT_MAX_NEW_CLUSTERS = 24

#: Suffix separating a parent cluster id from its refinement index, e.g. ``5`` -> ``5.1``.
#: A dot keeps the parent readable in the output, so a reader can see where a label came
#: from without consulting the report.
SUBCLUSTER_SEPARATOR = "."


class ClusterRefinementError(RuntimeError):
    """Refinement could not run. The caller falls back to the original clusters."""


def mixed_cluster_ids(
    metrics: dict[str, dict[str, Any]],
    *,
    min_cells: int = DEFAULT_MIN_SUBCLUSTER_CELLS * 2,
    cluster_sizes: Optional[dict[str, int]] = None,
) -> list[str]:
    """Clusters worth splitting, from the CellTypist heterogeneity metrics.

    Args:
        metrics: ``{cluster: metrics}`` from
            :func:`~.tools.summarize_celltypist_by_cluster`.
        min_cells: A cluster must hold at least this many cells to be worth splitting —
            below twice the minimum sub-cluster size there is no room for two children.
        cluster_sizes: ``{cluster: n_cells}``. When omitted the size check is skipped.

    Returns:
        Cluster ids flagged mixed and large enough to split, largest first so the
        ``max_new_clusters`` budget is spent where it buys the most.
    """
    flagged = [
        cluster
        for cluster, values in metrics.items()
        if bool(values.get("mixed_cluster_flag", False))
    ]
    if cluster_sizes:
        flagged = [c for c in flagged if int(cluster_sizes.get(c, 0)) >= min_cells]
        flagged.sort(key=lambda c: -int(cluster_sizes.get(c, 0)))
    else:
        flagged.sort()
    return flagged


def refine_mixed_clusters(
    adata,
    *,
    cluster_col: str,
    mixed_clusters: Sequence[str],
    key_added: Optional[str] = None,
    resolution: float = DEFAULT_REFINE_RESOLUTION,
    min_subcluster_cells: int = DEFAULT_MIN_SUBCLUSTER_CELLS,
    max_new_clusters: int = DEFAULT_MAX_NEW_CLUSTERS,
    seed: int = 0,
) -> dict[str, Any]:
    """Split the flagged clusters in place and write a refined cluster column.

    Args:
        adata: AnnData with `cluster_col` in ``obs`` and a neighbour graph already built
            (refinement reuses the existing graph; it does not recompute PCA/neighbours).
        cluster_col: Existing cluster column, e.g. ``leiden``.
        mixed_clusters: Cluster ids to split, from :func:`mixed_cluster_ids`.
        key_added: Column to write. Defaults to ``<cluster_col>_refined``.
        resolution: Leiden resolution applied inside each flagged cluster.
        min_subcluster_cells: Sub-clusters smaller than this are merged into their
            largest sibling rather than voted on.
        max_new_clusters: Ceiling on new clusters created across the run.
        seed: Leiden random_state, so refinement is reproducible.

    Returns:
        A report: the column written, per-parent children, counts, and whether the
        budget capped the split. ``{"refined": False, ...}`` when nothing was split.

    Raises:
        ClusterRefinementError: If `cluster_col` is absent, or the neighbour graph
            required by Leiden has not been computed.
    """
    import scanpy as sc

    target = key_added or f"{cluster_col}_refined"

    if cluster_col not in adata.obs.columns:
        raise ClusterRefinementError(
            f"cluster column {cluster_col!r} not in obs; cluster before refining."
        )
    if "neighbors" not in adata.uns:
        raise ClusterRefinementError(
            "no neighbour graph in adata.uns['neighbors']; refinement reuses the "
            "existing graph and cannot build one."
        )

    original = adata.obs[cluster_col].astype(str)
    # scanpy's restrict_to reads the column through the .cat accessor, so it must be a
    # pandas Categorical — a plain string column raises
    # "Can only use .cat accessor with a 'category' dtype".
    adata.obs[target] = pd.Categorical(original.to_numpy())

    if not mixed_clusters:
        logger.info(
            "[REFINE] no mixed clusters to split; %s == %s", target, cluster_col
        )
        return {
            "refined": False,
            "column": target,
            "reason": "no cluster flagged mixed",
            "n_clusters_before": int(original.nunique()),
            "n_clusters_after": int(original.nunique()),
            "children": {},
        }

    sizes = original.value_counts().to_dict()
    children: dict[str, list[str]] = {}
    merged_back: dict[str, int] = {}
    new_cluster_budget = int(max_new_clusters)
    budget_exhausted_on: list[str] = []

    for parent in mixed_clusters:
        if new_cluster_budget <= 0:
            budget_exhausted_on.append(parent)
            continue
        if str(parent) not in sizes:
            logger.warning("[REFINE] cluster %s not present; skipping.", parent)
            continue

        scratch = f"__refine_{parent}"
        try:
            # restrict_to re-runs Leiden on ONLY this cluster's cells and leaves every
            # other cell's label untouched, which is what keeps the edit local.
            sc.tl.leiden(
                adata,
                resolution=resolution,
                restrict_to=(target, [str(parent)]),
                key_added=scratch,
                random_state=seed,
                flavor="igraph",
                n_iterations=2,
                directed=False,
            )
        except TypeError:
            # Older scanpy without the igraph flavour arguments.
            sc.tl.leiden(
                adata,
                resolution=resolution,
                restrict_to=(target, [str(parent)]),
                key_added=scratch,
                random_state=seed,
            )
        except Exception as exc:  # noqa: BLE001 - one failed split must not lose the run
            logger.warning("[REFINE] could not split cluster %s: %s", parent, exc)
            continue

        # scanpy writes children as "<parent>,<n>"; rename to "<parent>.<n>" and merge
        # any child too small to vote on back into the largest sibling.
        refined = adata.obs[scratch].astype(str)
        in_parent = original == str(parent)
        raw_children = refined[in_parent].value_counts()

        keep = [c for c, n in raw_children.items() if n >= min_subcluster_cells]
        if len(keep) <= 1:
            logger.info(
                "[REFINE] cluster %s did not separate at resolution %.2f "
                "(%d child >= %d cells); left intact.",
                parent,
                resolution,
                len(keep),
                min_subcluster_cells,
            )
            del adata.obs[scratch]
            continue

        largest = str(raw_children.index[0])
        rename: dict[str, str] = {}
        for index, child in enumerate(keep, start=1):
            rename[str(child)] = f"{parent}{SUBCLUSTER_SEPARATOR}{index}"
        n_merged = 0
        for child, n in raw_children.items():
            if str(child) not in rename:
                rename[str(child)] = rename[largest]
                n_merged += int(n)

        values = adata.obs[target].astype(str).to_numpy()
        refined_values = refined.to_numpy()
        mask = in_parent.to_numpy()
        values[mask] = [rename.get(v, v) for v in refined_values[mask]]
        # Re-wrap as Categorical so a second pass through restrict_to still works.
        adata.obs[target] = pd.Categorical(values)

        created = sorted(set(rename[str(c)] for c in keep))
        children[str(parent)] = created
        if n_merged:
            merged_back[str(parent)] = n_merged
        # Splitting into k children costs k-1 against the budget: the parent already
        # existed.
        new_cluster_budget -= len(created) - 1
        del adata.obs[scratch]

        logger.info(
            "[REFINE] cluster %s -> %s (%d cells merged back as too small)",
            parent,
            ", ".join(created),
            n_merged,
        )

    if budget_exhausted_on:
        logger.warning(
            "[REFINE] max_new_clusters=%d reached; %d flagged cluster(s) left "
            "unsplit: %s",
            max_new_clusters,
            len(budget_exhausted_on),
            budget_exhausted_on,
        )

    after = adata.obs[target].astype(str)
    report = {
        "refined": bool(children),
        "column": target,
        "resolution": float(resolution),
        "min_subcluster_cells": int(min_subcluster_cells),
        "max_new_clusters": int(max_new_clusters),
        "n_clusters_before": int(original.nunique()),
        "n_clusters_after": int(after.nunique()),
        "n_flagged": len(list(mixed_clusters)),
        "n_split": len(children),
        "children": children,
        "cells_merged_back": merged_back,
        "not_split_budget": budget_exhausted_on,
        "seed": int(seed),
    }
    logger.info(
        "[REFINE] %d cluster(s) -> %d; %s now has %d clusters (was %d).",
        report["n_split"],
        sum(len(v) for v in children.values()),
        target,
        report["n_clusters_after"],
        report["n_clusters_before"],
    )
    return report


__all__ = [
    "ClusterRefinementError",
    "DEFAULT_REFINE_RESOLUTION",
    "DEFAULT_MIN_SUBCLUSTER_CELLS",
    "DEFAULT_MAX_NEW_CLUSTERS",
    "SUBCLUSTER_SEPARATOR",
    "mixed_cluster_ids",
    "refine_mixed_clusters",
]
