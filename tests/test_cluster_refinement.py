"""
test_cluster_refinement.py — splitting the clusters the pipeline already flags as mixed.

The consensus emits one label per cluster, so a cluster holding several cell types can
only be given one of them. On a real lung cohort 8 of 12 clusters were flagged
``mixed_cluster_flag=True`` and one held 2,852 smooth-muscle cells, 650 myofibroblasts
and 532 neurons under a single "Airway smooth muscle cell" label — the flags were
computed and then ignored.

These tests pin the properties that make acting on them safe:

* clusters that were NOT flagged keep their exact ids, so labels that were already right
  cannot change;
* no cell is lost or duplicated (the pipeline asserts total-in == total-out downstream);
* a sub-cluster too small for a voter to reason about is merged back, not voted on;
* the new-cluster budget is enforced and reported, because every sub-cluster costs an
  extra LLM/PubMed call;
* refinement is reproducible, so the pipeline's determinism guarantee survives it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.cluster_refinement import (  # noqa: E501
    SUBCLUSTER_SEPARATOR,
    ClusterRefinementError,
    mixed_cluster_ids,
    refine_mixed_clusters,
)

# --------------------------------------------------------------------------------------
# selecting which clusters to split
# --------------------------------------------------------------------------------------


def test_only_flagged_clusters_are_selected():
    metrics = {
        "0": {"mixed_cluster_flag": True},
        "1": {"mixed_cluster_flag": False},
        "2": {"mixed_cluster_flag": True},
    }
    assert mixed_cluster_ids(metrics) == ["0", "2"]


def test_clusters_too_small_to_split_are_skipped():
    """A cluster with no room for two viable children is not worth splitting."""
    metrics = {"0": {"mixed_cluster_flag": True}, "1": {"mixed_cluster_flag": True}}
    selected = mixed_cluster_ids(
        metrics, min_cells=100, cluster_sizes={"0": 500, "1": 10}
    )
    assert selected == ["0"]


def test_largest_first_so_the_budget_buys_the_most():
    metrics = {k: {"mixed_cluster_flag": True} for k in ("0", "1", "2")}
    selected = mixed_cluster_ids(
        metrics, min_cells=1, cluster_sizes={"0": 100, "1": 900, "2": 400}
    )
    assert selected == ["1", "2", "0"]


def test_no_flags_selects_nothing():
    assert mixed_cluster_ids({"0": {"mixed_cluster_flag": False}}) == []


# --------------------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------------------


def _adata_with_structure(n_per_block: int = 120, n_genes: int = 60, seed: int = 0):
    """AnnData whose cluster 0 contains TWO transcriptionally distinct blocks.

    Cluster 0 is mixed by construction — two disjoint gene programmes — while cluster 1
    is homogeneous. A correct refinement splits 0 and leaves 1 untouched.
    """
    anndata = pytest.importorskip("anndata")
    import scanpy as sc

    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    # cluster 0, sub-block A: genes 0-19 elevated
    first = rng.poisson(1.0, size=(n_per_block, n_genes)).astype(np.float32)
    first[:, 0:20] += rng.poisson(30, size=(n_per_block, 20))
    blocks.append(first)
    labels += ["0"] * n_per_block
    # cluster 0, sub-block B: genes 20-39 elevated
    second = rng.poisson(1.0, size=(n_per_block, n_genes)).astype(np.float32)
    second[:, 20:40] += rng.poisson(30, size=(n_per_block, 20))
    blocks.append(second)
    labels += ["0"] * n_per_block
    # cluster 1: genes 40-59 elevated, homogeneous
    third = rng.poisson(1.0, size=(n_per_block, n_genes)).astype(np.float32)
    third[:, 40:60] += rng.poisson(30, size=(n_per_block, 20))
    blocks.append(third)
    labels += ["1"] * n_per_block

    matrix = np.vstack(blocks)
    obs = pd.DataFrame(
        {"leiden": labels}, index=[f"cell_{i}" for i in range(matrix.shape[0])]
    )
    adata = anndata.AnnData(X=matrix, obs=obs)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=15)
    return adata


def test_mixed_cluster_is_split_and_clean_one_is_not():
    adata = _adata_with_structure()
    report = refine_mixed_clusters(
        adata, cluster_col="leiden", mixed_clusters=["0"], min_subcluster_cells=20
    )
    assert report["refined"] is True
    assert report["n_split"] == 1
    assert report["n_clusters_after"] > report["n_clusters_before"]

    refined = adata.obs[report["column"]].astype(str)
    # cluster 1 was never flagged, so its cells keep the exact original id
    untouched = adata.obs["leiden"].astype(str) == "1"
    assert set(refined[untouched]) == {"1"}, "an unflagged cluster was modified"
    # the flagged cluster's cells now carry parent-prefixed child ids
    was_zero = adata.obs["leiden"].astype(str) == "0"
    children = set(refined[was_zero])
    assert len(children) >= 2, f"cluster 0 did not split: {children}"
    assert all(c.startswith(f"0{SUBCLUSTER_SEPARATOR}") for c in children), children


def test_no_cell_is_lost_or_duplicated():
    """total-in == total-out; the pipeline asserts this downstream."""
    adata = _adata_with_structure()
    n_before = adata.n_obs
    report = refine_mixed_clusters(
        adata, cluster_col="leiden", mixed_clusters=["0"], min_subcluster_cells=20
    )
    refined = adata.obs[report["column"]]
    assert len(refined) == n_before
    assert refined.notna().all()
    assert (refined.astype(str) != "").all()


def test_children_partition_the_parent_exactly():
    adata = _adata_with_structure()
    report = refine_mixed_clusters(
        adata, cluster_col="leiden", mixed_clusters=["0"], min_subcluster_cells=20
    )
    original = adata.obs["leiden"].astype(str)
    refined = adata.obs[report["column"]].astype(str)
    parent_cells = int((original == "0").sum())
    child_cells = int(refined.str.startswith(f"0{SUBCLUSTER_SEPARATOR}").sum())
    assert child_cells == parent_cells


def test_empty_flag_list_is_a_noop_copy():
    adata = _adata_with_structure()
    report = refine_mixed_clusters(adata, cluster_col="leiden", mixed_clusters=[])
    assert report["refined"] is False
    assert (
        adata.obs[report["column"]].astype(str) == adata.obs["leiden"].astype(str)
    ).all()


def test_budget_is_recorded():
    adata = _adata_with_structure()
    report = refine_mixed_clusters(
        adata,
        cluster_col="leiden",
        mixed_clusters=["0", "1"],
        min_subcluster_cells=20,
        max_new_clusters=1,
    )
    assert report["max_new_clusters"] == 1
    assert report["n_split"] <= 2


def test_tiny_subclusters_are_merged_back():
    """A child too small for a voter to reason about must not become its own question."""
    adata = _adata_with_structure()
    report = refine_mixed_clusters(
        adata,
        cluster_col="leiden",
        mixed_clusters=["0"],
        min_subcluster_cells=100_000,  # nothing can qualify
    )
    assert report["refined"] is False, "split despite no child meeting the size floor"


def test_missing_cluster_column_raises():
    adata = _adata_with_structure()
    with pytest.raises(ClusterRefinementError, match="not in obs"):
        refine_mixed_clusters(adata, cluster_col="nope", mixed_clusters=["0"])


def test_missing_neighbour_graph_raises():
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=np.zeros((6, 3), dtype="float32"),
        obs=pd.DataFrame({"leiden": list("000111")}, index=[f"c{i}" for i in range(6)]),
    )
    with pytest.raises(ClusterRefinementError, match="neighbour graph"):
        refine_mixed_clusters(adata, cluster_col="leiden", mixed_clusters=["0"])


def test_refinement_is_reproducible():
    """Same seed, same split — the pipeline's determinism guarantee must survive this."""
    first = _adata_with_structure()
    second = _adata_with_structure()
    report_a = refine_mixed_clusters(
        first,
        cluster_col="leiden",
        mixed_clusters=["0"],
        min_subcluster_cells=20,
        seed=7,
    )
    report_b = refine_mixed_clusters(
        second,
        cluster_col="leiden",
        mixed_clusters=["0"],
        min_subcluster_cells=20,
        seed=7,
    )
    assert (
        first.obs[report_a["column"]].astype(str).to_numpy()
        == second.obs[report_b["column"]].astype(str).to_numpy()
    ).all()


def test_report_records_what_happened():
    adata = _adata_with_structure()
    report = refine_mixed_clusters(
        adata, cluster_col="leiden", mixed_clusters=["0"], min_subcluster_cells=20
    )
    for key in (
        "refined",
        "column",
        "resolution",
        "n_clusters_before",
        "n_clusters_after",
        "n_flagged",
        "n_split",
        "children",
        "seed",
    ):
        assert key in report, f"report is missing {key}"
    assert report["children"]["0"], "no children recorded for the split cluster"
