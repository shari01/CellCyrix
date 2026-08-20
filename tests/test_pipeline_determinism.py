"""
test_pipeline_determinism.py — the same input twice must give the same answer.

``set_global_seed`` plus ``random_state=SEED`` threaded through PCA/neighbors/UMAP/Leiden
is the *intent*. This asserts the outcome, which is a different claim: a single unseeded
call anywhere in the chain, a dict iteration order that leaks into a tie-break, or a
thread-count-dependent BLAS reduction all defeat the intent without touching the seed.

Two runs of the real drivers over the identical synthetic cohort, then a comparison of
every artifact a reader would cite:

  * cell-type labels, tiers, confidences and the ontology ids
  * cluster assignments
  * the pseudobulk DE table, to full float precision
  * the embeddings

A reviewer asking "is this reproducible?" is asking exactly this, and "we set a seed" is
not an answer to it.

Marked ``slow``: two full pipeline runs, ~2 minutes. Runs offline with no credentials and
no R — the LLM and SingleR voters are off, so what is under test is the deterministic
core plus CellTypist.

    pytest tests/test_pipeline_determinism.py -m slow
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

# Columns whose exact per-cell values must match between runs. Anything a downstream
# analysis or a figure reads belongs here.
LABEL_COLUMNS = (
    "leiden",
    "celltype_consensus",
    "consensus_tier",
    "celltype_celltypist",
    "celltype_subtype",
    "celltype_cl_id",
    "celltype_ontology_node_id",
    "lineage_coarse",
    "include_in_downstream_analysis",
)

# Numeric obs columns compared with exact equality. These are computed, not sampled.
NUMERIC_COLUMNS = (
    "celltype_subtype_confidence",
    "celltypist_label_entropy",
    "celltypist_dominant_fraction",
)

# DE columns that must match to full precision. A pipeline whose fold changes drift
# between runs cannot support a gene list.
DE_NUMERIC_COLUMNS = ("log2_fold_change", "p_value", "p_value_adj", "base_mean")


def _make_cohort(
    root: Path, *, n_cells_per_sample: int = 180, n_genes: int = 400
) -> Path:
    """Write a synthetic two-group 10x cohort to `root`.

    Deliberately built with an explicit ``default_rng(0)`` rather than by reusing the
    smoke test's builder, so the fixture itself cannot be a source of run-to-run
    variation and this test isolates the pipeline.

    Args:
        root: Directory to create the cohort under.
        n_cells_per_sample: Cells per sample.
        n_genes: Genes in the matrix.

    Returns:
        The cohort base directory.
    """
    import scipy.io
    import scipy.sparse

    rng = np.random.default_rng(0)
    base = root / "cohort"
    groups = {"CONTROL": ["C1", "C2"], "CASE": ["D1", "D2"]}

    # A block of genes elevated in CASE, so the DE table is not all noise and a sign
    # flip between runs would be visible.
    case_up = np.arange(0, 20)

    gene_ids = [f"ENSG{i:08d}" for i in range(n_genes)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    # Mitochondrial genes so the QC stage has something real to compute.
    for i in range(n_genes - 10, n_genes):
        gene_names[i] = f"MT-CO{i - (n_genes - 10) + 1}"

    for group, samples in groups.items():
        for sample in samples:
            sample_dir = base / group / sample
            sample_dir.mkdir(parents=True, exist_ok=True)

            counts = rng.poisson(1.5, size=(n_genes, n_cells_per_sample)).astype(
                np.int32
            )
            # Three cell populations, so clustering has structure to find.
            for population, gene_block in enumerate(
                (np.arange(40, 80), np.arange(80, 120), np.arange(120, 160))
            ):
                lo = population * (n_cells_per_sample // 3)
                hi = (population + 1) * (n_cells_per_sample // 3)
                counts[np.ix_(gene_block, np.arange(lo, hi))] += rng.poisson(
                    12, size=(gene_block.size, hi - lo)
                ).astype(np.int32)
            if group == "CASE":
                counts[case_up, :] += rng.poisson(
                    8, size=(case_up.size, n_cells_per_sample)
                ).astype(np.int32)

            scipy.io.mmwrite(
                str(sample_dir / "matrix.mtx"), scipy.sparse.csr_matrix(counts)
            )
            (sample_dir / "barcodes.tsv").write_text(
                "\n".join(f"{sample}_CELL{i}-1" for i in range(n_cells_per_sample))
                + "\n",
                encoding="utf-8",
            )
            (sample_dir / "features.tsv").write_text(
                "\n".join(
                    f"{gene_id}\t{name}\tGene Expression"
                    for gene_id, name in zip(gene_ids, gene_names, strict=True)
                )
                + "\n",
                encoding="utf-8",
            )

    rows = ["sample,group"]
    for group, samples in groups.items():
        rows += [f"{sample},{group}" for sample in samples]
    (base / "group_map.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return base


def _run(cohort: Path, output_root: Path, out_name: str) -> Path:
    """Run the multi-sample driver offline and return its output directory."""
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x import run_pipeline_multi

    return Path(
        run_pipeline_multi(
            multi_base_dir=cohort,
            out_name=out_name,
            output_root=output_root,
            # No `group_col` option exists: the group column comes from group_map.csv.
            batch_key="sample",
            reference_group="CONTROL",
            # Every network- or R-dependent voter off: this test is about the
            # deterministic core, and it must run in CI with no credentials.
            enable_knowledge_based=False,
            enable_singler=False,
            enable_pubmed=False,
            enable_celltypist=True,
            do_groupwise_de=True,
            do_pseudobulk_de=True,
            do_doublet_detection=False,
            do_dpt=False,
            do_pathway_clustering=False,
            skip_tsne=True,
            run_per_sample=False,
            # The report's narrative sections call OpenRouter even when the annotation
            # voters are off, which would make this test require a key, cost money, and
            # introduce the one non-deterministic component it exists to exclude.
            generate_report=False,
            prepare_for_bisque=False,
        )
    )


def _load_h5ad(out_dir: Path):
    """The run's processed AnnData.

    ``bisque_ready_*_processed_scanpy_output.h5ad`` also matches the suffix and sorts
    ahead of the real export, but it is a reduced deconvolution view without ``leiden``
    or the embeddings — so it must be excluded explicitly rather than by ordering.
    """
    import anndata

    matches = sorted(
        path
        for path in out_dir.glob("*_processed_scanpy_output.h5ad")
        if not path.name.startswith("bisque_ready_")
    )
    assert matches, f"no processed h5ad under {out_dir}"
    return anndata.read_h5ad(matches[0])


def _pseudobulk_table(out_dir: Path) -> pd.DataFrame | None:
    """The donor-level DE table, or None when the run did not write one."""
    matches = sorted(out_dir.rglob("pseudobulk_overall_de.csv"))
    if not matches:
        return None
    return pd.read_csv(matches[0])


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """Run the pipeline twice over one cohort; yield both output directories."""
    workspace = tmp_path_factory.mktemp("determinism")
    cohort = _make_cohort(workspace)
    first = _run(cohort, workspace / "out_a", "run_a")
    second = _run(cohort, workspace / "out_b", "run_b")
    yield first, second
    shutil.rmtree(workspace, ignore_errors=True)


def test_cell_counts_match(two_runs):
    """Same QC decisions, so the same cells survive."""
    first, second = (_load_h5ad(path) for path in two_runs)
    assert (first.n_obs, first.n_vars) == (second.n_obs, second.n_vars)
    assert list(first.obs_names) == list(second.obs_names)


@pytest.mark.parametrize("column", LABEL_COLUMNS)
def test_label_columns_are_identical(two_runs, column):
    """Every per-cell categorical a reader would cite matches exactly."""
    first, second = (_load_h5ad(path) for path in two_runs)
    if column not in first.obs.columns:
        pytest.skip(f"{column} not produced by this configuration")
    assert column in second.obs.columns, f"{column} present in run A but not run B"

    left = first.obs[column].astype(str).to_numpy()
    right = second.obs[column].astype(str).to_numpy()
    n_differing = int((left != right).sum())
    assert n_differing == 0, (
        f"{column} differs on {n_differing}/{len(left)} cells between two runs of the "
        f"same input. Examples: "
        f"{[(a, b) for a, b in zip(left, right, strict=True) if a != b][:5]}"
    )


@pytest.mark.parametrize("column", NUMERIC_COLUMNS)
def test_numeric_obs_columns_are_identical(two_runs, column):
    """Computed per-cell numerics match bit for bit."""
    first, second = (_load_h5ad(path) for path in two_runs)
    if column not in first.obs.columns:
        pytest.skip(f"{column} not produced by this configuration")
    left = pd.to_numeric(first.obs[column], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(second.obs[column], errors="coerce").to_numpy(dtype=float)
    np.testing.assert_array_equal(
        np.isnan(left), np.isnan(right), err_msg=f"{column}: NaN pattern differs"
    )
    mask = ~np.isnan(left)
    np.testing.assert_array_equal(
        left[mask], right[mask], err_msg=f"{column}: values differ between runs"
    )


def test_cluster_partition_is_identical(two_runs):
    """Leiden gives the same partition, not merely the same number of clusters.

    Checked as a partition rather than by label equality, so a harmless relabelling
    (cluster 0 and 1 swapping names) is not reported as non-determinism while a genuine
    change in which cells group together is.
    """
    first, second = (_load_h5ad(path) for path in two_runs)
    left = first.obs["leiden"].astype(str)
    right = second.obs["leiden"].astype(str)

    assert left.nunique() == right.nunique(), "different number of clusters"
    # A one-to-one mapping between the two labellings exists iff the partitions match.
    crosstab = pd.crosstab(left, right)
    non_zero_per_row = (crosstab > 0).sum(axis=1)
    non_zero_per_column = (crosstab > 0).sum(axis=0)
    assert (non_zero_per_row == 1).all() and (non_zero_per_column == 1).all(), (
        "Leiden partition differs between runs; clusters do not map one-to-one:\n"
        f"{crosstab}"
    )


def test_embeddings_are_identical(two_runs):
    """PCA and UMAP coordinates match, so a figure is reproducible."""
    first, second = (_load_h5ad(path) for path in two_runs)
    for key in ("X_pca", "X_umap"):
        if key not in first.obsm:
            continue
        assert key in second.obsm, f"{key} present in run A but not run B"
        np.testing.assert_array_equal(
            np.asarray(first.obsm[key]),
            np.asarray(second.obsm[key]),
            err_msg=f"{key} coordinates differ between two runs of the same input",
        )


def test_pseudobulk_de_table_is_identical(two_runs):
    """The donor-level DE table matches to full float precision.

    The strongest single assertion here: this table is the paper's gene list.
    """
    first, second = (_pseudobulk_table(path) for path in two_runs)
    if first is None or second is None:
        pytest.skip("no pseudobulk DE table produced by this configuration")

    assert list(first.columns) == list(second.columns), "DE table columns differ"
    assert len(first) == len(second), "DE table row counts differ"

    gene_column = "gene" if "gene" in first.columns else first.columns[0]
    left = first.sort_values(gene_column).reset_index(drop=True)
    right = second.sort_values(gene_column).reset_index(drop=True)
    assert left[gene_column].tolist() == right[gene_column].tolist()

    for column in DE_NUMERIC_COLUMNS:
        if column not in left.columns:
            continue
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(
            dtype=float
        )
        np.testing.assert_array_equal(
            np.isnan(left_values),
            np.isnan(right_values),
            err_msg=f"DE {column}: NaN pattern differs",
        )
        mask = ~np.isnan(left_values)
        np.testing.assert_array_equal(
            left_values[mask],
            right_values[mask],
            err_msg=(
                f"DE {column} differs between two runs of the same input — the gene "
                f"list is not reproducible"
            ),
        )


def test_manifest_records_the_same_seed_and_shape(two_runs):
    """Both runs record the same seed and dataset shape in provenance."""
    import json

    payloads = []
    for path in two_runs:
        manifest = path / "provenance" / "manifest.json"
        assert manifest.is_file(), f"no provenance manifest under {path}"
        payloads.append(json.loads(manifest.read_text(encoding="utf-8")))

    assert payloads[0]["seed"] == payloads[1]["seed"]
    assert payloads[0]["dataset"] == payloads[1]["dataset"]
    assert payloads[0]["package_versions"] == payloads[1]["package_versions"]
