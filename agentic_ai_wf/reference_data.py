"""Resolve large reference datasets from one configurable root.

Preferred layout (repository root = parent of ``agentic_ai_wf/``)::

    shared_reference/
        scRNA_reference_data/
        gwas_mr_reference/
        signor_reference_data/
        targeted_crispr_reference_data/
        pathway_agent_datasets/
        pathway_gsea_data/            # GSEA module: datasets/, .enrichr_cache/, pathway_consolidation_cache/
        pathway_enrichr_cache/        # optional; if present, overrides colocated .enrichr_cache (pathway_gsea)
        gene_prioritization_datasets/
        ipaa_causality_data/
        perturbation_dep_map_data/
        perturbation_l1000_data/
        perturbation_integration_data/
        mdp_pipeline_agent_data/
        mdp_multi_pathway_data/   # also default dir for MDP classification_memory.csv (see below)

    MDP ``pc2`` / ``pathway_classifier`` use :func:`mdp_pathway_classifier_dataset_dir`
    for ``classification_memory.csv`` (same bucket as ``mdp_multi_pathway_data`` when
    present; else legacy ``analysis_tools/`` if the csv already lives there).

Environment variables (first non-empty wins):

- ``SHARED_REFERENCE_ROOT`` — directory that **contains** the bucket folders above.
- ``AGENTIC_REFERENCE_DATA_ROOT`` — alias for the same.

If ``shared_reference/<bucket>`` does not exist, each resolver falls back to the
legacy path under the repo (e.g. ``gwas_mr_reference/``, ``agentic_ai_wf/...``)
so existing checkouts work until data is copied into ``shared_reference/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from decouple import config

SHARED_REFERENCE_ROOT = config("SHARED_REFERENCE_ROOT", default="shared_reference")

_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent


def get_shared_reference_root() -> Path:
    """Absolute root of the shared reference-data tree.

    Returns:
        `SHARED_REFERENCE_ROOT` if configured, else `<repo>/shared_reference`.
    """
    if SHARED_REFERENCE_ROOT:
        return Path(SHARED_REFERENCE_ROOT).expanduser().resolve()
    return (REPO_ROOT / "shared_reference").resolve()


def _legacy_path(rel: str) -> Path:
    rel = rel.replace("/", os.sep)
    return (REPO_ROOT / rel).resolve()


def resolve_reference_bucket(bucket_dirname: str, *legacy_repo_relative: str) -> Path:
    """Resolve a dataset bucket: prefer shared root, then legacy paths, else canonical shared path."""
    preferred = (get_shared_reference_root() / bucket_dirname).resolve()
    if preferred.is_dir():
        return preferred
    for leg in legacy_repo_relative:
        p = _legacy_path(leg)
        if p.is_dir():
            return p
    return preferred


def sc_rna_reference_dir() -> Path:
    """Directory of scRNA reference tables (marker panels, cell-type references).

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket("scRNA_reference_data", "scRNA_reference_data")


def gwas_mr_reference_dir() -> Path:
    """Directory of GWAS / Mendelian-randomisation reference data.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket("gwas_mr_reference", "gwas_mr_reference")


def signor_reference_dir() -> Path:
    """Directory of SIGNOR causal-interaction reference data.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket("signor_reference_data", "signor_reference_data")


def targeted_crispr_reference_dir() -> Path:
    """Directory of targeted-CRISPR reference data.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "targeted_crispr_reference_data",
        "targeted_crispr_reference_data",
    )


def pathway_agent_datasets_dir() -> Path:
    """Directory of the pathway agent's static datasets.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "pathway_agent_datasets",
        "agentic_ai_wf/pathway_agent/datasets",
    )


def pathway_gsea_module_data_dir() -> Path:
    """Root for pathway_gsea static data and caches (GSEA module).

    Prefer ``<SHARED_REFERENCE_ROOT>/pathway_gsea_data``, else the package tree
    ``agentic_ai_wf/pathway_agent/pathway_gsea`` (same tree that holds
    ``.enrichr_cache`` in legacy checkouts).
    """
    return resolve_reference_bucket(
        "pathway_gsea_data",
        "agentic_ai_wf/pathway_agent/pathway_gsea",
    )


def pathway_gsea_datasets_dir() -> Path:
    """Directory for ``classification_memory.csv`` and other GSEA tabular assets."""
    return pathway_gsea_module_data_dir() / "datasets"


def pathway_enrichr_cache_dir() -> Path:
    """Disk cache for Enrichr library gene-set JSON (pathway_gsea enrichment).

    If ``shared_reference/pathway_enrichr_cache`` exists (older layout), use it.
    Otherwise use ``<pathway_gsea_module_data_dir>/.enrichr_cache`` so the cache
    lives beside ``datasets/`` under the same GSEA module root.
    """
    standalone = (get_shared_reference_root() / "pathway_enrichr_cache").resolve()
    if standalone.is_dir():
        return standalone
    colocated = (pathway_gsea_module_data_dir() / ".enrichr_cache").resolve()
    return colocated


def gene_prioritization_datasets_dir() -> Path:
    """Directory of the gene-prioritisation agent's static datasets.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "gene_prioritization_datasets",
        "agentic_ai_wf/gene_prioritization/datasets",
    )


def ipaa_causality_data_dir() -> Path:
    """Directory of IPAA causality input data.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "ipaa_causality_data",
        "agentic_ai_wf/ipaa_causality/data",
    )


def perturbation_dep_map_data_dir() -> Path:
    """Directory of DepMap data for the perturbation pipeline.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "perturbation_dep_map_data",
        "agentic_ai_wf/perturbation_pipeline_agent/perturbation/dep_map/data",
    )


def perturbation_l1000_data_dir() -> Path:
    """Directory of L1000 data for the perturbation pipeline.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "perturbation_l1000_data",
        "agentic_ai_wf/perturbation_pipeline_agent/perturbation/l1000/data",
    )


def perturbation_integration_data_dir() -> Path:
    """Directory of integrated perturbation data (DepMap + L1000).

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "perturbation_integration_data",
        "agentic_ai_wf/perturbation_pipeline_agent/perturbation/integration/data",
    )


def mdp_pipeline_agent_data_dir() -> Path:
    """Directory of the MDP pipeline agent's static data.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "mdp_pipeline_agent_data",
        "agentic_ai_wf/mdp_pipeline_agent/data",
    )


def mdp_multi_pathway_data_dir() -> Path:
    """Directory of the MDP multi-pathway module's static data.

    Returns:
        The resolved bucket directory.
    """
    return resolve_reference_bucket(
        "mdp_multi_pathway_data",
        "agentic_ai_wf/mdp_pipeline_agent/multi_pathway/data",
    )


def mdp_pathway_classifier_dataset_dir() -> Path:
    """Directory where MDP ``pc2`` / ``pathway_classifier`` read/write ``classification_memory.csv``.

    Resolves like :func:`mdp_multi_pathway_data_dir` when that tree exists. If the
    legacy ``multi_pathway/data`` bucket is absent but ``classification_memory.csv``
    still lives next to those scripts (``analysis_tools/``), returns that folder so
    existing checkouts keep working until the cache file is moved under
    ``shared_reference/mdp_multi_pathway_data`` or ``multi_pathway/data``.
    """
    multip = mdp_multi_pathway_data_dir()
    if multip.is_dir():
        return multip
    tools = _legacy_path(
        "agentic_ai_wf/mdp_pipeline_agent/multi_pathway/analysis_tools"
    )
    if (tools / "classification_memory.csv").is_file():
        return tools
    return multip
