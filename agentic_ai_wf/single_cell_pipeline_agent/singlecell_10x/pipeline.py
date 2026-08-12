"""
pipeline.py — core Scanpy orchestration for a single AnnData object.

Given one (already loaded and, in multi mode, combined) AnnData, this module runs
the full analysis and writes every artifact: QC + filtering, doublet detection,
normalization/log1p (with raw counts preserved in ``layers['counts']``), HVG,
scaling + PCA, optional BBKNN batch integration, neighbors -> UMAP (+ optional
t-SNE), optional diffusion pseudotime, Leiden clustering, multi-method consensus
cell-type annotation, cluster markers, donor-level pseudobulk DESeq2 group DE,
pathway enrichment, the HTML/PDF report, and the Bisque-ready export.

Every stochastic step is threaded with a fixed ``random_state`` and a provenance
manifest is written, so a run is reproducible.

Import-time behaviour: selecting the matplotlib ``Agg`` backend is the ONE action
this module takes at import, and it is unavoidable — the backend must be chosen
before ``matplotlib.pyplot`` is imported, and pyplot is imported below. Everything
else, including the figure-style defaults, is applied inside
:func:`run_scanpy_pipeline` so that importing this module changes no global state a
caller did not ask for.
"""

from __future__ import annotations

import os

# MUST precede the `matplotlib.pyplot` import below: the backend cannot be changed
# once pyplot has bound one, and a run in a subprocess or on a head-less server has
# no display. Both the env var and use() are set because scanpy re-imports pyplot
# internally and the env var is what a fresh interpreter (the subprocess entry
# points) sees. This is the only statement in this module that runs at import time.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")

import gc
import re as _re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse as sp_sparse

from . import celltype_qc_plots, env_names, hvg_selection, qc_filters
from . import clustering as clustering_mod
from . import downstream_gating as gating
from . import integration as integration_policy
from .atomic_io import atomic_to_csv
from .config_cli import DO_PATHWAY_CLUSTERING as DEFAULT_DO_PATHWAY_CLUSTERING
from .config_cli import logger

# --- figure resolution / geometry ------------------------------------------
# Resolution, panel size, marker scaling and panel spacing all live in
# figure_style so the subprocess entry points (which run in fresh interpreters
# and so miss anything set here) can install exactly the same defaults.
from .figure_style import (
    FIGURE_DPI,
    PANEL_SIZE,
    apply_figure_style,
    clamp_fig_inches,
    panel_wspace,
    point_size,
    tick_rotation,
)
from .gene_names import update_gene_names
from .group_de import (
    compute_de_by_celltype,
    plot_group_specific_umaps,
    plot_groupwise_celltype_proportions,
)
from .markers import compute_celltype_markers
from .pathway_enrichment import run_cluster_marker_enrichment
from .pipeline_options import PipelineOptions
from .pseudobulk_de import compute_pseudobulk_de
from .reproducibility import DEFAULT_SEED, set_global_seed, write_run_manifest
from .safe_names import safe_filename
from .scanpy_params import (
    NEIGHBORS_N_NEIGHBORS,
    NEIGHBORS_N_PCS,
    NORMALIZE_TARGET_SUM,
    PCA_N_COMPS,
    SCALE_MAX_VALUE,
)
from .summary_ct_deg import summarize_celltype_degs_markers_pathways

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData


class _SkipDPT(Exception):
    """Internal sentinel: cleanly skip the rest of the DPT block (reason already logged)."""


def _resolve_dpt_root(adata, group_col, dpt_root_group, *, enable_llm, tissue, species):
    """Choose a pseudotime root: (iroot_cell_index, root_group).

    Root group: explicit `dpt_root_group` if it names a real group; else "auto"
    -> LLM picks the baseline/normal group, falling back to a name heuristic
    (normal/healthy/control/baseline/...). Root cell: the root-group cell at the
    extreme of the first diffusion component (deterministic, reproducible).

    Returns (None, None) when no valid baseline group exists — the caller then
    SKIPS DPT rather than rooting arbitrarily (pseudotime would be meaningless).
    """
    if group_col not in adata.obs.columns:
        return None, None
    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        return None, None

    root_group = None
    rg = str(dpt_root_group or "auto").strip()
    if rg.lower() != "auto" and rg in set(groups):
        root_group = rg
    elif rg.lower() == "auto":
        if enable_llm:
            try:
                from .celltype_consensus.agent import llm_select_root_group
                from .celltype_consensus.config import ConsensusConfigError, load_config

                try:
                    _cfg = load_config(enable_llm=True, tissue=tissue, species=species)
                    root_group = llm_select_root_group(_cfg, groups)
                except ConsensusConfigError:
                    root_group = None
            except Exception as e:
                logger.warning(
                    "[DPT] LLM root-group selection unavailable (%s); trying heuristic.",
                    e,
                    exc_info=True,
                )
        if root_group is None:
            kw = _re.compile(
                r"normal|healthy|control|ctrl|baseline|adjacent|non[-_ ]?tumou?r|"
                r"benign|naive|untreated|pre[-_ ]?infusion",
                _re.IGNORECASE,
            )
            hits = [g for g in groups if kw.search(g)]
            if hits:
                root_group = hits[0]

    if root_group is None:
        return None, None

    if "X_diffmap" not in adata.obsm:
        return None, root_group
    dc1 = np.asarray(adata.obsm["X_diffmap"])[:, 1].ravel()
    mask = (adata.obs[group_col].astype(str) == root_group).values
    if mask.sum() == 0:
        return None, root_group
    idxs = np.flatnonzero(mask)
    # root at the tip: the extreme DC1 on the side the root group sits.
    if float(dc1[mask].mean()) <= float(dc1.mean()):
        root = idxs[int(np.argmin(dc1[mask]))]
    else:
        root = idxs[int(np.argmax(dc1[mask]))]
    return int(root), root_group


# ---------------------------------------------------------------------
#  Annotation reuse gate
# ---------------------------------------------------------------------
# Columns that may legitimately BE a final annotation when reuse is requested.
# Deliberately EXCLUDES every per-voter column (celltype_celltypist,
# celltype_singler, celltype_knowledge_based, celltype_pubmed): those are audit
# evidence for one annotator, not a consensus-supported identity, and promoting one
# of them to `celltype` would silently replace the consensus with a single vote.
FINAL_ANNOTATION_COLUMNS = (
    "celltype_consensus",  # this pipeline's own final label — preferred
    "celltype",  # a previous run's final label (a copy of the above)
    "cell_type",  # externally curated final annotations
    "cell_ontology_class",
    "celltype_major",
    "cell_type_major",
)

# Per-voter columns that must NEVER be promoted to the final annotation.
_VOTER_ONLY_COLUMNS = (
    "celltype_celltypist",
    "celltype_singler",
    "celltype_knowledge_based",
    "celltype_pubmed",
    "celltype_subtype",
)

_EMPTY_LABELS = {"", "nan", "none", "null", "unassigned", "unknown", "na"}


def _annotation_column_is_complete(adata, col: str) -> tuple[bool, str]:
    """Is ``obs[col]`` a usable final annotation? Returns ``(ok, reason)``."""
    if col not in adata.obs.columns:
        return False, f"obs['{col}'] is absent"
    s = adata.obs[col]
    n_missing = int(s.isna().sum())
    vals = s.astype(str).str.strip().str.lower()
    n_empty = int(vals.isin(_EMPTY_LABELS).sum())
    if n_missing or n_empty:
        return False, (
            f"obs['{col}'] is incomplete ({n_missing} NaN, {n_empty} empty/Unassigned "
            f"of {adata.n_obs} cells)"
        )
    if s.astype(str).nunique() < 1:
        return False, f"obs['{col}'] has no labels"
    return True, f"obs['{col}'] is complete ({s.astype(str).nunique()} distinct labels)"


def _resolve_annotation_reuse(
    adata,
    *,
    reuse_existing_final_annotation: bool,
    final_annotation_column: str | None,
    analysis_name: str,
) -> dict[str, Any]:
    """Decide whether to reuse an existing annotation instead of running consensus.

    Returns ``{"column": <obs col or None>, "source": <str or None>, "reason": str}``.
    ``column=None`` means "run the consensus". Every decision — reuse or recompute —
    is logged with its explicit reason.
    """
    present_voters = [c for c in _VOTER_ONLY_COLUMNS if c in adata.obs.columns]

    if not reuse_existing_final_annotation:
        reason = (
            "reuse_existing_final_annotation=False -> recomputing consensus annotation."
        )
        if present_voters:
            reason += (
                f" Per-voter column(s) {present_voters} are present but are audit "
                f"evidence only and will NOT be promoted to '{'celltype'}'."
            )
        logger.info("[%s] [ANNOTATION-REUSE] %s", analysis_name, reason)
        return {"column": None, "source": None, "reason": reason}

    # Reuse requested: an explicit column wins, else the first complete known final column.
    if final_annotation_column:
        col = str(final_annotation_column)
        if col in _VOTER_ONLY_COLUMNS:
            reason = (
                f"final_annotation_column='{col}' is a single-voter column; refusing to "
                f"reuse it as the final annotation. Recomputing consensus."
            )
            logger.warning("[%s] [ANNOTATION-REUSE] %s", analysis_name, reason)
            return {"column": None, "source": None, "reason": reason}
        ok, why = _annotation_column_is_complete(adata, col)
        if not ok:
            reason = f"configured final_annotation_column rejected: {why}. Recomputing consensus."
            logger.warning("[%s] [ANNOTATION-REUSE] %s", analysis_name, reason)
            return {"column": None, "source": None, "reason": reason}
        reason = f"reusing configured final annotation column: {why}."
        logger.info("[%s] [ANNOTATION-REUSE] %s", analysis_name, reason)
        return {
            "column": col,
            "source": f"reused_final_annotation ({col})",
            "reason": reason,
        }

    rejected: list[str] = []
    for col in FINAL_ANNOTATION_COLUMNS:
        if col not in adata.obs.columns:
            continue
        ok, why = _annotation_column_is_complete(adata, col)
        if ok:
            reason = f"reuse_existing_final_annotation=True -> reusing {why}."
            logger.info("[%s] [ANNOTATION-REUSE] %s", analysis_name, reason)
            return {
                "column": col,
                "source": f"reused_final_annotation ({col})",
                "reason": reason,
            }
        rejected.append(why)
        logger.info(
            "[%s] [ANNOTATION-REUSE] skipping candidate: %s.", analysis_name, why
        )

    # Carry the per-candidate rejection reasons through, not just "none found" — this
    # string is what the provenance manifest records as `reuse_decision`.
    reason = (
        "reuse_existing_final_annotation=True but no complete final annotation column "
        f"was found among {list(FINAL_ANNOTATION_COLUMNS)}; recomputing consensus."
    )
    if rejected:
        reason += " Rejected candidate(s): " + "; ".join(rejected) + "."
    logger.info("[%s] [ANNOTATION-REUSE] %s", analysis_name, reason)
    return {"column": None, "source": None, "reason": reason}


def _annotation_provenance(
    adata,
    *,
    enable_celltypist: bool,
    enable_llm: bool,
    enable_singler,
    enable_pubmed: bool,
    tissue,
    species,
    celltypist_model: str,
    singler_reference: str,
    celltype_source,
    reuse_info: dict[str, Any],
    use_subtypes_for_downstream: bool,
) -> dict[str, Any]:
    """Annotation provenance with RESOLVED resources, not the requested ``"auto"``.

    Requested values come from the call arguments; resolved values are read from
    ``adata.uns[CONSENSUS_UNS_KEY]``, which ``run_consensus_annotation`` populates
    with the model/reference/tissue/species it actually used. When annotation did not
    run (reuse, or failure) the resolved fields stay None and the reason is recorded.
    """
    from .celltype_consensus import CONSENSUS_UNS_KEY

    resolved = adata.uns.get(CONSENSUS_UNS_KEY) or {}
    out = {
        # requested (as configured)
        "requested_celltypist_model": celltypist_model,
        "requested_singler_reference": singler_reference,
        "requested_tissue": tissue,
        "requested_species": species,
        # resolved (as executed)
        "resolved_celltypist_model": resolved.get("resolved_celltypist_model"),
        "resolved_singler_reference": resolved.get("resolved_singler_reference"),
        "annotation_tissue": resolved.get("annotation_tissue", tissue),
        "annotation_species": resolved.get("annotation_species", species),
        # voter enablement — as requested and as executed
        "celltypist_enabled": bool(enable_celltypist),
        "singler_enabled": enable_singler,
        "knowledge_based_enabled": bool(enable_llm),
        "pubmed_enabled": bool(enable_pubmed),
        "celltypist_enabled_effective": resolved.get("celltypist_enabled"),
        "singler_enabled_effective": resolved.get("singler_enabled"),
        "knowledge_based_enabled_effective": resolved.get("knowledge_based_enabled"),
        "pubmed_enabled_effective": resolved.get("pubmed_enabled"),
        "llm_model": resolved.get("openrouter_model"),
        # marker evidence + annotation-quality settings
        "marker_ranking_method": resolved.get("marker_ranking_method"),
        "marker_min_detection_fraction": resolved.get("marker_min_detection_fraction"),
        "top_n_markers": resolved.get("top_n_markers"),
        "expression_source": resolved.get("expression_source"),
        "mixed_cluster_min_dominant_fraction": resolved.get(
            "mixed_cluster_min_dominant_fraction"
        ),
        "mixed_cluster_second_label_fraction": resolved.get(
            "mixed_cluster_second_label_fraction"
        ),
        "n_mixed_clusters": resolved.get("n_mixed_clusters"),
        "tier_counts": resolved.get("tier_counts"),
        "clusters_with_empty_markers": resolved.get("clusters_with_empty_markers"),
        # reference data backing the annotation — which curated source was actually
        # used for the lineage panels and for label harmonization. Both degrade to
        # built-in tables when the reference is missing, so an unrecorded run is
        # indistinguishable from a fully-referenced one.
        "lineage_panel_source": resolved.get("lineage_panel_source"),
        "lineage_panel_table": resolved.get("lineage_panel_table"),
        "lineage_panel_sizes": resolved.get("lineage_panel_sizes"),
        "lineage_panel_fallback_reason": resolved.get("lineage_panel_fallback_reason"),
        "lineage_min_score": resolved.get("lineage_min_score"),
        "label_resolver": resolved.get("label_resolver"),
        "label_resolver_min_confidence": resolved.get("label_resolver_min_confidence"),
        # subtype policy + reuse decision
        "use_subtypes_for_downstream": bool(use_subtypes_for_downstream),
        "celltype_source": celltype_source,
        "reuse_existing_final_annotation": bool(reuse_info.get("column") is not None),
        "reuse_decision": reuse_info.get("reason"),
    }
    # Legacy keys (previous manifests carried these names); kept so older readers work.
    out["enable_llm"] = bool(enable_llm)
    out["enable_singler"] = enable_singler
    out["tissue"] = tissue
    out["species"] = species
    out["celltypist_model"] = celltypist_model
    out["singler_reference"] = singler_reference
    return out


def _contrast_provenance(
    adata,
    *,
    group_col: str,
    reference_group,
    lfc_threshold: float,
    alpha: float,
) -> dict[str, Any]:
    """Which group is the DE reference, how it was chosen, and what was tested.

    The sign of every log2FoldChange in the run depends on this single decision, so
    it belongs in the manifest next to the seed — not only in the CSVs. When no
    baseline can be identified, ``reference_group_resolved`` is None and
    ``reference_selection`` says the direction was alphabetical (i.e. arbitrary).
    """
    from .contrasts import ordered_contrasts

    if group_col not in adata.obs.columns:
        return {
            "group_col": group_col,
            "reference_group_requested": reference_group,
            "reference_group_resolved": None,
            "reference_selection": f"obs['{group_col}'] absent; no contrasts defined.",
            "contrasts": [],
            "lfc_threshold": float(lfc_threshold or 0.0),
            "alpha": float(alpha),
        }

    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    pairs, ref, reason = ordered_contrasts(groups, reference=reference_group)
    return {
        "group_col": group_col,
        "groups": groups,
        "reference_group_requested": reference_group,
        "reference_group_resolved": ref,
        "reference_selection": reason,
        "contrasts": [f"{focus}_vs_{r}" for focus, r in pairs],
        "contrast_direction": [
            f"{focus}_vs_{r}: positive log2FoldChange = higher in {focus}"
            for focus, r in pairs
        ],
        "lfc_threshold": float(lfc_threshold or 0.0),
        "alpha": float(alpha),
        "effect_size_rule": (
            f"DESeq2 formal null H0:|log2FC|<={float(lfc_threshold or 0.0):g} "
            f"(no post-hoc fold-change filter)"
            if (lfc_threshold or 0) > 0
            else "DESeq2 null H0:log2FC=0"
        ),
        "lfc_shrinkage": "apeGLM (pyDESeq2 lfc_shrink); p-values unaffected",
    }


# =====================================================================
#  MAIN SCANPY PIPELINE (verbatim copy, no logic changes)
# =====================================================================


def run_scanpy_pipeline(
    adata: AnnData,
    out_dir: Path,
    # Keyword-only (Rule 6.2). The ~40 analysis options arrive as ONE PipelineOptions
    # object rather than 41 separate parameters (Rule 7.2): they are declared once in
    # pipeline_options.py, which is also what the two drivers and main.py's config
    # loader resolve against, so there is a single place an option can be added.
    *,
    analysis_name: str,
    options: PipelineOptions,
    group_col: str = "group",
    cluster_col: str = "leiden",
) -> None:
    """Run the full Scanpy analysis on one AnnData and write all artifacts under ``out_dir``.

    Executes QC + filtering, optional Scrublet doublet removal, normalization/log1p
    (raw counts kept in ``layers['counts']``), HVG, scaling + PCA, optional BBKNN
    integration on ``batch_key``, neighbors -> UMAP (+ optional t-SNE), optional
    diffusion pseudotime, Leiden clustering, multi-method consensus annotation
    (CellTypist / SingleR / LLM / PubMed voters, each toggleable), cluster markers,
    donor-level pseudobulk DESeq2 group DE, optional pathway enrichment, the report,
    and the Bisque export. ``seed`` is set globally and threaded as ``random_state``
    through every stochastic step for reproducibility.

    Returns the path to the processed ``.h5ad`` written under ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `options.seed` is None when the config did not set one; DEFAULT_SEED is this
    # module's own value and stays authoritative in that case. Bound once here so
    # every `random_state=seed` below reads the same number.
    seed = DEFAULT_SEED if options.seed is None else int(options.seed)

    # Figure defaults (DPI, panel size, marker scaling, fonts). Applied here rather
    # than at import so importing this module does not mutate a caller's rcParams.
    # Every plotting helper reads rcParams inside its own body, so setting them once
    # at the start of the run reaches all of them.
    apply_figure_style()

    # Reproducibility: seed Python/NumPy up-front; random_state=seed is threaded
    # through PCA/neighbors/UMAP/tSNE/Leiden below.
    set_global_seed(seed)
    logger.info("[%s] Global seed set to %s (reproducible run).", analysis_name, seed)

    # "Not configured" falls back to the config default. Written back into `options`,
    # because that is what every later read consults — a bare local would be dropped
    # and pathway enrichment would follow the unresolved value.
    if options.do_pathway_clustering is None:
        options = options.merged(do_pathway_clustering=DEFAULT_DO_PATHWAY_CLUSTERING)

    # ---- Validate config-driven knobs UP-FRONT ----
    # Fail before any expensive computation if a threshold/resolution is unusable,
    # rather than silently coercing it. `None` everywhere means "not configured" and
    # resolves to the historical hard-coded value, so old configs are unaffected.
    resolution = clustering_mod.resolve_leiden_resolution(options.leiden_resolution)
    qc_thresholds = qc_filters.resolve_qc_thresholds(
        min_genes=options.qc_min_genes,
        max_genes=options.qc_max_genes,
        max_mito_percent=options.qc_max_mito_percent,
    )
    min_cluster_cells_eff = clustering_mod.resolve_positive_int(
        options.min_cluster_cells,
        name="clustering.min_cluster_cells",
        default=clustering_mod.DEFAULT_MIN_CLUSTER_CELLS,
    )
    silhouette_cap = clustering_mod.resolve_positive_int(
        options.resolution_silhouette_max_cells,
        name="clustering.resolution_silhouette_max_cells",
        default=clustering_mod.DEFAULT_SILHOUETTE_MAX_CELLS,
    )
    excluded_tiers = gating.resolve_excluded_tiers(options.excluded_consensus_tiers)
    logger.info(
        "[%s] Config: leiden_resolution=%s, evaluate_resolutions=%s, qc=%s, exclude_low_confidence_de=%s (tiers=%s).",
        analysis_name,
        resolution,
        options.evaluate_resolutions,
        qc_thresholds.as_dict(),
        options.exclude_low_confidence_de,
        excluded_tiers,
    )

    # Structured subfolders
    summary_dir = out_dir / "00_analysis_summary"
    qc_dir = out_dir / "01_qc_and_filtering"
    hvg_dir = out_dir / "02_highly_variable_genes"
    dimred_dir = out_dir / "03_dimensionality_reduction_and_embeddings"
    clustering_dir = out_dir / "04_clustering_and_cell_states"

    celltype_root_dir = out_dir / "05_celltype_analysis"
    celltype_anno_dir = celltype_root_dir / "celltype_annotation"
    celltype_markers_dir = celltype_root_dir / "celltype_specific_markers"

    deg_root_dir = out_dir / "06_groupwise_deg"
    pathway_root_dir = out_dir / "07_pathway_enrichment"
    reference_dir = out_dir / "08_reference_summary"

    for d in [
        summary_dir,
        qc_dir,
        hvg_dir,
        dimred_dir,
        clustering_dir,
        celltype_root_dir,
        celltype_anno_dir,
        celltype_markers_dir,
        reference_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=== Running Scanpy pipeline: %s ===", analysis_name)
    logger.info("Output root folder: %s", out_dir)

    initial_cells = adata.n_obs
    initial_genes = adata.n_vars

    # ---- Pre-flight data validation (safe auto-fix; fail-fast on fatal issues) ----
    # Runs on the freshly-loaded matrix BEFORE any processing: repairs mechanical
    # defects (non-unique barcodes/genes, all-zero rows/cols, Ensembl->symbol) and
    # raises on anything that would corrupt results (non-count matrix, NaN/Inf,
    # negatives, batch==group). Report: <out>/00_data_validation/.
    from .data_validation import run_preflight_validation

    adata = run_preflight_validation(
        adata,
        out_dir=out_dir / "00_data_validation",
        analysis_name=analysis_name,
        group_col=group_col,
        sample_col="sample",
        batch_key=options.batch_key,
        block_on_fail=True,
    )

    if "counts" not in adata.layers:
        if sp_sparse.issparse(adata.X):
            adata.layers["counts"] = adata.X.copy()
        else:
            adata.layers["counts"] = np.array(adata.X)

    adata = update_gene_names(adata)
    if getattr(adata, "raw", None) is not None:
        adata.raw = None

    sc.pp.filter_genes(adata, min_cells=3)
    genes_after_min_cells = adata.n_vars

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    groupby_col = "sample" if "sample" in adata.obs.columns else None
    sc.settings.figdir = qc_dir

    # One violin per sample, and sample IDs are wide: printed upright they ran into
    # each other under the axis, so they are rotated when they do not fit the slot.
    violin_rotation = None
    if groupby_col:
        n_slots = max(1, int(adata.obs[groupby_col].nunique()))
        violin_rotation = tick_rotation(
            adata.obs[groupby_col].astype(str).unique(),
            slot_inches=PANEL_SIZE[0] / n_slots,
        )

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        groupby=groupby_col,
        multi_panel=True,
        rotation=violin_rotation,
        show=False,
        save=f"_{analysis_name}_qc_violin.png",
    )
    plt.close("all")
    gc.collect()

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        show=False,
        save=f"_{analysis_name}_qc_total_vs_mito.png",
    )
    plt.close("all")
    gc.collect()

    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        show=False,
        save=f"_{analysis_name}_qc_total_vs_genes.png",
    )
    plt.close("all")
    gc.collect()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(adata.obs["n_genes_by_counts"], bins=60)
    axes[0].set_title("n_genes_by_counts")
    axes[1].hist(adata.obs["total_counts"], bins=60)
    axes[1].set_title("total_counts")
    axes[2].hist(adata.obs["pct_counts_mt"], bins=60)
    axes[2].set_title("pct_counts_mt")
    plt.tight_layout()
    plt.savefig(qc_dir / f"{analysis_name}_qc_metric_histograms.png", dpi=FIGURE_DPI)
    plt.close("all")
    gc.collect()

    # Cell-level QC. Thresholds come from the config (qc.*) and default to the
    # historical 200 / 6000 / 15.0 with the same comparison operators, so an
    # unchanged config filters exactly the same cells. `qc_report` records how many
    # cells each individual rule rejected for the summary and the manifest.
    adata, qc_report = qc_filters.apply_qc_filters(
        adata, qc_thresholds, analysis_name=analysis_name
    )

    # --- Doublet detection (Scrublet) on RAW counts, before normalization ---
    # Run per-sample (batch_key) so simulated doublets stay within a library.
    if options.do_doublet_detection:
        try:
            _db_batch = (
                "sample"
                if "sample" in adata.obs.columns and adata.obs["sample"].nunique() > 1
                else None
            )
            sc.pp.scrublet(adata, batch_key=_db_batch, random_state=seed)
            n_doub = int(
                adata.obs.get(
                    "predicted_doublet", pd.Series(False, index=adata.obs_names)
                ).sum()
            )
            logger.info(
                "[%s] Scrublet: %s/%s predicted doublets (batch_key=%r).",
                analysis_name,
                n_doub,
                adata.n_obs,
                _db_batch,
            )
            qc_report["predicted_doublets"] = n_doub
            if (
                options.remove_doublets
                and "predicted_doublet" in adata.obs.columns
                and n_doub > 0
            ):
                adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()
                qc_report["removed_doublets"] = n_doub
                logger.info(
                    "[%s] Removed doublets -> %s cells remain.",
                    analysis_name,
                    adata.n_obs,
                )
            else:
                qc_report["removed_doublets"] = 0
            qc_report["n_cells_after_doublets"] = int(adata.n_obs)
        except Exception as e:
            qc_report["doublet_detection_error"] = str(e)
            logger.warning(
                "[%s] Scrublet doublet detection failed (%s); continuing.",
                analysis_name,
                e,
                exc_info=True,
            )
    else:
        qc_report["predicted_doublets"] = None
        qc_report["removed_doublets"] = 0
        qc_report["n_cells_after_doublets"] = int(adata.n_obs)
    qc_report["remove_doublets_requested"] = bool(options.remove_doublets)
    qc_report["do_doublet_detection"] = bool(options.do_doublet_detection)

    sc.pp.normalize_total(adata, target_sum=NORMALIZE_TARGET_SUM)
    sc.pp.log1p(adata)
    # Copy so the later in-place sc.pp.scale() cannot corrupt .raw (raw.X and .X
    # can otherwise share one buffer). raw must stay log-normalized for DE/markers.
    adata.raw = adata.copy()

    n_cells = adata.n_obs
    if n_cells < 4000:
        N_HVG = 2000
    elif n_cells > 200000:
        N_HVG = 4000
    else:
        N_HVG = 4000

    # seurat_v3 fits one LOESS per batch and aborts the whole call if a single
    # library is too small to fit (see hvg_selection). Cohorts that fit today take
    # the identical first rung and select the identical genes.
    hvg_report = hvg_selection.select_hvgs(
        adata,
        n_top_genes=N_HVG,
        batch_key="sample" if "sample" in adata.obs.columns else None,
        layer="counts",
        analysis_name=analysis_name,
    )

    sc.settings.figdir = hvg_dir
    sc.pl.highly_variable_genes(
        adata, show=False, save=f"_{analysis_name}_highly_variable_genes_plot.png"
    )
    plt.close("all")
    gc.collect()

    hvg_count = int(adata.var["highly_variable"].sum())
    hvg_table = adata.var.copy()
    # index = gene name; index=False would make the table unusable.
    atomic_to_csv(
        hvg_table,
        hvg_dir / f"{analysis_name}_highly_variable_genes_table.csv",
        index=True,
    )

    sc.pp.scale(adata, max_value=SCALE_MAX_VALUE)
    sc.tl.pca(
        adata,
        n_comps=PCA_N_COMPS,
        svd_solver="arpack",
        use_highly_variable=True,
        random_state=seed,
    )

    sc.settings.figdir = dimred_dir
    sc.pl.pca_variance_ratio(
        adata, log=True, show=False, save=f"_{analysis_name}_pca_variance_explained.png"
    )
    plt.close("all")
    gc.collect()

    # ---- Integration policy: ANNOTATION-ONLY, warn-and-proceed ----
    # `sample` nested inside `group` is the standard case-control design, not a
    # metadata error, so this never aborts. It warns with the specific design facts
    # and records the scope: the corrected graph drives clustering/UMAP/annotation,
    # while DE reads raw counts per donor and never consults it. See integration.py.
    integration_record = integration_policy.resolve_integration_policy(
        adata,
        integration_method=options.integration_method,
        batch_key=options.batch_key,
        group_col=group_col,
        analysis_name=analysis_name,
    )

    # Track the method that ACTUALLY executed (not just the one requested), so a
    # silent BBKNN->neighbors fallback cannot be recorded as a successful "bbknn"
    # integration in the provenance manifest.
    integration_method_used = "none"
    if (
        options.integration_method == "bbknn"
        and options.batch_key is not None
        and options.batch_key in adata.obs.columns
    ):
        try:
            import scanpy.external as sce

            logger.info(
                "[INTEGRATION] Using BBKNN with batch_key='%s'.", options.batch_key
            )
            sce.pp.bbknn(adata, batch_key=options.batch_key)
            integration_method_used = "bbknn"
        except Exception:
            logger.warning(
                "[INTEGRATION] scanpy.external (bbknn) not available. Falling back to standard neighbors."
            )
            sc.pp.neighbors(
                adata,
                n_neighbors=NEIGHBORS_N_NEIGHBORS,
                n_pcs=NEIGHBORS_N_PCS,
                random_state=seed,
            )
            integration_method_used = "neighbors (bbknn requested; fell back)"
    else:
        logger.info("[INTEGRATION] Using standard neighbors (no explicit integration).")
        sc.pp.neighbors(
            adata,
            n_neighbors=NEIGHBORS_N_NEIGHBORS,
            n_pcs=NEIGHBORS_N_PCS,
            random_state=seed,
        )
        integration_method_used = "neighbors"

    integration_record["method_used"] = integration_method_used
    integration_policy.stamp_integration_provenance(adata, integration_record)

    sc.tl.umap(adata, random_state=seed)

    if options.skip_tsne:
        logger.info("[TSNE] Skipping t-SNE (skip_tsne=True).")
    elif adata.n_obs <= 50000:
        sc.tl.tsne(adata, n_pcs=NEIGHBORS_N_PCS, use_rep="X_pca", random_state=seed)
    else:
        logger.info("[TSNE] Skipping t-SNE because n_obs=%s > 50000.", adata.n_obs)

    if options.do_dpt:
        try:
            logger.info(
                "[%s] Computing diffusion map + DPT (trajectory inference)...",
                analysis_name,
            )
            sc.tl.diffmap(adata)

            # Pseudotime needs a biological starting point. Pick the baseline group
            # (LLM/heuristic) then the root cell at that group's diffusion tip. If
            # no baseline group exists, SKIP DPT rather than rooting arbitrarily.
            iroot, root_group = _resolve_dpt_root(
                adata,
                group_col,
                options.dpt_root_group,
                enable_llm=options.enable_knowledge_based,
                tissue=options.tissue,
                species=options.species,
            )
            if iroot is None:
                _grps = (
                    sorted(adata.obs[group_col].astype(str).unique().tolist())
                    if group_col in adata.obs.columns
                    else "N/A"
                )
                logger.warning(
                    "[%s] Skipping DPT: no identifiable baseline/normal root group among %s. Pseudotime needs a starting point — set dpt_root_group to a baseline group to enable it.",
                    analysis_name,
                    _grps,
                )
                raise _SkipDPT()

            adata.uns["iroot"] = iroot
            logger.info(
                "[%s] DPT root -> group='%s', cell index=%s.",
                analysis_name,
                root_group,
                iroot,
            )
            sc.tl.dpt(adata)
            if "dpt_pseudotime" not in adata.obs.columns:
                logger.warning(
                    "[%s] DPT produced no 'dpt_pseudotime'; skipping DPT plots.",
                    analysis_name,
                )
                raise _SkipDPT()

            dpt_pt_size = point_size(adata.n_obs)
            sc.pl.umap(
                adata,
                color=["dpt_pseudotime"],
                size=dpt_pt_size,
                show=False,
                save=f"_{analysis_name}_umap_dpt_pseudotime.png",
            )
            plt.close("all")
            gc.collect()

            if "X_tsne" in adata.obsm:
                sc.pl.tsne(
                    adata,
                    color=["dpt_pseudotime"],
                    size=dpt_pt_size,
                    show=False,
                    save=f"_{analysis_name}_tsne_dpt_pseudotime.png",
                )
                plt.close("all")
                gc.collect()

            sc.pl.diffmap(
                adata,
                color=["dpt_pseudotime"],
                size=dpt_pt_size,
                show=False,
                save=f"_{analysis_name}_diffmap_dpt_pseudotime.png",
            )
            plt.close("all")
            gc.collect()
        except _SkipDPT:
            pass  # reason already logged; not a failure
        except Exception as e:
            logger.warning(
                "[%s] DPT computation/plotting failed: %s",
                analysis_name,
                e,
                exc_info=True,
            )

    color_cols = []
    if "sample" in adata.obs.columns:
        color_cols.append("sample")
    if "group" in adata.obs.columns:
        color_cols.append("group")

    pt_size = point_size(adata.n_obs, base=15)

    if color_cols:
        # Sample IDs are long ("GSM8035466_OM"), so the gap between panels is
        # measured from the labels themselves rather than guessed.
        samples_wspace = panel_wspace(adata, color_cols)
        sc.pl.umap(
            adata,
            color=color_cols,
            wspace=samples_wspace,
            size=pt_size,
            show=False,
            save=f"_{analysis_name}_umap_samples_groups.png",
        )
        plt.close("all")
        gc.collect()

        if "X_tsne" in adata.obsm:
            sc.pl.tsne(
                adata,
                color=color_cols,
                wspace=samples_wspace,
                size=pt_size,
                show=False,
                save=f"_{analysis_name}_tsne_samples_groups.png",
            )
            plt.close("all")
            gc.collect()
    else:
        sc.pl.umap(adata, size=pt_size, show=False, save=f"_{analysis_name}_umap.png")
        plt.close("all")
        gc.collect()

        if "X_tsne" in adata.obsm:
            sc.pl.tsne(
                adata, size=pt_size, show=False, save=f"_{analysis_name}_tsne.png"
            )
            plt.close("all")
            gc.collect()

    for qc_col in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
        if qc_col in adata.obs.columns:
            sc.pl.umap(
                adata,
                color=[qc_col],
                size=pt_size,
                show=False,
                save=f"_{analysis_name}_umap_{qc_col}.png",
            )
            plt.close("all")
            gc.collect()

            if "X_tsne" in adata.obsm:
                sc.pl.tsne(
                    adata,
                    color=[qc_col],
                    size=pt_size,
                    show=False,
                    save=f"_{analysis_name}_tsne_{qc_col}.png",
                )
                plt.close("all")
                gc.collect()

    sc.settings.figdir = clustering_dir
    # PRIMARY clustering. Leiden stays the pipeline's structural partition; only the
    # resolution is now configurable. Run FIRST and never overwritten, so the optional
    # resolution audit below cannot perturb obs['leiden'] or its cluster IDs.
    clustering_mod.run_leiden(adata, resolution=resolution, seed=seed)

    # Optional, non-destructive resolution audit -> leiden_res_<r> columns + CSV.
    resolution_eval = None
    if options.evaluate_resolutions:
        resolution_eval = clustering_mod.evaluate_leiden_resolutions(
            adata,
            primary_resolution=resolution,
            candidates=options.resolution_candidates,
            seed=seed,
            out_dir=clustering_dir,
            analysis_name=analysis_name,
            min_cluster_cells=min_cluster_cells_eff,
            silhouette_max_cells=silhouette_cap,
        )
    else:
        logger.info(
            "[%s] Resolution audit skipped (evaluate_resolutions=False); ran only the configured resolution %s.",
            analysis_name,
            resolution,
        )

    clusters = sorted(adata.obs["leiden"].unique().tolist(), key=lambda x: int(x))
    cluster_sizes = adata.obs["leiden"].value_counts().sort_index()
    atomic_to_csv(
        cluster_sizes,
        clustering_dir / f"{analysis_name}_cluster_cell_counts_leiden.csv",
        header=["n_cells"],
        index=True,
    )
    _small = cluster_sizes[cluster_sizes < min_cluster_cells_eff]
    if len(_small) > 0:
        logger.warning(
            "[%s] %s cluster(s) have < %s cells %s — annotation for these is poorly supported.",
            analysis_name,
            len(_small),
            min_cluster_cells_eff,
            _small.to_dict(),
        )

    leiden_pt_size = point_size(adata.n_obs, base=15)
    sc.pl.umap(
        adata,
        color=["leiden"],
        legend_loc="on data",
        size=leiden_pt_size,
        show=False,
        save=f"_{analysis_name}_umap_leiden.png",
    )
    plt.close("all")
    gc.collect()

    if "X_tsne" in adata.obsm:
        sc.pl.tsne(
            adata,
            color=["leiden"],
            legend_loc="on data",
            size=leiden_pt_size,
            show=False,
            save=f"_{analysis_name}_tsne_leiden.png",
        )
        plt.close("all")
        gc.collect()

    if not options.skip_pca_cluster_plots:
        sc.pl.pca(
            adata,
            color=["leiden"],
            size=leiden_pt_size,
            show=False,
            save=f"_{analysis_name}_pca_leiden.png",
        )
        plt.close("all")
        gc.collect()

    # ========= Cell type detection / prediction =========
    celltype_col_raw = None
    celltype_source = None
    standard_celltype_col = "celltype"

    # Reuse is now OPT-IN and restricted to a FINAL annotation column.
    #
    # Previously any of a long candidate list — including the single-voter
    # `celltype_celltypist` — short-circuited the whole consensus, so re-running on
    # an already-annotated .h5ad silently promoted one voter's raw labels to
    # `celltype`. Per-voter columns are audit evidence, never a final annotation, so
    # they are excluded from reuse entirely.
    celltype_reuse_info = _resolve_annotation_reuse(
        adata,
        reuse_existing_final_annotation=options.reuse_existing_final_annotation,
        final_annotation_column=options.final_annotation_column,
        analysis_name=analysis_name,
    )
    celltype_col_raw = celltype_reuse_info["column"]
    celltype_source = celltype_reuse_info["source"]

    if celltype_col_raw is None:
        try:
            # Multi-method consensus annotation (CellTypist + SingleR + LLM).
            # Replaces the previous immune-only CellTypist call, which mislabeled
            # non-immune cells. Fully disease-agnostic: no disease/group label is
            # ever passed into annotation (only optional tissue context).
            from .celltype_consensus import (
                CELLTYPE_CONSENSUS_COL,
                load_config,
                run_consensus_annotation,
            )
            from .celltype_consensus.config import ConsensusConfigError

            # SingleR: explicit param wins; otherwise fall back to the env flag.
            if options.enable_singler is None:
                _enable_singler = str(
                    env_names.get_env(env_names.ENABLE_SINGLER, "") or ""
                ).strip().lower() in {"1", "true", "yes"}
            else:
                _enable_singler = bool(options.enable_singler)
            # PubMed literature voter: config-driven (on by default); env can force-disable.
            _enable_pubmed = bool(options.enable_pubmed) and str(
                env_names.get_env(env_names.ENABLE_PUBMED, "1")
            ).strip().lower() not in {"0", "false", "no", "off"}
            # Shared, config-driven annotation settings. Collected once so every
            # load_config call site below (including the SingleR-failure retry) uses
            # identical marker-ranking / mixed-cluster / subtype settings.
            _anno_kwargs = dict(
                tissue=options.tissue,
                species=options.species,
                celltypist_model=options.celltypist_model,
                singler_reference=options.singler_reference,
                cluster_col="leiden",
                mixed_cluster_min_dominant_fraction=options.mixed_cluster_min_dominant_fraction,
                mixed_cluster_second_label_fraction=options.mixed_cluster_second_label_fraction,
                use_subtypes_for_downstream=options.use_subtypes_for_downstream,
            )
            try:
                _cfg = load_config(
                    enable_celltypist=options.enable_celltypist,
                    enable_llm=options.enable_knowledge_based,
                    enable_singler=_enable_singler,
                    enable_pubmed=_enable_pubmed,
                    **_anno_kwargs,
                )
            except ConsensusConfigError as e:
                logger.warning(
                    "[CELLTYPE-CONSENSUS] LLM voter disabled for this run (reason: %s). Proceeding with CellTypist%s.",
                    e,
                    " + SingleR" if _enable_singler else "",
                    exc_info=True,
                )
                _cfg = load_config(
                    enable_celltypist=options.enable_celltypist,
                    enable_llm=False,
                    enable_singler=_enable_singler,
                    enable_pubmed=_enable_pubmed,
                    **_anno_kwargs,
                )

            logger.info(
                "[CELLTYPE-CONSENSUS] Running multi-method consensus annotation (llm=%s, singler=%s) for %s.",
                _cfg.enable_llm,
                _cfg.enable_singler,
                analysis_name,
            )
            try:
                adata = run_consensus_annotation(
                    adata,
                    out_dir=celltype_anno_dir,
                    analysis_name=analysis_name,
                    config=_cfg,
                    geo_json_path=options.geo_json_path,
                )
            except Exception as e_singler:
                # SingleR raises loudly on any non-zero R exit and would otherwise
                # sink the whole annotation (CellTypist + LLM included). If it was
                # on, degrade to CellTypist + LLM rather than losing everything.
                if _cfg.enable_singler:
                    logger.error(
                        f"[CELLTYPE-CONSENSUS] SingleR-enabled consensus failed for "
                        f"{analysis_name} ({e_singler}); retrying WITHOUT SingleR "
                        f"(CellTypist{' + LLM' if _cfg.enable_llm else ''}).",
                        exc_info=True,
                    )
                    _cfg = load_config(
                        enable_celltypist=_cfg.enable_celltypist,
                        enable_llm=_cfg.enable_llm,
                        enable_singler=False,
                        enable_pubmed=_cfg.enable_pubmed,
                        **_anno_kwargs,
                    )
                    adata = run_consensus_annotation(
                        adata,
                        out_dir=celltype_anno_dir,
                        analysis_name=analysis_name,
                        config=_cfg,
                        geo_json_path=options.geo_json_path,
                    )
                else:
                    raise
            celltype_col_raw = CELLTYPE_CONSENSUS_COL
            celltype_source = (
                "Consensus (CellTypist"
                + (" + SingleR" if _cfg.enable_singler else "")
                + (" + LLM" if _cfg.enable_llm else "")
                + ")"
            )
        except Exception as e:
            # Keep the pipeline resilient: annotation is optional for downstream
            # steps. The failure is logged explicitly (not swallowed silently).
            logger.error(
                f"Consensus annotation failed for {analysis_name}: {e}", exc_info=True
            )

    celltype_col = None
    if celltype_col_raw is not None:
        if not pd.api.types.is_categorical_dtype(adata.obs[celltype_col_raw]):
            adata.obs[celltype_col_raw] = adata.obs[celltype_col_raw].astype("category")
        adata.obs[standard_celltype_col] = adata.obs[celltype_col_raw].astype(
            "category"
        )
        celltype_col = standard_celltype_col

    # ---- Downstream inclusion flag (marks; never deletes) ----
    # Writes obs['include_in_downstream_analysis'] for EVERY run (all-True when
    # filtering is off) so the exported h5ad always carries one unambiguous field,
    # and back-fills the same decision into the consensus CSV.
    gating_report = gating.annotate_downstream_inclusion(
        adata,
        exclude_low_confidence=options.exclude_low_confidence_de,
        excluded_tiers=excluded_tiers,
        cluster_col="leiden",
    )
    _consensus_csv = celltype_anno_dir / f"{analysis_name}_consensus_annotation.csv"
    gating.stamp_consensus_table_inclusion(_consensus_csv, adata, cluster_col="leiden")

    # Annotation QC figures. Both answer "is this label right?", which no existing
    # output does: the dotplot shows whether each label has its defining genes, the
    # agreement grid shows which annotator dissented and whether the final call
    # followed them. Never allowed to fail the run.
    try:
        celltype_qc_plots.plot_annotation_marker_dotplot(
            adata,
            celltype_col=celltype_col,
            out_dir=celltype_anno_dir,
            analysis_name=analysis_name,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[%s] annotation marker dotplot failed (%s).",
            analysis_name,
            e,
            exc_info=True,
        )
    try:
        celltype_qc_plots.plot_voter_agreement(
            _consensus_csv, out_dir=celltype_anno_dir, analysis_name=analysis_name
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[%s] voter-agreement plot failed (%s).", analysis_name, e, exc_info=True
        )

    if options.use_subtypes_for_downstream and "celltype_subtype" in adata.obs.columns:
        # Opt-in only. Subtypes come from ONE annotator (see celltype_subtype_source)
        # and are not consensus-validated, so this is off by default.
        logger.warning(
            "[%s] use_subtypes_for_downstream=True: downstream analyses will key off 'celltype_subtype' (single-annotator labels), not the consensus 'celltype'. 'celltype' itself is unchanged.",
            analysis_name,
        )

    if celltype_col is not None:
        sc.settings.figdir = celltype_anno_dir
        ct_pt_size = point_size(adata.n_obs, base=15)
        sc.pl.umap(
            adata,
            color=[celltype_col],
            legend_loc="right margin",
            size=ct_pt_size,
            show=False,
            save=f"_{analysis_name}_umap_celltypes.png",
        )
        plt.close("all")
        gc.collect()

        if "X_tsne" in adata.obsm:
            sc.pl.tsne(
                adata,
                color=[celltype_col],
                legend_loc="right margin",
                size=ct_pt_size,
                show=False,
                save=f"_{analysis_name}_tsne_celltypes.png",
            )
            plt.close("all")
            gc.collect()

        if not options.skip_pca_cluster_plots:
            sc.pl.pca(
                adata,
                color=[celltype_col],
                size=ct_pt_size,
                show=False,
                save=f"_{analysis_name}_pca_celltypes.png",
            )
            plt.close("all")
            gc.collect()

        ct_counts = adata.obs[celltype_col].value_counts().sort_values(ascending=False)
        # One bar per cell type: a fixed width crushed the rotated names together
        # once there were more than a handful, so the canvas grows with the count.
        bar_width = clamp_fig_inches(0.55 * len(ct_counts) + 2.5, minimum=10.0)
        fig, ax = plt.subplots(figsize=(bar_width, 5.0))
        ax.bar(ct_counts.index.astype(str), ct_counts.values)
        ax.set_xticks(range(len(ct_counts)))
        ax.set_xticklabels(ct_counts.index.astype(str), rotation=90)
        ax.set_ylabel("cells")
        ax.set_title(f"Cell types ({celltype_col})")
        plt.tight_layout()
        plt.savefig(
            celltype_anno_dir / f"{analysis_name}_celltype_composition_barplot.png",
            dpi=FIGURE_DPI,
        )
        plt.close("all")
        gc.collect()

        logger.info(
            "[%s] Computing celltype-specific marker genes (Scanpy)...", analysis_name
        )
        compute_celltype_markers(
            adata,
            celltype_col=celltype_col,
            out_dir=celltype_markers_dir,
            analysis_name=analysis_name,
            n_markers_per_type=50,
            reference_dir=reference_dir,
            skip_per_celltype_plots=options.skip_per_celltype_plots,
            skip_per_celltype_csvs=options.skip_per_celltype_csvs,
        )

        if "leiden" in adata.obs.columns:
            logger.info(
                "[%s] Making side-by-side embeddings (leiden vs cell type).",
                analysis_name,
            )
            sc.settings.figdir = celltype_anno_dir

            # Cell-type names are the long labels here; leiden's legend is what has
            # to clear the second panel's axis label, so the gap is measured from it.
            side_by_side = ["leiden", celltype_col]
            side_wspace = panel_wspace(adata, side_by_side)

            if not options.skip_pca_cluster_plots:
                sc.pl.pca(
                    adata,
                    color=side_by_side,
                    wspace=side_wspace,
                    size=ct_pt_size,
                    show=False,
                    save=f"_{analysis_name}_pca_leiden_vs_celltype.png",
                )
                plt.close("all")
                gc.collect()

            if "X_tsne" in adata.obsm:
                sc.pl.tsne(
                    adata,
                    color=side_by_side,
                    wspace=side_wspace,
                    size=ct_pt_size,
                    show=False,
                    save=f"_{analysis_name}_tsne_leiden_vs_celltype.png",
                )
                plt.close("all")
                gc.collect()

            sc.pl.umap(
                adata,
                color=side_by_side,
                wspace=side_wspace,
                size=ct_pt_size,
                show=False,
                save=f"_{analysis_name}_umap_leiden_vs_celltype.png",
            )
            plt.close("all")
            gc.collect()

    else:
        logger.info(
            "[CELLTYPE] No cell type column found for %s and no ML-based cell-type prediction could be applied.",
            analysis_name,
        )

    # Build mapping: Leiden cluster → "<clusterID>_<MajorCellType>"
    cluster_celltype_map = None
    if celltype_col is not None and "leiden" in adata.obs.columns:
        tmp = adata.obs[["leiden", celltype_col]].dropna()
        if not tmp.empty:
            cluster_celltype_map = {}
            ref_rows = []
            for cl, sub in tmp.groupby("leiden"):
                top_ct = sub[celltype_col].astype(str).value_counts().idxmax()
                # This label is carried into cluster_<label>_markers.csv below, so
                # it must be filename-safe at the point it is minted.
                safe_ct = safe_filename(top_ct)
                label = f"{cl}_{safe_ct}"
                cluster_celltype_map[str(cl)] = label
                ref_rows.append(
                    {
                        "leiden_cluster": cl,
                        "major_celltype_label": top_ct,
                        "cluster_celltype_label": label,
                    }
                )
            logger.info(
                "[%s] Cluster → celltype map: %s", analysis_name, cluster_celltype_map
            )

            reference_dir.mkdir(parents=True, exist_ok=True)
            ref_df = pd.DataFrame(ref_rows)
            ref_map_file = (
                reference_dir / f"{analysis_name}_cluster_to_celltype_map.csv"
            )
            atomic_to_csv(ref_df, ref_map_file, index=False)
            logger.info(
                "[%s] Saved cluster→celltype reference table: %s",
                analysis_name,
                ref_map_file,
            )

    # ========= Cluster marker genes + intercluster DEG =========
    try:
        logger.info(
            "[%s] Computing cluster marker genes (leiden, t-test) via subprocess for Celery compatibility...",
            analysis_name,
        )

        intercluster_dir = clustering_dir / "intercluster_analysis_deg"
        intercluster_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL FOR CELERY: Run rank_genes_groups in isolated subprocess
        # This avoids deadlocks with matplotlib/R resources in Celery prefork mode
        # Save adata to temporary h5ad file
        temp_h5ad = intercluster_dir / f"{analysis_name}_temp_for_rank_genes.h5ad"
        logger.info(
            "[%s] Saving temporary h5ad for subprocess: %s", analysis_name, temp_h5ad
        )
        adata.write_h5ad(temp_h5ad)

        # Build subprocess command
        # Use file path instead of module path for cross-environment compatibility
        _rank_genes_script = Path(__file__).parent / "rank_genes_subprocess.py"
        cmd = [
            sys.executable,
            str(_rank_genes_script),
            "--h5ad",
            str(temp_h5ad),
            "--output-dir",
            str(intercluster_dir),
            "--analysis-name",
            analysis_name,
            "--method",
            "t-test",
            "--n-genes",
            "50",
            "--groupby",
            "leiden",
        ]

        logger.info(
            "[%s] Running rank_genes_groups in isolated subprocess...", analysis_name
        )
        logger.debug("[%s] Command: %s", analysis_name, " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.stdout:
                logger.info("[%s] Subprocess stdout:\n%s", analysis_name, result.stdout)
            if result.stderr:
                logger.warning(
                    "[%s] Subprocess stderr:\n%s", analysis_name, result.stderr
                )

            logger.info(
                "[%s] rank_genes_groups subprocess completed successfully",
                analysis_name,
            )

        except subprocess.TimeoutExpired as e:
            logger.error(
                "[%s] rank_genes_groups subprocess timed out after 1 hour",
                analysis_name,
            )
            raise RuntimeError("rank_genes_groups execution timed out") from e

        except subprocess.CalledProcessError as e:
            logger.exception(
                "[%s] rank_genes_groups subprocess failed with exit code %s",
                analysis_name,
                e.returncode,
            )
            if e.stdout:
                logger.exception("[%s] Subprocess stdout:\n%s", analysis_name, e.stdout)
            if e.stderr:
                logger.exception("[%s] Subprocess stderr:\n%s", analysis_name, e.stderr)
            raise RuntimeError(f"rank_genes_groups execution failed: {e.stderr}") from e

        # Load results from subprocess output
        markers_csv = intercluster_dir / "intercluster_cluster_markers.csv"
        if not markers_csv.exists():
            raise FileNotFoundError(f"Markers CSV not found: {markers_csv}")

        markers_all = pd.read_csv(markers_csv)
        logger.info(
            "[%s] Loaded %s marker gene records from subprocess",
            analysis_name,
            len(markers_all),
        )

        # Add cluster column and cluster_celltype_label mapping
        markers_all["cluster"] = markers_all["group"].astype(str)
        if cluster_celltype_map is not None:
            markers_all["cluster_celltype_label"] = (
                markers_all["cluster"]
                .map(cluster_celltype_map)
                .fillna(markers_all["cluster"])
            )

        # Load updated h5ad with rank_genes_groups results
        updated_h5ad = intercluster_dir / f"{analysis_name}_with_rank_genes.h5ad"
        if updated_h5ad.exists():
            logger.info(
                "[%s] Loading updated h5ad with rank_genes_groups results...",
                analysis_name,
            )
            adata_updated = sc.read_h5ad(updated_h5ad)
            # Copy rank_genes_groups results back to main adata
            if "rank_genes_groups" in adata_updated.uns:
                adata.uns["rank_genes_groups"] = adata_updated.uns["rank_genes_groups"]
                logger.info(
                    "[%s] Copied rank_genes_groups results back to main adata",
                    analysis_name,
                )
            # Clean up temporary files
            updated_h5ad.unlink()

        # Clean up temporary h5ad
        if temp_h5ad.exists():
            temp_h5ad.unlink()
            logger.debug("[%s] Cleaned up temporary h5ad: %s", analysis_name, temp_h5ad)

        # Save markers CSV with cluster mapping (overwrite the one from subprocess)
        intercluster_csv = intercluster_dir / "intercluster_cluster_markers.csv"
        atomic_to_csv(markers_all, intercluster_csv, index=False)
        logger.info(
            "[%s] Wrote intercluster markers with cluster mapping: %s",
            analysis_name,
            intercluster_csv,
        )

        markers_all["cluster"] = markers_all["group"].astype(str)
        if cluster_celltype_map is not None:
            markers_all["cluster_celltype_label"] = (
                markers_all["cluster"]
                .map(cluster_celltype_map)
                .fillna(markers_all["cluster"])
            )

        intercluster_csv = intercluster_dir / "intercluster_cluster_markers.csv"
        atomic_to_csv(markers_all, intercluster_csv, index=False)
        logger.info(
            "[%s] Wrote intercluster markers: %s", analysis_name, intercluster_csv
        )

        if options.skip_per_cluster_marker_csvs:
            logger.info(
                "[%s] Skipping per-cluster marker CSVs (skip_per_cluster_marker_csvs=True). Combined CSV retained.",
                analysis_name,
            )
        else:
            for cl, subdf in markers_all.groupby("cluster"):
                label = (
                    markers_all.loc[
                        markers_all["cluster"] == cl, "cluster_celltype_label"
                    ].iloc[0]
                    if "cluster_celltype_label" in markers_all.columns
                    else cl
                )
                safe_label = safe_filename(label)
                out_f = intercluster_dir / f"cluster_{safe_label}_markers.csv"
                atomic_to_csv(subdf, out_f, index=False)

        if options.do_pathway_clustering:
            pathway_root_dir.mkdir(parents=True, exist_ok=True)
            cluster_pathway_dir = pathway_root_dir / "cluster_marker_enrichment"
            logger.info(
                "[%s] Running pathway enrichment for cluster markers...", analysis_name
            )
            run_cluster_marker_enrichment(
                markers_all,
                out_dir=cluster_pathway_dir,
                analysis_name=analysis_name,
                # Rank-based selection. The clusters were defined from this matrix,
                # so a marker p-value cutoff would launder a selection-biased
                # statistic into the pathway results (see marker_stats).
                pval_col=None,
                top_n=200,
                cluster_celltype_map=cluster_celltype_map,
            )
        else:
            logger.info(
                "[%s] Skipping cluster-marker pathway enrichment (do_pathway_clustering=False).",
                analysis_name,
            )

    except Exception as e:
        logger.warning(
            "[%s] intercluster marker computation failed: %s",
            analysis_name,
            e,
            exc_info=True,
        )

    # ========= Save processed AnnData =========
    processed_h5ad = out_dir / f"{analysis_name}_processed_scanpy_output.h5ad"
    adata.write_h5ad(processed_h5ad)

    # ========= Provenance manifest (reproducibility) =========
    # Verify the annotation-only integration boundary before recording it: the DE
    # steps below read layers['counts'] / .raw / the design columns, none of which an
    # integration step writes. Checked on `adata` because the DE object is a cell
    # subset of it and carries the identical layers/raw/obs schema.
    integration_record["de_boundary"] = integration_policy.check_de_inputs_uncorrected(
        adata,
        celltype_col=celltype_col,
        analysis_name=analysis_name,
    )

    try:
        manifest_path = write_run_manifest(
            out_dir,
            analysis_name=analysis_name,
            seed=seed,
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars),
            params={
                # Preprocessing numerics, recorded so a run's numbers stay traceable
                # to the parameters that produced them (see scanpy_params.py).
                "preprocessing": {
                    "normalize_target_sum": NORMALIZE_TARGET_SUM,
                    "scale_max_value": SCALE_MAX_VALUE,
                    "pca_n_comps": PCA_N_COMPS,
                    "neighbors_n_pcs": NEIGHBORS_N_PCS,
                    "neighbors_n_neighbors": NEIGHBORS_N_NEIGHBORS,
                },
                "batch_key": options.batch_key,
                "integration_method": options.integration_method,  # requested
                "integration_method_used": integration_method_used,  # actually executed
                # Full integration policy: the batch/condition design, the
                # annotation-only scope, what the corrected graph did and did not
                # touch, and the verified DE boundary.
                "integration": integration_record,
                "hvg_selection": hvg_report,  # rung actually used + any excluded batch
                "do_groupwise_de": options.do_groupwise_de,
                "do_pseudobulk_de": options.do_pseudobulk_de,
                # --- contrast design: which arm is the reference, and why ---
                # Recorded because the SIGN of every log fold change depends on it.
                "de_contrast": _contrast_provenance(
                    adata,
                    group_col=group_col,
                    reference_group=options.reference_group,
                    lfc_threshold=options.de_lfc_threshold,
                    alpha=options.de_alpha,
                ),
                "exclude_low_confidence_de": options.exclude_low_confidence_de,
                "do_doublet_detection": options.do_doublet_detection,
                "remove_doublets": options.remove_doublets,
                "do_dpt": options.do_dpt,
                "dpt_root_group": options.dpt_root_group,
                "do_pathway_clustering": options.do_pathway_clustering,
                # --- clustering: the resolution ACTUALLY used, plus the audit ---
                "cluster_resolution": float(resolution),
                "leiden_resolution": float(resolution),
                "clustering": {
                    "leiden_resolution": float(resolution),
                    "cluster_col": "leiden",
                    "n_clusters": len(clusters),
                    "evaluate_resolutions": bool(options.evaluate_resolutions),
                    "resolution_candidates": (
                        [float(r) for r in resolution_eval["resolution"].tolist()]
                        if resolution_eval is not None
                        else None
                    ),
                    "resolution_evaluation_csv": (
                        str(clustering_dir / clustering_mod.EVALUATION_CSV_NAME)
                        if resolution_eval is not None
                        else None
                    ),
                    "min_cluster_cells": int(min_cluster_cells_eff),
                    "selection_rule": "user_configured (no automatic re-selection)",
                },
                # --- QC: requested vs applied, with per-rule removal counts ---
                "qc": {
                    "requested": {
                        "min_genes": options.qc_min_genes,
                        "max_genes": options.qc_max_genes,
                        "max_mito_percent": options.qc_max_mito_percent,
                        "remove_doublets": options.remove_doublets,
                    },
                    "applied": qc_thresholds.as_dict(),
                    "cells_removed": qc_report,
                    # legacy keys retained so older manifest readers keep working
                    "min_genes": qc_thresholds.min_genes,
                    "max_genes": qc_thresholds.max_genes,
                    "max_pct_mt": qc_thresholds.max_mito_percent,
                },
                # --- annotation: RESOLVED resources, not the literal "auto" ---
                "annotation": _annotation_provenance(
                    adata,
                    enable_celltypist=options.enable_celltypist,
                    enable_llm=options.enable_knowledge_based,
                    enable_singler=options.enable_singler,
                    enable_pubmed=options.enable_pubmed,
                    tissue=options.tissue,
                    species=options.species,
                    celltypist_model=options.celltypist_model,
                    singler_reference=options.singler_reference,
                    celltype_source=celltype_source,
                    reuse_info=celltype_reuse_info,
                    use_subtypes_for_downstream=options.use_subtypes_for_downstream,
                ),
                # --- confidence filtering configuration + what it actually did ---
                "confidence_filtering": {
                    "exclude_low_confidence_de": bool(
                        options.exclude_low_confidence_de
                    ),
                    "excluded_consensus_tiers": list(excluded_tiers),
                    "report": gating_report,
                },
            },
            extra={
                "initial_cells": int(initial_cells),
                "initial_genes": int(initial_genes),
            },
        )
        logger.info("[%s] Wrote provenance manifest: %s", analysis_name, manifest_path)
    except Exception as e:
        logger.warning(
            "[%s] Failed to write provenance manifest (%s).",
            analysis_name,
            e,
            exc_info=True,
        )

    # ========= Summary report =========
    summary_lines = [
        f"=== {analysis_name} ===",
        f"Output folder: {out_dir}",
        f"Initial cells: {initial_cells}",
        f"Initial genes: {initial_genes}",
        f"Genes after min_cells filter: {genes_after_min_cells}",
        f"Cells after QC filters: {adata.n_obs}",
        f"HVGs used: {hvg_count} (method: {hvg_report['method']})",
        *(
            [
                f"HVG ranking excluded {len(hvg_report['excluded_batches'])} batch(es) too "
                f"small to fit — cells retained: "
                f"{', '.join(hvg_report['excluded_batches'])}"
            ]
            if hvg_report.get("excluded_batches")
            else []
        ),
        f"Final shape (post-HVG selection for embeddings, ALL genes retained for DE): "
        f"{adata.n_obs} cells x {adata.n_vars} genes",
        f"Leiden clusters: {len(clusters)}",
        f"Leiden resolution: {resolution}",
        f"Integration: {integration_method_used} (batch_key={options.batch_key!r}); scope="
        f"{integration_record['scope']} — drives clustering/UMAP/annotation only, "
        f"DE reads raw counts per donor.",
        f"Batch/condition design: {integration_record['batch_design']['verdict']} — "
        f"{integration_record['batch_design']['interpretation']}",
        *(
            [
                f"INTEGRATION WARNING: {w}"
                for w in integration_record.get("warnings", [])
                if "CONFOUNDED" in w or "batch_key ==" in w
            ]
        ),
        f"QC thresholds applied: min_genes>{qc_thresholds.min_genes}, "
        f"max_genes<{qc_thresholds.max_genes}, pct_mt<{qc_thresholds.max_mito_percent}",
        f"Cells removed by QC: {qc_report.get('removed_total')} "
        f"(min_genes={qc_report.get('removed_min_genes')}, "
        f"max_genes={qc_report.get('removed_max_genes')}, "
        f"mito={qc_report.get('removed_max_mito_percent')}); "
        f"doublets removed: {qc_report.get('removed_doublets')}",
        f"Processed AnnData (Scanpy): {processed_h5ad}",
    ]
    if resolution_eval is not None:
        summary_lines.append(
            f"Resolution audit: {clustering_dir / clustering_mod.EVALUATION_CSV_NAME} "
            f"(primary resolution retained: {resolution})"
        )
    if "celltype" in adata.obs.columns:
        celltype_col_used = "celltype"
        ct_counts2 = adata.obs[celltype_col_used].value_counts()
        summary_lines.append(f"Celltype column (standard): {celltype_col_used}")
        if celltype_source is not None:
            summary_lines.append(f"Celltype annotation source: {celltype_source}")
        # Annotation-quality accounting: tiers, mixed clusters, and what was gated
        # out of inferential analyses. Reported here because this file is what the
        # HTML/PDF report reads back.
        if "consensus_tier" in adata.obs.columns:
            _tier_by_cluster = adata.obs.groupby(
                adata.obs["leiden"].astype(str), observed=True
            )["consensus_tier"].agg(lambda s: str(s.iloc[0]))
            _tc = _tier_by_cluster.value_counts().to_dict()
            summary_lines.append(
                "Consensus tiers (clusters): "
                + ", ".join(f"{k}={v}" for k, v in sorted(_tc.items()))
            )
        if "mixed_cluster_flag" in adata.obs.columns:
            _mixed_clusters = sorted(
                adata.obs.loc[adata.obs["mixed_cluster_flag"].astype(bool), "leiden"]
                .astype(str)
                .unique()
                .tolist()
            )
            summary_lines.append(
                f"Mixed/heterogeneous clusters (CellTypist): {len(_mixed_clusters)} "
                f"{_mixed_clusters}"
            )
        summary_lines.append(
            f"Downstream exclusion: exclude_low_confidence_de="
            f"{options.exclude_low_confidence_de} (tiers={excluded_tiers}); "
            f"{gating_report.get('n_cells_excluded', 0)} cell(s) in "
            f"{gating_report.get('n_clusters_excluded', 0)} cluster(s) "
            f"{gating_report.get('excluded_clusters', [])} excluded from "
            f"DE/composition; all cells retained in the h5ad."
        )
        summary_lines.append(
            f"Downstream gating detail: {gating_report.get('reason', '')}"
        )
        summary_lines.append("Celltype counts:")
        for ct, n in ct_counts2.items():
            summary_lines.append(f"  {ct}: {int(n)} cells")
    else:
        summary_lines.append("Celltype annotation: not available")

    summary_file = summary_dir / f"{analysis_name}_analysis_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logger.info("\n".join(summary_lines))

    # ========= Group-wise DE & downstream (single-cell only) =========
    celltype_col = "celltype" if "celltype" in adata.obs.columns else None

    # Which cells may carry an inferential claim. Driven by
    # obs['include_in_downstream_analysis'] (written above by downstream_gating), so
    # the DE object and the exported flag can never disagree. The FULL object — all
    # cells, all labels, all tiers — was already written to disk above and is what
    # the UMAPs and audit tables use. Default (filtering off) is byte-identical to
    # the previous behaviour.
    adata_de, de_subset_report = gating.subset_for_downstream(
        adata,
        group_col=group_col,
        sample_col="sample",
        require_replication=options.do_pseudobulk_de,
        analysis_name=analysis_name,
    )
    gating_report["downstream_subset"] = de_subset_report

    if options.do_groupwise_de:
        deg_root_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[%s] Comparing cell type proportions between groups...", analysis_name
        )
        plot_groupwise_celltype_proportions(
            adata_de,
            group_col=group_col,
            celltype_col=celltype_col,
            out_dir=celltype_anno_dir,
        )
        # Donor-level version of the same comparison. The pooled fractions above
        # treat every CELL as an independent observation, so between-donor variation
        # reads as a group difference; this makes the donor the unit of replication
        # and reports a test. See celltype_qc_plots for the measured case.
        try:
            celltype_qc_plots.plot_per_donor_proportions(
                adata_de,
                group_col=group_col,
                celltype_col=celltype_col,
                sample_col="sample",
                out_dir=celltype_anno_dir,
                analysis_name=analysis_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[%s] per-donor proportion plot failed (%s).",
                analysis_name,
                e,
                exc_info=True,
            )

        ct_deg_pathway_dir = pathway_root_dir / "celltype_deg_enrichment"

        # PRIMARY group DE: donor/sample-level pseudobulk DESeq2 (the statistically
        # correct CASE-vs-CONTROL test — sample is the unit of replication, so it
        # avoids the pseudoreplication of cell-level Wilcoxon).
        pseudobulk_dir = deg_root_dir / "pseudobulk_deg"
        if options.do_pseudobulk_de:
            logger.info(
                "[%s] Running PSEUDOBULK (sample-level) DESeq2 group DE...",
                analysis_name,
            )
            try:
                compute_pseudobulk_de(
                    adata_de,
                    group_col=group_col,
                    sample_col="sample",
                    celltype_col=celltype_col,
                    out_dir=pseudobulk_dir,
                    reference_group=options.reference_group,
                    lfc_threshold=options.de_lfc_threshold,
                    alpha=options.de_alpha,
                )
            except Exception as e:
                logger.warning(
                    "[%s] Pseudobulk DE failed (%s); continuing.",
                    analysis_name,
                    e,
                    exc_info=True,
                )
            # The DE tables are multi-MB CSVs; a volcano makes each one readable.
            try:
                celltype_qc_plots.plot_pseudobulk_volcanoes(
                    pseudobulk_dir / "per_celltype",
                    out_dir=pseudobulk_dir / "volcano",
                    analysis_name=analysis_name,
                    padj_cut=options.de_alpha,
                    lfc_cut=options.de_lfc_threshold,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[%s] volcano plots failed (%s).", analysis_name, e, exc_info=True
                )

        if celltype_col is not None:
            # SECONDARY / exploratory: cell-level Wilcoxon per cell type. Kept for
            # per-cell effect exploration ONLY — NOT valid for cohort significance
            # (pseudoreplication). Use the pseudobulk_deg/ results for group claims.
            logger.info(
                "[%s] Running exploratory cell-level DE per cell type (Scanpy Wilcoxon)...",
                analysis_name,
            )
            compute_de_by_celltype(
                adata_de,
                celltype_col=celltype_col,
                group_col=group_col,
                deg_root_dir=deg_root_dir,
                reference_group=options.reference_group,
            )

            if options.do_pathway_clustering:
                pathway_root_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "[%s] (Optional) run pathway enrichment for celltype-specific DEGs here if desired.",
                    analysis_name,
                )
                # If you want: run_celltype_deg_enrichment_from_files(...) can be added.

            # The integrated per-cell-type summary reads the DONOR-LEVEL pseudobulk
            # tables. It previously read celltype_specific_deg/ — cell-level Wilcoxon,
            # which pseudoreplicates donors — so 08_reference_summary/ presented cohort
            # claims built on inflated p-values. Only fall back to the exploratory
            # tables when pseudobulk did not run (too few donors), and the summariser
            # then stamps that limitation into its own output.
            _summary_deg_dir = pseudobulk_dir / "per_celltype"
            if not (
                options.do_pseudobulk_de
                and _summary_deg_dir.exists()
                and any(_summary_deg_dir.glob("*.csv"))
            ):
                _summary_deg_dir = deg_root_dir / "celltype_specific_deg"
                logger.warning(
                    "[%s] No pseudobulk per-cell-type DE available; the reference summary falls back to the EXPLORATORY cell-level tables and will be labelled as such (not cohort-valid).",
                    analysis_name,
                )
            try:
                summarize_celltype_degs_markers_pathways(
                    out_dir=out_dir,
                    analysis_name=analysis_name,
                    deg_dir=_summary_deg_dir,
                    celltype_dir=celltype_markers_dir,
                    ct_deg_pathway_dir=ct_deg_pathway_dir,
                    padj_cutoff=options.de_alpha,
                )
            except Exception as e:
                logger.warning(
                    "[SUMMARY-CT-DEG] Failed to summarise celltype DEGs+markers+pathways: %s",
                    e,
                    exc_info=True,
                )

        logger.info("[%s] Building group-specific UMAPs...", analysis_name)
        color_col_for_umap = celltype_col if celltype_col is not None else cluster_col
        group_umap_dir = dimred_dir / "groupwise_embeddings"
        group_umap_dir.mkdir(parents=True, exist_ok=True)
        plot_group_specific_umaps(
            adata,
            group_col=group_col,
            color_col=color_col_for_umap,
            out_dir=group_umap_dir,
        )

    logger.info("=== Done Scanpy pipeline: %s ===", analysis_name)
