"""
pipeline_options.py — the single declaration of the options both drivers accept.

Why this exists
---------------
``run_pipeline`` (single sample) and ``run_pipeline_multi`` (cohort) each declared 53
parameters, and 51 of them were IDENTICAL — same name, same default, same meaning.
``run_scanpy_pipeline`` repeated 41 of those a third time. That is the DRY problem
Rule 7.1 describes: adding an option meant editing three signatures, and adding it to
only two of them produced a silent behaviour DIFFERENCE between single and multi mode
rather than an error.

:class:`PipelineOptions` is now the one place those options are declared. Both drivers
AND ``pipeline.run_scanpy_pipeline`` take ``options: PipelineOptions`` and pass the same
object down, so a new option is added here, once — and ``run_scanpy_pipeline`` itself went
from 47 parameters to 6.

The drivers still accept loose keywords (``**overrides``) so every existing call site
keeps working unchanged. That path is NOT permissive: an unknown keyword raises
:class:`UnknownPipelineOption` naming the closest real option, because a silently
ignored ``qc_max_mito`` (the real name is ``qc_max_mito_percent``) means a run that
looks configured and is not.

``main.py`` resolves YAML config against :data:`SHARED_PARAMS` rather than against
``inspect.signature``, which is why that filtering keeps working now that the options
live in a dataclass instead of in the signatures.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

from .exceptions import PipelineInputError

# Parameters that only make sense for one input shape, and so are expected to differ.
SINGLE_ONLY_PARAMS: frozenset[str] = frozenset(
    {
        "single_10x_dir",  # the one 10x folder
        "sample_label",  # obs['sample'] for that folder
        "group_label",  # obs['group'] for that folder
    }
)

MULTI_ONLY_PARAMS: frozenset[str] = frozenset(
    {
        "multi_base_dir",  # the <group>/<sample> tree
        "group_map_path",  # metadata.csv / group_map.csv override
        "run_per_sample",  # also run each sample on its own
    }
)

# Options that must be present, identically, on BOTH drivers. Kept as a literal so a
# rename or a dropped option is a diff in this file rather than a silent divergence.
SHARED_PARAMS: frozenset[str] = frozenset(
    {
        # output location
        "out_name",
        "output_root",
        # stage toggles
        "do_pathway_clustering",
        "do_groupwise_de",
        "do_dpt",
        "dpt_root_group",
        "generate_report",
        "prepare_for_bisque",
        # integration
        "batch_key",
        "integration_method",
        # context / assets
        "geo_json_path",
        "logos_dir",
        # annotation voters
        "enable_celltypist",
        "enable_knowledge_based",
        "enable_singler",
        "enable_pubmed",
        "tissue",
        "species",
        "celltypist_model",
        "singler_reference",
        # clustering
        "leiden_resolution",
        "evaluate_resolutions",
        "resolution_candidates",
        "min_cluster_cells",
        "resolution_silhouette_max_cells",
        # QC
        "qc_min_genes",
        "qc_max_genes",
        "qc_max_mito_percent",
        "do_doublet_detection",
        "remove_doublets",
        # annotation quality / reuse
        "mixed_cluster_min_dominant_fraction",
        "mixed_cluster_second_label_fraction",
        "use_subtypes_for_downstream",
        "reuse_existing_final_annotation",
        "final_annotation_column",
        # downstream confidence gating
        "exclude_low_confidence_de",
        "excluded_consensus_tiers",
        "do_pseudobulk_de",
        # contrast design
        "reference_group",
        "de_lfc_threshold",
        "de_alpha",
        "seed",
        # output-volume skip flags
        "skip_tsne",
        "skip_pca_cluster_plots",
        "skip_per_celltype_plots",
        "skip_per_celltype_csvs",
        "skip_per_cluster_marker_csvs",
        # cleanup flags
        "cleanup_raw_pathway_csvs",
        "cleanup_dedup_logs",
        "cleanup_pipeline_log",
        "cleanup_per_cluster_marker_csvs",
    }
)


class UnknownPipelineOption(PipelineInputError):
    """A keyword passed to a driver is not a pipeline option."""


@dataclass(frozen=True)
class PipelineOptions:
    """Every option both pipeline drivers accept, declared once.

    Defaults are the SINGLE-sample defaults. :meth:`for_multi` overrides the four that
    are legitimately different for a cohort (a cohort has groups to contrast and
    batches to correct; one sample has neither).

    ``None`` on a numeric option means "not configured" and resolves to the historical
    hard-coded value inside the pipeline, so a config written before an option existed
    keeps producing identical results.
    """

    # --- output location ---
    out_name: str = "SC_RESULTS"
    output_root: Optional[str | Path] = None
    # --- stage toggles ---
    do_pathway_clustering: bool = True
    do_groupwise_de: bool = False
    do_dpt: bool = False
    dpt_root_group: str = "auto"
    generate_report: bool = True
    prepare_for_bisque: bool = True
    # --- integration ---
    batch_key: Optional[str] = None
    integration_method: Optional[str] = None  # "bbknn" or None
    # --- context / assets ---
    geo_json_path: Optional[str | Path] = None
    logos_dir: Optional[str | Path] = None
    # --- cell-type annotation (multi-method consensus; toggle any voter) ---
    enable_celltypist: bool = True
    enable_knowledge_based: bool = True  # AI marker-reasoning voter
    enable_singler: Optional[bool] = None  # None -> SCPIPE_ENABLE_SINGLER
    enable_pubmed: bool = True
    tissue: Optional[str] = None  # organ context; None/"auto" -> infer from GEO JSON
    species: Optional[str] = None  # None/"auto" -> infer from GEO taxon
    celltypist_model: str = "auto"
    singler_reference: str = "auto"
    # --- clustering (config: clustering.*) ---
    leiden_resolution: Optional[float] = None  # None -> 0.5
    evaluate_resolutions: bool = False
    resolution_candidates: Optional[list[float]] = (
        None  # None -> (0.2,0.4,0.5,0.6,0.8,1.0)
    )
    min_cluster_cells: Optional[int] = None  # None -> 20
    resolution_silhouette_max_cells: Optional[int] = None  # None -> 5000
    # --- QC thresholds (config: qc.*) ---
    qc_min_genes: Optional[int] = None  # None -> 200
    qc_max_genes: Optional[int] = None  # None -> 6000
    qc_max_mito_percent: Optional[float] = None  # None -> 15.0
    do_doublet_detection: bool = True
    remove_doublets: bool = True
    # --- annotation quality / reuse (config: annotation.*) ---
    mixed_cluster_min_dominant_fraction: Optional[float] = None  # None -> 0.70
    mixed_cluster_second_label_fraction: Optional[float] = None  # None -> 0.20
    use_subtypes_for_downstream: bool = False
    reuse_existing_final_annotation: bool = False
    final_annotation_column: Optional[str] = None
    # --- downstream confidence filtering (config: downstream.*) ---
    exclude_low_confidence_de: bool = False  # default OFF = pre-gating behaviour
    excluded_consensus_tiers: Optional[list[str]] = None  # None -> ["Low/Review"]
    do_pseudobulk_de: bool = True
    # --- contrast design (config: de.*) ---
    reference_group: Optional[str] = None  # CONTROL arm; None = detect from names
    de_lfc_threshold: float = 1.0  # DESeq2 formal null H0:|log2FC| <= this
    de_alpha: float = 0.05
    seed: Optional[int] = None  # None -> run_scanpy_pipeline's DEFAULT_SEED
    # --- skip flags (reduce output volume) ---
    skip_tsne: bool = True
    skip_pca_cluster_plots: bool = True
    skip_per_celltype_plots: bool = True
    skip_per_celltype_csvs: bool = True
    skip_per_cluster_marker_csvs: bool = True
    # --- cleanup flags (remove intermediates after processing) ---
    cleanup_raw_pathway_csvs: bool = True
    cleanup_dedup_logs: bool = True
    cleanup_pipeline_log: bool = False  # pipeline.log is a retained execution record
    cleanup_per_cluster_marker_csvs: bool = True

    @classmethod
    def field_names(cls) -> frozenset[str]:
        """Every option name this class declares."""
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def for_single(cls, **overrides: Any) -> PipelineOptions:
        """Options for a single-sample run.

        Args:
            **overrides: Any option name declared on this class.

        Returns:
            The single-sample defaults with `overrides` applied.

        Raises:
            UnknownPipelineOption: If an override is not a declared option.
        """
        return cls().merged(**overrides)

    @classmethod
    def for_multi(cls, **overrides: Any) -> PipelineOptions:
        """Options for a cohort run.

        Four defaults differ from single mode on purpose: a cohort has two or more
        groups to contrast (`do_groupwise_de`) and per-sample batch structure to
        correct (`batch_key`, `integration_method`), and its results go to a
        differently-named folder.

        Args:
            **overrides: Any option name declared on this class.

        Returns:
            The cohort defaults with `overrides` applied.

        Raises:
            UnknownPipelineOption: If an override is not a declared option.
        """
        base = cls(
            out_name="SC_RESULTS_MULTI",
            do_groupwise_de=True,
            batch_key="sample",
            integration_method="bbknn",
        )
        return base.merged(**overrides)

    def merged(self, **overrides: Any) -> PipelineOptions:
        """Return a copy with `overrides` applied, rejecting unknown names.

        Args:
            **overrides: Option names declared on this class.

        Returns:
            A new `PipelineOptions`; this instance is unchanged (the class is frozen).

        Raises:
            UnknownPipelineOption: If any name is not a declared option. The message
                names the closest match, because the realistic mistake is a near-miss
                (`qc_max_mito` for `qc_max_mito_percent`) and silently dropping it
                would leave the run mis-configured but apparently fine.
        """
        if not overrides:
            return self
        known = self.field_names()
        unknown = sorted(set(overrides) - known)
        if unknown:
            hints = []
            for name in unknown:
                close = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.6)
                hints.append(
                    f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
                )
            raise UnknownPipelineOption(
                f"{len(unknown)} unknown pipeline option(s): {', '.join(hints)}. "
                "Declared options are in singlecell_10x/pipeline_options.py."
            )
        return replace(self, **overrides)


@dataclass(frozen=True)
class ParameterDrift:
    """What a driver's real signature has that the declared contract does not, or vice versa.

    Attributes:
        function_name: The driver inspected.
        missing: Declared shared options absent from the signature.
        undeclared: Signature parameters that are neither shared nor mode-specific.
    """

    function_name: str
    missing: frozenset[str]
    undeclared: frozenset[str]

    @property
    def ok(self) -> bool:
        """True when the signature matches the declared contract exactly."""
        return not self.missing and not self.undeclared

    def describe(self) -> str:
        """Human-readable summary for a test failure message."""
        parts = [f"{self.function_name}:"]
        if self.missing:
            parts.append(f"missing shared options {sorted(self.missing)}")
        if self.undeclared:
            parts.append(
                f"parameters not declared in pipeline_options {sorted(self.undeclared)}"
            )
        return " ".join(parts) if len(parts) > 1 else f"{self.function_name}: ok"


def check_parameter_contract(
    function: Callable[..., object], *, mode_only: frozenset[str]
) -> ParameterDrift:
    """Check a driver exposes the declared options and nothing undeclared.

    Shared options reach a driver through its ``options`` / ``**overrides``
    parameters rather than as named parameters, so they are checked against
    :meth:`PipelineOptions.field_names` — the declaration — while the signature is
    checked for stray parameters that belong in neither set.

    Args:
        function: `run_pipeline` or `run_pipeline_multi`.
        mode_only: The parameters this driver is allowed to have on its own
            (`SINGLE_ONLY_PARAMS` or `MULTI_ONLY_PARAMS`).

    Returns:
        The drift between the driver and the contract; `ParameterDrift.ok` when clean.
    """
    signature_params = frozenset(inspect.signature(function).parameters)
    plumbing = frozenset({"options", "overrides"})
    return ParameterDrift(
        function_name=function.__name__,
        missing=SHARED_PARAMS - PipelineOptions.field_names(),
        undeclared=signature_params - mode_only - plumbing,
    )


__all__ = [
    "SHARED_PARAMS",
    "SINGLE_ONLY_PARAMS",
    "MULTI_ONLY_PARAMS",
    "PipelineOptions",
    "UnknownPipelineOption",
    "ParameterDrift",
    "check_parameter_contract",
]
