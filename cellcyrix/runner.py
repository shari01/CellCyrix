"""
runner.py — config loading and driver dispatch, shared by both entry points.

This is the pipeline's single run path. Two thin front ends call it and neither adds
behaviour:

  * the repository's root ``main.py`` — for a clone, with repo-relative defaults
  * ``cellcyrix.cli`` — the ``cellcyrix`` console script, for an installed copy

The logic lives here rather than in ``main.py`` because ``main.py`` sits at the
repository root and is not part of the package: ``pip install`` puts ``cellcyrix`` on
the path but puts ``main.py`` nowhere, so an installed copy had no way to start a run
and a console script could not reach the code. Moving the body into the package fixes
that without duplicating it — there is one ``run_from_config`` underneath both.

Neither ``config_path`` nor ``output_root`` has a default here on purpose. The defaults
are repo-relative (``<repo>/config.yaml``, ``<repo>/outputs``) and only the caller knows
where its own root is; a package-relative default would resolve inside site-packages.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from cellcyrix.single_cell_pipeline_agent.singlecell_10x import (
    run_pipeline,
    run_pipeline_multi,
)
from cellcyrix.single_cell_pipeline_agent.singlecell_10x.pipeline_options import (
    PipelineOptions,
)

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and lightly validate the YAML config file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        The parsed config mapping (empty dict if the file is empty).

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If the YAML top level is not a mapping.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Create one (see config.yaml) or point CONFIG_FILE at your own YAML."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got {type(config).__name__}: {config_path}"
        )
    return config


# Nested config sections mapped onto the flat driver parameters.
#
# The drivers take flat keyword arguments, and ``_build_kwargs`` filters on
# ``inspect.signature``, so a nested block would otherwise be silently dropped as an
# unknown option. This table is what makes ``clustering:`` / ``qc:`` / ``annotation:``
# / ``downstream:`` actually reach the pipeline. A section key not listed here is
# reported as unknown instead of being ignored quietly.
#
# ``None`` on the right means "same name, no prefix".
_SECTION_KEY_MAP: dict[str, dict[str, str | None]] = {
    "clustering": {
        "leiden_resolution": None,
        "evaluate_resolutions": None,
        "resolution_candidates": None,
        "min_cluster_cells": None,
        "resolution_silhouette_max_cells": None,
    },
    "qc": {
        "min_genes": "qc_min_genes",
        "max_genes": "qc_max_genes",
        "max_mito_percent": "qc_max_mito_percent",
        "remove_doublets": None,
        "do_doublet_detection": None,
    },
    "annotation": {
        "mixed_cluster_min_dominant_fraction": None,
        "mixed_cluster_second_label_fraction": None,
        "use_subtypes_for_downstream": None,
        "reuse_existing_final_annotation": None,
        "final_annotation_column": None,
    },
    "downstream": {
        "exclude_low_confidence_de": None,
        "excluded_consensus_tiers": None,
        "do_pseudobulk_de": None,
    },
    "de": {
        "reference_group": None,
        "lfc_threshold": "de_lfc_threshold",
        "alpha": "de_alpha",
    },
}


def flatten_sections(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested config sections into flat driver keyword arguments.

    Sections are read from the top level of the config AND from ``common:`` (so
    either placement works). Unrecognized keys inside a known section are reported
    rather than dropped in silence. Flat keys elsewhere in the config are untouched
    and still win, because they are merged after this in `run_from_config`.

    Args:
        config: The parsed YAML config mapping.

    Returns:
        Flat keyword arguments for the pipeline drivers.
    """
    flat: dict[str, Any] = {}
    sources = [config] + [config.get("common") or {}]
    for src in sources:
        if not isinstance(src, dict):
            continue
        for section, keymap in _SECTION_KEY_MAP.items():
            block = src.get(section)
            if block is None:
                continue
            if not isinstance(block, dict):
                logger.warning(
                    "Section '%s:' must be a mapping, got %s; ignoring.",
                    section,
                    type(block).__name__,
                )
                continue
            for key, value in block.items():
                if key not in keymap:
                    logger.warning("Unknown option '%s.%s'; ignoring.", section, key)
                    continue
                flat[keymap[key] or key] = value
    if flat:
        logger.info("Applied nested sections -> %s", sorted(flat))
    return flat


def _build_kwargs(fn: Callable[..., Any], *blocks: dict[str, Any]) -> dict[str, Any]:
    """Merge config blocks and keep only the keys the driver accepts.

    Later blocks win. Unknown keys are reported and dropped so a stray or misplaced
    YAML option never crashes the run.

    Accepted keys are the driver's own named parameters PLUS every field of
    `PipelineOptions` — the ~50 shared options reach a driver through its
    ``**overrides``, so they are not visible to `inspect.signature` and filtering on
    the signature alone would silently discard the entire config.

    Args:
        fn: The driver function to build arguments for.
        *blocks: Config blocks to merge, in increasing order of precedence.

    Returns:
        Keyword arguments `fn` accepts.
    """
    merged: dict[str, Any] = {}
    for block in blocks:
        if block:
            merged.update(block)

    accepted = set(inspect.signature(fn).parameters) | PipelineOptions.field_names()
    kwargs = {k: v for k, v in merged.items() if k in accepted}

    unknown = sorted(set(merged) - accepted)
    if unknown:
        logger.warning("Ignoring options not used by %s: %s", fn.__name__, unknown)
    return kwargs


def run_from_config(config_path: Path, output_root: Path) -> Path:
    """Run the pipeline described by a YAML config.

    Args:
        config_path: YAML config to drive the run.
        output_root: Directory that the config's `out_name` is resolved against, so
            outputs never land in the process working directory.

    Returns:
        The run's output directory.

    Raises:
        ValueError: If `mode:` is missing/unknown, or the mode's required input
            directory key is absent.
    """
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    logger.info("Loaded: %s", config_path)

    mode = str(config.get("mode", "")).strip().lower()
    # Nested sections resolve first; the flat `common:`/mode blocks are merged after,
    # so an explicit flat key still overrides a section value.
    sections = flatten_sections(config)
    # Drop the nested blocks from `common:` once flattened, so they are not reported
    # as "unknown options" (they were consumed by flatten_sections).
    common = {
        k: v
        for k, v in (config.get("common") or {}).items()
        if k not in _SECTION_KEY_MAP
    }

    if mode == "single":
        block = dict(config.get("single") or {})
        if not block.get("single_10x_dir"):
            raise ValueError(
                "single mode requires 'single_10x_dir' under the 'single:' section."
            )
        block["single_10x_dir"] = Path(block["single_10x_dir"])
        block["output_root"] = output_root
        kwargs = _build_kwargs(run_pipeline, sections, common, block)
        logger.info("Running SINGLE pipeline on: %s", kwargs["single_10x_dir"])
        output_dir = run_pipeline(**kwargs)

    elif mode == "multi":
        block = dict(config.get("multi") or {})
        if not block.get("multi_base_dir"):
            raise ValueError(
                "multi mode requires 'multi_base_dir' under the 'multi:' section."
            )
        block["multi_base_dir"] = Path(block["multi_base_dir"])
        block["output_root"] = output_root
        kwargs = _build_kwargs(run_pipeline_multi, sections, common, block)
        logger.info("Running MULTI pipeline on: %s", kwargs["multi_base_dir"])
        output_dir = run_pipeline_multi(**kwargs)

    else:
        raise ValueError(
            f"Unknown mode: {mode!r} in {config_path} (expected 'single' or 'multi')."
        )

    logger.info("Pipeline completed! Results saved to: %s", output_dir)
    logger.info("Report (if generated) saved to: %s", output_dir / "singlecell_report")
    return output_dir


__all__ = [
    "load_config",
    "flatten_sections",
    "run_from_config",
    "_SECTION_KEY_MAP",
    "_build_kwargs",
]
