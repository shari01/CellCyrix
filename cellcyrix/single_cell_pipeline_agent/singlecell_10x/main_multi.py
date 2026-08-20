"""
main_multi.py — multi-sample cohort driver.

Entry point for a CASE-vs-CONTROL / multi-stage cohort. It walks the
``<GROUP>/<SAMPLE>/`` input tree (honoring an auto-detected ``metadata.csv`` /
``group_map.csv`` as the source of truth for group assignment), loads and combines
every sample into one AnnData with ``sample`` and ``group`` in ``.obs``, then calls
``run_scanpy_pipeline`` once on the combined object (enabling batch integration and
donor-level pseudobulk group DE). Can optionally also run each sample individually.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import pandas as pd

from .loader_10x import load_10x_feature_barcode_matrix
from .main_single import _cleanup_intermediates, _find_geo_json
from .output_paths import resolve_output_dir
from .pipeline import run_scanpy_pipeline
from .pipeline_options import PipelineOptions
from .sc_to_bisq import process_h5ad_file

logger = logging.getLogger(__name__)

# Group labels that mean "do NOT analyze this sample". The upstream cohort tool
# (bridge_to_analysis.py / --auto-group) emits these markers in the group column.
_EXCLUDE_MARKERS = {"_excluded", "_review", "_non_expression"}

# `analysis_name` for the joint run over all samples. It becomes the prefix of every
# combined output file, so downstream readers and the report depend on this exact
# string — it is a named constant rather than a literal at the call site.
COMBINED_ANALYSIS_NAME = "combined_all_samples"


def _is_excluded_group(group: Optional[str]) -> bool:
    """True if a group label marks a sample that should NOT be analyzed."""
    if group is None:
        return True
    g = str(group).strip()
    if not g or g.lower() == "nan":
        return True
    if g.lower() in _EXCLUDE_MARKERS:
        return True
    # Treat any leading-underscore sentinel (_EXCLUDED, _REVIEW, ...) as skip.
    return g.startswith("_")


def _load_group_map(
    base_dir: Path,
    group_map_path: Optional[Path] = None,
) -> dict[str, str]:
    """
    Load an optional ``sample -> group`` mapping (upstream cohort-tool handshake).

    The cohort/download tool lays an analysis-ready tree next to a
    ``metadata.csv`` (columns ``sample, group, gse, gsm, ...``). When present,
    that file is the source of truth for which biological arm each sample
    belongs to. Auto-detected in ``base_dir`` unless an explicit
    ``group_map_path`` is given. ``metadata.csv``/``group_map.csv`` are
    preferred; ``.xlsx`` is a fallback.

    Returns
    -------
    dict
        ``{sample_label: group_name}``. Empty if no usable file is found, in
        which case the folder-derived group is used (legacy behaviour).
    """
    log = logging.getLogger()
    candidates: List[Path] = []
    if group_map_path is not None:
        candidates.append(Path(group_map_path))
    else:
        for name in (
            "metadata.csv",
            "group_map.csv",
            "metadata.xlsx",
            "group_map.xlsx",
        ):
            candidates.append(base_dir / name)

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                # Excel is a fallback, never the preferred source: it silently
                # reformats identifiers (SEPT9 -> 2-Sep, long GSM/barcode strings
                # -> floats in scientific notation), and a mangled sample id here
                # assigns a donor to the wrong arm. CSV is checked first above.
                log.warning(
                    "[MULTI-10X] Reading the group map from Excel (%s). Excel can "
                    "silently reformat sample identifiers; export it as .csv and "
                    "keep the CSV as the source of truth.",
                    path.name,
                )
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            log.warning("[MULTI-10X] Could not read group map %s: %s", path.name, e)
            continue

        cols = {str(c).lower().strip(): c for c in df.columns}
        if "sample" not in cols or "group" not in cols:
            log.warning(
                "[MULTI-10X] %s lacks required 'sample'/'group' columns; ignoring.",
                path.name,
            )
            continue

        s_col, g_col = cols["sample"], cols["group"]
        mapping: dict[str, str] = {}
        for row in df.to_dict("records"):
            sample = str(row[s_col]).strip()
            if not sample or sample.lower() == "nan":
                continue
            mapping[sample] = row[g_col]
        log.info(
            "[MULTI-10X] Loaded group map from %s: %s sample(s).",
            path.name,
            len(mapping),
        )
        return mapping

    return {}


def _discover_multi_samples(base_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Discover 10x samples arranged as ``<base>/<group>/<sample>/``.

    Each ``<sample>`` folder is expected to hold a 10x feature-barcode matrix
    (matrix.mtx, barcodes.tsv, features.tsv). If a ``<group>`` folder has no
    sub-folders, it is treated as a single sample (group name == sample name).

    Returns
    -------
    list of (group_name, sample_label, sample_dir)
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Multi base directory not found: {base_dir}")

    group_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not group_dirs:
        raise FileNotFoundError(f"No group folders found in: {base_dir}")

    samples: List[Tuple[str, str, Path]] = []
    for gdir in group_dirs:
        group_name = gdir.name
        sample_dirs = sorted([d for d in gdir.iterdir() if d.is_dir()])
        if not sample_dirs:
            # No sample sub-folders — treat the group folder itself as one sample.
            sample_dirs = [gdir]
        for sdir in sample_dirs:
            samples.append((group_name, sdir.name, sdir))

    return samples


def run_pipeline_multi(
    multi_base_dir: str | Path,
    # Keyword-only (Rule 6.2). The ~50 shared options live on PipelineOptions
    # (singlecell_10x/pipeline_options.py) so they are declared ONCE for both
    # drivers instead of being restated in each signature. Pass an options
    # object, or loose keywords — an unknown keyword raises rather than being
    # silently dropped.
    *,
    group_map_path: Optional[str | Path] = None,
    run_per_sample: bool = False,
    options: Optional[PipelineOptions] = None,
    **overrides: object,
) -> Path:
    """
    Run the single-cell 10x pipeline across MANY samples (multi-cohort).

    Samples are discovered as ``<multi_base_dir>/<group>/<sample>/`` 10x folders,
    loaded, tagged with ``obs['sample']`` and ``obs['group']``, concatenated into
    one AnnData, and analysed jointly with group-wise differential expression
    (CASE vs CONTROL, etc.).

    This is the multi-sample counterpart to
    :func:`main_single.run_pipeline`. Unlike a single dataset (one group only),
    a multi run has >=2 groups, so ``do_groupwise_de`` is meaningful here.

    Parameters
    ----------
    multi_base_dir : str | Path
        Base directory containing a ``<group>/<sample>`` tree of 10x folders.
    options : PipelineOptions, optional
        The ~50 shared options (output location, stage toggles, QC thresholds,
        voter flags, contrast design, skip/cleanup flags). Declared once in
        ``pipeline_options.PipelineOptions``; defaults come from
        ``PipelineOptions.for_multi()``. Loose keywords are accepted too and
        applied on top, and an unknown keyword raises ``UnknownPipelineOption``.
    do_pathway_clustering : bool, default True
        Whether to run pathway enrichment analysis.
    do_groupwise_de : bool, default True
        Whether to run group-wise differential expression (needs >=2 groups).
        Automatically disabled with a warning if only one group is discovered.
    do_dpt : bool, default False
        Whether to compute diffusion pseudotime.
    batch_key : str, optional, default "sample"
        Batch key for integration (per-sample batch correction).
    integration_method : str, optional, default "bbknn"
        Integration method ("bbknn" or None).
    geo_json_path : str | Path, optional
        Path to a GEO metadata JSON file. If None, searches the base dir.
    logos_dir : str | Path, optional
        Directory containing logo files for the report.
    group_map_path : str | Path, optional
        Path to a ``metadata.csv``/``group_map.csv`` with ``sample,group``
        columns (upstream cohort-tool handshake). If None, auto-detects
        ``metadata.csv``/``group_map.csv`` in ``multi_base_dir``. When present,
        it is the source of truth for group assignment and overrides the
        folder-derived group; samples whose group is a ``_EXCLUDED``/
        ``_REVIEW``/``_NON_EXPRESSION`` marker are skipped.
    generate_report : bool, default True
        Whether to generate the HTML/PDF report after pipeline completion.
    prepare_for_bisque : bool, default True
        Whether to prepare the combined h5ad output for Bisque deconvolution.
    run_per_sample : bool, default False
        If True, also run the full pipeline on each sample individually
        (written under ``<out>/per_sample/<group>_<sample>``). Off by default
        because it multiplies runtime and output volume.
    skip_tsne, skip_pca_cluster_plots, skip_per_celltype_plots,
    skip_per_celltype_csvs, skip_per_cluster_marker_csvs : bool
        Output-reduction flags (see :func:`main_single.run_pipeline`).
    cleanup_raw_pathway_csvs, cleanup_dedup_logs, cleanup_pipeline_log,
    cleanup_per_cluster_marker_csvs : bool
        Post-run cleanup flags (see :func:`main_single.run_pipeline`).

    Returns
    -------
    Path
        Path to the combined output directory.

    Examples
    --------
    >>> from singlecell_10x.main_multi import run_pipeline_multi
    >>> out = run_pipeline_multi(
    ...     multi_base_dir="/data/cohort_base",  # <base>/CASE/S1, <base>/CONTROL/S2, ...
    ...     out_name="cervical_multi",
    ... )
    """
    # Resolve the option set: an explicit PipelineOptions if given, else the cohort
    # defaults, with any loose keywords applied on top. `merged` raises on an
    # unknown name rather than dropping it, so a misspelled option fails the run
    # instead of leaving it quietly mis-configured.
    options = (options or PipelineOptions.for_multi()).merged(**overrides)

    # Force UTF-8 console I/O so emoji status prints don't raise on Windows cp1252.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # A wrapped, detached or closed stream cannot be reconfigured; the run
            # is unaffected, only non-ASCII console output would be mangled.
            pass

    multi_base_dir = Path(multi_base_dir)

    # Set up output directory. Resolved via output_paths so a relative out_name
    # can never be interpreted against the process working directory.
    combined_out_dir = resolve_output_dir(options.out_name, options.output_root)

    # Set up logging
    log_file = combined_out_dir / "pipeline.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t[%(levelname)s]\t%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    pipeline_logger = logging.getLogger()
    pipeline_logger.info("Logging to file: %s", log_file)
    pipeline_logger.info("Multi base directory: %s", multi_base_dir)
    pipeline_logger.info("Output directory: %s", combined_out_dir)

    # Locate the study metadata JSON up-front so annotation can auto-derive
    # tissue/species from it (same file the report uses later).
    _geo_json_ctx = (
        Path(options.geo_json_path)
        if options.geo_json_path
        else _find_geo_json(multi_base_dir)
    )
    if _geo_json_ctx:
        pipeline_logger.info("Biocontext metadata JSON: %s", _geo_json_ctx)

    # Discover samples
    samples = _discover_multi_samples(multi_base_dir)
    pipeline_logger.info("Discovered %s sample folder(s).", len(samples))

    # Optional metadata.csv / group_map.csv is the source of truth for group
    # assignment and sample exclusion (upstream cohort-tool handshake).
    group_map = _load_group_map(
        multi_base_dir, Path(group_map_path) if group_map_path else None
    )
    if not group_map:
        pipeline_logger.info(
            "[MULTI-10X] No metadata.csv/group_map.csv found — using folder-derived groups."
        )

    # Load + tag each sample
    all_adatas: List[ad.AnnData] = []
    for group_name, sample_label, sample_dir in samples:
        # metadata.csv group (if present) wins over the folder-derived group.
        effective_group = group_map.get(sample_label, group_name)

        if _is_excluded_group(effective_group):
            pipeline_logger.info(
                "[MULTI-10X] Skipping %s: group '%s' marked do-not-analyze.",
                sample_label,
                effective_group,
            )
            continue

        pipeline_logger.info(
            "[MULTI-10X] Loading sample=%s, group=%s, dir=%s",
            sample_label,
            effective_group,
            sample_dir,
        )
        try:
            adata_raw = load_10x_feature_barcode_matrix(sample_dir)
        except FileNotFoundError as e:
            pipeline_logger.warning("[MULTI-10X] Skipping %s: %s", sample_label, e)
            continue

        adata_raw.obs["sample"] = sample_label
        adata_raw.obs["group"] = effective_group

        if run_per_sample:
            sample_out_dir = (
                combined_out_dir / "per_sample" / f"{effective_group}_{sample_label}"
            )
            pipeline_logger.info("[MULTI-10X] Per-sample run -> %s", sample_out_dir)
            run_scanpy_pipeline(
                adata_raw.copy(),
                sample_out_dir,
                analysis_name=f"{effective_group}_{sample_label}",
                batch_key=None,
                integration_method=None,
                # A single sample has one group and no batch structure, so the
                # cohort-level contrast and integration are switched off for it.
                options=options.merged(
                    geo_json_path=_geo_json_ctx,
                    do_groupwise_de=False,
                    do_dpt=False,
                ),
            )

        all_adatas.append(adata_raw)

    if not all_adatas:
        raise RuntimeError(f"No valid 10x samples loaded from: {multi_base_dir}")

    # Make barcodes unique per sample BEFORE concat (prefix with the sample label).
    for adata in all_adatas:
        adata.obs["barcode"] = adata.obs_names.astype(str)
        adata.obs_names = (
            adata.obs["sample"].astype(str) + "_" + adata.obs["barcode"].astype(str)
        )
        adata.obs_names_make_unique()

    adata_all = ad.concat(all_adatas, join="outer", fill_value=0)
    adata_all.var_names_make_unique()
    pipeline_logger.info(
        "[MULTI-10X] Combined shape: %s cells x %s genes",
        adata_all.n_obs,
        adata_all.n_vars,
    )

    # Group-wise DE only makes sense with >=2 groups.
    n_groups = adata_all.obs["group"].nunique()
    groups = sorted(adata_all.obs["group"].unique().tolist())
    pipeline_logger.info("[MULTI-10X] Groups (%s): %s", n_groups, groups)
    if options.do_groupwise_de and n_groups < 2:
        pipeline_logger.warning(
            "Only one group present — disabling group-wise DE "
            "(need >=2 groups for CASE vs CONTROL comparison)."
        )
        # Must go back into `options`, because that is what is forwarded below.
        # A bare local would be silently ignored and a one-group cohort would
        # still attempt a CASE-vs-CONTROL contrast.
        options = options.merged(do_groupwise_de=False)

    # One options object all the way down: run_scanpy_pipeline takes it directly, so
    # nothing has to be unpacked and re-listed here.
    run_scanpy_pipeline(
        adata_all,
        combined_out_dir,
        analysis_name=COMBINED_ANALYSIS_NAME,
        # The resolved GEO JSON replaces the raw option: the driver auto-detects it.
        options=options.merged(geo_json_path=_geo_json_ctx),
    )

    pipeline_logger.info(
        "DONE — Full single-cell pipeline (10x-only, multi-sample) finished with structured outputs."
    )

    # Prepare for Bisque deconvolution if requested
    if options.prepare_for_bisque:
        try:
            processed_h5ad = (
                combined_out_dir / "combined_all_samples_processed_scanpy_output.h5ad"
            )
            if processed_h5ad.exists():
                pipeline_logger.info("Preparing h5ad file for Bisque deconvolution...")
                process_h5ad_file(processed_h5ad)
                pipeline_logger.info("Bisque preparation completed successfully.")
            else:
                pipeline_logger.warning(
                    "Expected h5ad file not found: %s. Skipping Bisque preparation.",
                    processed_h5ad,
                )
        except Exception as e:
            pipeline_logger.exception("Error preparing for Bisque: %s", e)
            pipeline_logger.warning("Continuing without Bisque preparation.")

    # Generate report if requested
    if options.generate_report:
        try:
            from .singlecell_sc_report_generation import build_singlecell_report

            detected_geo_json = None
            if options.geo_json_path is None:
                detected_geo_json = _find_geo_json(multi_base_dir)
                if detected_geo_json:
                    pipeline_logger.info(
                        "Auto-detected GEO JSON file: %s", detected_geo_json
                    )
                else:
                    pipeline_logger.info(
                        "No GEO JSON file found. Report will be generated without GEO metadata."
                    )
            else:
                detected_geo_json = Path(options.geo_json_path)
                if not detected_geo_json.exists():
                    pipeline_logger.warning(
                        "Specified GEO JSON file not found: %s. Attempting to auto-detect...",
                        detected_geo_json,
                    )
                    detected_geo_json = _find_geo_json(multi_base_dir)

            pipeline_logger.info("Generating single-cell report...")
            logos_path = Path(options.logos_dir) if options.logos_dir else None
            build_singlecell_report(
                sc_root=combined_out_dir,
                geo_json_path=detected_geo_json,
                case_id="combined_all_samples",
                logos_dir=logos_path,
            )
            pipeline_logger.info("Report generation completed successfully.")
        except Exception as e:
            pipeline_logger.exception("Error generating report: %s", e)
            pipeline_logger.warning("Continuing without report generation.")

    # --- Post-pipeline cleanup ---
    any_cleanup = (
        options.cleanup_raw_pathway_csvs
        or options.cleanup_dedup_logs
        or options.cleanup_pipeline_log
        or options.cleanup_per_cluster_marker_csvs
    )
    if any_cleanup:
        pipeline_logger.info("Running post-pipeline cleanup...")
        if options.cleanup_pipeline_log:
            for h in pipeline_logger.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
                    pipeline_logger.removeHandler(h)

        removed = _cleanup_intermediates(
            combined_out_dir,
            cleanup_raw_pathway_csvs=options.cleanup_raw_pathway_csvs,
            cleanup_dedup_logs=options.cleanup_dedup_logs,
            cleanup_pipeline_log=options.cleanup_pipeline_log,
            cleanup_per_cluster_marker_csvs=options.cleanup_per_cluster_marker_csvs,
        )
        if removed:
            pipeline_logger.info(
                "Cleanup complete — removed %s intermediate file(s).", removed
            )
        else:
            pipeline_logger.info("Cleanup complete — nothing to remove.")

    return combined_out_dir
