"""
main_single.py — single-sample driver.

Entry point for running the pipeline on ONE 10x feature-barcode folder: it loads
the matrix, auto-detects any GEO metadata JSON for species/tissue context, calls
``run_scanpy_pipeline`` on the single AnnData, optionally writes the Bisque-ready
export, and cleans up bulky intermediates. ``run_pipeline`` is the public function
the subprocess wrapper and the top-level ``main.py`` call.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from .loader_10x import load_10x_feature_barcode_matrix
from .output_paths import resolve_output_dir
from .pipeline import run_scanpy_pipeline
from .pipeline_options import PipelineOptions
from .sc_to_bisq import process_h5ad_file


def _find_geo_json(single_10x_dir: Path) -> Optional[Path]:
    """
    Automatically find GEO metadata JSON file in the single_10x_dir.
    Looks for files matching patterns like GSE*_metadata.json or *_metadata.json
    """
    if not single_10x_dir.exists():
        return None

    # Common patterns for GEO metadata JSON files
    patterns = [
        "GSE*_metadata.json",
        "*_metadata.json",
        "GSE*.json",
    ]

    for pattern in patterns:
        matches = list(single_10x_dir.glob(pattern))
        if matches:
            # Prefer files with "metadata" in the name
            metadata_files = [f for f in matches if "metadata" in f.name.lower()]
            if metadata_files:
                return metadata_files[0]
            return matches[0]

    return None


def run_pipeline(
    single_10x_dir: str | Path,
    # Keyword-only (Rule 6.2). The ~50 shared options live on PipelineOptions
    # (singlecell_10x/pipeline_options.py) so they are declared ONCE for both
    # drivers instead of being restated in each signature. Pass an options
    # object, or loose keywords — an unknown keyword raises rather than being
    # silently dropped.
    *,
    sample_label: Optional[str] = None,
    group_label: Optional[str] = None,
    options: Optional[PipelineOptions] = None,
    **overrides: object,
) -> Path:
    """
    Run the single-cell 10x pipeline programmatically.

    Parameters
    ----------
    single_10x_dir : str | Path
        Path to the 10x Genomics data folder (containing matrix.mtx, barcodes.tsv, features.tsv)
    sample_label : str, optional
        Sample label to store in obs['sample']. If None, will be extracted from single_10x_dir name.
    group_label : str, optional
        Group label to store in obs['group'] (e.g., "CASE", "CONTROL", "TUMOR", "NORMAL")
    out_name : str, default "SC_RESULTS"
        Name of the output folder. Resolved against `output_root` when relative.
    output_root : str | Path, optional
        Directory that a relative `out_name` is resolved against. Required unless
        `out_name` is already absolute — a relative name with no root would land
        wherever the process happens to be running, so that case raises instead of
        scattering results across the filesystem.
    do_pathway_clustering : bool, default True
        Whether to run pathway enrichment analysis
    do_groupwise_de : bool, default False
        Whether to run group-wise differential expression analysis
    do_dpt : bool, default False
        Whether to compute diffusion pseudotime
    batch_key : str, optional
        Batch key for integration (if None, no batch correction)
    integration_method : str, optional
        Integration method ("bbknn" or None)
    geo_json_path : str | Path, optional
        Path to GEO metadata JSON file (e.g., GSE208653_metadata.json).
        If None, will automatically search for GEO JSON files in single_10x_dir.
    logos_dir : str | Path, optional
        Directory containing logo files for the report. If None, uses default logos directory.
    generate_report : bool, default True
        Whether to generate the HTML/PDF report after pipeline completion.
        Report will be generated even if GEO JSON is not found (with limited metadata).
    prepare_for_bisque : bool, default True
        Whether to automatically prepare the output h5ad file for Bisque deconvolution.
        If True, will create a Bisque-ready version of the processed h5ad file.
    skip_tsne : bool, default False
        Skip t-SNE computation and all tSNE plots (UMAP is usually sufficient).
    skip_pca_cluster_plots : bool, default False
        Skip PCA-based cluster/celltype overlay plots (PCA embeddings rarely
        add insight beyond UMAP/tSNE for visualization).
    skip_per_celltype_plots : bool, default False
        Skip individual per-celltype dotplots and rankplots in sc_dot_plot_vis/
        and sc_rank_plot_vis/ (summary plots are always generated).
    skip_per_celltype_csvs : bool, default False
        Skip individual per-celltype marker CSV files (the combined ALL.csv
        already contains the same data).
    skip_per_cluster_marker_csvs : bool, default False
        Skip individual per-cluster marker CSV files in intercluster_analysis_deg/
        (the combined intercluster_cluster_markers.csv is always kept).
    cleanup_raw_pathway_csvs : bool, default False
        Delete *_combined_pathways_raw.csv intermediates after processing
        (the deduplicated DEDUP versions are kept).
    cleanup_dedup_logs : bool, default False
        Delete *_pathway_dedup_log.txt debug logs after processing.
    cleanup_pipeline_log : bool, default False
        Delete the pipeline.log file after successful completion.
    cleanup_per_cluster_marker_csvs : bool, default False
        Delete per-cluster marker CSV files (cluster_*_markers.csv) in
        intercluster_analysis_deg/ after processing, keeping only the
        combined intercluster_cluster_markers.csv.

    Returns
    -------
    Path
        Path to the output directory

    Examples
    --------
    >>> from singlecell_10x import run_pipeline
    >>> output_dir = run_pipeline(
    ...     single_10x_dir="/path/to/10x/data",
    ...     sample_label="Sample1",
    ...     group_label="CASE",
    ...     out_name="my_results",
    ...     skip_tsne=True,
    ...     skip_per_celltype_plots=True,
    ...     cleanup_raw_pathway_csvs=True,
    ... )
    """
    # Resolve the option set: an explicit PipelineOptions if given, else the
    # mode's defaults, with any loose keywords applied on top. `merged` raises on an
    # unknown name rather than dropping it, so a misspelled option fails the run
    # instead of leaving it quietly mis-configured.
    options = (options or PipelineOptions.for_single()).merged(**overrides)

    # Force UTF-8 console I/O so emoji status prints (CellTypist, our own logs)
    # don't raise UnicodeEncodeError on Windows cp1252 consoles.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # A wrapped, detached or closed stream cannot be reconfigured; the run
            # is unaffected, only non-ASCII console output would be mangled.
            pass

    # Convert to Path
    single_10x_dir = Path(single_10x_dir)

    # Extract sample_label from directory name if not provided
    if sample_label is None:
        sample_label = single_10x_dir.name

    analysis_name = sample_label

    if not single_10x_dir.exists():
        raise FileNotFoundError(f"10x data directory not found: {single_10x_dir}")

    # Set up output directory. Resolved via output_paths so a relative out_name can
    # never be interpreted against the process working directory.
    combined_out_dir = resolve_output_dir(options.out_name, options.output_root)

    # Set up logging
    log_file = combined_out_dir / "pipeline.log"

    # Remove existing handlers to avoid duplicates
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
    pipeline_logger.info("Input directory: %s", single_10x_dir)
    pipeline_logger.info("Output directory: %s", combined_out_dir)
    pipeline_logger.info("Sample label: %s", sample_label)
    if group_label:
        pipeline_logger.info("Group label: %s", group_label)

    # Load data
    adata_single_raw = load_10x_feature_barcode_matrix(single_10x_dir)

    # Add metadata
    adata_single_raw.obs["sample"] = sample_label
    if group_label is not None:
        adata_single_raw.obs["group"] = group_label

    # Locate the study metadata JSON up-front so annotation can auto-derive
    # tissue/species from it (same file the report uses later).
    _geo_json_ctx = (
        Path(options.geo_json_path)
        if options.geo_json_path
        else _find_geo_json(single_10x_dir)
    )
    if _geo_json_ctx:
        pipeline_logger.info("Biocontext metadata JSON: %s", _geo_json_ctx)

    # One options object all the way down: run_scanpy_pipeline takes it directly, so
    # nothing has to be unpacked and re-listed here.
    run_scanpy_pipeline(
        adata_single_raw,
        combined_out_dir,
        analysis_name=analysis_name,
        # The resolved GEO JSON replaces the raw option: the driver auto-detects it.
        options=options.merged(geo_json_path=_geo_json_ctx),
    )

    pipeline_logger.info(
        "DONE — Full single-cell pipeline (10x-only, single dataset) finished with structured outputs."
    )

    # Prepare for Bisque deconvolution if requested
    if options.prepare_for_bisque:
        try:
            processed_h5ad = (
                combined_out_dir / f"{analysis_name}_processed_scanpy_output.h5ad"
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

            # Auto-detect GEO JSON if not provided
            detected_geo_json = None
            if options.geo_json_path is None:
                detected_geo_json = _find_geo_json(single_10x_dir)
                if detected_geo_json:
                    pipeline_logger.info(
                        "Auto-detected GEO JSON file: %s", detected_geo_json
                    )
                else:
                    pipeline_logger.info(
                        "No GEO JSON file found in input directory. Report will be generated without GEO metadata."
                    )
            else:
                detected_geo_json = Path(options.geo_json_path)
                if not detected_geo_json.exists():
                    pipeline_logger.warning(
                        "Specified GEO JSON file not found: %s. Attempting to auto-detect...",
                        detected_geo_json,
                    )
                    detected_geo_json = _find_geo_json(single_10x_dir)
                    if detected_geo_json:
                        pipeline_logger.info(
                            "Auto-detected GEO JSON file: %s", detected_geo_json
                        )
                    else:
                        pipeline_logger.info(
                            "No GEO JSON file found. Report will be generated without GEO metadata."
                        )

            pipeline_logger.info("Generating single-cell report...")
            logos_path = Path(options.logos_dir) if options.logos_dir else None

            build_singlecell_report(
                sc_root=combined_out_dir,
                geo_json_path=detected_geo_json,
                case_id=sample_label,
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

        # Close FileHandlers before cleanup so pipeline.log isn't locked on Windows
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


def _cleanup_intermediates(
    out_dir: Path,
    *,
    cleanup_raw_pathway_csvs: bool = False,
    cleanup_dedup_logs: bool = False,
    cleanup_pipeline_log: bool = False,
    cleanup_per_cluster_marker_csvs: bool = False,
    logger=None,
) -> int:
    """
    Delete intermediate/debug files from a completed pipeline run.

    Returns the number of files removed.
    """
    removed = 0

    if cleanup_raw_pathway_csvs:
        # Both spellings: the lowercase name is current, the uppercase one is what
        # runs before the Rule 5.1 rename left behind. On a case-insensitive
        # filesystem the two patterns match the same files, hence the set.
        raw_csvs = set(out_dir.rglob("*_combined_pathways_raw.csv")) | set(
            out_dir.rglob("*_combined_pathways_RAW.csv")
        )
        for f in sorted(raw_csvs):
            f.unlink()
            removed += 1
            if logger:
                logger.debug("  Removed: %s", f)

    if cleanup_dedup_logs:
        for f in out_dir.rglob("*_pathway_dedup_log.txt"):
            f.unlink()
            removed += 1
            if logger:
                logger.debug("  Removed: %s", f)

    if cleanup_per_cluster_marker_csvs:
        intercluster_dir = (
            out_dir / "04_clustering_and_cell_states" / "intercluster_analysis_deg"
        )
        if intercluster_dir.is_dir():
            for f in intercluster_dir.glob("cluster_*_markers.csv"):
                f.unlink()
                removed += 1
                if logger:
                    logger.debug("  Removed: %s", f)

    if cleanup_pipeline_log:
        log_file = out_dir / "pipeline.log"
        if log_file.exists():
            log_file.unlink()
            removed += 1
            if logger:
                logger.debug("  Removed: %s", log_file)

    return removed
