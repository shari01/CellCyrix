#!/usr/bin/env python3
"""
Subprocess wrapper for rank_genes_groups computation and plotting (Celery fork-safe).

This module runs rank_genes_groups computation and plotting in an isolated subprocess
to avoid Celery prefork deadlocks with matplotlib/R resources.

Usage:
    python -m cellcyrix.single_cell_pipeline_agent.singlecell_10x.rank_genes_subprocess \
        --h5ad <input.h5ad> \
        --output_dir <output_dir> \
        --analysis_name <name> \
        --method <t-test|logreg> \
        --n_genes <50>
"""

import logging
import os

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")

import argparse
import gc
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr so/status prints don't crash on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError, ValueError):
    # Module-level bootstrap: no logger exists yet, and a stream that cannot be
    # reconfigured only affects non-ASCII console output, never the analysis.
    pass

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)

# This module runs as a FRESH interpreter under subprocess.run(capture_output=True)
# (pipeline.py:1447), so nothing has configured logging for it — without a handler
# every record below would be dropped and the parent's
# `logger.info("Subprocess stdout:\n%s", result.stdout)` would log an empty string.
# STDOUT is the deliberate target: these records replaced bare print() calls that
# went to stdout, and that is the stream the parent reports as "Subprocess stdout".
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# This file is executed as a standalone script (subprocess.run on its PATH), so the
# package-relative import does not resolve. figure_style, marker_stats and atomic_io are
# deliberately leaf-level (stdlib + pandas/matplotlib only) so importing them by
# directory always works. EVERY sibling import must go in this block — a bare
# `from .x import y` anywhere else in this module raises "attempted relative import with
# no known parent package" and kills the subprocess, which the parent reports only as
# "rank_genes_groups subprocess failed with exit code 1".
try:
    from .atomic_io import write_table
    from .figure_style import apply_figure_style, clamp_fig_inches, panel_grid_budget
    from .marker_stats import drop_selection_biased_pvalues, write_marker_stats_note
except ImportError:  # pragma: no cover - the standalone-script path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from atomic_io import write_table
    from figure_style import apply_figure_style, clamp_fig_inches, panel_grid_budget
    from marker_stats import drop_selection_biased_pvalues, write_marker_stats_note


def run_rank_genes_groups_isolated(
    h5ad_path: str,
    output_dir: str,
    analysis_name: str,
    method: str = "t-test",
    n_genes: int = 50,
    groupby: str = "leiden",
) -> None:
    """
    Run rank_genes_groups computation and plotting in isolated process.

    Args:
        h5ad_path: Path to input h5ad file
        output_dir: Output directory for plots and results
        analysis_name: Analysis name for file naming
        method: Statistical method ('t-test' or 'logreg')
        n_genes: Number of top genes per group
        groupby: Column name for grouping (default: 'leiden')
    """
    # This module runs in a FRESH interpreter, so pipeline.py's figure settings do
    # not reach it — the cluster-marker heatmap and rank plot written below would
    # otherwise be the only figures in the run still at scanpy's 80 dpi screen
    # default. Called here rather than at import so the module has no import-time
    # side effects.
    apply_figure_style()

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set scanpy figure directory
    sc.settings.figdir = output_dir

    logger.info("Loading AnnData from: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    logger.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)
    # Prefer log-normalized .raw for valid statistics/logFC; .X is z-scaled here.
    _use_raw = getattr(adata, "raw", None) is not None
    logger.info("rank_genes_groups use_raw=%s (raw=log-norm; X=scaled)", _use_raw)

    # Verify groupby column exists
    if groupby not in adata.obs.columns:
        raise ValueError(f"Groupby column '{groupby}' not found in adata.obs")

    clusters = sorted(adata.obs[groupby].unique().tolist(), key=lambda x: int(x))
    logger.info("Found %d clusters: %s", len(clusters), clusters)

    # Compute rank_genes_groups
    logger.info("Computing rank_genes_groups using method='%s'...", method)
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            method=method,
            n_genes=n_genes,
            use_raw=_use_raw,
        )
        logger.info("Successfully computed rank_genes_groups using %s", method)
    except Exception as e:
        logger.warning("rank_genes_groups failed with %s: %s", method, e, exc_info=True)
        if method == "t-test":
            logger.info("Trying fallback method 'logreg'...")
            sc.tl.rank_genes_groups(
                adata,
                groupby=groupby,
                method="logreg",
                n_genes=n_genes,
                use_raw=_use_raw,
            )
            logger.info("Successfully computed rank_genes_groups using logreg")
        else:
            raise

    # Generate plots with cleanup after each
    logger.info("Generating rank plot...")
    # One panel per cluster, each the size of a full embedding panel, so the grid
    # is the one figure here that can outgrow what is rasterisable.
    with panel_grid_budget(len(clusters)):
        sc.pl.rank_genes_groups(
            adata,
            n_genes=20,
            sharey=False,
            show=False,
            save=f"_{analysis_name}_cluster_markers_rankplot.png",
        )
    plt.close("all")
    gc.collect()

    logger.info("Generating heatmap...")
    # Height grows with the cluster count; clamped because at FIGURE_DPI an
    # unbounded inch value turns into a PNG matplotlib refuses to rasterise.
    heatmap_height_clusters = clamp_fig_inches(0.6 * len(clusters), minimum=8.0)
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=10,
        show=False,
        save=f"_{analysis_name}_cluster_markers_heatmap.png",
        figsize=(12, heatmap_height_clusters),
    )
    plt.close("all")
    gc.collect()

    logger.info("Generating dotplot...")
    sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=10,
        show=False,
        save=f"_{analysis_name}_cluster_markers_dotplot.png",
    )
    plt.close("all")
    gc.collect()

    # Extract markers dataframe
    logger.info("Extracting marker genes dataframe...")
    try:
        markers_all = sc.get.rank_genes_groups_df(adata, None)
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        # Fallback manual extraction
        logger.debug("%s: falling back after %r", __name__, exc)
        rg = adata.uns["rank_genes_groups"]
        groups = rg["names"].dtype.names
        rows = []
        for g in groups:
            names = rg["names"][g]
            scores = rg["scores"][g]
            for rank, (gene, score) in enumerate(
                zip(names, scores, strict=True), start=1
            ):
                rows.append(
                    {
                        "group": g,
                        "names": gene,
                        "scores": score,
                        "rank": rank,
                    }
                )
        markers_all = pd.DataFrame(rows)

    # Leiden clusters were DERIVED from this expression matrix, so testing genes
    # between them re-uses the data that defined the groups. The p-values are
    # anti-conservative by construction (double dipping) and are dropped; the rank
    # ordering, which is what a marker table is actually for, is kept.
    markers_all = drop_selection_biased_pvalues(markers_all, group_col="group")
    write_marker_stats_note(output_dir)

    # Save markers CSV
    markers_csv = output_dir / "intercluster_cluster_markers.csv"
    write_table(markers_all, markers_csv, index=False)
    logger.info("Saved markers CSV: %s", markers_csv)

    # Save updated h5ad with rank_genes_groups results
    updated_h5ad = output_dir / f"{analysis_name}_with_rank_genes.h5ad"
    adata.write_h5ad(updated_h5ad)
    logger.info("Saved updated h5ad: %s", updated_h5ad)

    logger.info("rank_genes_groups computation and plotting completed successfully")


def main() -> None:
    """CLI entry: run ``rank_genes_groups`` in an isolated subprocess and emit JSON status.

    Kept fork-safe (matplotlib/R) by running out-of-process; on success/failure it
    prints a machine-readable result line for the parent process to consume.
    """
    parser = argparse.ArgumentParser(
        description="Run rank_genes_groups in isolated subprocess (Celery fork-safe)"
    )
    parser.add_argument("--h5ad", required=True, help="Path to input h5ad file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--analysis-name", required=True, help="Analysis name")
    parser.add_argument(
        "--method",
        default="t-test",
        choices=["t-test", "logreg"],
        help="Statistical method (default: t-test)",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=50,
        help="Number of top genes per group (default: 50)",
    )
    parser.add_argument(
        "--groupby",
        default="leiden",
        help="Column name for grouping (default: leiden)",
    )

    args = parser.parse_args()

    try:
        run_rank_genes_groups_isolated(
            h5ad_path=args.h5ad,
            output_dir=args.output_dir,
            analysis_name=args.analysis_name,
            method=args.method,
            n_genes=args.n_genes,
            groupby=args.groupby,
        )
        sys.exit(0)
    except Exception as e:
        # logger.exception attaches the traceback, replacing the previous
        # print(file=sys.stderr) + traceback.print_exc() pair.
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
