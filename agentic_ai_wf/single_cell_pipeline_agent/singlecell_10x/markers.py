"""
markers.py — per-cell-type marker gene detection and export.

Runs ``sc.tl.rank_genes_groups`` grouped by the consensus cell-type label to find
the genes that distinguish each cell type from the rest, then writes tidy
per-cell-type marker tables (and supporting plots) used by the report and the
downstream DEG/pathway summaries. Identity markers are one-vs-rest and naturally
up-regulation biased — they describe *what a cluster is*, not condition DE.

No p-values are exported (see :data:`MARKER_STATS_NOTE`). The groups being compared
were DEFINED by the same expression matrix the test then reads, so the p-values are
anti-conservative by construction — classic double dipping / selection bias. The
gene ORDER is still informative and is what the tables carry: ``scores``,
``logfoldchanges`` and an explicit 1-based ``rank`` per cell type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc

from .atomic_io import write_table
from .config_cli import logger
from .figure_style import clamp_fig_inches, panel_grid_budget
from .marker_stats import drop_selection_biased_pvalues, write_marker_stats_note
from .safe_names import safe_filename

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData


def compute_celltype_markers(
    adata: AnnData,
    celltype_col: str,
    out_dir: Path,
    analysis_name: str,
    n_markers_per_type: int = 50,
    reference_dir: Path | None = None,
    skip_per_celltype_plots: bool = False,
    skip_per_celltype_csvs: bool = False,
) -> None:
    """
    Celltype-specific markers: for each cell type, DE vs all other cell types.

    Writes:
      - global rankplot / heatmap / dotplot
      - celltype_marker_genes_{celltype_col}_all.csv
      - per-celltype CSVs + dotplots/rankplots (unless skip flags are set)
      - optional copy of ALL markers in reference_dir
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[CT-MARKERS] compute_celltype_markers called for column '%s'.", celltype_col
    )

    if celltype_col not in adata.obs.columns:
        logger.info("[CT-MARKERS] '%s' not in obs → skip.", celltype_col)
        return

    sc.settings.figdir = out_dir

    if not pd.api.types.is_categorical_dtype(adata.obs[celltype_col]):
        adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")

    celltypes = adata.obs[celltype_col].cat.categories.tolist()
    logger.info("[CT-MARKERS] Found %s cell types: %s", len(celltypes), celltypes)

    # Height grows with the cell-type count; clamped because at FIGURE_DPI an
    # unbounded inch value turns into a PNG matplotlib refuses to rasterise.
    heatmap_height = clamp_fig_inches(0.6 * len(celltypes), minimum=8.0)

    if not skip_per_celltype_plots:
        dotplot_dir = out_dir / "sc_dot_plot_vis"
        rankplot_dir = out_dir / "sc_rank_plot_vis"
        dotplot_dir.mkdir(parents=True, exist_ok=True)
        rankplot_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "[CT-MARKERS] Running rank_genes_groups by '%s' (wilcoxon)...", celltype_col
        )
        sc.tl.rank_genes_groups(
            adata,
            groupby=celltype_col,
            method="wilcoxon",
            n_genes=n_markers_per_type,
        )

        # One panel per cell type, each the size of a full embedding panel, so the
        # grid is the one figure in the run that can outgrow what is rasterisable.
        with panel_grid_budget(len(celltypes)):
            sc.pl.rank_genes_groups(
                adata,
                n_genes=20,
                sharey=False,
                show=False,
                save=f"_{analysis_name}_celltype_markers_rankplot.png",
            )

        sc.pl.rank_genes_groups_heatmap(
            adata,
            n_genes=10,
            show=False,
            save=f"_{analysis_name}_celltype_markers_heatmap.png",
            figsize=(12, heatmap_height),
        )
        sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=10,
            show=False,
            save=f"_{analysis_name}_celltype_markers_dotplot.png",
        )

        try:
            markers_all = sc.get.rank_genes_groups_df(adata, None)
        except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
            logger.debug("%s: falling back after %r", __name__, exc)
            rg = adata.uns["rank_genes_groups"]
            groups = rg["names"].dtype.names
            rows = []
            for g in groups:
                names = rg["names"][g]
                scores = rg["scores"][g]
                pvals_adj = rg["pvals_adj"][g]
                logfc = rg.get("logfoldchanges", {}).get(g, None)
                for idx, gene in enumerate(names):
                    row = {
                        "group": g,
                        "names": gene,
                        "scores": scores[idx] if scores is not None else np.nan,
                        "pvals_adj": pvals_adj[idx]
                        if pvals_adj is not None
                        else np.nan,
                        "rank": idx + 1,
                    }
                    if logfc is not None:
                        row["logfoldchanges"] = logfc[idx]
                    rows.append(row)
            markers_all = pd.DataFrame(rows)

        if "logfoldchanges" in markers_all.columns:
            markers_all["direction_simple"] = np.where(
                markers_all["logfoldchanges"] > 0,
                "upregulated",
                np.where(
                    markers_all["logfoldchanges"] < 0,
                    "downregulated",
                    "no_expression_change",
                ),
            )

        # Ranking survives; the p-values do not (see MARKER_STATS_NOTE).
        markers_all = drop_selection_biased_pvalues(markers_all, group_col="group")
        write_marker_stats_note(out_dir)

        all_file = out_dir / f"celltype_marker_genes_{celltype_col}_all.csv"
        write_table(markers_all, all_file, index=False)
        logger.info("[CT-MARKERS] Wrote all celltype markers: %s", all_file)

        if reference_dir is not None:
            reference_dir = Path(reference_dir)
            reference_dir.mkdir(parents=True, exist_ok=True)
            ref_file = (
                reference_dir
                / f"{analysis_name}_celltype_markers_{celltype_col}_all.csv"
            )
            write_table(markers_all, ref_file, index=False)
            logger.info(
                "[CT-MARKERS] Copied celltype markers to reference dir: %s", ref_file
            )

        if skip_per_celltype_csvs and skip_per_celltype_plots:
            logger.info(
                "[CT-MARKERS] Skipping per-celltype CSVs and plots (skip flags enabled)."
            )
        else:
            for ct in celltypes:
                sub = markers_all[markers_all["group"] == ct].copy()
                # safe_filename, not an ad-hoc replace: a label like
                # "Other: GABAergic" would otherwise write into an NTFS alternate
                # data stream and vanish from listings (see safe_names.py).
                ct_safe = safe_filename(ct)

                if not skip_per_celltype_csvs:
                    ct_file = (
                        out_dir / f"celltype_marker_genes_{celltype_col}_{ct_safe}.csv"
                    )
                    write_table(sub, ct_file, index=False)
                    logger.info(
                        "[CT-MARKERS] Wrote markers for celltype '%s': %s", ct, ct_file
                    )

                if not skip_per_celltype_plots:
                    # Rank order comes straight from rank_genes_groups; sorting by a
                    # p-value here would reintroduce the statistic we just removed.
                    sub_sorted = (
                        sub.sort_values("rank") if "rank" in sub.columns else sub
                    )
                    top_genes = (
                        sub_sorted["names"]
                        .head(min(n_markers_per_type, len(sub_sorted)))
                        .tolist()
                    )
                    if not top_genes:
                        continue

                    sc.settings.figdir = dotplot_dir
                    try:
                        sc.pl.dotplot(
                            adata,
                            var_names=top_genes,
                            groupby=celltype_col,
                            show=False,
                            save=f"_{analysis_name}_dotplot_{ct_safe}.png",
                        )
                    except Exception as e:
                        logger.warning(
                            "[CT-MARKERS] Dotplot failed for %s: %s",
                            ct,
                            e,
                            exc_info=True,
                        )

                    sc.settings.figdir = rankplot_dir
                    try:
                        sc.pl.rank_genes_groups(
                            adata,
                            groups=[ct],
                            n_genes=20,
                            sharey=False,
                            show=False,
                            save=f"_{analysis_name}_rankplot_{ct_safe}.png",
                        )
                    except Exception as e:
                        logger.warning(
                            "[CT-MARKERS] Rankplot (single celltype) failed for %s: %s",
                            ct,
                            e,
                            exc_info=True,
                        )

        sc.settings.figdir = out_dir

    except Exception as e:
        logger.warning(
            "[CT-MARKERS] Failed to compute celltype markers: %s", e, exc_info=True
        )
