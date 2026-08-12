"""
group_de.py — cell-level group comparisons and cohort composition views.

Complements the statistically-primary pseudobulk DE (``pseudobulk_de.py``) with
descriptive, per-cluster group contrasts and the cohort composition plots the
report uses: per-group cell-type proportions and group-split UMAPs. These are
exploratory/visual — condition claims should rest on the sample-level pseudobulk
DESeq2 output, not on cell-level tests, which pseudoreplicate donors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from .atomic_io import write_table
from .config_cli import logger
from .contrasts import ordered_contrasts, stamp_contrast
from .figure_style import FIGURE_DPI, clamp_fig_inches, point_size
from .safe_names import safe_filename

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

# ============================================================
# 1. compute_de_between_groups_per_cluster
# ============================================================


def compute_de_between_groups_per_cluster(
    adata: AnnData,
    group_col: str = "group",
    cluster_col: str = "leiden",
    out_dir: Path | None = None,
    reference_group: str | None = None,
) -> None:
    """
    For each cluster, run DE for ALL group pairs as ``focus_vs_reference``.

    The reference is the CONTROL/baseline arm (see :mod:`contrasts`), not the
    alphabetically-first group, and the literal contrast is stamped onto every row.
    Output: one CSV with all comparisons per cluster.
    """
    if out_dir is None:
        raise ValueError(
            "out_dir must be provided for compute_de_between_groups_per_cluster"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_col not in adata.obs.columns:
        logger.info("[DE-CLUSTER-GROUP] '%s' not in obs → skipping.", group_col)
        return

    if cluster_col not in adata.obs.columns:
        logger.info("[DE-CLUSTER-GROUP] '%s' not in obs → skipping.", cluster_col)
        return

    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        logger.info(
            "[DE-CLUSTER-GROUP] Need at least 2 groups in '%s', found: %s",
            group_col,
            groups,
        )
        return

    pairs, ref_group, ref_reason = ordered_contrasts(groups, reference=reference_group)
    if ref_group is None:
        logger.warning("[DE-CLUSTER-GROUP] %s", ref_reason)
    logger.info(
        "[DE-CLUSTER-GROUP] Comparing ALL group pairs within each cluster (%s). Groups: %s; reference: %r (%s)",
        cluster_col,
        groups,
        ref_group,
        ref_reason,
    )

    all_rows = []
    clusters = sorted(adata.obs[cluster_col].astype(str).unique().tolist())

    for cl in clusters:
        sub = adata[adata.obs[cluster_col].astype(str) == cl].copy()
        sub_group_counts = sub.obs[group_col].astype(str).value_counts()

        if sub_group_counts.size < 2:
            logger.info(
                "[DE-CLUSTER-GROUP] Cluster %s: only one group present %s → skip.",
                cl,
                sub_group_counts.to_dict(),
            )
            continue

        for group_focus, group_ref in pairs:
            if (
                group_ref not in sub_group_counts.index
                or group_focus not in sub_group_counts.index
            ):
                continue

            logger.info(
                "[DE-CLUSTER-GROUP] Cluster %s: %s vs %s, counts=%s",
                cl,
                group_focus,
                group_ref,
                sub_group_counts.to_dict(),
            )

            try:
                sc.tl.rank_genes_groups(
                    sub,
                    groupby=group_col,
                    groups=[group_focus],
                    reference=group_ref,
                    method="wilcoxon",
                    n_genes=sub.n_vars,
                )

                try:
                    df = sc.get.rank_genes_groups_df(sub, group_focus)
                except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
                    logger.debug("%s: falling back after %r", __name__, exc)
                    rg = sub.uns["rank_genes_groups"]
                    names = rg["names"][group_focus]
                    scores = rg["scores"][group_focus]
                    pvals_adj = rg["pvals_adj"][group_focus]
                    logfc = None
                    if "logfoldchanges" in rg:
                        logfc = rg["logfoldchanges"][group_focus]
                    df_dict = {
                        "names": names,
                        "scores": scores,
                        "pvals_adj": pvals_adj,
                    }
                    if logfc is not None:
                        df_dict["logfoldchanges"] = logfc
                    df = pd.DataFrame(df_dict)

                df["cluster"] = cl
                df["group_focus"] = group_focus
                df["group_ref"] = group_ref
                stamp_contrast(
                    df, focus=group_focus, ref=group_ref, reference_selection=ref_reason
                )
                df["validity"] = "cell_level_exploratory_pseudoreplicated"
                df["unit_of_replication"] = "cell (PSEUDOREPLICATED — not cohort-valid)"

                if "logfoldchanges" in df.columns and "pvals_adj" in df.columns:
                    df["regulation"] = "no_change"
                    up_mask = (df["logfoldchanges"] > 1.0) & (df["pvals_adj"] < 0.05)
                    down_mask = (df["logfoldchanges"] < -1.0) & (df["pvals_adj"] < 0.05)
                    df.loc[up_mask, "regulation"] = "up"
                    df.loc[down_mask, "regulation"] = "down"

                    df["direction_simple"] = np.where(
                        df["logfoldchanges"] > 0,
                        "upregulated",
                        np.where(
                            df["logfoldchanges"] < 0,
                            "downregulated",
                            "no_expression_change",
                        ),
                    )

                all_rows.append(df)

            except Exception as e:
                logger.warning(
                    "[DE-CLUSTER-GROUP] DE failed for cluster %s, %s vs %s: %s",
                    cl,
                    group_focus,
                    group_ref,
                    e,
                    exc_info=True,
                )
                continue

    if all_rows:
        de_all = pd.concat(all_rows, axis=0, ignore_index=True)
        out_file = out_dir / f"de_{cluster_col}_all_group_pairs.csv"
        write_table(de_all, out_file, index=False)
        logger.info(
            "[DE-CLUSTER-GROUP] Wrote group-wise DE per cluster (all pairs): %s",
            out_file,
        )
    else:
        logger.info("[DE-CLUSTER-GROUP] No DE tables were generated.")


# ============================================================
# 2. plot_groupwise_celltype_proportions
# ============================================================


def plot_groupwise_celltype_proportions(
    adata: AnnData,
    group_col: str = "group",
    celltype_col: str | None = None,
    out_dir: Path | None = None,
) -> None:
    """Plot and save the cell-type composition (proportions) of each group.

    Writes a stacked/grouped bar chart of ``celltype_col`` proportions per
    ``group_col`` into ``out_dir`` so shifts in cellular composition between
    conditions are visible at a glance. ``out_dir`` is required.
    """
    if out_dir is None:
        raise ValueError(
            "out_dir must be provided for plot_groupwise_celltype_proportions"
        )
    out_dir = Path(out_dir)

    if group_col not in adata.obs.columns:
        logger.info("[CT-PROP] '%s' not in obs → skip.", group_col)
        return

    if celltype_col is None or celltype_col not in adata.obs.columns:
        logger.info("[CT-PROP] celltype column missing → skip.")
        return

    df = (
        adata.obs[[group_col, celltype_col]]
        .astype(str)
        .value_counts()
        .reset_index(name="n_cells")
    )
    total_per_group = df.groupby(group_col)["n_cells"].sum().rename("total_cells")
    df = df.merge(total_per_group, on=group_col, how="left")
    df["fraction"] = df["n_cells"] / df["total_cells"]

    prop_file = out_dir / "celltype_proportions_by_group.csv"
    write_table(df, prop_file, index=False)
    logger.info("[CT-PROP] cell type proportions: %s", prop_file)

    pivot = df.pivot(index=celltype_col, columns=group_col, values="fraction").fillna(
        0.0
    )

    # One group of bars per cell type: the canvas grows with the count so the
    # rotated names below the axis stay separated instead of merging.
    bar_width = clamp_fig_inches(0.7 * len(pivot.index) + 3.0, minimum=12.0)
    fig, ax = plt.subplots(figsize=(bar_width, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("fraction of cells")
    ax.set_title("Cell type proportions by group")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / "celltype_proportions_by_group.png", dpi=FIGURE_DPI)
    plt.close(fig)


# ============================================================
# 3. plot_group_specific_umaps
# ============================================================


def plot_group_specific_umaps(
    adata: AnnData,
    group_col: str = "group",
    color_col: str = "leiden",
    out_dir: Path | None = None,
) -> None:
    """Save one UMAP per group, colored by ``color_col`` (default Leiden cluster).

    Splits the embedding by ``group_col`` and renders a UMAP for each group so
    per-condition structure can be compared side by side. ``out_dir`` is required.
    """
    if out_dir is None:
        raise ValueError("out_dir must be provided for plot_group_specific_umaps")
    out_dir = Path(out_dir)

    if group_col not in adata.obs.columns:
        logger.info("[UMAP-BY-GROUP] '%s' not in obs → skip.", group_col)
        return

    sc.settings.figdir = out_dir

    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    for g in groups:
        sub = adata[adata.obs[group_col].astype(str) == g, :].copy()
        logger.info("[UMAP-BY-GROUP] Plotting group=%s, color=%s", g, color_col)
        try:
            sc.pl.umap(
                sub,
                color=[color_col],
                size=point_size(sub.n_obs, base=15),
                show=False,
                save=f"_group_{g}_{color_col}.png",
            )
        except Exception as e:
            logger.warning(
                "[UMAP-BY-GROUP] Failed to plot group %s: %s", g, e, exc_info=True
            )


# ============================================================
# 4. compute_de_by_celltype
# ============================================================

_CELL_LEVEL_DE_NOTE = (
    "These per-cell-type DE tables are CELL-LEVEL Scanpy Wilcoxon results.\n"
    "They treat individual cells as independent replicates, which PSEUDOREPLICATES\n"
    "donors/samples and inflates significance. They are EXPLORATORY ONLY and must\n"
    "NOT be used for cohort (case-vs-control) significance.\n\n"
    "For cohort significance use the sample-level pseudobulk DESeq2 output in:\n"
    "    06_groupwise_deg/pseudobulk_deg/\n"
)


def _write_cell_level_de_note(out_dir: Path) -> None:
    """Drop a README next to the cell-level DE CSVs so they can't be mistaken for
    cohort-valid DE (see the pseudoreplication caveat)."""
    try:
        (out_dir / "readme_cell_level_de.txt").write_text(
            _CELL_LEVEL_DE_NOTE, encoding="utf-8"
        )
    except Exception as e:
        logger.warning(
            "[DE-CELLTYPE] could not write cell-level DE note (%s).", e, exc_info=True
        )


def compute_de_by_celltype(
    adata: AnnData,
    celltype_col: str,
    group_col: str = "group",
    deg_root_dir: Path | None = None,
    reference_group: str | None = None,
) -> None:
    """EXPLORATORY per-cell-type, per-group-pair DE at the CELL level (Scanpy Wilcoxon).

    STATISTICAL CAVEAT (why this is exploratory, not cohort DE): this treats each
    cell as an independent replicate, which PSEUDOREPLICATES donors and inflates
    significance. Cohort (case-vs-control) significance MUST come from the
    sample-level pseudobulk DESeq2 output (``pseudobulk_deg/``), never from these
    files. To make that unmissable, every row is stamped with a ``validity`` column
    and a ``readme_cell_level_de.txt`` note is written beside the CSVs.

    Contrasts are DETERMINISTIC and EXHAUSTIVE: every group pair is tested as
    ``focus_vs_reference`` with the CONTROL/baseline arm as the reference (resolved by
    :mod:`contrasts`) — the SAME convention as the pseudobulk DE, so the two tables
    are directly comparable and a positive logFC means the same thing in both. The
    literal contrast is stamped onto every row.
    """
    if deg_root_dir is None:
        raise ValueError("deg_root_dir must be provided for compute_de_by_celltype")

    if celltype_col not in adata.obs.columns or group_col not in adata.obs.columns:
        logger.info(
            "[DE-CELLTYPE] Missing '%s' or '%s' → skip.", celltype_col, group_col
        )
        return

    # Baseline-oriented, exhaustive pairs (identical convention to pseudobulk_de).
    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        logger.info(
            "[DE-CELLTYPE] Need >=2 groups in '%s', found: %s", group_col, groups
        )
        return
    pairs, ref_group, ref_reason = ordered_contrasts(groups, reference=reference_group)
    if ref_group is None:
        logger.warning("[DE-CELLTYPE] %s", ref_reason)
    else:
        logger.info("[DE-CELLTYPE] reference group = %r — %s", ref_group, ref_reason)
    celltype_deg_root = deg_root_dir / "celltype_specific_deg"
    celltype_deg_root.mkdir(parents=True, exist_ok=True)
    _write_cell_level_de_note(celltype_deg_root)

    celltypes = sorted(adata.obs[celltype_col].astype(str).unique().tolist())
    for ct in celltypes:
        sub = adata[adata.obs[celltype_col].astype(str) == ct, :].copy()
        counts_by_group = sub.obs[group_col].astype(str).value_counts()

        for group_focus, group_ref in pairs:
            n_ref = int(counts_by_group.get(group_ref, 0))
            n_focus = int(counts_by_group.get(group_focus, 0))
            if n_ref < 2 or n_focus < 2:
                logger.info(
                    "[DE-CELLTYPE] celltype=%s: %s vs %s skipped (cells/group: %s=%s, %s=%s).",
                    ct,
                    group_focus,
                    group_ref,
                    group_ref,
                    n_ref,
                    group_focus,
                    n_focus,
                )
                continue

            logger.info(
                "[DE-CELLTYPE] celltype=%s: %s vs %s (CELL-LEVEL, exploratory — not cohort-valid) ...",
                ct,
                group_focus,
                group_ref,
            )
            try:
                sc.tl.rank_genes_groups(
                    sub,
                    groupby=group_col,
                    groups=[group_focus],
                    reference=group_ref,
                    method="wilcoxon",
                    n_genes=sub.n_vars,
                )
                df = sc.get.rank_genes_groups_df(sub, group_focus)
                if "names" not in df.columns:
                    logger.warning(
                        "[DE-CELLTYPE] %s %s vs %s: no 'names' column; skip.",
                        ct,
                        group_focus,
                        group_ref,
                    )
                    continue

                keep_cols = [
                    c
                    for c in ["names", "logfoldchanges", "pvals_adj"]
                    if c in df.columns
                ]
                df = df[keep_cols].copy()
                df["celltype"] = ct
                # Literal, auditable contrast identity (same convention as pseudobulk).
                stamp_contrast(
                    df, focus=group_focus, ref=group_ref, reference_selection=ref_reason
                )
                # Stamp every row so downstream readers cannot mistake this for cohort DE.
                df["validity"] = "cell_level_exploratory_pseudoreplicated"
                df["unit_of_replication"] = "cell (PSEUDOREPLICATED — not cohort-valid)"
                if "logfoldchanges" in df.columns:
                    df["logFC_str"] = df["logfoldchanges"].map(
                        lambda x: f"{x:+.2f}" if pd.notnull(x) else ""
                    )

                ct_safe = safe_filename(ct)
                ct_dir = celltype_deg_root / ct_safe
                ct_dir.mkdir(parents=True, exist_ok=True)

                out_file = (
                    ct_dir
                    / f"{ct_safe}_{group_focus}_vs_{group_ref}_cell_level_exploratory.csv"
                )
                write_table(df, out_file, index=False)
                logger.info(
                    "[DE-CELLTYPE] Wrote exploratory cell-level DEG file: %s", out_file
                )

            except Exception as e:
                logger.warning(
                    "[DE-CELLTYPE] Failed for celltype=%s %s vs %s: %s",
                    ct,
                    group_focus,
                    group_ref,
                    e,
                    exc_info=True,
                )
                continue
