"""
summary_ct_deg.py — integrated per-cell-type DEG / marker / pathway summaries.

Joins the three per-cell-type evidence streams — donor-level pseudobulk DE hits,
identity markers, and enriched pathways — into compact, human-readable summary
tables that the report and reviewers use to see, per cell type, what changed between
groups and which pathways it implicates. Pure post-processing of already-written
CSVs; no recomputation of statistics.

Source of the DE stream
-----------------------
This reads the **pseudobulk DESeq2** tables (``pseudobulk_deg/per_celltype/``), where
the donor is the unit of replication. It previously read the cell-level Wilcoxon
tables, which pseudoreplicate donors and inflate significance — so ``08_reference_
summary/`` presented cohort claims built on cell-level p-values, and the ``validity``
stamp that ``group_de`` writes was dropped by the column selection on the way through.

Both schemas are accepted (``_normalize_de_frame``) so the module still works if it
is ever pointed at the exploratory tables deliberately, but the provenance of every
row is recorded in a ``source`` / ``unit_of_replication`` column and echoed into the
text summary. A summary that does not say which test produced it is how an invalid
number acquires authority.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .atomic_io import read_table, write_table
from .column_names import GENE_COLUMNS, LOG2FC_COLUMNS, PADJ_COLUMNS
from .config_cli import logger

# Accepted column spellings, canonical first. `read_table` already maps the library
# spellings to the canonical names, so these tuples are the belt to that braces: a
# table produced outside this pipeline still resolves.
_GENE_COLS = GENE_COLUMNS
_LFC_COLS = LOG2FC_COLUMNS
_PADJ_COLS = PADJ_COLUMNS


def _first_existing(*candidates: Path) -> Path:
    """First candidate path that exists, else the first one.

    Lets a reader accept both the current lowercase output name and the
    pre-rename uppercase one without duplicating the read logic.

    Args:
        *candidates: Paths in preference order.

    Returns:
        The first existing path, or `candidates[0]` when none exist (so the
        caller reports the expected name in its error).
    """
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _first_col(df: pd.DataFrame, candidates) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def _celltype_from_filename(path: Path) -> str:
    """Cell type for tables that carry it in the filename, not a column.

    ``per_celltype/<ct>_pseudobulk_de.csv`` -> ``<ct>``.
    """
    stem = path.stem
    for suffix in ("_pseudobulk_DE", "_pseudobulk_de"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _normalize_de_frame(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Read one DE CSV into a common schema. Returns ``(df, reason_if_skipped)``.

    Normalized columns: ``gene``, ``logFC``, ``padj``, ``celltype``, ``comparison``,
    plus the provenance passed through from the source table when present
    (``reference_group``, ``contrast_direction``, ``significance_rule``, ``validity``,
    ``unit_of_replication``).
    """
    try:
        df = read_table(path)
    except Exception as e:  # noqa: BLE001
        logger.debug("%s: falling back after %r", __name__, e)
        return None, f"unreadable ({e})"
    if df.empty:
        return None, "empty"

    gene_c = _first_col(df, _GENE_COLS)
    lfc_c = _first_col(df, _LFC_COLS)
    padj_c = _first_col(df, _PADJ_COLS)
    if not (gene_c and lfc_c and padj_c):
        return None, (
            f"missing required columns (gene={gene_c}, logFC={lfc_c}, padj={padj_c})"
        )

    out = pd.DataFrame(
        {
            "gene": df[gene_c].astype(str),
            "logFC": pd.to_numeric(df[lfc_c], errors="coerce"),
            "padj": pd.to_numeric(df[padj_c], errors="coerce"),
        }
    )
    out["celltype"] = (
        df["celltype"].astype(str)
        if "celltype" in df.columns
        else _celltype_from_filename(path)
    )
    out["comparison"] = (
        df["comparison"].astype(str) if "comparison" in df.columns else path.stem
    )
    for col in (
        "reference_group",
        "focus_group",
        "contrast_direction",
        "significance_rule",
        "validity",
        "unit_of_replication",
        "regulation",
        "lfc_threshold",
    ):
        if col in df.columns:
            out[col] = df[col]
    return out, ""


def summarize_celltype_degs_markers_pathways(
    out_dir: Path,
    analysis_name: str,
    deg_dir: Path,
    celltype_dir: Path,
    ct_deg_pathway_dir: Path | None = None,
    top_n_genes: int = 30,
    top_n_pathways: int = 10,
    padj_cutoff: float = 0.05,
    min_abs_logfc: float | None = None,
) -> None:
    """
    Summary across:
      - per-cell-type DE hits (donor-level pseudobulk DESeq2)
      - whether each DE gene is also a cell-type identity marker
      - a simple score (2 = DE + marker, 1 = DE only)
      - top pathways per celltype comparison

    ``min_abs_logfc`` is ``None`` by default: the pseudobulk tables are produced by a
    formal effect-size test (``H0: |log2FC| <= threshold``), so ``padj`` already
    encodes the fold-change requirement and stacking another filter on top would
    double-count it. Pass a number only when summarising tables whose p-values were
    computed against zero.

    Outputs in ``08_reference_summary``.
    """
    out_dir = Path(out_dir)
    deg_dir = Path(deg_dir)
    celltype_dir = Path(celltype_dir)
    ref_dir = out_dir / "08_reference_summary"
    ref_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[SUMMARY-CT-DEG] Building celltype DEG + marker + pathway summary for %s",
        analysis_name,
    )

    markers_all_path = _first_existing(
        celltype_dir / "celltype_marker_genes_celltype_all.csv",
        celltype_dir / "celltype_marker_genes_celltype_ALL.csv",  # pre-rename runs
    )
    marker_pairs = set()
    if markers_all_path.exists():
        try:
            markers_all = read_table(markers_all_path)
            gene_col = _first_col(markers_all, _GENE_COLS)
            if "group" in markers_all.columns and gene_col:
                for row in markers_all.to_dict("records"):
                    ct = str(row["group"])
                    g = str(row[gene_col])
                    marker_pairs.add((ct, g))
                logger.info(
                    "[SUMMARY-CT-DEG] Loaded %s marker rows from %s",
                    len(markers_all),
                    markers_all_path.name,
                )
            else:
                logger.warning(
                    "[SUMMARY-CT-DEG] Marker file %s missing a 'group' column or any "
                    "of the gene columns %s.",
                    markers_all_path.name,
                    list(_GENE_COLS),
                )
        except Exception as e:
            logger.warning(
                "[SUMMARY-CT-DEG] Failed to read %s: %s",
                markers_all_path,
                e,
                exc_info=True,
            )
    else:
        logger.info(
            "[SUMMARY-CT-DEG] No marker ALL file found at %s; DEGs will still be summarised but without marker scores.",
            markers_all_path,
        )

    pathways_combined_dir = None
    if ct_deg_pathway_dir is not None:
        ct_deg_pathway_dir = Path(ct_deg_pathway_dir)
        candidate = ct_deg_pathway_dir / "pathways" / "combined"
        if candidate.exists():
            pathways_combined_dir = candidate
            logger.info(
                "[SUMMARY-CT-DEG] Using combined pathway dir: %s", pathways_combined_dir
            )

    # Pseudobulk writes per_celltype/<ct>_pseudobulk_de.csv (flat); the cell-level
    # tables live one directory deeper. Accept both layouts.
    ct_deg_files = sorted(set(deg_dir.glob("*.csv")) | set(deg_dir.glob("*/*.csv")))
    if not ct_deg_files:
        logger.info("[SUMMARY-CT-DEG] No per-celltype DE CSVs found under %s.", deg_dir)
        return

    genelevel_rows = []
    overview_rows = []
    sources = set()
    summary_txt_lines = [
        f"=== Cell-type DEGs + markers + pathways summary for {analysis_name} ===",
        "",
        f"DE source directory: {deg_dir}",
        f"Significance: padj < {padj_cutoff}"
        + (
            f" and |logFC| > {min_abs_logfc}"
            if min_abs_logfc is not None
            else " (effect size already tested by the DE model; no post-hoc logFC filter)"
        ),
        "",
        "Score legend:",
        "  2 = gene is DE AND cell-type marker",
        "  1 = gene is DE only",
        "",
    ]

    for f in ct_deg_files:
        norm, why = _normalize_de_frame(f)
        if norm is None:
            logger.warning("[SUMMARY-CT-DEG] Skipping %s: %s.", f.name, why)
            continue

        # One pseudobulk file can hold SEVERAL contrasts (all group pairs are
        # concatenated), so never assume a single comparison per file.
        for comparison_label, block in norm.groupby("comparison", sort=True):
            ct_name = str(block["celltype"].iloc[0])
            unit = (
                str(block["unit_of_replication"].iloc[0])
                if "unit_of_replication" in block.columns
                else "unspecified"
            )
            sources.add(unit)
            direction = (
                str(block["contrast_direction"].iloc[0])
                if "contrast_direction" in block.columns
                else ""
            )

            df_deg = block.dropna(subset=["padj", "logFC"]).copy()
            df_deg = df_deg[df_deg["padj"] < padj_cutoff]
            if min_abs_logfc is not None:
                df_deg = df_deg[df_deg["logFC"].abs() > float(min_abs_logfc)]

            if df_deg.empty:
                summary_txt_lines.append(
                    f"{ct_name} ({comparison_label}) → 0 DE genes passing filters. [{unit}]"
                )
                summary_txt_lines.append("")
                continue

            df_deg["is_marker_for_celltype"] = [
                int((ct_name, str(g)) in marker_pairs) for g in df_deg["gene"]
            ]
            df_deg["celltype_DEG_marker_score"] = np.where(
                df_deg["is_marker_for_celltype"] == 1, 2, 1
            )

            keep = [
                "celltype",
                "comparison",
                "gene",
                "logFC",
                "padj",
                "is_marker_for_celltype",
                "celltype_DEG_marker_score",
            ]
            # Carry provenance THROUGH — the previous version selected a fixed column
            # list that silently dropped `validity`, so an exploratory table became
            # indistinguishable from a cohort-valid one downstream.
            for col in (
                "reference_group",
                "focus_group",
                "contrast_direction",
                "regulation",
                "significance_rule",
                "validity",
                "unit_of_replication",
            ):
                if col in df_deg.columns:
                    keep.append(col)
            genelevel_rows.append(df_deg[keep].rename(columns={"padj": "pval_adj"}))

            n_deg = int(df_deg.shape[0])
            n_markers = int(df_deg["is_marker_for_celltype"].sum())

            df_markers_only = df_deg[df_deg["is_marker_for_celltype"] == 1].sort_values(
                "padj"
            )
            top_marker_genes = (
                df_markers_only["gene"].head(min(10, len(df_markers_only))).tolist()
            )
            top_marker_genes_str = (
                "; ".join(map(str, top_marker_genes)) if top_marker_genes else ""
            )

            top_pathway_str_list = []
            if pathways_combined_dir is not None:
                prefix = f.stem
                candidate_files = [
                    pathways_combined_dir / f"{prefix}_combined_pathways_dedup.csv",
                    pathways_combined_dir / f"{prefix}_combined_pathways_raw.csv",
                    pathways_combined_dir / f"{prefix}_combined_pathways.csv",
                    # Pre-rename runs (Rule 5.1 lowercased these on 2026-08-11), kept
                    # so an existing output directory is still summarisable.
                    pathways_combined_dir / f"{prefix}_combined_pathways_DEDUP.csv",
                    pathways_combined_dir / f"{prefix}_combined_pathways_RAW.csv",
                ]
                comb_file = next((cf for cf in candidate_files if cf.exists()), None)
                if comb_file is not None:
                    try:
                        pdf = read_table(comb_file)
                        if not pdf.empty:
                            if "combined_score" in pdf.columns:
                                pdf = pdf.sort_values("combined_score", ascending=False)
                            elif "p_value_adj" in pdf.columns:
                                pdf = pdf.sort_values("p_value_adj", ascending=True)
                            for prow in pdf.head(top_n_pathways).to_dict("records"):
                                gs = str(prow.get("biological_database", "NA"))
                                term = str(prow.get("pathway", "NA"))
                                adjp = (
                                    float(prow.get("p_value_adj", np.nan))
                                    if "p_value_adj" in pdf.columns
                                    else np.nan
                                )
                                top_pathway_str_list.append(
                                    f"{gs}:: {term}"
                                    if np.isnan(adjp)
                                    else f"{gs}:: {term} (adj_p={adjp:.2e})"
                                )
                    except Exception as e:
                        logger.warning(
                            "[SUMMARY-CT-DEG] Failed to load pathways for %s: %s",
                            prefix,
                            e,
                            exc_info=True,
                        )

            top_pathways_str = (
                "; ".join(top_pathway_str_list) if top_pathway_str_list else ""
            )

            overview_rows.append(
                {
                    "celltype": ct_name,
                    "comparison": comparison_label,
                    "contrast_direction": direction,
                    "unit_of_replication": unit,
                    "n_deg_filtered": n_deg,
                    "n_deg_markers": n_markers,
                    "top_deg_marker_genes": top_marker_genes_str,
                    "top_pathways": top_pathways_str,
                }
            )

            summary_txt_lines.append(
                f"{ct_name} ({comparison_label}) → {n_deg} DE genes. [{unit}]"
            )
            if direction:
                summary_txt_lines.append(f"  direction: {direction}")
            summary_txt_lines.append(
                f"  DE genes that are also cell-type markers: {n_markers}"
            )
            if top_marker_genes:
                summary_txt_lines.append("  Top marker DE genes:")
                for g in top_marker_genes[: min(5, len(top_marker_genes))]:
                    summary_txt_lines.append(f"    - {g} (score=2)")
            else:
                summary_txt_lines.append(
                    "  Top marker DE genes: none (no overlap with markers)."
                )

            if top_pathway_str_list:
                summary_txt_lines.append("  Top pathways:")
                for pw in top_pathway_str_list[: min(5, len(top_pathway_str_list))]:
                    summary_txt_lines.append(f"    - {pw}")
            else:
                summary_txt_lines.append(
                    "  Top pathways: none (no enriched terms found)."
                )
            summary_txt_lines.append("")

    if sources:
        summary_txt_lines.insert(
            4, f"Unit of replication in the source tables: {', '.join(sorted(sources))}"
        )
        if any("PSEUDOREPLICATED" in s for s in sources):
            warning = (
                "WARNING: this summary was built from CELL-LEVEL tables, which "
                "pseudoreplicate donors and inflate significance. It is EXPLORATORY "
                "and must not be used for cohort claims — point deg_dir at "
                "06_groupwise_deg/pseudobulk_deg/per_celltype/ instead."
            )
            summary_txt_lines.insert(5, warning)
            logger.warning("[SUMMARY-CT-DEG] %s", warning)

    if genelevel_rows:
        genelevel_df = pd.concat(genelevel_rows, axis=0, ignore_index=True)
        genelevel_file = ref_dir / "celltype_deg_marker_genelevel_summary.csv"
        write_table(genelevel_df, genelevel_file, index=False)
        logger.info(
            "[SUMMARY-CT-DEG] Wrote gene-level DE+marker summary: %s (%s rows).",
            genelevel_file,
            len(genelevel_df),
        )

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        overview_file = ref_dir / "celltype_deg_marker_pathway_overview.csv"
        write_table(overview_df, overview_file, index=False)
        logger.info(
            "[SUMMARY-CT-DEG] Wrote celltype DE+marker+pathway overview: %s",
            overview_file,
        )

    summary_txt_file = ref_dir / "celltype_deg_marker_pathway_summary.txt"
    summary_txt_file.write_text("\n".join(summary_txt_lines), encoding="utf-8")
    logger.info("[SUMMARY-CT-DEG] Wrote text summary: %s", summary_txt_file)
