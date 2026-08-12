"""
pathway_enrichment.py — gene-set / pathway over-representation analysis.

Takes ranked gene lists (cluster markers or DE hits) and runs enrichment against
GO / KEGG / Reactome / WikiPathways via gseapy/Enrichr, then de-duplicates
semantically similar terms so the report shows distinct biology rather than many
near-identical pathway names. Network access is required for the Enrichr call;
the step is optional (``do_pathway_clustering``) and degrades gracefully when
gseapy is unavailable or the request fails.
"""

from __future__ import annotations

import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .atomic_io import write_table
from .column_names import to_canonical_columns
from .config_cli import logger
from .figure_style import FIGURE_DPI, clamp_fig_inches

# Optional gseapy for multi-database pathway enrichment (Enrichr)
try:
    import gseapy as gp

    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False

# Configuration for enrichment API calls
ENRICHR_TIMEOUT = 10  # seconds per API call
ENRICHR_MAX_RETRIES = 2  # max retries per database
ENRICHR_MAX_CONSECUTIVE_FAILURES = 5  # stop after N consecutive failures


def deduplicate_pathways_semantic(
    df: pd.DataFrame,
    combined_dir: Path,
    prefix: str,
    sim_threshold: float = 0.9,
) -> pd.DataFrame:
    """
    Optional semantic deduplication using SentenceTransformer (MiniLM) + FAISS.
    Falls back to simple string-based dedup if libraries are not available.
    Writes a log file describing which pathways were removed and why.
    """
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    log_file = combined_dir / f"{prefix}_pathway_dedup_log.txt"

    df = df.reset_index(drop=True)

    def _score(row):
        p = row.get("p_value_adj", np.nan)
        cs = row.get("combined_score", 0.0)
        if pd.isna(p):
            p = 1.0
        return (p, -cs)

    order = sorted(range(len(df)), key=lambda i: _score(df.iloc[i]))

    log_lines = []
    log_lines.append("=== Pathway deduplication (semantic) ===")
    log_lines.append(f"Original rows: {len(df)}")
    log_lines.append(f"Similarity threshold: {sim_threshold}")
    log_lines.append("Priority: min(p_value_adj), then max(combined_score)")
    log_lines.append("")

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        names = df["pathway"].astype(str).tolist()
        embeddings = model.encode(names, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        keep_mask = np.zeros(len(df), dtype=bool)

        kept_indices = []
        for idx in order:
            if not kept_indices:
                index.add(embeddings[idx : idx + 1])
                kept_indices.append(idx)
                keep_mask[idx] = True
                log_lines.append(
                    f"KEEP idx={idx}: {df.loc[idx, 'biological_database']} :: {df.loc[idx, 'pathway']}"
                )
                continue

            distances, indices = index.search(embeddings[idx : idx + 1], k=1)
            sim = float(distances[0][0])
            dup_idx = int(indices[0][0])

            if sim >= sim_threshold:
                log_lines.append(
                    f"REMOVE idx={idx} (sim={sim:.3f} vs idx={dup_idx}) → "
                    f"{df.loc[idx, 'biological_database']} :: {df.loc[idx, 'pathway']}"
                )
            else:
                index.add(embeddings[idx : idx + 1])
                kept_indices.append(idx)
                keep_mask[idx] = True
                log_lines.append(
                    f"KEEP idx={idx}: {df.loc[idx, 'biological_database']} :: {df.loc[idx, 'pathway']}"
                )

        new_df = df.loc[keep_mask].reset_index(drop=True)
        log_lines.append("")
        log_lines.append(f"Final rows after semantic dedup: {len(new_df)}")

    except Exception as e:
        logger.debug("%s: falling back after %r", __name__, e)
        log_lines.append("")
        log_lines.append(f"Semantic dedup skipped (reason: {e}).")
        log_lines.append("Using simple exact-string dedup instead.")
        df["_score_adj"] = df["p_value_adj"].fillna(1.0)
        df["_score_comb"] = -df.get("combined_score", 0.0).fillna(0.0)
        df = df.sort_values(
            ["biological_database", "pathway", "_score_adj", "_score_comb"]
        )
        new_df = df.drop_duplicates(
            subset=["biological_database", "pathway"], keep="first"
        ).copy()
        new_df = new_df.drop(columns=["_score_adj", "_score_comb"])
        new_df = new_df.reset_index(drop=True)
        log_lines.append(f"Final rows after string dedup: {len(new_df)}")

    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    logger.info("[ENRICHR-DEDUP] Wrote pathway deduplication log: %s", log_file)
    return new_df


def run_enrichr_multidb(
    gene_list: Sequence[str],
    out_dir: Path,
    prefix: str,
    pval_cutoff: float = 0.05,
) -> None:
    """
    Run Enrichr-based enrichment on a list of gene symbols across multiple databases.

    Output columns (per file), lower_snake_case per Rule 5.4 — gseapy's own headers
    ("Adjusted P-value", "Combined Score") contain spaces and are mapped through
    ``column_names.to_canonical_columns`` at the producer boundary:

        biological_database    pathway    overlap    p_value    p_value_adj
        odds_ratio    combined_score    genes
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not GSEAPY_AVAILABLE:
        logger.warning(
            "[ENRICHR] gseapy not available; skipping enrichment for %s.", prefix
        )
        return

    genes = [g for g in set(map(str, gene_list)) if g not in (None, "", "nan")]
    if len(genes) < 5:
        logger.info(
            "[ENRICHR] Not enough genes for enrichment (%s) for %s. Skipping.",
            len(genes),
            prefix,
        )
        return

    gene_sets = [
        "GO_Biological_Process_2021",
        "GO_Molecular_Function_2021",
        "GO_Cellular_Component_2021",
        "KEGG_2021_Human",
        "Reactome_2022",
        "WikiPathways_2019_Human",
    ]

    gs_meta = {
        "GO_Biological_Process_2021": (
            "Gene Ontology – Biological Process (GO BP)",
            "GO_BP",
        ),
        "GO_Molecular_Function_2021": (
            "Gene Ontology – Molecular Function (GO MF)",
            "GO_MF",
        ),
        "GO_Cellular_Component_2021": (
            "Gene Ontology – Cellular Component (GO CC)",
            "GO_CC",
        ),
        "KEGG_2021_Human": (
            "KEGG Pathway Database",
            "KEGG",
        ),
        "Reactome_2022": (
            "Reactome Pathway Database",
            "Reactome",
        ),
        "WikiPathways_2019_Human": (
            "WikiPathways Database",
            "WikiPathways",
        ),
    }

    logger.info(
        "[ENRICHR] Running enrichment for %s on %s genes...", prefix, len(genes)
    )

    base_pathway_dir = out_dir / "pathways"
    base_pathway_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = base_pathway_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_results = []
    consecutive_failures = 0

    def _plot_top_bar(df: pd.DataFrame, out_png: Path, top_n: int = 20):
        if "combined_score" not in df.columns:
            return
        df_plot = df.sort_values("combined_score", ascending=False).head(top_n)
        if df_plot.empty:
            return

        # Pathway names run long ("Regulation Of Cytokine Production ..."), and on a
        # single line they either ran into each other or ate the plotting area.
        # Wrapping them and growing the canvas per wrapped LINE keeps every term
        # readable and separated.
        labels = [textwrap.fill(str(p), 58) for p in df_plot["pathway"]]
        n_lines = sum(lbl.count("\n") + 1 for lbl in labels)
        plt.figure(figsize=(14, clamp_fig_inches(0.42 * n_lines + 2.0, minimum=5.0)))
        colors = plt.cm.tab20(np.linspace(0, 1, len(df_plot)))
        plt.barh(labels, df_plot["combined_score"], color=colors)
        plt.gca().invert_yaxis()
        plt.xlabel("combined_score", fontsize=13)
        plt.ylabel("pathway", fontsize=12)
        plt.title(out_png.stem.replace("_", " "), fontsize=14)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png, dpi=FIGURE_DPI)
        plt.close()

    for gs in gene_sets:
        # Early exit if too many consecutive failures (likely network issue)
        if consecutive_failures >= ENRICHR_MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "[ENRICHR] Stopping enrichment for %s after %s consecutive failures (likely network/server issue). Remaining databases will be skipped.",
                prefix,
                consecutive_failures,
            )
            break

        friendly_name, db_folder = gs_meta.get(gs, (gs, "Other"))

        # Retry logic with timeout
        enr = None
        last_error = None
        start_time = time.time()
        for attempt in range(ENRICHR_MAX_RETRIES + 1):
            try:
                # Set a timeout for the API call
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=gs,
                    outdir=None,
                    cutoff=pval_cutoff,
                )
                elapsed = time.time() - start_time
                if elapsed > ENRICHR_TIMEOUT:
                    logger.warning(
                        "[ENRICHR] Enrichment for %s, gene_set=%s took %ss (exceeded timeout of %ss)",
                        prefix,
                        gs,
                        format(elapsed, ".1f"),
                        ENRICHR_TIMEOUT,
                    )
                consecutive_failures = 0  # Reset on success
                break
            except Exception as e:
                last_error = e
                elapsed = time.time() - start_time
                if attempt < ENRICHR_MAX_RETRIES:
                    wait_time = min(2**attempt, 5)  # Exponential backoff, max 5s
                    logger.info(
                        "[ENRICHR] Attempt %s/%s failed for %s, gene_set=%s (took %ss). Retrying in %ss...",
                        attempt + 1,
                        ENRICHR_MAX_RETRIES + 1,
                        prefix,
                        gs,
                        format(elapsed, ".1f"),
                        wait_time,
                    )
                    time.sleep(wait_time)
                    start_time = time.time()  # Reset timer for retry
                else:
                    consecutive_failures += 1
                    logger.warning(
                        "[ENRICHR] Enrichment failed for %s, gene_set=%s after %s attempts (took %ss): %s",
                        prefix,
                        gs,
                        ENRICHR_MAX_RETRIES + 1,
                        format(elapsed, ".1f"),
                        last_error,
                    )

        if enr is None:
            continue

        if enr is None or getattr(enr, "results", None) is None or enr.results.empty:
            logger.info(
                "[ENRICHR] No significant terms for %s, gene_set=%s.", prefix, gs
            )
            continue

        df_res = enr.results.copy()

        for col in ["Old P-value", "Old Adjusted P-value"]:
            if col in df_res.columns:
                df_res = df_res.drop(columns=[col])

        if "Gene_set" not in df_res.columns:
            df_res["Gene_set"] = gs

        if "genes" not in df_res.columns:
            df_res["genes"] = ""

        df_res["biological_database"] = friendly_name

        if "Gene_set" in df_res.columns:
            df_res = df_res.drop(columns=["Gene_set"])

        # Canonicalise here, at the producer boundary, rather than only on write:
        # gseapy's headers contain SPACES ("p_value_adj", "combined_score"),
        # which are unusable as column references downstream, and this module sorts,
        # dedups and plots the frame before it is written. Doing it once here means
        # the module, its figures and its CSVs all speak one vocabulary.
        df_res = to_canonical_columns(df_res)

        desired_cols = [
            "biological_database",
            "pathway",
            "overlap",
            "p_value",
            "p_value_adj",
            "odds_ratio",
            "combined_score",
            "genes",
        ]
        cols = [c for c in desired_cols if c in df_res.columns] + [
            c for c in df_res.columns if c not in desired_cols
        ]
        df_res = df_res[cols]

        db_dir = base_pathway_dir / db_folder
        db_dir.mkdir(parents=True, exist_ok=True)

        out_file = db_dir / f"{prefix}_{db_folder}_enrichment.csv"
        write_table(df_res, out_file, index=False)
        logger.info("[ENRICHR] Saved %s with %s rows.", out_file, df_res.shape[0])

        barplot_file = db_dir / f"{prefix}_{db_folder}_top20_barplot.png"
        _plot_top_bar(df_res, barplot_file, top_n=20)

        combined_results.append(df_res)

    if combined_results:
        combined_df = pd.concat(combined_results, axis=0, ignore_index=True)

        combined_dedup = deduplicate_pathways_semantic(
            combined_df,
            combined_dir=combined_dir,
            prefix=prefix,
        )

        combined_file_raw = combined_dir / f"{prefix}_combined_pathways_raw.csv"
        write_table(combined_df, combined_file_raw, index=False)

        combined_file_clean = combined_dir / f"{prefix}_combined_pathways_dedup.csv"
        write_table(combined_dedup, combined_file_clean, index=False)

        logger.info(
            "[ENRICHR] Saved combined pathways for %s: %s (raw), %s (deduplicated)",
            prefix,
            combined_file_raw.name,
            combined_file_clean.name,
        )
    else:
        logger.info("[ENRICHR] No enrichment results to combine for %s.", prefix)


def run_cluster_marker_enrichment(
    markers_all: pd.DataFrame,
    out_dir: Path,
    analysis_name: str,
    pval_col: Optional[str] = None,
    pval_cutoff: float = 0.05,
    top_n: int = 200,
    cluster_celltype_map: Optional[Dict[str, str]] = None,
    max_clusters: Optional[int] = None,
) -> None:
    """
    For each Leiden cluster (markers_all['group']),
    run multi-DB Enrichr enrichment on top marker genes.

    Gene selection is by ``rank`` (the order ``rank_genes_groups`` emitted), NOT by a
    marker p-value: the clusters were defined from the same matrix the test read, so
    those p-values are anti-conservative and filtering on them would launder a
    selection-biased statistic into the pathway results. See
    :mod:`marker_stats`. ``pval_col`` is retained for callers that pass a table whose
    groups came from the experimental design; it is ``None`` (rank-based) by default.

    Parameters
    ----------
    max_clusters : int, optional
        Maximum number of clusters to process. If None, processes all clusters.
        Useful for testing or when network is unreliable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "group" not in markers_all.columns or "names" not in markers_all.columns:
        logger.warning(
            "[CLUSTER-ENRICH] markers_all must have 'group' and 'names' columns."
        )
        return

    use_pval = bool(pval_col) and pval_col in markers_all.columns
    if not use_pval:
        logger.info(
            "[CLUSTER-ENRICH] Selecting the top %s marker(s) per cluster by rank (marker p-values are selection-biased and are not used as a filter).",
            top_n,
        )

    clusters = sorted(markers_all["group"].unique())
    if max_clusters is not None:
        clusters = clusters[:max_clusters]
        logger.info(
            "[CLUSTER-ENRICH] Processing %s clusters (limited from %s)",
            len(clusters),
            len(markers_all["group"].unique()),
        )

    total_failures = 0
    max_total_failures = len(clusters) * 2  # Allow some failures but not all

    for idx, cl in enumerate(clusters):
        # Early exit if too many total failures
        if total_failures > max_total_failures:
            logger.warning(
                "[CLUSTER-ENRICH] Stopping enrichment after %s failures (processed %s/%s clusters). Network/server may be unavailable.",
                total_failures,
                idx,
                len(clusters),
            )
            break
        sub = markers_all[markers_all["group"] == cl].copy()

        if use_pval:
            sub = sub.sort_values(pval_col).dropna(subset=[pval_col])
            sig = sub[sub[pval_col] < pval_cutoff]
            if sig.empty:
                logger.info(
                    "[CLUSTER-ENRICH] No significant markers for cluster %s. Skipping.",
                    cl,
                )
                continue
            gene_list = sig["names"].head(top_n).tolist()
        else:
            if "rank" in sub.columns:
                sub = sub.sort_values("rank")
            elif "scores" in sub.columns:
                sub = sub.sort_values("scores", ascending=False)
            else:
                logger.warning(
                    "[CLUSTER-ENRICH] No 'rank' or 'scores' column for cluster %s; using the table order as given.",
                    cl,
                )
            gene_list = sub["names"].head(top_n).tolist()

        if not gene_list:
            logger.info(
                "[CLUSTER-ENRICH] Empty gene list for cluster %s. Skipping.", cl
            )
            continue

        cl_str = str(cl)
        if cluster_celltype_map is not None and cl_str in cluster_celltype_map:
            cl_label = cluster_celltype_map[cl_str]
        else:
            cl_label = cl_str

        prefix = f"{analysis_name}_cluster_{cl_label}"
        try:
            run_enrichr_multidb(
                gene_list=gene_list,
                out_dir=out_dir,
                prefix=prefix,
                pval_cutoff=pval_cutoff,
            )
        except Exception as e:
            total_failures += 1
            logger.warning(
                "[CLUSTER-ENRICH] Failed to run enrichment for cluster %s: %s",
                cl,
                e,
                exc_info=True,
            )


def run_group_cluster_deg_enrichment_from_file(
    deg_table_path: Path,
    out_dir: Path,
    analysis_name: str,
    pval_col: str = "pvals_adj",
    logfc_col: str = "logfoldchanges",
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
) -> None:
    """
    Use the table de_<cluster_col>_all_group_pairs.csv (from
    compute_de_between_groups_per_cluster) and run Enrichr (gseapy)
    for EACH (cluster, comparison) pair.
    """
    deg_table_path = Path(deg_table_path)
    if not deg_table_path.exists():
        logger.info("[GRP-CL-DEG-ENRICH] DEG table not found: %s", deg_table_path)
        return

    df = pd.read_csv(deg_table_path)
    required_cols = {"names", "cluster", "comparison", pval_col, logfc_col}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(
            "[GRP-CL-DEG-ENRICH] Missing columns %s in %s; skipping.",
            missing,
            deg_table_path.name,
        )
        return

    wrote_any = False
    for (cl, comp), sub in df.groupby(["cluster", "comparison"]):
        sub = sub.dropna(subset=[pval_col, logfc_col])
        sub = sub[sub[pval_col] < pval_cutoff]
        sub = sub[sub[logfc_col].abs() >= logfc_cutoff]

        if sub.empty:
            continue

        if not wrote_any:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            wrote_any = True

        prefix = f"{analysis_name}_cluster{cl}_{comp}"
        gene_list = sub["names"].tolist()
        logger.info(
            "[GRP-CL-DEG-ENRICH] Running enrichment for cluster=%s, comparison=%s (%s genes, prefix=%s)",
            cl,
            comp,
            len(gene_list),
            prefix,
        )
        run_enrichr_multidb(
            gene_list=gene_list,
            out_dir=out_dir,
            prefix=prefix,
            pval_cutoff=pval_cutoff,
        )
