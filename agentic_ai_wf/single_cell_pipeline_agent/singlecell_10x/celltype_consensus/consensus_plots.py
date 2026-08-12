"""
consensus_plots.py — visual summaries of the consensus annotation.

Two figures (+ their CSVs), written next to the annotation outputs:

 1. Per-sample / per-group cell-type COMPOSITION — which cell types occur in
    which sample (and biological group), as stacked-proportion bars. Answers
    "from which sample does this cell type come, and how much".

 2. Cross-method CONFIDENCE / AGREEMENT — per Leiden cluster, the CellTypist /
    SingleR / Knowledge-based calls beside the consensus, coloured by tier. Where the
    methods agree (tier High) the label is strong; disagreement (Low/Review)
    is flagged. Answers "how confident is each cell-type call".

Plotting never raises into the caller — every entry point is best-effort and
logs on failure, so a plotting problem can't sink an annotation run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless / subprocess-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..atomic_io import atomic_to_csv
from ..figure_style import FIGURE_DPI  # noqa: E402

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)


def _harmonized(label) -> str:
    """Harmonize a raw method label to the controlled vocabulary so a method's call
    can be compared to the (already-harmonized) consensus. A method 'agrees' when
    its HARMONIZED label equals the consensus — 'T_cells'/'Th' must count as
    agreeing with a 'T cell' consensus, not be treated as a mismatch because the
    raw strings differ. Lazily imported to keep this module free of the heavy
    scanpy import at load time; falls back to the raw string if unavailable."""
    try:
        from .tools import harmonize_label

        return harmonize_label(label)
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        logger.debug("%s: falling back after %r", __name__, exc)
        return str(label)


# Tier -> colour for the confidence figure.
_TIER_COLORS = {
    "High": "#2e7d32",  # green  — all methods agree
    "Medium": "#f9a825",  # amber  — majority
    "Low/Review": "#c62828",  # red    — disagreement / adjudicated
}


def _cat_colors(n: int):
    """n distinct categorical colours (tab20 cycled)."""
    base = plt.get_cmap("tab20").colors
    return [base[i % len(base)] for i in range(n)]


def plot_sample_celltype_composition(
    adata: AnnData,
    out_dir: Path,
    analysis_name: str,
    celltype_col: str,
    sample_col: str = "sample",
    group_col: str = "group",
) -> None:
    """Stacked-proportion bars of cell-type composition per sample (and per
    group when >1 group). Also writes the count/proportion crosstabs as CSV."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        obs = adata.obs
        if celltype_col not in obs.columns:
            logger.warning(
                "[CONSENSUS-PLOT] '%s' missing; skipping composition plot.",
                celltype_col,
            )
            return
        if sample_col not in obs.columns:
            logger.info(
                "[CONSENSUS-PLOT] '%s' missing; skipping composition plot.", sample_col
            )
            return

        # order cell types by overall abundance (stable, readable stacks)
        ct_order = obs[celltype_col].astype(str).value_counts().index.tolist()
        colors = dict(zip(ct_order, _cat_colors(len(ct_order)), strict=True))

        def _panel(ax, by_col: str, title: str):
            ct = pd.crosstab(obs[by_col].astype(str), obs[celltype_col].astype(str))
            ct = ct.reindex(columns=ct_order, fill_value=0)
            prop = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            left = np.zeros(len(prop))
            ypos = np.arange(len(prop))
            for ctype in ct_order:
                vals = prop[ctype].values
                ax.barh(
                    ypos,
                    vals,
                    left=left,
                    color=colors[ctype],
                    label=ctype,
                    edgecolor="white",
                    linewidth=0.3,
                )
                left += vals
            ax.set_yticks(ypos)
            ax.set_yticklabels(
                [f"{idx}  (n={int(ct.loc[idx].sum())})" for idx in prop.index]
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("cell-type proportion")
            ax.set_title(title)
            ax.invert_yaxis()
            return ct, prop

        has_group = (
            group_col in obs.columns and obs[group_col].astype(str).nunique() > 1
        )
        n_samples = obs[sample_col].astype(str).nunique()
        nrows = 2 if has_group else 1
        fig, axes = plt.subplots(
            nrows,
            1,
            figsize=(11, max(3, 0.45 * n_samples) + (3 if has_group else 0)),
            squeeze=False,
        )
        ct_s, prop_s = _panel(
            axes[0][0], sample_col, f"{analysis_name} — cell types per sample"
        )
        # index = cell type, which is data here, so it is written deliberately.
        atomic_to_csv(
            ct_s, out_dir / f"{analysis_name}_celltype_by_sample_counts.csv", index=True
        )
        atomic_to_csv(
            prop_s.round(4),
            out_dir / f"{analysis_name}_celltype_by_sample_proportions.csv",
            index=True,
        )
        if has_group:
            ct_g, _ = _panel(
                axes[1][0], group_col, f"{analysis_name} — cell types per group"
            )
            atomic_to_csv(
                ct_g,
                out_dir / f"{analysis_name}_celltype_by_group_counts.csv",
                index=True,
            )

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            title="cell type",
            frameon=False,
        )
        fig.tight_layout()
        fig_path = out_dir / f"{analysis_name}_celltype_composition_by_sample.png"
        fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("[CONSENSUS-PLOT] composition figure -> %s", fig_path)
    except Exception as e:
        logger.exception("[CONSENSUS-PLOT] composition plot failed (%s).", e)


def plot_method_agreement(
    table_rows: List[Dict[str, object]],
    out_dir: Path,
    analysis_name: str,
    enable_singler: bool = True,
    enable_llm: bool = True,
    enable_pubmed: bool = False,
) -> None:
    """Per-cluster method-vs-consensus table coloured by tier. Cells whose method
    label matches the consensus are bold-bordered (agreement = strong confidence).
    Includes every enabled voter (CellTypist / SingleR / Knowledge-based / PubMed).
    Also writes the agreement summary CSV."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not table_rows:
            logger.info("[CONSENSUS-PLOT] no cluster rows; skipping agreement plot.")
            return

        _display = {
            "celltypist": "CellTypist",
            "singler": "SingleR",
            "knowledge_based": "Knowledge-based",
            "pubmed": "PubMed",
        }
        methods = ["celltypist"]
        if enable_singler:
            methods.append("singler")
        if enable_llm:
            methods.append("knowledge_based")
        if enable_pubmed:
            methods.append("pubmed")
        cols = methods + ["consensus"]
        col_labels = [_display.get(m, m.capitalize()) for m in methods] + ["Consensus"]

        rows = sorted(table_rows, key=lambda r: str(r.get("cluster", "")))
        n = len(rows)

        # agreement count per cluster = # methods matching the consensus label
        summary = []
        for r in rows:
            cons = str(r.get("consensus", ""))
            agree = sum(
                1 for m in methods if cons and _harmonized(str(r.get(m, ""))) == cons
            )
            summary.append(
                {
                    "cluster": r.get("cluster"),
                    "consensus": cons,
                    "tier": r.get("tier"),
                    "methods_agreeing": agree,
                    "n_methods": len(methods),
                }
            )
        atomic_to_csv(
            pd.DataFrame(summary),
            out_dir / f"{analysis_name}_method_agreement.csv",
            index=False,
        )

        fig_h = max(2.0, 0.45 * n + 1.6)  # extra headroom for the reserved header band
        fig, ax = plt.subplots(figsize=(2.3 * len(cols) + 2, fig_h))
        ax.set_xlim(0, len(cols))
        ax.set_ylim(0, n + 1.3)  # reserve a band ABOVE the data rows for column headers
        ax.axis("off")
        # Title sits above the axes with padding so it can never overlap the column
        # headers (previously the 2-line title collided with the header row).
        ax.set_title(
            f"{analysis_name} — cell-type calls per method vs consensus\n"
            "(row colour = consensus tier; ● = method matches consensus)",
            fontsize=10,
            pad=26,
        )

        # column headers live in the reserved band — clear of both rows and title
        for j, cl_label in enumerate(col_labels):
            ax.text(
                j + 0.5,
                n + 0.3,
                cl_label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        for i, r in enumerate(rows):
            y = n - 1 - i
            cons = str(r.get("consensus", ""))
            tier = str(r.get("tier", ""))
            row_color = _TIER_COLORS.get(tier, "#9e9e9e")
            # cluster id + tier band on the far left
            ax.text(
                -0.05,
                y + 0.5,
                f"cl {r.get('cluster')}",
                ha="right",
                va="center",
                fontsize=8,
            )
            for j, m in enumerate(cols):
                label = str(r.get(m, ""))
                matches = (
                    (m != "consensus") and bool(cons) and _harmonized(label) == cons
                )
                face = row_color if m == "consensus" else "white"
                ax.add_patch(
                    plt.Rectangle(
                        (j, y),
                        1,
                        1,
                        facecolor=face,
                        edgecolor=row_color,
                        linewidth=2.0 if (matches or m == "consensus") else 0.6,
                        alpha=0.85 if m == "consensus" else 1.0,
                    )
                )
                txt = ("● " if matches else "") + (label if label else "—")
                ax.text(
                    j + 0.5,
                    y + 0.5,
                    txt[:26],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if m == "consensus" else "black",
                )

        # tier legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=c)
            for c in _TIER_COLORS.values()
        ]
        fig.legend(
            handles,
            list(_TIER_COLORS.keys()),
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
            title="consensus tier (agreement)",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig_path = out_dir / f"{analysis_name}_method_agreement.png"
        fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("[CONSENSUS-PLOT] agreement figure -> %s", fig_path)
    except Exception as e:
        logger.exception("[CONSENSUS-PLOT] agreement plot failed (%s).", e)


_METHOD_DISPLAY = {
    "celltypist": "CellTypist",
    "singler": "SingleR",
    "knowledge_based": "Knowledge-based",
    "pubmed": "PubMed",
}


def plot_per_method_calls(
    table_rows: List[Dict[str, object]],
    out_dir: Path,
    analysis_name: str,
    enable_celltypist: bool = True,
    enable_singler: bool = True,
    enable_llm: bool = True,
    enable_pubmed: bool = False,
) -> None:
    """One STANDALONE figure per method (SingleR, CellTypist, Knowledge-based,
    PubMed): each cluster's call for that method, bar length = the method's native
    confidence (bar=1.0 where the method reports none), coloured by whether it
    agrees with the consensus. Answers 'what did each method say on its own'."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not table_rows:
            return
        enabled = []
        if enable_celltypist:
            enabled.append("celltypist")
        if enable_singler:
            enabled.append("singler")
        if enable_llm:
            enabled.append("knowledge_based")
        if enable_pubmed:
            enabled.append("pubmed")

        rows = sorted(table_rows, key=lambda r: str(r.get("cluster", "")))
        n = len(rows)
        _absent = {"", "—", "(disabled)", "Unassigned"}

        for m in enabled:
            labels, confs, colors, n_ok = [], [], [], 0
            for r in rows:
                lab = str(r.get(m, "") or "")
                cons = str(r.get("consensus", "") or "")
                try:
                    conf = float(r.get(f"{m}_conf"))
                    if conf != conf:  # NaN
                        conf = None
                except (TypeError, ValueError):
                    conf = None
                abstain = lab in _absent
                match = (not abstain) and bool(cons) and _harmonized(lab) == cons
                if match:
                    n_ok += 1
                colors.append(
                    "#c62828" if abstain else ("#2e7d32" if match else "#9e9e9e")
                )
                confs.append(conf if conf is not None else 1.0)
                labels.append(lab if lab else "—")

            fig, ax = plt.subplots(figsize=(8.5, max(2.0, 0.45 * n + 1.0)))
            y = np.arange(n)
            ax.barh(y, confs, color=colors, edgecolor="white", linewidth=0.4)
            ax.set_yticks(y)
            ax.set_yticklabels([f"cl {r.get('cluster')}" for r in rows])
            ax.invert_yaxis()
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("native confidence (bar = 1.0 when the method reports none)")
            ax.set_title(
                f"{analysis_name} — {_METHOD_DISPLAY.get(m, m)} calls "
                f"({n_ok}/{n} agree with consensus)",
                fontsize=10,
            )
            for yi, lab, cf in zip(y, labels, confs, strict=True):
                ax.text(
                    min(cf, 1.0) + 0.01,
                    yi,
                    f" {lab[:28]}",
                    va="center",
                    ha="left",
                    fontsize=7,
                )
            handles = [
                plt.Rectangle((0, 0), 1, 1, color=c)
                for c in ("#2e7d32", "#9e9e9e", "#c62828")
            ]
            ax.legend(
                handles,
                ["agrees with consensus", "differs", "abstained/none"],
                loc="lower right",
                fontsize=7,
                frameon=False,
            )
            fig.tight_layout()
            fig_path = out_dir / f"{analysis_name}_method_{m}_calls.png"
            fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("[CONSENSUS-PLOT] per-method figure (%s) -> %s", m, fig_path)
    except Exception as e:
        logger.exception("[CONSENSUS-PLOT] per-method plot failed (%s).", e)
