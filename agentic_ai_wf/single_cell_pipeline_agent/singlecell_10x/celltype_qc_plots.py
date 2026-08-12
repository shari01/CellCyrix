"""
celltype_qc_plots.py — figures for judging whether the annotation is right, and
whether a composition difference is real.

Why these three
---------------
**Per-donor proportions.** The pipeline already writes
``celltype_proportions_by_group.csv`` — one pooled fraction per group. Pooling
treats every cell as an independent observation, so a cohort of 21 donors is
reported as if it were 158,084 samples, and ordinary between-donor variation
reads as a group difference. Measured on GSE157827: pooled fractions put
excitatory neurons at 30.04% control vs 27.19% AD, which looks like the textbook
finding. Re-tested per donor, that is 28.47% vs 23.91% with **p = 0.337** —
control donors alone span 13%–47%, so the shift is inside the noise. The pooled
number was not wrong, it was untestable. This module makes the unit of
replication the donor, which is what a reviewer will ask for.

**Annotation marker dotplot.** The fastest way to see a wrong label. Canonical
markers on one axis, final cell types on the other: a T-cell row with an empty
CD3D/CD3E/TRAC column is a mislabel you can spot in one glance. On GSE337706,
cluster 0 (24,272 cells, 27.8% of the run) shipped as "T cell" with zero T-cell
markers — this plot would have shown it immediately.

**Voter agreement.** Which annotator disagreed, on which cluster, and whether the
final call followed the majority. Makes a saturated voter obvious as a solid
band (GSE157827: SingleR returned 'Astrocyte' for 18 of 20 clusters).

Every function is defensive: missing columns, absent groups or too few donors
produce a logged skip and ``None``, never an exception. A figure is never worth
failing a pipeline run over.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .atomic_io import read_table, write_table
from .column_names import GENE_COLUMNS, LOG2FC_COLUMNS, PADJ_COLUMNS
from .figure_style import FIGURE_DPI  # noqa: E402
from .safe_names import safe_filename

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

# A proportion test needs enough donors per group to mean anything. Below this a
# p-value is theatre, so the values are still written but no test is reported.
MIN_DONORS_PER_GROUP: int = 3
# Cell types shown in the per-donor panel grid, ranked by mean abundance. The
# long tail is still written to the CSV.
MAX_PANELS: int = 12

_GROUP_COLORS: Sequence[str] = (
    "#4C72B0",
    "#C44E52",
    "#55A868",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
)

# Canonical markers for the annotation dotplot, grouped by the cell type they
# support. Kept short and unambiguous on purpose: this figure answers "does the
# label have its defining genes", not "what else is expressed".
CANONICAL_MARKERS: Dict[str, List[str]] = {
    "T cell": ["CD3D", "CD3E", "TRAC", "IL7R"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
    "B cell": ["MS4A1", "CD79A", "CD19"],
    "Plasma cell": ["MZB1", "JCHAIN", "SDC1"],
    "Monocyte": ["LYZ", "CD14", "VCAN", "FCN1"],
    "Macrophage": ["CD68", "CSF1R", "MRC1", "C1QA"],
    "Microglia": ["P2RY12", "CX3CR1", "TMEM119", "APBB1IP"],
    "Dendritic cell": ["FCER1A", "CD1C", "CLEC9A", "LILRA4"],
    "Granulocyte": ["TPSAB1", "CPA3", "FCGR3B", "S100A8"],
    "Platelet": ["PF4", "PPBP", "ITGA2B", "GP9"],
    "Erythrocyte": ["HBB", "HBA1", "ALAS2"],
    "Epithelial cell": ["EPCAM", "KRT18", "KRT8", "CDH1"],
    "Endothelial cell": ["PECAM1", "CLDN5", "FLT1", "VWF"],
    "Mural cell": ["PDGFRB", "ACTA2", "RGS5", "NOTCH3"],
    "Fibroblast": ["COL1A1", "DCN", "LUM", "PDGFRA"],
    "Excitatory neuron": ["SLC17A7", "SATB2", "RBFOX3", "SNAP25"],
    "Inhibitory neuron": ["GAD1", "GAD2", "SLC32A1", "LHX6"],
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2", "GJA1"],
    "Oligodendrocyte": ["PLP1", "MBP", "MOBP", "MOG"],
    "OPC": ["PDGFRA", "CSPG4", "PCDH15"],
    "Schwann cell": ["MPZ", "PMP22", "S100B"],
}


def _safe(name: str) -> str:
    """Filesystem-safe path component — see safe_names for why the colon
    matters on Windows."""
    return safe_filename(name)


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from scipy import stats
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001 - scipy is present in practice; degrade quietly
        logger.debug("%s: falling back after %r", __name__, exc)
        return None
    if len(a) < MIN_DONORS_PER_GROUP or len(b) < MIN_DONORS_PER_GROUP:
        return None
    if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.isclose(a[0], b[0]):
        return 1.0
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001
        logger.debug("%s: falling back after %r", __name__, exc)
        return None


def _bh_fdr(pvals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg across the cell types tested in one figure.

    One test per cell type is still a family of tests; reporting raw p-values for
    a dozen of them invites reading the smallest as a finding.
    """
    idx = [i for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return list(pvals)
    ordered = sorted(idx, key=lambda i: pvals[i])
    n = len(ordered)
    out: List[Optional[float]] = list(pvals)
    prev = 1.0
    for rank, i in enumerate(reversed(ordered), start=1):
        q = min(prev, float(pvals[i]) * n / (n - rank + 1))
        out[i] = prev = min(1.0, q)
    return out


def _stars(q: Optional[float]) -> str:
    if q is None:
        return "n/a"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


# ===========================================================================
# 1. Per-donor cell-type proportions
# ===========================================================================
def plot_per_donor_proportions(
    adata: AnnData,
    *,
    group_col: str,
    celltype_col: str,
    sample_col: str = "sample",
    out_dir: Path,
    analysis_name: str = "",
) -> Optional[Path]:
    """Donor-level proportions with a per-cell-type test. Returns the CSV path.

    Writes ``celltype_proportions_per_donor.csv`` (one row per donor x cell type)
    and ``celltype_proportions_per_donor_stats.csv`` (per cell type: group means,
    difference, Mann-Whitney p and BH q), plus a panel figure.
    """
    tag = f"[{analysis_name}] " if analysis_name else ""
    for col in (group_col, celltype_col, sample_col):
        if col not in adata.obs.columns:
            logger.info("%s[CT-PROP-DONOR] '%s' missing → skip.", tag, col)
            return None

    obs = adata.obs[[sample_col, group_col, celltype_col]].astype(str)
    counts = obs.groupby([sample_col, celltype_col]).size().unstack(fill_value=0)
    if counts.empty:
        logger.info("%s[CT-PROP-DONOR] no cells → skip.", tag)
        return None
    prop = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    donor_group = obs.groupby(sample_col)[group_col].first()

    groups = sorted(donor_group.unique())
    if len(groups) < 2:
        logger.info("%s[CT-PROP-DONOR] only one group → nothing to compare.", tag)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    long = (
        prop.stack()
        .rename("fraction")
        .reset_index()
        .rename(columns={"level_1": celltype_col})
    )
    long[group_col] = long[sample_col].map(donor_group)
    long["n_cells"] = counts.stack().rename("n").reset_index()["n"].values
    long_path = out_dir / "celltype_proportions_per_donor.csv"
    write_table(long, long_path, index=False)

    # --- per-cell-type test, only meaningful for a two-group comparison -------
    order = prop.mean().sort_values(ascending=False).index.tolist()
    rows, pvals = [], []
    g0, g1 = groups[0], groups[1]
    s0 = donor_group[donor_group == g0].index
    s1 = donor_group[donor_group == g1].index
    for ct in order:
        a = prop.loc[prop.index.intersection(s0), ct].to_numpy() * 100
        b = prop.loc[prop.index.intersection(s1), ct].to_numpy() * 100
        p = _mannwhitney(a, b) if len(groups) == 2 else None
        pvals.append(p)
        rows.append(
            {
                "celltype": ct,
                f"n_donors_{g0}": len(a),
                f"n_donors_{g1}": len(b),
                f"mean_pct_{g0}": round(float(a.mean()) if len(a) else np.nan, 4),
                f"mean_pct_{g1}": round(float(b.mean()) if len(b) else np.nan, 4),
                "diff_pct": round(
                    float(b.mean() - a.mean()) if len(a) and len(b) else np.nan, 4
                ),
                "mannwhitney_p": p,
            }
        )
    qvals = _bh_fdr(pvals)
    for r, q in zip(rows, qvals, strict=True):
        r["bh_q"] = q
        r["significant_q05"] = bool(q is not None and q < 0.05)
    stats_df = pd.DataFrame(rows)
    stats_path = out_dir / "celltype_proportions_per_donor_stats.csv"
    write_table(stats_df, stats_path, index=False)

    n_small = min(len(s0), len(s1))
    if n_small < MIN_DONORS_PER_GROUP:
        logger.warning(
            "%s[CT-PROP-DONOR] only %s donor(s) in the smaller group; proportions written but NOT tested (need >= %s).",
            tag,
            n_small,
            MIN_DONORS_PER_GROUP,
        )
    n_sig = int(stats_df["significant_q05"].sum())
    logger.info(
        "%s[CT-PROP-DONOR] %s donors x %s cell types -> %s, %s (%s significant at BH q<0.05).",
        tag,
        len(prop),
        len(order),
        long_path.name,
        stats_path.name,
        n_sig,
    )

    # --- figure ---------------------------------------------------------------
    shown = order[:MAX_PANELS]
    ncol = min(4, max(1, len(shown)))
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow), squeeze=False
    )
    rng = np.random.default_rng(0)  # jitter must not change between runs
    colors = {g: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(groups)}
    for ax, ct in zip(
        axes.ravel(), shown, strict=False
    ):  # grid is padded past len(shown)
        series = []
        for i, g in enumerate(groups):
            idx = prop.index.intersection(donor_group[donor_group == g].index)
            v = prop.loc[idx, ct].to_numpy() * 100
            series.append(v)
            if len(v):
                ax.scatter(
                    rng.normal(i, 0.055, len(v)),
                    v,
                    s=34,
                    zorder=3,
                    color=colors[g],
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.7,
                )
        if any(len(v) for v in series):
            ax.boxplot(
                [v for v in series],
                positions=list(range(len(groups))),
                widths=0.5,
                showfliers=False,
                medianprops=dict(color="#222", lw=1.6),
                boxprops=dict(color="#888"),
                whiskerprops=dict(color="#888"),
                capprops=dict(color="#888"),
            )
        top = max([float(v.max()) for v in series if len(v)] or [1.0])
        if len(groups) == 2:
            q = stats_df.loc[stats_df.celltype == ct, "bh_q"].iloc[0]
            lbl = _stars(q) + ("" if q is None else f"  q={q:.3f}")
            ax.plot([0, 1], [top * 1.10] * 2, color="#444", lw=1)
            ax.text(0.5, top * 1.13, lbl, ha="center", fontsize=8.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[:14] for g in groups], fontsize=8.5, rotation=0)
        ax.set_title(ct if len(ct) <= 32 else ct[:29] + "…", fontsize=9.5)
        ax.set_ylim(0, top * 1.30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(shown) :]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r, 0].set_ylabel("% of donor's cells", fontsize=9.5)
    fig.suptitle(
        f"Per-donor cell-type proportions — each dot is ONE donor\n"
        f"{' vs '.join(f'{g} (n={int((donor_group == g).sum())})' for g in groups)}"
        f"   ·   Mann-Whitney U, Benjamini-Hochberg across {len(order)} cell types",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99 - 0.035 * nrow / max(nrow, 1)])
    fig.savefig(
        out_dir / "celltype_proportions_per_donor.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    return stats_path


# ===========================================================================
# 2. Canonical-marker dotplot per final cell type
# ===========================================================================
def plot_annotation_marker_dotplot(
    adata: AnnData,
    *,
    celltype_col: str,
    out_dir: Path,
    analysis_name: str = "",
) -> Optional[Path]:
    """Does each final label actually express its defining genes?

    Only marker groups whose cell type is present in this run are drawn, so a blood
    cohort does not get an oligodendrocyte column. Uses ``adata.raw`` when present
    (log-normalized, all genes) — ``.X`` at this stage is scaled, where a dotplot's
    colour scale is meaningless.
    """
    tag = f"[{analysis_name}] " if analysis_name else ""
    if celltype_col not in adata.obs.columns:
        logger.info("%s[CT-DOTPLOT] '%s' missing → skip.", tag, celltype_col)
        return None

    labels = adata.obs[celltype_col].astype(str)
    present = set(labels.unique())

    def _relevant(ct_key: str) -> bool:
        k = ct_key.lower().replace(" cell", "")
        return any(k in p.lower() for p in present)

    var_names = set(
        map(str, adata.raw.var_names if adata.raw is not None else adata.var_names)
    )
    groups: Dict[str, List[str]] = {}
    for ct_key, genes in CANONICAL_MARKERS.items():
        keep = [g for g in genes if g in var_names]
        if keep and (_relevant(ct_key) or ct_key in ("T cell", "Monocyte")):
            groups[ct_key] = keep
    # Fall back to every marker group we can plot rather than draw nothing.
    if len(groups) < 2:
        groups = {
            k: [g for g in v if g in var_names] for k, v in CANONICAL_MARKERS.items()
        }
        groups = {k: v for k, v in groups.items() if v}
    if not groups:
        logger.info(
            "%s[CT-DOTPLOT] no canonical markers found in var_names → skip.", tag
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{_safe(analysis_name)}_annotation_marker_dotplot.png"
    try:
        import scanpy as sc

        n_ct = labels.nunique()
        width = min(28.0, 0.32 * sum(len(v) for v in groups.values()) + 4.0)
        height = min(20.0, 0.42 * n_ct + 2.5)
        dp = sc.pl.dotplot(
            adata,
            groups,
            groupby=celltype_col,
            use_raw=adata.raw is not None,
            standard_scale="var",
            show=False,
            return_fig=True,
            figsize=(width, height),
            dendrogram=False,
        )
        dp.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close("all")
    except Exception as e:  # noqa: BLE001 - a figure must never fail the run
        logger.warning(
            "%s[CT-DOTPLOT] could not draw dotplot (%s); skipped.",
            tag,
            e,
            exc_info=True,
        )
        return None

    logger.info(
        "%s[CT-DOTPLOT] canonical-marker dotplot -> %s (%s marker groups x %s cell types).",
        tag,
        out_png.name,
        len(groups),
        labels.nunique(),
    )
    return out_png


# ===========================================================================
# 3. Voter agreement
# ===========================================================================
_VOTER_COLS = [
    ("celltypist", "celltypist_label"),
    ("singler", "singler_label"),
    ("knowledge_based", "knowledge_based_label"),
    ("pubmed", "pubmed_label"),
]


def plot_voter_agreement(
    consensus_csv: Path,
    *,
    out_dir: Path,
    analysis_name: str = "",
) -> Optional[Path]:
    """Cluster x voter grid: did each annotator match the label that shipped?

    Reads the consensus table rather than re-deriving anything, so it reflects
    exactly what was written. A saturated voter shows up as a near-solid column of
    mismatches; an overridden majority as a row that disagrees with most voters.
    """
    tag = f"[{analysis_name}] " if analysis_name else ""
    try:
        df = pd.read_csv(consensus_csv)
    except Exception as e:  # noqa: BLE001
        logger.info(
            "%s[VOTER-AGREE] cannot read %s (%s) → skip.", tag, consensus_csv, e
        )
        return None
    if "final_celltype" not in df.columns or "cluster" not in df.columns:
        logger.info(
            "%s[VOTER-AGREE] consensus table lacks required columns → skip.", tag
        )
        return None

    try:
        from .celltype_consensus import tools as _t

        harm = _t.harmonize_label
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001
        logger.debug("%s: falling back after %r", __name__, exc)
        harm = lambda x: str(x)  # noqa: E731 - identity fallback

    voters = [(n, c) for n, c in _VOTER_COLS if c in df.columns]
    if not voters:
        logger.info("%s[VOTER-AGREE] no voter columns → skip.", tag)
        return None

    # 1 = matches the final call, 0 = disagrees, NaN = abstained / disabled
    mat = np.full((len(df), len(voters)), np.nan)
    for i, row in enumerate(df.to_dict("records")):
        final = str(row["final_celltype"])
        for j, (_, col) in enumerate(voters):
            raw = row.get(col)
            if pd.isna(raw) or str(raw).strip() in ("", "Unassigned", "(disabled)"):
                continue
            mat[i, j] = 1.0 if harm(str(raw)) == final else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.5 * len(voters) + 4.5, 0.34 * len(df) + 2.2))
    cmap = matplotlib.colors.ListedColormap(["#C44E52", "#55A868"])
    cmap.set_bad("#DDDDDD")
    ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(voters)))
    ax.set_xticklabels([n for n, _ in voters], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(df)))
    lbl = [
        f"{r['cluster']}  {str(r['final_celltype'])[:30]}"
        for r in df.to_dict("records")
    ]
    ax.set_yticklabels(lbl, fontsize=8)
    for i in range(len(df)):
        for j in range(len(voters)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#666")
    agree = np.nansum(mat, axis=0)
    tested = np.sum(~np.isnan(mat), axis=0)
    ax.set_xlabel(
        "  ".join(
            f"{n}: {int(a)}/{int(t)}"
            for (n, _), a, t in zip(voters, agree, tested, strict=True)
        ),
        fontsize=8.5,
    )
    ax.set_title(
        "Voter agreement with the final call\n"
        "green = matches   ·   red = disagrees   ·   grey/– = abstained",
        fontsize=10.5,
    )
    fig.tight_layout()
    out_png = out_dir / f"{_safe(analysis_name)}_voter_agreement.png"
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "%s[VOTER-AGREE] %s %s",
        tag,
        out_png.name,
        ", ".join(
            f"{n}={int(a)}/{int(t)}"
            for (n, _), a, t in zip(voters, agree, tested, strict=True)
        ),
    )
    return out_png


# ===========================================================================
# 4. Volcano per cell type, from the pseudobulk DE tables
# ===========================================================================
def plot_pseudobulk_volcanoes(
    pseudobulk_dir: Path,
    *,
    out_dir: Path,
    analysis_name: str = "",
    padj_cut: float = 0.05,
    lfc_cut: float = 1.0,
    n_label: int = 10,
) -> List[Path]:
    """One volcano per ``*_pseudobulk_de.csv``. Returns the figures written.

    The pipeline writes these tables but no plots, so a reader has to open a
    4 MB CSV to see whether anything moved.
    """
    tag = f"[{analysis_name}] " if analysis_name else ""
    # Both spellings: the lowercase name is current, the uppercase one is what
    # runs before the Rule 5.1 rename produced.
    files = sorted(
        set(Path(pseudobulk_dir).glob("*_pseudobulk_de.csv"))
        | set(Path(pseudobulk_dir).glob("*_pseudobulk_DE.csv"))
    )
    if not files:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for f in files:
        try:
            raw = read_table(f)
        except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001
            logger.debug("%s: falling back after %r", __name__, exc)
            continue
        gene_c = next(
            (c for c in (*GENE_COLUMNS, "Unnamed: 0") if c in raw.columns),
            None,
        )
        lfc_c = next(
            (c for c in LOG2FC_COLUMNS if c in raw.columns),
            None,
        )
        p_c = next((c for c in PADJ_COLUMNS if c in raw.columns), None)
        if not (gene_c and lfc_c and p_c):
            continue

        name = f.name.replace("_pseudobulk_de.csv", "").replace(
            "_pseudobulk_DE.csv", ""
        )
        # One CSV holds EVERY group pair, so plot one volcano per contrast — overlaying
        # them would mix contradictory directions onto a single axis.
        blocks = (
            raw.groupby("comparison", sort=True)
            if "comparison" in raw.columns
            else [(None, raw)]
        )
        for comparison, block in blocks:
            keep = [gene_c, lfc_c, p_c] + [
                c for c in ("regulation",) if c in block.columns
            ]
            d = block[keep].dropna(subset=[gene_c, lfc_c, p_c])
            if d.empty:
                continue
            d = d.copy()
            d["_y"] = -np.log10(d[p_c].clip(lower=1e-300))
            if "regulation" in d.columns:
                # Trust the DE model's own call: padj already encodes the tested
                # effect size, so re-filtering on |LFC| would drop genes that passed
                # the formal H0:|log2FC|<=threshold test with a shrunken estimate.
                sig = d["regulation"].isin(["up", "down"])
                rule = "model call (padj + tested effect size)"
            else:
                sig = (d[p_c] < padj_cut) & (d[lfc_c].abs() > lfc_cut)
                rule = f"padj<{padj_cut}, |LFC|>{lfc_cut}"

            direction = ""
            if "contrast_direction" in block.columns and len(block):
                direction = str(block["contrast_direction"].iloc[0])
            elif comparison:
                direction = f"positive = higher in {str(comparison).split('_vs_')[0]}"

            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            ax.scatter(
                d.loc[~sig, lfc_c],
                d.loc[~sig, "_y"],
                s=7,
                color="#BBBBBB",
                alpha=0.55,
                linewidths=0,
            )
            ax.scatter(
                d.loc[sig, lfc_c],
                d.loc[sig, "_y"],
                s=11,
                color="#C44E52",
                alpha=0.85,
                linewidths=0,
            )
            ax.axhline(-np.log10(padj_cut), color="#888", lw=0.8, ls="--")
            for x in (-lfc_cut, lfc_cut):
                ax.axvline(x, color="#888", lw=0.8, ls="--")
            for r in d[sig].nlargest(n_label, "_y").to_dict("records"):
                ax.annotate(
                    str(r[gene_c]),
                    (r[lfc_c], r["_y"]),
                    fontsize=7,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
            title = f"{name} — pseudobulk DE"
            if comparison:
                title += f"\n{comparison}"
            title += f"\n{int(sig.sum())} genes ({rule})"
            if direction:
                title += f"\n{direction}"
            ax.set_xlabel("log2 fold change (shrunken)")
            ax.set_ylabel("-log10 adjusted p")
            ax.set_title(title, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            suffix = f"_{_safe(str(comparison))}" if comparison else ""
            p = out_dir / f"volcano_{_safe(name)}{suffix}.png"
            fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            written.append(p)
    if written:
        logger.info(
            "%s[VOLCANO] wrote %s pseudobulk volcano plot(s).", tag, len(written)
        )
    return written
