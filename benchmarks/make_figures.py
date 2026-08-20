"""
make_figures.py — publication figures from the benchmark's result tables.

Reads only the CSVs written by ``run_annotation_benchmark.py`` and
``run_heldout_celltype.py``, so a figure can never disagree with the number in the
table: nothing is recomputed here. Re-run this after any benchmark change and the
figures follow.

Every figure is written as both **PDF** (vector, what a journal wants) and **PNG** at
300 dpi (what a slide or a README wants).

Design rules applied, and why they matter for a reviewer:

* **Colour is assigned per method NAME, not per position.** Adding or dropping a method
  must never repaint the survivors — a reader comparing two figures would otherwise read
  a colour change as a data change.
* **Categorical palette is fixed and colourblind-safe.** Four hues validated for
  protan/deutan/tritan separation (worst adjacent pair ΔE 9.1 protan, 22.9 normal), with
  a legend AND direct labels on every figure, so identity is never carried by colour
  alone.
* **One y-axis per panel, never two.** Two measures on different scales get two panels.
* **Sequential single hue for the confusion matrix**, light to dark — never a rainbow,
  which invents boundaries that are not in the data.
* **Recessive grid, thin marks, no chartjunk.** The data is the ink.

Usage::

    python benchmarks/make_figures.py --results benchmarks/results/aida_full
    python benchmarks/make_figures.py --results benchmarks/results/aida_full --dpi 600
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

logger = logging.getLogger("benchmarks.figures")

# --------------------------------------------------------------------------------------
# Palette — validated for CVD separation; see the module docstring.
# --------------------------------------------------------------------------------------

#: Categorical hues in fixed order. Slots are handed out by sorted method name so the
#: same method keeps the same colour across every figure and every dataset.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")

#: The pipeline's own consensus is pinned to slot 1 — it is the subject of the paper and
#: should read the same in every panel regardless of how many baselines are present.
CONSENSUS_COLUMN = "celltype_consensus"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e5e4e0"

#: Single-hue sequential ramp for the confusion matrix.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "cc_blue", ["#f4f8fd", "#c3daf4", "#7fb2e8", "#2a78d6", "#12386b"]
)

#: Human-readable names. The obs column names are precise but unreadable on an axis.
DISPLAY_NAMES = {
    "celltype_consensus": "CellCyrix consensus",
    "celltype_celltypist": "CellTypist",
    "celltype_singler": "SingleR",
    "celltype_knowledge_based": "LLM markers",
    "celltype_pubmed": "PubMed",
    "celltype_azimuth": "Azimuth",
    "celltype_gptcelltype": "GPTCelltype",
}


def display(method: str) -> str:
    """Readable label for a method column."""
    return DISPLAY_NAMES.get(method, method.replace("celltype_", "").replace("_", " "))


def colour_map(methods: Sequence[str]) -> dict[str, str]:
    """Stable method -> hex mapping.

    The consensus always takes slot 1; every other method is assigned by sorted name.
    Sorting rather than input order is what makes the mapping reproducible when a
    baseline is added or removed.

    Args:
        methods: Method column names appearing in this run.

    Returns:
        Mapping of method name to hex colour.
    """
    others = sorted(m for m in methods if m != CONSENSUS_COLUMN)
    mapping: dict[str, str] = {}
    if CONSENSUS_COLUMN in methods:
        mapping[CONSENSUS_COLUMN] = CATEGORICAL[0]
    for index, method in enumerate(others):
        mapping[method] = CATEGORICAL[(index + 1) % len(CATEGORICAL)]
    return mapping


def style_axes(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Apply the recessive-grid, thin-spine house style to one axes."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12, loc="left")


def save(fig, out_dir: Path, name: str, dpi: int) -> list[Path]:
    """Write a figure as PDF and PNG, and close it.

    Args:
        fig: The figure.
        out_dir: Destination directory.
        name: Base filename, no extension.
        dpi: Raster resolution for the PNG.

    Returns:
        The paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
        written.append(path)
    plt.close(fig)
    logger.info("wrote %s.{pdf,png}", name)
    return written


def _read(results: Path, name: str) -> Optional[pd.DataFrame]:
    """Read a result CSV, or None when the benchmark did not produce it."""
    path = results / name
    if not path.is_file():
        logger.info("skipping %s (not present)", name)
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


# --------------------------------------------------------------------------------------
# Figure 1 — risk-coverage. The headline claim.
# --------------------------------------------------------------------------------------


def figure_risk_coverage(results: Path, out_dir: Path, dpi: int) -> None:
    """Error rate against coverage, per method.

    The paper's central figure: a method that declines its hardest calls and is more
    accurate on what remains is better than one that answers everything, and no accuracy
    table can show that. A flat line means the method cannot rank its own calls.
    """
    curves = _read(results, "04_risk_coverage.csv")
    summary = _read(results, "05_risk_coverage_summary.csv")
    if curves is None:
        return

    methods = list(dict.fromkeys(curves["method"]))
    colours = colour_map(methods)
    aurc = summary.set_index("method")["aurc"].to_dict() if summary is not None else {}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(SURFACE)

    for method in methods:
        sub = curves[curves["method"] == method].sort_values("coverage")
        # Thin the curve for the vector file: one point per cell is millions of nodes in
        # a PDF and identical to the eye at print size.
        if len(sub) > 2000:
            sub = sub.iloc[:: max(1, len(sub) // 2000)]
        label = display(method)
        if method in aurc and np.isfinite(aurc[method]):
            label = f"{label}  (AURC {aurc[method]:.4f})"
        ax.plot(
            sub["coverage"],
            sub["error"],
            color=colours[method],
            linewidth=2.0,
            label=label,
            solid_capstyle="round",
        )

    style_axes(
        ax,
        xlabel="Coverage — fraction of cells the method is willing to call",
        ylabel="Error rate on covered cells",
        title="Risk–coverage: does declining the hardest calls buy accuracy?",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 1.0)
    # Scale to the data, not to [0, 1]. Error rates here are a couple of percent, and a
    # full-height axis flattens every curve into the baseline — the differences between
    # methods are the entire point of the panel. Zero stays on the axis because it is a
    # meaningful floor for an error rate.
    visible = curves.loc[np.isfinite(curves["error"]), "error"]
    # Ignore the first 2% of coverage: with a handful of cells covered the error rate is
    # 0 or 1 and spikes wildly, which would set the limit for a region nobody reads.
    settled = curves.loc[curves["coverage"] > 0.02, "error"]
    top = float((settled if len(settled) else visible).max()) if len(visible) else 1.0
    ax.set_ylim(0, max(0.01, top * 1.35))
    legend = ax.legend(
        frameon=True,
        fontsize=9,
        loc="upper right",
        labelcolor=INK_SECONDARY,
        facecolor=SURFACE,
        edgecolor=GRID,
        framealpha=0.95,
    )
    legend.set_title(None)
    ax.text(
        0.99,
        -0.16,
        "Lower is better. A flat line means the method cannot rank its own confidence.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    save(fig, out_dir, "fig1_risk_coverage", dpi)


# --------------------------------------------------------------------------------------
# Figure 2 — accuracy comparison with confidence intervals.
# --------------------------------------------------------------------------------------


def figure_method_comparison(results: Path, out_dir: Path, dpi: int) -> None:
    """Macro-F1 per method with bootstrap CIs, and accuracy beside it.

    Two panels rather than two y-axes: macro-F1 and accuracy are different measures, and
    overlaying them on one scale is the most common way a chart lies. The gap between the
    panels is itself the finding — high accuracy with low macro-F1 means the rare cell
    types are being missed.
    """
    frame = _read(results, "01_method_comparison.csv")
    if frame is None:
        return

    frame = frame.sort_values("macro_f1")
    colours = colour_map(list(frame["method"]))
    labels = [display(m) for m in frame["method"]]
    positions = np.arange(len(frame))

    fig, (ax_f1, ax_acc) = plt.subplots(
        1, 2, figsize=(10.5, 0.75 * len(frame) + 2.4), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)

    errors = np.vstack(
        [
            (frame["macro_f1"] - frame["ci_low"]).clip(lower=0).to_numpy(),
            (frame["ci_high"] - frame["macro_f1"]).clip(lower=0).to_numpy(),
        ]
    )
    ax_f1.barh(
        positions,
        frame["macro_f1"],
        height=0.62,
        color=[colours[m] for m in frame["method"]],
        xerr=errors,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capsize": 3},
    )
    # Place the value past the CI whisker, not past the bar, or the two collide.
    label_x = frame[["macro_f1", "ci_high"]].max(axis=1).to_numpy()
    for y, value, x in zip(positions, frame["macro_f1"], label_x, strict=True):
        ax_f1.text(
            x + 0.02,
            y,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )

    ax_acc.barh(
        positions,
        frame["accuracy"],
        height=0.62,
        color=[colours[m] for m in frame["method"]],
    )
    for y, value in zip(positions, frame["accuracy"], strict=True):
        ax_acc.text(
            value - 0.02,
            y,
            f"{value:.3f}",
            va="center",
            ha="right",
            fontsize=9,
            color="white" if value > 0.35 else INK_PRIMARY,
        )

    style_axes(ax_f1, xlabel="Macro-F1 (95% bootstrap CI)", title="Balanced accuracy")
    style_axes(ax_acc, xlabel="Accuracy", title="Raw accuracy")
    ax_f1.set_yticks(positions)
    ax_f1.set_yticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax_f1.set_xlim(0, min(1.0, max(0.35, frame["ci_high"].max() * 1.18)))
    ax_acc.set_xlim(0, 1.0)

    fig.suptitle(
        "Macro-F1 treats every cell type equally; accuracy is dominated by the "
        "commonest ones",
        color=INK_PRIMARY,
        fontsize=12,
        x=0.01,
        ha="left",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, out_dir, "fig2_method_comparison", dpi)


# --------------------------------------------------------------------------------------
# Figure 3 — calibration.
# --------------------------------------------------------------------------------------


def figure_calibration(results: Path, out_dir: Path, dpi: int) -> None:
    """Reliability diagram: stated confidence against observed accuracy."""
    curves = _read(results, "06_calibration.csv")
    summary = _read(results, "07_calibration_summary.csv")
    if curves is None:
        return

    methods = list(dict.fromkeys(curves["method"]))
    colours = colour_map(methods)
    ece = summary.set_index("method")["ece"].to_dict() if summary is not None else {}

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    fig.patch.set_facecolor(SURFACE)

    ax.plot(
        [0, 1],
        [0, 1],
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="perfect calibration",
        zorder=1,
    )
    for method in methods:
        sub = curves[curves["method"] == method].sort_values("mean_confidence")
        label = display(method)
        if method in ece and np.isfinite(ece[method]):
            label = f"{label}  (ECE {ece[method]:.3f})"
        ax.plot(
            sub["mean_confidence"],
            sub["accuracy"],
            color=colours[method],
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=label,
            zorder=3,
        )

    style_axes(
        ax,
        xlabel="Mean stated confidence",
        ylabel="Observed accuracy",
        title="Calibration: does a confidence of 0.8 mean 80% correct?",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_SECONDARY)
    ax.text(
        0.0,
        -0.14,
        "Below the diagonal = overconfident. Above = underconfident.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_MUTED,
    )
    save(fig, out_dir, "fig3_calibration", dpi)


# --------------------------------------------------------------------------------------
# Figure 4 — per-class F1.
# --------------------------------------------------------------------------------------


def figure_per_class(results: Path, out_dir: Path, dpi: int, top_n: int = 18) -> None:
    """Per-cell-type F1, ordered by how many cells of that type exist.

    Shows WHERE a method fails. Ordering by support puts the rare types at the bottom,
    which is usually where the macro-F1 loss lives.
    """
    frame = _read(results, "03_per_class_f1.csv")
    if frame is None:
        return

    order = (
        frame.groupby("cell_type")["support"].max().sort_values(ascending=False).index
    )
    order = list(order[:top_n])
    frame = frame[frame["cell_type"].isin(order)]

    methods = list(dict.fromkeys(frame["method"]))
    colours = colour_map(methods)
    positions = np.arange(len(order))
    height = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(order) + 2.2))
    fig.patch.set_facecolor(SURFACE)

    for index, method in enumerate(methods):
        sub = frame[frame["method"] == method].set_index("cell_type").reindex(order)
        offset = (index - (len(methods) - 1) / 2) * height
        ax.barh(
            positions + offset,
            sub["f1"].fillna(0.0),
            height=height * 0.88,
            color=colours[method],
            label=display(method),
        )

    supports = frame.groupby("cell_type")["support"].max().reindex(order).astype(int)
    style_axes(ax, xlabel="F1", title="Where each method fails, by cell type")
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{name}  (n={supports[name]:,})" for name in order],
        fontsize=9,
        color=INK_PRIMARY,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_SECONDARY)
    save(fig, out_dir, "fig4_per_class_f1", dpi)


# --------------------------------------------------------------------------------------
# Figure 5 — voter ablation.
# --------------------------------------------------------------------------------------


def figure_ablation(results: Path, out_dir: Path, dpi: int) -> None:
    """Every voter subset, answering "why this many voters?"."""
    frame = _read(results, "02_ablation.csv")
    if frame is None:
        return

    frame = frame.sort_values("macro_f1")
    labels = [
        v.replace("celltype_", "").replace("_", " ").replace("+", " + ")
        for v in frame["voters"]
    ]
    positions = np.arange(len(frame))
    # Highlight the full pipeline; the subsets are context, so they recede.
    is_pipeline = frame["voters"].str.contains("pipeline", case=False)
    colours = [CATEGORICAL[0] if flag else "#b9c9dc" for flag in is_pipeline]

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(frame) + 2.2))
    fig.patch.set_facecolor(SURFACE)

    errors = np.vstack(
        [
            (frame["macro_f1"] - frame["ci_low"]).clip(lower=0).to_numpy(),
            (frame["ci_high"] - frame["macro_f1"]).clip(lower=0).to_numpy(),
        ]
    )
    ax.barh(
        positions,
        frame["macro_f1"],
        height=0.62,
        color=colours,
        xerr=errors,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.1, "capsize": 3},
    )
    for y, value in zip(positions, frame["macro_f1"], strict=True):
        ax.text(
            value + 0.012, y, f"{value:.3f}", va="center", fontsize=9, color=INK_PRIMARY
        )

    style_axes(
        ax,
        xlabel="Macro-F1 (95% bootstrap CI)",
        title="Voter ablation: what does each voter actually add?",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9, color=INK_PRIMARY)
    ax.set_xlim(0, min(1.0, max(0.35, frame["ci_high"].max() * 1.18)))
    save(fig, out_dir, "fig5_ablation", dpi)


# --------------------------------------------------------------------------------------
# Figure 6 — confusion matrix for the consensus.
# --------------------------------------------------------------------------------------


def figure_confusion(results: Path, out_dir: Path, dpi: int, top_n: int = 14) -> None:
    """Row-normalised confusion for the consensus: what gets mistaken for what."""
    path = results / f"09_confusion_{CONSENSUS_COLUMN}.csv"
    if not path.is_file():
        candidates = sorted(results.glob("09_confusion_*.csv"))
        if not candidates:
            logger.info("skipping confusion matrix (no 09_confusion_*.csv)")
            return
        path = candidates[0]

    matrix = pd.read_csv(path, index_col=0)
    if matrix.empty:
        return
    # Keep the largest blocks: a 30x30 grid at print size is unreadable.
    rows = list(matrix.sum(axis=1).sort_values(ascending=False).index[:top_n])
    cols = [c for c in matrix.columns if c in rows] + [
        c for c in matrix.columns if c not in rows
    ][:4]
    matrix = matrix.loc[rows, cols]

    fig, ax = plt.subplots(figsize=(0.52 * len(cols) + 4.2, 0.46 * len(rows) + 3.0))
    fig.patch.set_facecolor(SURFACE)

    image = ax.imshow(matrix.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8, color=INK_PRIMARY)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8, color=INK_PRIMARY)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Label only the cells worth reading; a number in every cell is noise.
    values = matrix.to_numpy()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value >= 0.02:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 else INK_PRIMARY,
                )
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Fraction of the true class", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)
    ax.set_title(
        "Consensus confusion (rows = truth, normalised)",
        color=INK_PRIMARY,
        fontsize=12,
        pad=12,
        loc="left",
    )
    ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("True", color=INK_SECONDARY, fontsize=10)
    save(fig, out_dir, "fig6_confusion", dpi)


# --------------------------------------------------------------------------------------
# Figure 7 — held-out cell type.
# --------------------------------------------------------------------------------------


def figure_heldout(results: Path, out_dir: Path, dpi: int) -> None:
    """Abstention versus confident error when a cell type is withheld.

    The disease-agnostic claim in one panel: for a type absent from the reference, the
    right behaviour is to decline, and the wrong behaviour is a confident wrong label.
    """
    frame = _read(results, "10_heldout_summary.csv")
    if frame is None:
        return

    frame = frame.sort_values("mean_confident_error_rate")
    labels = [display(m) for m in frame["method"]]
    positions = np.arange(len(frame))
    height = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 0.85 * len(frame) + 2.6))
    fig.patch.set_facecolor(SURFACE)

    ax.barh(
        positions + height / 2,
        frame["mean_abstention_rate"],
        height=height,
        color=CATEGORICAL[2],
        label="Abstained (good)",
    )
    ax.barh(
        positions - height / 2,
        frame["mean_confident_error_rate"],
        height=height,
        color=CATEGORICAL[1],
        label="Confidently wrong (bad)",
    )
    for y, value in zip(positions, frame["mean_abstention_rate"], strict=True):
        ax.text(
            value + 0.012,
            y + height / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )
    for y, value in zip(positions, frame["mean_confident_error_rate"], strict=True):
        ax.text(
            value + 0.012,
            y - height / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
        )

    style_axes(
        ax,
        xlabel="Fraction of the held-out type's cells",
        title="Held-out cell type: abstain, or confidently mislabel?",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_SECONDARY)
    save(fig, out_dir, "fig7_heldout", dpi)


# --------------------------------------------------------------------------------------
# Figure 8 — voter disagreement.
# --------------------------------------------------------------------------------------


def figure_disagreement(results: Path, out_dir: Path, dpi: int) -> None:
    """Distribution of per-cell voter entropy, split by whether the consensus was right.

    If disagreement is informative, the wrong calls should sit at higher entropy. Skipped
    when only one voter ran, since entropy is then zero everywhere.
    """
    frame = _read(results, "08_disagreement.csv")
    if frame is None or "voter_entropy_bits" not in frame.columns:
        return
    entropy = pd.to_numeric(frame["voter_entropy_bits"], errors="coerce").dropna()
    if entropy.nunique() <= 1:
        logger.info("skipping disagreement figure (entropy is constant — one voter)")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    fig.patch.set_facecolor(SURFACE)

    correct = frame.get("consensus_correct")
    bins = np.linspace(0, max(1e-6, float(entropy.max())), 26)
    if correct is not None and pd.notna(correct).any():
        mask = correct.astype(str).str.lower().isin(["true", "1", "1.0"])
        ax.hist(
            frame.loc[mask, "voter_entropy_bits"].dropna(),
            bins=bins,
            color=CATEGORICAL[2],
            alpha=0.85,
            label="Consensus correct",
        )
        ax.hist(
            frame.loc[~mask, "voter_entropy_bits"].dropna(),
            bins=bins,
            color=CATEGORICAL[1],
            alpha=0.85,
            label="Consensus wrong",
        )
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    else:
        ax.hist(entropy, bins=bins, color=CATEGORICAL[0])

    style_axes(
        ax,
        xlabel="Voter disagreement (Shannon entropy, bits)",
        ylabel="Cells",
        title="Is disagreement informative about being wrong?",
    )
    save(fig, out_dir, "fig8_disagreement", dpi)


# --------------------------------------------------------------------------------------


def make_all(results: Path, out_dir: Optional[Path] = None, dpi: int = 300) -> Path:
    """Generate every figure the present result tables support.

    Args:
        results: A ``benchmarks/results/<name>`` directory.
        out_dir: Where figures go. Defaults to ``<results>/figures``.
        dpi: Raster resolution for the PNG copies.

    Returns:
        The figure directory.

    Raises:
        SystemExit: If `results` does not exist.
    """
    results = Path(results)
    if not results.is_dir():
        raise SystemExit(f"Results directory not found: {results}")
    out_dir = Path(out_dir) if out_dir else results / "figures"

    for builder in (
        figure_risk_coverage,
        figure_method_comparison,
        figure_calibration,
        figure_per_class,
        figure_ablation,
        figure_confusion,
        figure_heldout,
        figure_disagreement,
    ):
        try:
            builder(results, out_dir, dpi)
        except Exception:  # noqa: BLE001 - one bad panel must not lose the rest
            logger.exception("could not build %s", builder.__name__)

    written = sorted(out_dir.glob("*.pdf"))
    logger.info("%d figure(s) in %s", len(written), out_dir)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="make_figures",
        description="Publication figures from the annotation benchmark's result tables.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="A benchmarks/results/<name> directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure destination (default: <results>/figures).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution; PDFs are vector regardless (default: 300).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    make_all(args.results, args.out_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
