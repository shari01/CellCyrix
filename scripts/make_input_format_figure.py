"""
make_input_format_figure.py — render the input-format diagram for docs/INPUT_FORMATS.md.

Draws the three things a user has to get right before a run: the Cell Ranger file
trio, the single- vs multi-mode directory layout, and the load/validate path that
turns those files into the AnnData every downstream stage reads. Output is written
to ``docs/figures/input_formats.png``.

Run:  python scripts/make_input_format_figure.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

logger = logging.getLogger(__name__)

# House palette — matches the other figures in docs/figures/.
NAVY = "#14304d"
NAVY_FILL = "#eef4f8"
TEAL = "#17808f"
TEAL_FILL = "#e7f3f4"
ORANGE = "#f5a623"
ORANGE_FILL = "#fdf3e1"
RED = "#c0453f"
RED_FILL = "#fbeceb"
ARROW = "#d5dce2"
MUTED = "#6b7a88"

MONO = "DejaVu Sans Mono"

OUT_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "figures" / "input_formats.png"
)


def box(ax, x, y, w, h, edge, fill, lw=2.0, zorder=2):
    """Draw one sharp-cornered panel and return its (centre_x, centre_y)."""
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=zorder,
        )
    )
    return x + w / 2.0, y + h / 2.0


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=3.0, zorder=1):
    """Thick soft arrow used for every flow edge."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def section(ax, x, y, text):
    """Small teal column header."""
    ax.text(
        x,
        y,
        text,
        fontsize=11,
        fontweight="bold",
        color=TEAL,
        ha="left",
        va="center",
    )


def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(20, 9.4))
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 94)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------ column 1
    section(ax, 4, 89, "1 · FILES IN EVERY SAMPLE FOLDER  (Cell Ranger trio)")

    files = [
        (
            "matrix.mtx[.gz]",
            "Matrix Market, genes × cells",
            "also: *.matrix.mtx[.gz] · *.mtx[.gz]",
        ),
        (
            "barcodes.tsv[.gz]",
            "headerless TSV, col 1 = cell barcode",
            "also: *barcodes.tsv[.gz] · barcode.tsv[.gz]",
        ),
        (
            "features.tsv[.gz]",
            "headerless TSV: feature_id, feature_name, feature_type",
            "also: genes.tsv[.gz] · *features.tsv[.gz]",
        ),
    ]
    tops = [76.0, 60.0, 44.0]
    # strict=True: the two lists are hand-maintained side by side, so a mismatch is an
    # editing error that should raise rather than silently drop the last file's box.
    for (name, meaning, alt), top in zip(files, tops, strict=True):
        box(ax, 4, top - 13, 50, 13, NAVY, NAVY_FILL)
        ax.text(
            7,
            top - 3.6,
            name,
            fontsize=13,
            fontweight="bold",
            family=MONO,
            color=NAVY,
            va="center",
        )
        ax.text(7, top - 7.6, meaning, fontsize=9.5, color=NAVY, va="center")
        ax.text(
            7,
            top - 10.9,
            alt,
            fontsize=8.5,
            style="italic",
            color=MUTED,
            va="center",
        )

    ax.text(
        4,
        27,
        "All three required — a missing file raises FileNotFoundError,\n"
        "a corrupt one raises ValueError naming the path.",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )
    ax.text(
        4,
        17.5,
        "Plain or .gz · bare or sample-prefixed\n(GSM6567157_matrix.mtx.gz both work).",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )

    # ------------------------------------------------------------------ column 2
    section(ax, 62, 89, "2 · DIRECTORY LAYOUT  (two supported shapes)")

    # single mode
    box(ax, 62, 63, 62, 20, NAVY, "white")
    ax.text(
        65, 80.0, "SINGLE MODE", fontsize=10, fontweight="bold", color=NAVY, va="center"
    )
    ax.text(
        65,
        76.3,
        "single_10x_dir  →  one 10x folder",
        fontsize=9.5,
        color=MUTED,
        va="center",
    )
    ax.text(
        65,
        69.5,
        "GSM8664023_NST8_CD8skin/\n"
        "├── matrix.mtx.gz\n"
        "├── barcodes.tsv.gz\n"
        "└── features.tsv.gz",
        fontsize=10,
        family=MONO,
        color=NAVY,
        va="center",
        linespacing=1.45,
    )

    # multi mode
    box(ax, 62, 20, 62, 38, TEAL, "white")
    ax.text(
        65, 54.6, "MULTI MODE", fontsize=10, fontweight="bold", color=TEAL, va="center"
    )
    ax.text(
        65,
        50.8,
        "multi_base_dir  →  <GROUP>/<SAMPLE>/ tree",
        fontsize=9.5,
        color=MUTED,
        va="center",
    )
    ax.text(
        65,
        35.0,
        "multi_base_dir/\n"
        "├── metadata.csv            sample,group  (optional)\n"
        "├── PDAC/\n"
        "│   ├── GSM6567157_PDAC1/   matrix + barcodes + features\n"
        "│   └── GSM6567159_PDAC2/   matrix + barcodes + features\n"
        "└── adjacent_normal/\n"
        "    ├── GSM6567169_N1/      matrix + barcodes + features\n"
        "    └── GSM6567170_N2/      matrix + barcodes + features",
        fontsize=9.5,
        family=MONO,
        color=NAVY,
        va="center",
        linespacing=1.5,
    )

    # metadata.csv note
    box(ax, 62, 5, 62, 10, ORANGE, ORANGE_FILL)
    ax.text(
        65,
        12.2,
        "metadata.csv / group_map.csv  (.xlsx fallback)",
        fontsize=9.5,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    ax.text(
        65,
        8.0,
        "sample + group columns override the folder-derived group.\n"
        "Group starting with _  (_EXCLUDED, _REVIEW, …)  → sample skipped.",
        fontsize=8.8,
        color=MUTED,
        va="center",
        linespacing=1.4,
    )

    # ------------------------------------------------------------------ column 3
    section(ax, 132, 89, "3 · LOAD AND VALIDATE")

    box(ax, 132, 68, 64, 15, NAVY, NAVY_FILL)
    ax.text(
        164,
        79.0,
        "load_10x_feature_barcode_matrix()",
        fontsize=11,
        fontweight="bold",
        family=MONO,
        color=NAVY,
        ha="center",
        va="center",
    )
    ax.text(
        164,
        73.5,
        "reads Matrix Market · transposes to cells × genes\n"
        "var_names = gene symbols, made unique",
        fontsize=9.5,
        color=NAVY,
        ha="center",
        va="center",
        linespacing=1.4,
    )

    box(ax, 132, 48, 64, 14, ORANGE, ORANGE_FILL)
    ax.text(
        164,
        58.5,
        "RAW-COUNT GATE",
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
    )
    ax.text(
        164,
        53.0,
        "finite  ·  non-negative  ·  integer-valued\n"
        "normalized / log / scaled input is rejected, not repaired",
        fontsize=9.5,
        color=NAVY,
        ha="center",
        va="center",
        linespacing=1.4,
    )

    box(ax, 132, 30, 29, 12, RED, RED_FILL)
    ax.text(
        146.5,
        38.0,
        "ValueError",
        fontsize=10.5,
        fontweight="bold",
        family=MONO,
        color=RED,
        ha="center",
        va="center",
    )
    ax.text(
        146.5,
        33.4,
        "run stops before\nany processing",
        fontsize=9,
        color=RED,
        ha="center",
        va="center",
        linespacing=1.4,
    )

    box(ax, 167, 4, 29, 38, TEAL, TEAL_FILL)
    ax.text(
        181.5,
        38.0,
        "AnnData",
        fontsize=12,
        fontweight="bold",
        color=TEAL,
        ha="center",
        va="center",
    )
    ax.text(
        169.5,
        22.0,
        "X          cells × genes\n"
        "           raw counts\n\n"
        "obs        barcode\n"
        "           sample\n"
        "           group\n\n"
        "var        feature_id\n"
        "           gene_symbol\n"
        "           feature_type",
        fontsize=9,
        family=MONO,
        color=NAVY,
        va="center",
        linespacing=1.5,
    )

    # ------------------------------------------------------------------ arrows
    arrow(ax, 54.5, 69.5, 61.0, 73.0)  # files → single
    arrow(ax, 54.5, 53.5, 61.0, 45.0)  # files → multi
    arrow(ax, 124.5, 73.0, 131.0, 76.0)  # single → loader
    arrow(ax, 124.5, 39.0, 131.0, 70.5)  # multi  → loader
    arrow(ax, 164, 67.5, 164, 62.5)  # loader → gate
    arrow(ax, 155, 47.5, 150, 42.5, color="#eccfcd")  # gate → reject
    arrow(ax, 175, 47.5, 181.5, 42.5, color="#bfe0e2")  # gate → AnnData

    ax.text(
        149.5,
        45.0,
        "fail",
        fontsize=9,
        style="italic",
        color=RED,
        ha="right",
        va="center",
    )
    ax.text(
        180.5,
        45.0,
        "pass",
        fontsize=9,
        style="italic",
        color=TEAL,
        ha="left",
        va="center",
    )
    ax.text(
        132,
        1.5,
        "One AnnData per run — multi mode concatenates every sample before "
        "QC, integration and pseudobulk DE.",
        fontsize=9,
        style="italic",
        color=MUTED,
        va="center",
    )

    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    # Rule 9 is "logging, never print()", with no production exemptions — including in
    # this developer utility. basicConfig is safe here because a figure script owns its
    # whole process and inherits no handlers, unlike the pipeline entry points.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = build()
    fig.savefig(OUT_PATH, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
