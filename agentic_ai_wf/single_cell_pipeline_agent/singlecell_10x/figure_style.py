"""Figure resolution, panel geometry and panel spacing, in one place.

Two problems this module owns.

RESOLUTION. Every figure used to be composed on scanpy's 4x4 inch panel default
and saved at 300 dpi, so a UMAP left here as a ~1.1k x 1.1k px PNG. Opened
full-screen that is soft, and zooming in to see which cells actually carry the
signal only magnifies the pixels. Panels are now ``PANEL_SIZE`` at
``FIGURE_DPI`` (~2.6k x 2.2k px for a single panel, ~4x the pixels), and marker
size scales with the panel so the extra canvas buys separation between cells
rather than bigger blobs.

SPACING. scanpy draws a categorical legend into the gap on the right of each
panel of a multi-panel grid — and the *next* panel's axis label into that same
gap. The gap is ``wspace``, a fraction of panel width that knows nothing about
how long the labels are, so it was hand-tuned to 0.4 at each call site: sample
IDs like "GSM8035466_OM" overran it and printed straight through the
neighbouring "UMAP2". ``panel_wspace`` measures the widest label that has to fit
there and sizes the gap from it.

Import is deliberately side-effect free: call ``apply_figure_style()`` once per
process. Plotting also happens in FRESH interpreters (``rank_genes_subprocess``,
``pipeline_subprocess``), and each of those must call it too or its figures come
out at scanpy's 80 dpi screen default. Keep this module leaf-level — stdlib +
matplotlib, with scanpy imported lazily — because ``rank_genes_subprocess``
imports it by directory when it runs as a script.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache, wraps
from math import ceil, sqrt
from typing import TYPE_CHECKING, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401  (import side effect: backend)
import pandas as pd  # noqa: E402  (must follow the backend selection above)
from matplotlib import rcParams  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

# --- resolution / geometry -------------------------------------------------
# 300 dpi is the print floor; 400 keeps that headroom while roughly doubling the
# pixels per axis, so the PNGs stay legible when zoomed rather than just when
# printed.
FIGURE_DPI: int = 400

# Per-panel size for every scanpy embedding (UMAP/t-SNE/PCA/diffmap) and for any
# figure that does not pass an explicit figsize.
PANEL_SIZE: tuple[float, float] = (6.4, 5.6)

# scanpy's own figure.figsize default — the baseline everything here scales from.
_SCANPY_PANEL_WIDTH: float = 4.0

# Marker sizes scale with the panel's LINEAR growth, not its area: dots stay in
# proportion to the axis without swelling into blobs, and the surplus canvas
# shows up as separation between cells.
POINT_SCALE: float = PANEL_SIZE[0] / _SCANPY_PANEL_WIDTH

# One size for all text. scanpy's set_figure_params derives legend/label/tick
# sizes from this; setting only rcParams["font.size"] (as this pipeline used to)
# left axis labels and legends at scanpy's unrelated 14 pt.
BASE_FONTSIZE: float = 13.0

# Guard rail for figures whose height is computed from a row count (marker
# heatmaps, enrichment bars): at 400 dpi an unbounded inch value turns into a
# 100-megapixel PNG, and matplotlib hard-fails past 2**16 px per side.
MAX_FIG_PX: int = 20000

# Total pixels allowed in one rasterised figure — a hard ceiling enforced at save
# time (see _install_raster_cap). Only wide dotplots and per-cluster panel grids
# come near it.
MAX_TOTAL_PX: int = 40_000_000

# What fraction of that ceiling a figure may claim BEFORE bbox_inches="tight"
# expands it to fit overflowing labels.
_TIGHT_BBOX_HEADROOM: float = 0.55

# Breathing room left around the tight bounding box, so no glyph ends up flush
# against the image edge.
SAVEFIG_PAD_INCHES: float = 0.15

# Clear space kept between a panel's legend and the next panel's axis label.
_LEGEND_GAP_MARGIN: float = 0.22

# A gap wider than this stops being spacing and starts being wasted canvas; past
# it the labels are the thing to shorten.
_MAX_WSPACE: float = 1.5

_applied: bool = False


def apply_figure_style(*, force: bool = False) -> None:
    """Install the pipeline's figure defaults on this interpreter (idempotent)."""
    global _applied
    if _applied and not force:
        return

    import scanpy as sc  # lazy: keeps this module cheap to import

    sc.settings.set_figure_params(
        dpi=110,  # on-screen only; cheap while composing
        dpi_save=FIGURE_DPI,
        format="png",
        vector_friendly=True,
        transparent=False,
        fontsize=BASE_FONTSIZE,
        figsize=PANEL_SIZE,
    )
    # scanpy's own savefig reads rcParams["savefig.dpi"], and bare plt.savefig()
    # calls elsewhere in the pipeline read it too — one value covers both.
    rcParams["savefig.dpi"] = FIGURE_DPI
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = SAVEFIG_PAD_INCHES
    _install_raster_cap()
    _applied = True


def _install_raster_cap() -> None:
    """Cap every saved figure at ``MAX_TOTAL_PX``, lowering dpi only where needed.

    Some figures are enormous in INCHES before dpi even enters: scanpy sizes a
    dotplot from its gene count (180 genes is ~60 inches wide) and a rank-plot grid
    from its cluster count, and ``bbox_inches="tight"`` then grows the canvas
    further to fit overflowing labels. At ``FIGURE_DPI`` those turn into 60-100
    megapixel PNGs — minutes of rasterising for a multi-megabyte file nobody can
    open. Such a figure saves at the dpi that fits the budget instead; every normal
    figure is far below it and keeps the full ``FIGURE_DPI``.

    The bound is on the figure's own canvas: a tight bounding box can still add a
    margin for labels that overflow it, so ``MAX_TOTAL_PX`` is set with that slack
    in mind and composition-level budgeting (``panel_grid_budget``) handles the one
    case — per-cluster grids — where the overflow is large.

    Patching ``Figure.savefig`` is the only central hook available: most figures
    here are written by scanpy's own ``save=`` path, which reads rcParams and calls
    savefig itself.
    """
    if getattr(Figure.savefig, "_sc_raster_capped", False):
        return

    original = Figure.savefig

    @wraps(original)
    def savefig(self, *args, **kwargs) -> object:
        """Cap the requested dpi so a large grid cannot exceed ``MAX_TOTAL_PX``."""
        requested = kwargs.get("dpi") or rcParams["savefig.dpi"]
        try:
            requested = float(requested)  # may be the keyword "figure"
        except (TypeError, ValueError):
            requested = float(FIGURE_DPI)
        width, height = (float(v) for v in self.get_size_inches())
        capped = sqrt(MAX_TOTAL_PX / max(width * height, 1e-6))
        if capped < requested:
            kwargs["dpi"] = capped
        return original(self, *args, **kwargs)

    savefig._sc_raster_capped = True
    Figure.savefig = savefig


def point_size(n_cells: int, *, base: float | None = None) -> float:
    """Marker size in pt^2 for an embedding scatter, scaled to ``PANEL_SIZE``.

    ``base`` is the size that was tuned for scanpy's 4x4 panel; omit it to scale
    scanpy's own default (120000 / n_cells), which tracks cell count but knows
    nothing about panel size.
    """
    if base is None:
        base = 120000.0 / max(int(n_cells), 1)
    return float(base) * POINT_SCALE


def clamp_fig_inches(inches: float, *, minimum: float = 2.0) -> float:
    """Clamp a computed figure dimension to something ``FIGURE_DPI`` can render."""
    return float(min(max(float(inches), minimum), MAX_FIG_PX / FIGURE_DPI))


@contextmanager
def panel_grid_budget(n_panels: int, *, ncols: int = 4) -> Iterator[None]:
    """Shrink the panel size for the duration if an ``n_panels`` grid is too big.

    ``sc.pl.rank_genes_groups`` sizes its grid as n_panels_x * n_panels_y copies of
    ``figure.figsize`` — one panel per cluster. A 20-cluster run on the panel size
    used for single embeddings would ask matplotlib to rasterise ~115 megapixels.
    Shrinking the panel keeps the grid inside the budget while every panel holds
    full-size TEXT, which is what makes a 20-panel grid readable; the savefig cap
    would instead have to drop the dpi and shrink the labels with it.

    Budgets against a fraction of ``MAX_TOTAL_PX`` because ``bbox_inches="tight"``
    grows the canvas past the requested figsize to fit the gene labels that
    overflow each panel (measured at ~1.8x on an 18-cluster grid).
    """
    n_x = max(1, min(int(ncols), max(1, int(n_panels))))
    n_y = int(ceil(max(1, int(n_panels)) / n_x))
    width, height = (float(v) for v in rcParams["figure.figsize"])
    total_px = (n_x * width * FIGURE_DPI) * (n_y * height * FIGURE_DPI)
    budget = MAX_TOTAL_PX * _TIGHT_BBOX_HEADROOM
    if total_px <= budget:
        yield
        return

    scale = sqrt(budget / total_px)
    previous = list(rcParams["figure.figsize"])
    rcParams["figure.figsize"] = [width * scale, height * scale]
    try:
        yield
    finally:
        rcParams["figure.figsize"] = previous


# --- panel spacing ---------------------------------------------------------


def panel_wspace(
    adata: AnnData,
    color: str | Sequence[str],
    *,
    legend_loc: str | None = "right margin",
    ncols: int = 4,
) -> float:
    """Horizontal gap for a multi-panel ``sc.pl.<embedding>`` grid.

    Returned as scanpy's ``wspace`` (a fraction of panel width) sized so the
    widest legend that has to sit *between* two panels clears the next panel's
    axis label. Single-panel calls get scanpy's default — their legend hangs off
    the right edge of the figure, where ``bbox_inches="tight"`` makes room.
    """
    colors = [color] if isinstance(color, str) else [c for c in color]
    if len(colors) < 2:
        return _default_wspace()

    grid_cols = max(1, min(int(ncols), len(colors)))
    # A panel in the last grid column draws its legend outside the figure, so it
    # does not constrain the gap. Every other panel is followed by a neighbour.
    followed = [c for i, c in enumerate(colors) if (i + 1) % grid_cols != 0]
    if not followed:
        return _default_wspace()

    legend_w = max(
        _legend_width_inches(adata, c, legend_loc=legend_loc) for c in followed
    )
    gap = legend_w + _ylabel_width_inches() + _LEGEND_GAP_MARGIN
    panel_w = float(rcParams["figure.figsize"][0]) or _SCANPY_PANEL_WIDTH
    return float(min(max(gap / panel_w, _default_wspace()), _MAX_WSPACE))


def _default_wspace() -> float:
    """scanpy's own heuristic (scatterplots.embedding), for the cases it handles."""
    panel_w = float(rcParams["figure.figsize"][0]) or _SCANPY_PANEL_WIDTH
    return 0.75 / panel_w + 0.02


def _legend_width_inches(adata, col: str, *, legend_loc: str | None) -> float:
    """Width the legend for ``col`` needs to the right of its panel, in inches."""
    if legend_loc in (None, "none", "on data", "on data export"):
        return 0.0

    fontsize = _resolve_fontsize(rcParams["legend.fontsize"])
    em = fontsize / 72.0
    labels = _categories(adata, col)

    if labels is None:
        # Continuous colour: a colorbar, which scanpy steals from the panel's own
        # area. Only its tick labels reach past the panel edge.
        return _text_width_inches("0.00", fontsize) + 0.25

    # scanpy wraps the legend into columns at these counts (_add_categorical_legend).
    ncol = 1 if len(labels) <= 14 else 2 if len(labels) <= 30 else 3
    widest = max((_text_width_inches(str(x), fontsize) for x in labels), default=0.0)
    handle = (rcParams["legend.handlelength"] + rcParams["legend.handletextpad"]) * em
    spacing = (ncol - 1) * float(rcParams["legend.columnspacing"]) * em
    pad = (
        2 * float(rcParams["legend.borderpad"])
        + float(rcParams["legend.borderaxespad"])
    ) * em
    return ncol * (widest + handle) + spacing + pad


def _ylabel_width_inches() -> float:
    """Space the neighbouring panel's rotated y-axis label occupies, in inches."""
    fontsize = _resolve_fontsize(rcParams["axes.labelsize"])
    return (fontsize * 1.25 + float(rcParams["axes.labelpad"])) / 72.0


def _categories(adata, col: str) -> list | None:
    """Legend labels for ``col``, or ``None`` when it is coloured continuously."""
    obs = getattr(adata, "obs", None)
    if obs is None or col not in getattr(obs, "columns", []):
        return None  # a gene / obsm key -> continuous colour
    series = obs[col]
    dtype = series.dtype
    if hasattr(dtype, "categories"):
        return list(dtype.categories)
    if pd.api.types.is_bool_dtype(dtype):
        return ["False", "True"]
    if pd.api.types.is_object_dtype(dtype) or str(dtype) == "string":
        return [str(x) for x in series.unique()]
    return None


def _resolve_fontsize(size) -> float:
    """Points for an rcParam font size, which may be a keyword like ``"medium"``."""
    try:
        return float(size)
    except (TypeError, ValueError):
        return float(FontProperties(size=size).get_size_in_points())


@lru_cache(maxsize=4096)
def _text_width_inches(text: str, fontsize_pt: float) -> float:
    """Rendered width of ``text``, measured rather than guessed from length."""
    try:
        from matplotlib.textpath import TextPath

        path = TextPath((0, 0), text or " ", size=fontsize_pt)
        return float(path.get_extents().width) / 72.0
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # noqa: BLE001 - a figure is never worth failing a run over
        logger.debug("%s: falling back after %r", __name__, exc)
        return len(text or " ") * 0.6 * fontsize_pt / 72.0


def tick_rotation(labels: Iterable, *, slot_inches: float) -> float | None:
    """``rotation`` for categorical x tick labels: ``None`` (upright) if they fit
    their ``slot_inches`` slot, else 90 so long names stack instead of colliding.

    The QC violins group by sample, and sample IDs are wide enough that six of them
    printed over each other under the axis.
    """
    fontsize = _resolve_fontsize(rcParams["xtick.labelsize"])
    widest = max((_text_width_inches(str(x), fontsize) for x in labels), default=0.0)
    return None if widest <= slot_inches * 0.95 else 90.0


def widest_label_inches(labels: Iterable, *, fontsize_pt: float | None = None) -> float:
    """Width of the longest of ``labels`` — for sizing hand-built figures."""
    fontsize = (
        _resolve_fontsize(rcParams["font.size"])
        if fontsize_pt is None
        else float(fontsize_pt)
    )
    return max((_text_width_inches(str(x), fontsize) for x in labels), default=0.0)
