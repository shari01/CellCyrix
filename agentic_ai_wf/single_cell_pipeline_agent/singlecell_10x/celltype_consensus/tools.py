"""
tools.py — pure-logic annotation tools for the consensus pipeline.

INVARIANT: this module contains ZERO LLM / OpenRouter / HTTP-to-LLM calls.
Everything here is deterministic logic: marker computation, CellTypist, the
SingleR subprocess bridge, label harmonization, the lineage sanity gate, and
vote counting. (Verify with: `grep -Ei "openrouter|chat/completions|requests\\.post|openai" tools.py` → no hits.)

Disease-agnostic: no function reads a disease name. `tissue` (organ context) is
cell biology and may be used to pick models/references.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # annotation only — anndata is not imported at module load
    from anndata import AnnData

import os
import shutil

import numpy as np
import pandas as pd
import scanpy as sc

from ..atomic_io import atomic_to_csv
from ..exceptions import PipelineComputationError
from .model_integrity import MANIFEST_NAME, ensure_model_file, verify_model_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical pan-lineage markers (coarse, disease-agnostic) for the sanity gate
# ---------------------------------------------------------------------------
# Hand-written fallback panels. These are the historical gate markers and remain
# the floor: :mod:`lineage_panels` always unions them into whatever it derives from
# the reference data, and returns them unchanged if that data is unavailable.
#
# On their own they are too narrow — notably there is no mast-cell and no
# dendritic-cell marker here, which is why such clusters used to score ~0 on every
# panel and get handed to whichever panel was least negative.
BUILTIN_LINEAGE_MARKERS: Dict[str, List[str]] = {
    "Immune": [
        "PTPRC",
        "CD3D",
        "CD3E",
        "CD8A",
        "CD4",
        "MS4A1",
        "CD79A",
        "NKG7",
        "GNLY",
        "LYZ",
        "CD68",
        "CD14",
        "FCGR3A",
        "ITGAM",
    ],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "PDGFRB"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "FLT1"],
    "Mural": ["RGS5", "ACTA2", "MYH11", "TAGLN"],
}


# Lazily-built lineage panels.
#
# WHY LAZY: building these reads a 1.2 MB reference CSV. Doing that at import time
# made `import tools` perform disk IO, and — because the reference root can be given
# as a relative path — made the result depend on the working directory at import
# rather than at call time. Both are now deferred to first use. The public names
# `LINEAGE_MARKERS` and `LINEAGE_MARKERS_PROVENANCE` still resolve exactly as before
# via the module __getattr__ below, so no caller changes.
_LINEAGE_STATE: Dict[str, object] = {}


def _load_lineage_markers() -> Dict[str, List[str]]:
    """Reference-derived panels, falling back to :data:`BUILTIN_LINEAGE_MARKERS`.

    Also records where the panels came from, readable as
    :data:`LINEAGE_MARKERS_PROVENANCE`.
    """
    try:
        from .lineage_panels import build_lineage_markers

        panels, prov = build_lineage_markers(
            tuple((k, tuple(v)) for k, v in BUILTIN_LINEAGE_MARKERS.items())
        )
        _LINEAGE_STATE["provenance"] = prov
        return panels
    except Exception as e:  # noqa: BLE001 - the gate must run even with no reference
        logger.warning(
            "[LINEAGE] panel build unavailable (%s); using built-ins.", e, exc_info=True
        )
        _LINEAGE_STATE["provenance"] = {
            "source": "builtin",
            "reason": f"{type(e).__name__}: {e}",
        }
        return {k: list(v) for k, v in BUILTIN_LINEAGE_MARKERS.items()}


def lineage_markers() -> Dict[str, List[str]]:
    """The active lineage panels, built once on first call."""
    if "markers" not in _LINEAGE_STATE:
        _LINEAGE_STATE["markers"] = _load_lineage_markers()
    return _LINEAGE_STATE["markers"]  # type: ignore[return-value]


def lineage_markers_provenance() -> Dict[str, object]:
    """Where the active panels came from. Builds them if not built yet."""
    lineage_markers()
    return _LINEAGE_STATE.get("provenance", {})  # type: ignore[return-value]


def __getattr__(name: str) -> object:
    """Keep ``tools.LINEAGE_MARKERS`` / ``...PROVENANCE`` working as module globals.

    PEP 562: only consulted when normal attribute lookup fails, so it fires exactly
    once per name and costs nothing afterwards.
    """
    if name == "LINEAGE_MARKERS":
        return lineage_markers()
    if name == "LINEAGE_MARKERS_PROVENANCE":
        return lineage_markers_provenance()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CellConservationError(PipelineComputationError):
    """A step lost, duplicated, or failed to label cells.

    Annotation must be a pure relabelling: every cell in must be a cell out, and
    every cell must carry a consensus label. A violation means a downstream table
    would silently describe a different cell population than the one analysed, so
    it is raised rather than warned about. This was previously an ``assert``, which
    is stripped under ``python -O`` — exactly the deployment where it matters most.
    """


IMMUNE_GATE_GENE = "PTPRC"  # CD45 — must be expressed to keep an Immune gate call

# Minimum cluster-mean ``sc.tl.score_genes`` value for the lineage gate to make a
# call at all. score_genes subtracts a background of similarly-expressed genes, so
# 0.0 means "indistinguishable from random genes" — but sampling noise alone can
# push a panel slightly positive, so a bare >0 test is not enough.
#
# Measured on synthetic controls: clusters with NO matching panel (mast cell,
# dendritic cell, pure background) peak at +0.040, while genuine lineage calls
# start at +0.85 — a >20x gap. 0.1 sits an order of magnitude clear of both sides.
# Applied to the CLUSTER MEAN, not per cell: per-cell scores are noisy, the gate's
# output is a per-cluster call, and the cluster mean is the stable statistic.
MIN_LINEAGE_SCORE: float = 0.1

# ---------------------------------------------------------------------------
# Controlled vocabulary for Stage 6 harmonization.
# Each canonical node has a coarse lineage and word-boundary keyword aliases.
# Order matters only for near-collisions; keywords are matched as whole words.
# ---------------------------------------------------------------------------
CANONICAL_TYPES: List[Tuple[str, str, List[str]]] = [
    (
        "T cell",
        "Immune",
        [
            "t cell",
            "t-cell",
            "t_cell",
            "cd4",
            "cd8",
            "treg",
            "regulatory t",
            "cytotoxic t",
            "helper t",
            "naive t",
            "memory t",
            "tcm",
            "tem",
            "th1",
            "th2",
            "th17",
            "tfh",
            "gamma delta",
            "mait",
        ],
    ),
    ("NK cell", "Immune", ["nk cell", "natural killer", "nk_", "cd56"]),
    (
        "B cell",
        "Immune",
        ["b cell", "b-cell", "b_cell", "plasma cell", "plasmablast", "germinal center"],
    ),
    # Microglia are checked BEFORE macrophage and kept as their own node. They are
    # yolk-sac-derived brain-resident cells, not infiltrating monocyte-derived
    # macrophages, and in a neuro cohort the distinction IS the finding. Measured on
    # GSE157827 cluster 8 (6,785 cells, CSF1R/P2RY12/APBB1IP/TLR2/CD86): CellTypist,
    # the knowledge voter and PubMed all said microglia, yet the cluster shipped as
    # "Macrophage" because every route — this keyword row and the hierarchy's
    # main_cell_type (see _PRESERVE_HIERARCHY_NODES) — collapsed it into macrophage.
    ("Microglia", "Immune", ["microglia", "microglial"]),
    # Monocyte and macrophage are separate nodes so the keyword fallback emits the
    # SAME strings the hierarchy does. They previously shared one
    # "Monocyte/Macrophage" node while the hierarchy returned plain "Monocyte" /
    # "Macrophage", so the two routes produced different labels for one cell type and
    # votes failed to aggregate. Measured on GSE337706 cluster 7: CellTypist
    # "Monocyte CD14+" took the keyword route to "Monocyte/Macrophage" while the
    # knowledge and PubMed voters resolved to "Monocyte" — a real 3-vote agreement
    # counted as 2-vs-1-vs-1, lost its majority, and was adjudicated to "Dendritic
    # cell" on a single dissent. They remain reconcilable at 'Myeloid cell' in the
    # hierarchy (see TestHarmonization.test_monocyte_and_macrophage_share_a_parent).
    ("Monocyte", "Immune", ["monocyte"]),
    ("Macrophage", "Immune", ["macrophage", "kupffer", "histiocyte", "tam"]),
    ("Dendritic cell", "Immune", ["dendritic", "pdc", "cdc", "langerhans"]),
    (
        "Granulocyte",
        "Immune",
        ["neutrophil", "eosinophil", "basophil", "mast cell", "granulocyte"],
    ),
    (
        "Hematopoietic progenitor",
        "Immune",
        ["hsc", "progenitor", "cmp", "gmp", "mpp", "hematopoietic stem"],
    ),
    # "carcinoma"/"adenocarcinoma" ARE epithelial by definition. Generic
    # "malignant"/"tumor cell" are NOT — sarcomas/lymphomas/melanomas are
    # malignant but non-epithelial — so those terms are deliberately NOT here;
    # an unqualified malignant label falls through to "Other" (lineage unknown).
    (
        "Epithelial cell",
        "Epithelial",
        [
            "epithel",
            "keratinocyte",
            "enterocyte",
            "goblet",
            "acinar",
            "ductal",
            "luminal",
            "basal cell",
            "alveolar",
            "hepatocyte",
            "secretory",
            "ciliated",
            "club cell",
            "carcinoma",
            "adenocarcinoma",
        ],
    ),
    (
        "Fibroblast",
        "Fibroblast",
        [
            "fibroblast",
            "stromal",
            "myofibroblast",
            "caf",
            "mesenchym",
            "chondrocyte",
            "osteoblast",
        ],
    ),
    ("Endothelial cell", "Endothelial", ["endothel", "vascular ec", "lymphatic ec"]),
    # "myocyte" (cardiomyocyte / skeletal myocyte) is NOT a mural cell — only
    # pericytes and (vascular) smooth muscle are — so it is deliberately omitted.
    ("Mural cell", "Mural", ["pericyte", "smooth muscle", "mural", "vascular smooth"]),
    ("Erythrocyte", "Other", ["erythrocyte", "erythroid", "red blood"]),
    ("Melanocyte", "Other", ["melanocyte"]),
    # Neural / glial nodes, named EXACTLY as the hierarchy names them so the two
    # resolution routes cannot disagree. These replace a single "Neuron/Glia"
    # catch-all, which lumped neurons, astrocytes, oligodendrocytes and Schwann
    # cells together while the hierarchy returned each separately — the same
    # split-vote failure as the old Monocyte/Macrophage node.
    #
    # "interneuron" needs its own row: \bneuron does NOT match "interneuron" (no
    # word boundary inside the word), so voter labels like "GABAergic interneuron"
    # and "Interneuron" — which the hierarchy also fails to resolve — fell through
    # to "Other: ...". Measured on GSE157827, that left clusters 6/9/11/13 spread
    # across "Other: GABAergic interneuron" and "Other: Interneuron" instead of one
    # inhibitory-neuron population, fragmenting the per-cell-type DE.
    #
    # Specific classes precede the generic "neuron" row, which is the fallback.
    (
        "Excitatory (glutamatergic) neuron",
        "Other",
        ["excitatory neuron", "glutamatergic", "pyramidal"],
    ),
    (
        "Inhibitory (GABAergic) neuron",
        "Other",
        ["interneuron", "gabaergic", "inhibitory neuron"],
    ),
    ("Astrocyte", "Other", ["astrocyte", "astroglia"]),
    (
        "Oligodendrocyte precursor cell",
        "Other",
        ["oligodendrocyte precursor", "oligodendrocyte progenitor"],
    ),
    ("Oligodendrocyte", "Other", ["oligodendrocyte"]),
    ("Schwann cell", "Other", ["schwann"]),
    ("Neuron", "Other", ["neuron", "neuronal"]),
    ("Glial cell", "Other", ["glial", "glia"]),
]

_LABEL_TO_LINEAGE: Dict[str, str] = {c: lin for c, lin, _ in CANONICAL_TYPES}
_LINEAGE_KEYWORDS: Dict[str, List[str]] = {}
for _canon, _lin, _kws in CANONICAL_TYPES:
    _LINEAGE_KEYWORDS.setdefault(_lin, []).extend(_kws)

UNASSIGNED = "Unassigned"

# ---------------------------------------------------------------------------
# Marker ranking (Stage 2) — recorded in provenance so a run's marker evidence
# is attributable to a specific selection rule.
# ---------------------------------------------------------------------------
# Effect size first, Scanpy's test statistic second, adjusted p-value only as a
# tie-breaker. Sorting primarily by pvals_adj (the previous behaviour) collapses
# in any well-powered dataset: thousands of genes underflow to padj == 0.0, so the
# "top" markers handed to the marker-reasoning voters were effectively arbitrary
# among the tied set rather than the strongest.
MARKER_RANKING_METHOD = (
    "positive_logFC + min_detection_fraction, then logfoldchanges desc, "
    "scores desc, pvals_adj asc (stable sort)"
)
MARKER_SORT_COLUMNS = ("logfoldchanges", "scores", "pvals_adj")
MARKER_SORT_ASCENDING = (False, False, True)

# Fraction of cells in a cluster that must express a gene for it to be usable as
# that cluster's marker. Filters "significant but barely detected" genes, which
# read as convincing in a prompt and are not.
DEFAULT_MIN_DETECTION_FRACTION = 0.10


# ===========================================================================
# Log-normalized working matrix (robust to scale-corrupted adata.raw)
# ===========================================================================
def get_lognorm(adata: AnnData) -> tuple["AnnData", str]:
    """Return (adata_ln, source) — a log1p(CP10k) AnnData for annotation.

    WHY: `adata.raw = adata` followed by in-place `sc.pp.scale` corrupts
    `adata.raw` (raw.X and X can share one buffer), and `adata.X` is scaled at
    annotation time. Both are unusable for CellTypist/marker/score logic. The
    `counts` layer is never mutated by scaling, so we re-normalize from it. Falls
    back to raw (if it still looks log-normed) then X, with an explicit warning.
    """
    import anndata as _ad

    if "counts" in adata.layers:
        Xc = adata.layers["counts"]
        src = _ad.AnnData(X=Xc.copy(), obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.normalize_total(src, target_sum=1e4)
        sc.pp.log1p(src)
        return src, "counts-layer"

    if getattr(adata, "raw", None) is not None:
        r = adata.raw.to_adata()
        sample = r.X[: min(50, r.n_obs)]
        sample = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
        looks_lognorm = (
            float(np.nanmax(sample)) < 50.0 and float(np.nanmin(sample)) >= -1e-6
        )
        if looks_lognorm:
            r.obs = adata.obs.copy()
            return r, "raw"
        logger.warning(
            "[LOGNORM] adata.raw is not log-normalized (likely scale-corrupted); "
            "no counts layer either — falling back to adata.X (results may be degraded)."
        )
    else:
        logger.warning("[LOGNORM] no counts layer and no raw — using adata.X as-is.")
    return adata.copy(), "X-fallback"


# ===========================================================================
# Stage 2 — marker computation
# ===========================================================================
def rank_cluster_marker_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 50,
    min_detection_fraction: float = DEFAULT_MIN_DETECTION_FRACTION,
) -> pd.DataFrame:
    """Rank ONE cluster's ``rank_genes_groups_df`` rows by biological effect size.

    Pure DataFrame logic (no AnnData), so it is directly unit-testable. Order of
    operations — see MARKER_RANKING_METHOD:

      1. drop rows with a missing/non-finite gene name, logFC, or score;
      2. keep only positive markers (``logfoldchanges > 0``);
      3. require ``pct_nz_group >= min_detection_fraction`` when Scanpy supplied
         detection fractions (``pts=True``) — relaxed automatically if it would
         empty the cluster, so a sparse cluster still gets markers;
      4. sort logFC desc -> scores desc -> pvals_adj asc, with a STABLE sort so
         the result is reproducible for a fixed input.

    Returns the ranked rows (at most ``top_n``).
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    sub = df.copy()

    # 1. valid rows only — an inf/NaN effect size must never win a ranking.
    sub = sub[sub["names"].notna()]
    sub = sub[sub["names"].astype(str).str.len() > 0]
    sub = sub[~sub["names"].astype(str).str.lower().isin({"nan", "none"})]
    for col in ("logfoldchanges", "scores", "pvals_adj"):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            if col != "pvals_adj":
                sub = sub[np.isfinite(sub[col])]
    if sub.empty:
        return sub

    # 2. positive direction only (identity markers are up-regulated by definition)
    if "logfoldchanges" in sub.columns:
        sub = sub[sub["logfoldchanges"] > 0]
    if sub.empty:
        return sub

    # 3. detection-fraction floor, relaxed rather than allowed to empty the cluster
    if "pct_nz_group" in sub.columns and min_detection_fraction > 0:
        detected = sub[
            pd.to_numeric(sub["pct_nz_group"], errors="coerce")
            >= float(min_detection_fraction)
        ]
        if not detected.empty:
            sub = detected
        else:
            logger.info(
                "[MARKERS] detection floor %.2f would empty this cluster; relaxed.",
                min_detection_fraction,
            )

    # 4. effect size first; stable so ties keep Scanpy's own ordering
    sort_cols = [c for c in MARKER_SORT_COLUMNS if c in sub.columns]
    if sort_cols:
        asc = [MARKER_SORT_ASCENDING[MARKER_SORT_COLUMNS.index(c)] for c in sort_cols]
        sub = sub.sort_values(sort_cols, ascending=asc, kind="mergesort")
    return sub.head(top_n)


def compute_cluster_markers(
    adata: AnnData,
    cluster_col: str = "leiden",
    top_n: int = 50,
    *,
    min_detection_fraction: float = DEFAULT_MIN_DETECTION_FRACTION,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Top-N up-regulated markers per cluster (Wilcoxon), ranked by effect size.

    Returns (markers_by_cluster, empty_clusters). Empty clusters are reported
    explicitly (not skipped). Runs on the log-normalized working copy built by
    ``get_lognorm`` (adata.X may be scaled at annotation time), hence use_raw=False.

    Ranking is delegated to ``rank_cluster_marker_frame`` — effect size first, not
    adjusted p-value. ``pts=True`` is requested so a detection-fraction floor can
    be applied; if the installed Scanpy does not support it the floor is skipped
    and everything else is unchanged.
    """
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"cluster_col '{cluster_col}' not in adata.obs")

    # Rank a deeper pool than we keep: the positive-direction and detection filters
    # each remove candidates, and the effect-size sort needs material to choose from.
    n_candidates = int(min(adata.n_vars, max(top_n * 4, 200)))
    _kw = dict(
        groupby=cluster_col, method="wilcoxon", n_genes=n_candidates, use_raw=False
    )
    try:
        sc.tl.rank_genes_groups(adata, pts=True, **_kw)
    except TypeError as e:  # older scanpy without `pts` — no detection fractions
        logger.info(
            "[MARKERS] scanpy rank_genes_groups has no `pts` (%s); detection-fraction floor unavailable for this run.",
            e,
        )
        sc.tl.rank_genes_groups(adata, **_kw)

    df = sc.get.rank_genes_groups_df(adata, None)

    markers: Dict[str, List[str]] = {}
    empty: List[str] = []
    for cl in sorted(df["group"].astype(str).unique(), key=_natural_key):
        sub = rank_cluster_marker_frame(
            df[df["group"].astype(str) == cl],
            top_n=top_n,
            min_detection_fraction=min_detection_fraction,
        )
        genes = [] if sub.empty else sub["names"].astype(str).tolist()
        markers[cl] = genes
        if not genes:
            empty.append(cl)
            logger.warning(
                "[MARKERS] cluster %s: no up-regulated markers (markers_empty=True).",
                cl,
            )
    logger.info(
        "[MARKERS] %s clusters, %s with empty marker sets (ranking: %s).",
        len(markers),
        len(empty),
        MARKER_RANKING_METHOD,
    )
    return markers, empty


# ===========================================================================
# Stage 3 — Annotator A: CellTypist (per cluster)
# ===========================================================================
# Defaults for the mixed-cluster heuristic (overridable from config).
DEFAULT_MIXED_MIN_DOMINANT_FRACTION = 0.70
DEFAULT_MIXED_SECOND_LABEL_FRACTION = 0.20

# --- bundled model set (offline / air-gapped runs) -------------------------
# CellTypist keeps its .pkl models in ~/.celltypist and fetches anything missing
# over HTTP. On a fresh machine with no network that leaves this voter abstaining
# on every cluster, quietly costing the consensus one of its four opinions. A
# release package may therefore ship the model set beside the reference tables:
#
#     <shared_reference_root>/celltypist_models/data/models/<Model_Name>.pkl
#
# The `data/models/` nesting is CellTypist's own cache layout, so that folder can
# equally be handed to the tool directly via `CELLTYPIST_FOLDER`. Nothing here is
# required: when the bundle — or the requested model within it — is absent, the
# bare model name is passed through and CellTypist behaves exactly as before,
# which is the case in a source checkout that carries no bundle.
_BUNDLED_CELLTYPIST_DIRNAME = "celltypist_models"


@lru_cache(maxsize=None)
def _bundled_celltypist_model(model_name: str) -> Optional[str]:
    """Absolute POSIX path to a bundled copy of ``model_name``, else None.

    THE FORWARD SLASHES ARE LOAD-BEARING. ``celltypist.models.Model.load`` treats
    its argument as a bare model name unless it contains ``'/'``, and that
    bare-name branch calls ``get_all_models()`` -> ``download_if_required()``,
    which pulls the ENTIRE upstream repertoire when the home cache is empty —
    precisely the network round-trip a bundled model exists to avoid. A path
    containing '/' short-circuits the test and is opened directly, so build it
    with ``as_posix()`` and not with the native Windows separator.
    """
    try:
        from .lineage_panels import shared_reference_root

        base = shared_reference_root() / _BUNDLED_CELLTYPIST_DIRNAME
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # pragma: no cover
        logger.debug("%s: falling back after %r", __name__, exc)
        return None
    for cand in (base / "data" / "models" / model_name, base / model_name):
        if cand.is_file():
            return cand.as_posix()

    # Not on disk. The models are referenced externally rather than committed, so a
    # fresh clone reaches here on its first run: fetch just this model, verify it
    # against the bundle's SHA256SUMS.txt, and cache it in place. Subsequent runs take
    # the is_file() path above and need no network. `ensure_model_file` returns None if
    # it cannot be fetched, which leaves the caller's existing fallback intact.
    for candidate_dir in (base / "data" / "models", base):
        if not (candidate_dir / MANIFEST_NAME).is_file():
            continue
        fetched = ensure_model_file(candidate_dir, model_name)
        if fetched is not None:
            return fetched.as_posix()
    return None


# Keys of the per-cluster CellTypist heterogeneity metrics, in table order.
CELLTYPIST_METRIC_KEYS = (
    "celltypist_dominant_label",
    "celltypist_dominant_fraction",
    "celltypist_second_label",
    "celltypist_second_fraction",
    "celltypist_label_entropy",
    "celltypist_unique_label_count",
    "mixed_cluster_flag",
)


def label_entropy(counts: pd.Series) -> float:
    """Shannon entropy (bits) of a label-count distribution. 0.0 for one label."""
    total = float(counts.sum())
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts.to_numpy(dtype=float) / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def summarize_celltypist_by_cluster(
    per_cell_labels: pd.Series,
    cluster_labels: pd.Series,
    *,
    min_dominant_fraction: float = DEFAULT_MIXED_MIN_DOMINANT_FRACTION,
    second_label_fraction: float = DEFAULT_MIXED_SECOND_LABEL_FRACTION,
) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, dict]]:
    """Reduce per-CELL CellTypist labels to per-CLUSTER call + heterogeneity metrics.

    The dominant label is the mode, exactly as before — the consensus vote is
    unchanged. What is new is that the discarded information is now quantified: a
    cluster where CellTypist is split 55/45 between two types is a very different
    object from one where it is unanimous, and both used to collapse to the same
    single label with only the winning fraction surviving.

    A cluster is flagged mixed when::

        dominant_fraction < min_dominant_fraction
        or second_label_fraction >= second_label_fraction

    Returns ``(votes, metrics)`` where ``votes`` is the historical
    ``{cluster: (label, fraction)}`` and ``metrics`` is
    ``{cluster: {**CELLTYPIST_METRIC_KEYS, celltypist_n_cells}}``.
    """
    votes: Dict[str, Tuple[str, float]] = {}
    metrics: Dict[str, dict] = {}

    per_cell = per_cell_labels.astype(str)
    for cl, s in per_cell.groupby(cluster_labels.astype(str)):
        counts = s.value_counts()
        if counts.empty:
            continue
        total = float(counts.sum())
        dom_label = str(counts.index[0])
        dom_frac = float(counts.iloc[0] / total)
        if counts.size > 1:
            second_label = str(counts.index[1])
            second_frac = float(counts.iloc[1] / total)
        else:
            second_label, second_frac = "", 0.0

        mixed = bool(
            dom_frac < float(min_dominant_fraction)
            or second_frac >= float(second_label_fraction)
        )

        votes[str(cl)] = (dom_label, dom_frac)
        metrics[str(cl)] = {
            "celltypist_dominant_label": dom_label,
            "celltypist_dominant_fraction": round(dom_frac, 4),
            "celltypist_second_label": second_label,
            "celltypist_second_fraction": round(second_frac, 4),
            "celltypist_label_entropy": round(label_entropy(counts), 4),
            "celltypist_unique_label_count": int(counts.size),
            "mixed_cluster_flag": mixed,
            "celltypist_n_cells": int(total),
        }
        if mixed:
            logger.info(
                "[CELLTYPIST] cluster %s MIXED: %s=%s, %s=%s, %s distinct labels, entropy=%s bits.",
                cl,
                dom_label,
                format(dom_frac, ".2f"),
                second_label or "n/a",
                format(second_frac, ".2f"),
                counts.size,
                format(metrics[str(cl)]["celltypist_label_entropy"], ".2f"),
            )
    return votes, metrics


def _celltypist_per_cell_labels(
    adata,
    cluster_col: str,
    model_name: str,
) -> Optional[pd.Series]:
    """Per-cell CellTypist labels aligned to ``adata.obs_names``, or None on failure.

    NOTE on clustering: ``majority_voting=True`` is passed WITHOUT
    ``over_clustering``, so CellTypist smooths its predictions over its OWN internal
    Leiden partition. That partition is internal to CellTypist and is never read
    back — the pipeline's structural clustering remains ``adata.obs[cluster_col]``,
    which is what the returned labels are aggregated over by the caller.
    """
    try:
        import celltypist
        from celltypist import models as ct_models
    except ImportError as e:
        logger.warning(
            "[CELLTYPIST] not installed (%s); voter abstains.", e, exc_info=True
        )
        return None

    bundled = _bundled_celltypist_model(model_name)
    if bundled:
        # A CellTypist model is a pickle, so loading it executes whatever it contains.
        # Verify it against the bundle's SHA256SUMS.txt BEFORE handing it over.
        # Deliberately outside the try/except below: a checksum mismatch must abort
        # the run, not be downgraded to "voter abstains" like a missing dependency.
        verify_model_file(bundled, required=True)
    try:
        if bundled:
            logger.info("[CELLTYPIST] using bundled model, no download: %s", bundled)
        else:
            try:
                ct_models.download_models(model=[model_name])
            except ValueError as e:
                # The locally cached models.json predates this model, so it is not
                # in "the celltypist model repertoire" as far as the stale index is
                # concerned and no amount of retrying the same call will fetch it.
                # Refresh the index once. Skipped when the file is already on disk,
                # because force_update=True re-downloads existing models too.
                if (Path(ct_models.models_path) / model_name).is_file():
                    logger.info(
                        "[CELLTYPIST] %s absent from the cached index but present on disk.",
                        model_name,
                    )
                else:
                    logger.info(
                        "[CELLTYPIST] %s absent from the cached index (%s); refreshing it.",
                        model_name,
                        e,
                    )
                    try:
                        ct_models.download_models(model=[model_name], force_update=True)
                    except Exception as e2:
                        logger.warning(
                            "[CELLTYPIST] index refresh failed (%s); assuming cached.",
                            e2,
                            exc_info=True,
                        )
            except (
                Exception
            ) as e:  # network / cache issue is non-fatal for a cached model
                logger.warning(
                    "[CELLTYPIST] model download check failed (%s); assuming cached.",
                    e,
                    exc_info=True,
                )

        # `adata` is already a log1p(CP10k) working copy (see get_lognorm), which
        # is exactly what CellTypist expects in .X.
        src = adata.copy()
        src.obs[cluster_col] = adata.obs[cluster_col].values
        preds = celltypist.annotate(
            src, model=(bundled or model_name), majority_voting=True
        )
        pl = preds.predicted_labels
        col = (
            "majority_voting" if "majority_voting" in pl.columns else "predicted_labels"
        )
        return pl[col].reindex(adata.obs_names).astype(str)
    except Exception as e:
        logger.warning(
            "[CELLTYPIST] annotation failed (%s); voter abstains.", e, exc_info=True
        )
        return None


def annotate_celltypist_detailed(
    adata: AnnData,
    cluster_col: str = "leiden",
    model_name: str = "Immune_All_Low.pkl",
    *,
    min_dominant_fraction: float = DEFAULT_MIXED_MIN_DOMINANT_FRACTION,
    second_label_fraction: float = DEFAULT_MIXED_SECOND_LABEL_FRACTION,
) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, dict]]:
    """CellTypist voter + per-cluster heterogeneity metrics.

    Returns ``({cluster: (label, fraction)}, {cluster: metrics})``. On failure both
    are empty (documented fallback: the voter abstains, it does not crash the
    consensus) and the abstention is surfaced in provenance.
    """
    per_cell = _celltypist_per_cell_labels(adata, cluster_col, model_name)
    if per_cell is None:
        return {}, {}
    votes, metrics = summarize_celltypist_by_cluster(
        per_cell,
        adata.obs[cluster_col],
        min_dominant_fraction=min_dominant_fraction,
        second_label_fraction=second_label_fraction,
    )
    n_mixed = sum(1 for m in metrics.values() if m["mixed_cluster_flag"])
    logger.info(
        "[CELLTYPIST] labeled %s clusters with '%s' (%s flagged mixed).",
        len(votes),
        model_name,
        n_mixed,
    )
    return votes, metrics


def annotate_celltypist(
    adata: AnnData,
    cluster_col: str = "leiden",
    model_name: str = "Immune_All_Low.pkl",
) -> Dict[str, Tuple[str, float]]:
    """Per-cluster CellTypist label = mode of per-cell majority-voting labels.

    Backward-compatible wrapper around ``annotate_celltypist_detailed``: returns
    only ``{cluster: (label, fraction)}``. Kept so external callers and older tests
    keep working; the consensus orchestrator uses the detailed variant.
    """
    votes, _ = annotate_celltypist_detailed(adata, cluster_col, model_name)
    return votes


# ===========================================================================
# Stage 4 — Annotator B: SingleR (R subprocess bridge)
# ===========================================================================
# R side of the SingleR call. Runs in-process via rpy2 on a genes x n_clusters
# pseudobulk matrix pushed in as `test_mat` (+ `ref_name`). Uses message() so the
# progress is streamed to the terminal by the console callback set below.
_SINGLER_R = r"""
suppressWarnings(suppressMessages({
  library(SingleR); library(celldex); library(SummarizedExperiment)
}))
ref <- switch(ref_name,
  "BlueprintEncodeData"              = celldex::BlueprintEncodeData(),
  "HumanPrimaryCellAtlasData"        = celldex::HumanPrimaryCellAtlasData(),
  "MonacoImmuneData"                 = celldex::MonacoImmuneData(),
  "DatabaseImmuneCellExpressionData" = celldex::DatabaseImmuneCellExpressionData(),
  "NovershternHematopoieticData"     = celldex::NovershternHematopoieticData(),
  "MouseRNAseqData"                  = celldex::MouseRNAseqData(),
  stop(paste("Unknown reference", ref_name)))
message(sprintf("[R] reference '%s' loaded (%d reference profiles)", ref_name, ncol(ref)))
common <- intersect(rownames(test_mat), rownames(ref))
message(sprintf("[R] %d genes overlap the reference", length(common)))
if (length(common) < 10L) stop("Too few overlapping genes between data and reference.")
message(sprintf("[R] classifying %d cluster profiles with SingleR ...", ncol(test_mat)))
pred <- SingleR::SingleR(test = test_mat, ref = ref, labels = ref$label.main)
score_vec <- tryCatch(apply(pred$scores, 1, function(r) max(r, na.rm = TRUE)),
                      error = function(e) rep(NA_real_, nrow(pred)))
final_labels <- pred$labels
if (!is.null(pred$pruned.labels)) {
  pl <- pred$pruned.labels
  final_labels <- ifelse(is.na(pl), pred$labels, pl)
}
message("[R] SingleR classification complete.")
data.frame(cluster = rownames(pred), label = final_labels,
           score = as.numeric(score_vec), stringsAsFactors = FALSE)
"""


def _ensure_r_home(rscript_exe: str = "Rscript") -> str:
    """Locate the R install and make its DLLs importable by rpy2.

    Sets R_HOME (if unset) and, on Windows, adds R's bin dir to the DLL search
    path so ``import rpy2.robjects`` can find R.dll. Must be called BEFORE rpy2
    is imported. Returns the resolved R_HOME.
    """
    r_home = os.environ.get("R_HOME", "").strip()
    if not (r_home and Path(r_home).exists()):
        # Rscript lives in <R_HOME>/bin/Rscript(.exe) -> go two levels up.
        rp = shutil.which(rscript_exe) or shutil.which("Rscript")
        if rp:
            r_home = str(Path(rp).resolve().parent.parent)
        if not (r_home and Path(r_home).exists()):
            r_exe = shutil.which("R")
            if r_exe:
                try:
                    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
                        [r_exe, "RHOME"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,  # probing for R; a failure is handled below
                    )
                    cand = (out.stdout or "").strip().splitlines()
                    if cand and Path(cand[0].strip()).exists():
                        r_home = cand[0].strip()
                except (OSError, subprocess.SubprocessError) as exc:
                    logger.debug("could not probe R for RHOME: %s", exc)
        if not (r_home and Path(r_home).exists()):
            raise RuntimeError(
                "Could not locate R. Set the R_HOME environment variable to your R "
                "install (e.g. 'C:\\Program Files\\R\\R-4.5.1') and retry."
            )
        os.environ["R_HOME"] = r_home

    if os.name == "nt":
        for sub in ("bin\\x64", "bin"):
            p = Path(r_home) / sub
            if p.exists():
                try:
                    os.add_dll_directory(str(p))
                except (AttributeError, OSError):
                    pass
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    return r_home


def run_singler(
    adata: AnnData,
    cluster_col: str,
    reference: str,
    bridge_script: Path | None = None,  # kept for back-compat; unused (rpy2 path)
    rscript_exe: str = "Rscript",
    timeout_s: int = 3600,  # kept for back-compat; unused (in-process)
) -> Dict[str, Tuple[str, float]]:
    """Per-cluster SingleR labels via IN-PROCESS R (rpy2) — no giant temp files.

    SingleR is run on per-cluster PSEUDOBULK profiles (mean log-norm expression).
    That is exactly what SingleR's own ``clusters=`` mode computes internally, but
    we aggregate in Python so embedded R only ever receives a genes x n_clusters
    matrix (kilobytes) instead of the full genes x n_cells matrix (which, dumped
    to a MatrixMarket text file, was multi-GB and made R hang on readMM).

    R console output is streamed live to this module's logger, so progress is
    visible on the terminal. NO SILENT FALLBACK: any R error propagates as
    RuntimeError; callers decide whether SingleR is required.

    Returns {cluster: (label, score)}.
    """
    from scipy import sparse as sp

    # ---- 1. per-cluster pseudobulk (genes x clusters), computed in Python ----
    # `adata` is a log-normalized working copy (see get_lognorm).
    X = adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    genes = adata.var_names.astype(str).to_numpy()
    clusters_series = adata.obs[cluster_col].astype(str)

    def _clkey(c):  # numeric-aware ordering so '2' sorts before '10'
        return (0, int(c)) if str(c).isdigit() else (1, str(c))

    cluster_ids = sorted(clusters_series.unique(), key=_clkey)
    codes = clusters_series.map({c: i for i, c in enumerate(cluster_ids)}).to_numpy()

    logger.info(
        "[SINGLER] aggregating %s cells -> %s cluster pseudobulk profiles (%s genes); running SingleR in-process (rpy2).",
        X.shape[0],
        len(cluster_ids),
        X.shape[1],
    )
    agg = np.zeros((X.shape[1], len(cluster_ids)), dtype=np.float64)  # genes x clusters
    for i in range(len(cluster_ids)):
        mask = codes == i
        if mask.any():
            agg[:, i] = np.asarray(X[mask].mean(axis=0)).ravel()

    # ---- 2. bring up embedded R and stream its console to our logger ---------
    _ensure_r_home(rscript_exe)
    try:
        import rpy2.rinterface_lib.callbacks as rcb
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as e:  # pragma: no cover - env/setup issue
        raise RuntimeError(
            f"Could not import rpy2 ({e}). Install it in this environment "
            f"(`pip install rpy2`) or disable SingleR (enable_singler: false)."
        ) from e

    _buf = {"s": ""}  # buffer partial R console writes into whole lines

    def _r_console(s):
        _buf["s"] += s
        *lines, _buf["s"] = _buf["s"].split("\n")
        for ln in lines:
            if ln.strip():
                logger.info("[SINGLER][R] %s", ln.rstrip())

    rcb.consolewrite_print = _r_console
    rcb.consolewrite_warnerror = _r_console

    # rpy2 3.6 removed activate(); scope numpy<->R conversion to the matrix push
    # only. Running SingleR OUTSIDE the numpy converter keeps its return value an
    # R data.frame (so .rx2 works) instead of a numpy recarray.
    try:
        with localconverter(default_converter + numpy2ri.converter):
            r_mat = ro.conversion.get_conversion().py2rpy(agg)  # -> R matrix
        ro.globalenv["test_mat"] = r_mat
        ro.globalenv["gene_names"] = ro.StrVector([str(g) for g in genes])
        ro.globalenv["cl_names"] = ro.StrVector([str(c) for c in cluster_ids])
        ro.globalenv["ref_name"] = reference
        ro.r("rownames(test_mat) <- gene_names; colnames(test_mat) <- cl_names")

        res = ro.r(_SINGLER_R)  # data.frame(cluster, label, score)

        clusters_out = [str(x) for x in res.rx2("cluster")]
        labels_out = [str(x) for x in res.rx2("label")]
        scores_out = [float(x) for x in res.rx2("score")]
    except Exception as e:
        raise RuntimeError(f"SingleR (rpy2) failed: {e}") from e
    finally:
        if _buf["s"].strip():  # flush trailing partial line
            logger.info("[SINGLER][R] %s", _buf["s"].rstrip())

    def _score(x):
        try:
            return float(x)
        except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
            logger.debug("%s: falling back after %r", __name__, exc)
            return float("nan")

    result: Dict[str, Tuple[str, float]] = {}
    for cl, lab, sco in zip(clusters_out, labels_out, scores_out, strict=True):
        result[str(cl)] = (str(lab), _score(sco))
    logger.info("[SINGLER] labeled %s clusters against '%s'.", len(result), reference)
    return result


# ===========================================================================
# Stage 6 — label harmonization (controlled vocabulary)
# ===========================================================================
# --------------------------------------------------------------------------- #
#  Hierarchy-backed label resolution (subtype-ref)
#
#  The keyword table below (CANONICAL_TYPES) compares label STRINGS, so three
#  voters saying "Mast_cell", "Mast cell" and "mast cell" read as three different
#  answers and a unanimous call was scored as a disagreement. The cell_hierarchy
#  resolver maps each label onto a node in a 404-node tree with a 965-entry
#  cross-vocabulary crosswalk, so granularity and spelling stop masquerading as
#  scientific disagreement.
#
#  `main_cell_type` is used, not the leaf label: it is the level at which voters of
#  different granularity ("CD16+ NK cells" vs "NK cells") should be counted as
#  agreeing. Anything the resolver cannot place at high confidence falls through to
#  the original keyword matching, which is never removed.
# --------------------------------------------------------------------------- #
MIN_RESOLVE_CONFIDENCE: float = 0.95

_HIERARCHY_TO_GATE_LINEAGE: Dict[str, str] = {
    "Haematopoietic cell": "Immune",
    "Epithelial cell": "Epithelial",
    "Endothelial cell": "Endothelial",
    "Stromal / mesenchymal cell": "Fibroblast",
}
_MURAL_MAIN_TYPES = {"Pericyte", "Vascular smooth muscle cell", "Mural cell"}


@lru_cache(maxsize=1)
def _hierarchy():
    """The shared CellHierarchy, or None if subtype-ref is unavailable."""
    try:
        from .cell_hierarchy import CellHierarchy

        h = CellHierarchy.from_spec()
        logger.info("[HARMONIZE] cell_hierarchy resolver active (subtype-ref).")
        return h
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "[HARMONIZE] cell_hierarchy unavailable (%s); falling back to keyword matching only.",
            e,
            exc_info=True,
        )
        return None


@lru_cache(maxsize=4096)
def _resolve_cached(raw: str):
    """Resolution for `raw`, or None. Cached: the same labels recur per cluster."""
    h = _hierarchy()
    if h is None:
        return None
    try:
        r = h.resolve(raw)
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below  # pragma: no cover - a bad label must never break voting
        logger.debug("%s: falling back after %r", __name__, exc)
        return None
    if not getattr(r, "node_id", None):
        return None
    if float(getattr(r, "confidence", 0.0) or 0.0) < MIN_RESOLVE_CONFIDENCE:
        return None
    return r


# Hierarchy nodes whose own name must survive instead of being collapsed into
# ``main_cell_type``. The tree knows microglia exactly (node_id 'microglial_cell',
# canonical 'Microglial cell') but files them under main_cell_type 'Macrophage', so
# taking main_cell_type throws the answer away. Deliberately tiny: a distinction
# earns a row here only when collapsing it changes the biological claim.
_PRESERVE_HIERARCHY_NODES: Dict[str, str] = {
    "microglial_cell": "Microglia",
}

# Leading-token class codes used by the brain CellTypist models, mapped to the
# EXACT canonical node names above. Longest prefix first so "L5-6"/"L2-3" are
# tested before "L5"/"L2".
#
# Scope is deliberately limited to NEURONS AND GLIA — the populations no
# vocabulary in this system otherwise covers. The vascular codes ("Endo ...",
# "PC ...", "SMC ...", "VLMC ...") are excluded on purpose: Endothelial cell,
# Mural cell and Fibroblast are already first-class nodes WITH lineage-gate
# panels, so those clusters are decided correctly by the other voters, and
# admitting CellTypist's vascular codes here only amplifies its mistakes.
# Measured on GSE157827: CellTypist called cluster 14 "Endo CLDN5 SLC7A5" at
# 0.977 and cluster 10 at 0.283, both on unmistakable neurons
# (NEFL/NEFM/SNAP25/NRGN/RBFOX1, no CLDN5/PECAM1/FLT1). Mapping those to a clean
# "Endothelial cell" turned two correct calls into wrong ones in replay; leaving
# them unresolved keeps the marker-driven voters in charge.
_CLASSIFIER_PREFIXES: List[Tuple[str, str]] = [
    ("astro", "Astrocyte"),
    ("opc", "Oligodendrocyte precursor cell"),
    ("cop", "Oligodendrocyte precursor cell"),  # committed oligo precursor
    ("oligo", "Oligodendrocyte"),
    ("micro", "Microglia"),
    ("macro", "Macrophage"),
    ("inn", "Inhibitory (GABAergic) neuron"),
    ("exn", "Excitatory (glutamatergic) neuron"),
    # Cortical layer prefixes are excitatory projection neurons by convention.
    ("l2-3", "Excitatory (glutamatergic) neuron"),
    ("l3-5", "Excitatory (glutamatergic) neuron"),
    ("l4-5", "Excitatory (glutamatergic) neuron"),
    ("l5-6", "Excitatory (glutamatergic) neuron"),
    ("l2", "Excitatory (glutamatergic) neuron"),
    ("l3", "Excitatory (glutamatergic) neuron"),
    ("l4", "Excitatory (glutamatergic) neuron"),
    ("l5", "Excitatory (glutamatergic) neuron"),
    ("l6", "Excitatory (glutamatergic) neuron"),
]


def harmonize_label(raw_label: Optional[str]) -> str:
    """Map a raw method label onto a canonical controlled-vocabulary node.

    Resolution order: the cell_hierarchy tree first (spelling- and
    granularity-insensitive), then the CANONICAL_TYPES keyword table. Unmatched
    labels are returned prefixed with "Other: " so voting never silently equates an
    unknown string with a canonical node.
    """
    if raw_label is None:
        return UNASSIGNED
    raw = str(raw_label).strip()
    # Idempotent: an already-"Other: ..." label must not be re-prefixed to
    # "Other: Other: ..." when harmonize runs a second time (e.g. on adjudicator
    # output that fell through to an unmatched label).
    if raw.startswith("Other:"):
        return raw
    # Normalize separators so classifier tokens like "Mast_cell" / "NK_cell" match
    # the space-form keywords ("mast cell" / "nk cell"); both forms live in the
    # keyword lists, so this only widens matching, never breaks it.
    s = raw.lower().replace("_", " ")
    if not s or s in {"nan", "none", "unassigned", "unknown", "unclear"}:
        return UNASSIGNED

    # 1. Hierarchy first — handles separators, plurals, granularity and the
    #    cross-vocabulary aliases (CellTypist/SingleR/Azimuth/PanglaoDB/ScType).
    r = _resolve_cached(raw)
    if r is not None:
        node_id = str(getattr(r, "node_id", "") or "").strip()
        if node_id in _PRESERVE_HIERARCHY_NODES:
            # main_cell_type would erase a distinction that matters (see the map).
            return _PRESERVE_HIERARCHY_NODES[node_id]
        node = str(
            getattr(r, "main_cell_type", "") or getattr(r, "canonical_label", "") or ""
        ).strip()
        if node:
            return node

    # 2. Compact classifier codes. The brain CellTypist models
    #    (Adult_Human_PrefrontalCortex, Adult_Human_MTG, Human_*_Hippocampus) name
    #    their classes "<class> <marker> <marker>" — "Micro P2RY12 APBB1IP",
    #    "Astro AQP4 SLC1A2", "L2-3 CUX2 ACVR1C THSD7A". The marker suffix makes
    #    every label unique, so no alias table can enumerate them and the hierarchy
    #    resolves none of them; on GSE157827 that left 9 clusters / 29,624 cells
    #    labelled "Other: <raw code>", which also fragmented the neurons across 9
    #    names and put a colon in every derived filename. Only the LEADING token is
    #    the class, so it is matched anchored — a bare-substring rule would map
    #    "microvascular" to microglia and "PC" to pericyte almost everywhere.
    for prefix, canon in _CLASSIFIER_PREFIXES:
        if re.match(r"^" + re.escape(prefix) + r"(?:[\s\-_]|$)", s):
            return canon

    # 3. Keyword table (unchanged) for anything the tree cannot place.
    for canon, _lin, kws in CANONICAL_TYPES:
        for kw in kws:
            # Leading boundary (matches plurals/suffixes: "t cell" -> "t cells")
            # plus a "no trailing digit" guard so numeric marker tokens stay
            # distinct: "cd4" must NOT match "cd40"/"cd45", "cd8" not "cd80", etc.
            if re.search(r"\b" + re.escape(kw) + r"(?!\d)", s):
                return canon
    return f"Other: {raw}"


def coarse_lineage_of(label: Optional[str]) -> str:
    """Coarse lineage for a (harmonized or raw) label. 'Other' if unknown.

    Returns one of the gate's coarse buckets — Immune / Epithelial / Fibroblast /
    Endothelial / Mural / Other — resolving through the hierarchy first and falling
    back to the keyword table.
    """
    if label is None:
        return "Other"
    if label in _LABEL_TO_LINEAGE:
        return _LABEL_TO_LINEAGE[label]

    r = _resolve_cached(str(label).strip())
    if r is not None:
        if str(getattr(r, "main_cell_type", "") or "") in _MURAL_MAIN_TYPES:
            return "Mural"
        gate = _HIERARCHY_TO_GATE_LINEAGE.get(str(getattr(r, "lineage", "") or ""))
        if gate:
            return gate
    s = str(label).strip().lower()
    for lin, kws in _LINEAGE_KEYWORDS.items():
        for kw in kws:
            # See harmonize_label: leading boundary + "no trailing digit" guard.
            if re.search(r"\b" + re.escape(kw) + r"(?!\d)", s):
                return lin
    return "Other"


# ===========================================================================
# Stage 7 — lineage sanity gate (per cluster, disease-agnostic)
# ===========================================================================
def lineage_gate_per_cluster(
    adata: AnnData, cluster_col: str = "leiden"
) -> Dict[str, str]:
    """Coarse lineage per cluster from canonical pan-lineage markers.

    Scores each lineage's marker set (log-norm), takes the per-cell argmax, gates
    Immune calls on PTPRC>0, then assigns each cluster its majority lineage.

    The gate ABSTAINS ("Other") for a CLUSTER whose winning lineage does not score
    above :data:`MIN_LINEAGE_SCORE` on average. ``sc.tl.score_genes`` subtracts a
    background of similarly-expressed genes, so a near-zero score means "no better
    than random genes" — there is no lineage evidence to act on. A bare ``idxmax``
    still crowns a winner, which is how cell types absent from
    :data:`LINEAGE_MARKERS` (mast cells, dendritic cells) were handed whichever
    panel happened to be least negative. "Other" is already understood downstream
    as "the gate has no opinion", so the voters decide instead.
    """
    # `adata` is a log-normalized working copy (see get_lognorm); score on .X.
    var_ns = adata.var_names

    score_cols = {}
    for lin, genes in lineage_markers().items():
        present = [g for g in genes if g in var_ns]
        if not present:
            logger.warning("[LINEAGE] no markers present for %s; skipping.", lin)
            continue
        key = f"_lin_score_{lin}"
        sc.tl.score_genes(adata, present, score_name=key, use_raw=False)
        score_cols[lin] = key

    if not score_cols:
        logger.warning(
            "[LINEAGE] no lineage markers found at all; gate returns 'Other'."
        )
        return {str(cl): "Other" for cl in adata.obs[cluster_col].astype(str).unique()}

    scores = adata.obs[list(score_cols.values())].copy()
    scores.columns = list(score_cols.keys())
    call = scores.idxmax(axis=1)
    noimm = scores.drop(columns=["Immune"], errors="ignore")

    if IMMUNE_GATE_GENE in var_ns:
        cd45 = sc.get.obs_df(adata, keys=[IMMUNE_GATE_GENE], use_raw=False)[
            IMMUNE_GATE_GENE
        ]
        false_imm = (call == "Immune") & (cd45 <= 0)
        if noimm.shape[1] > 0:
            # PTPRC is dropout-prone: it reads zero in roughly half of real
            # monocytes/macrophages and most mast cells. Ejecting on that alone
            # pushed genuine immune cells onto whichever non-immune panel was
            # least negative. Only eject when an alternative lineage has POSITIVE
            # support — never into noise.
            false_imm &= noimm.max(axis=1) > MIN_LINEAGE_SCORE
        if false_imm.any() and noimm.shape[1] > 0:
            call.loc[false_imm] = noimm.loc[false_imm].idxmax(axis=1)
            logger.info(
                "[LINEAGE] PTPRC-gated %s cells out of Immune.", int(false_imm.sum())
            )

    clusters = adata.obs[cluster_col].astype(str)
    maj = call.groupby(clusters).agg(lambda s: s.value_counts().idxmax())

    # Abstain per cluster when the winning panel is not distinguishable from random
    # genes across the cluster as a whole (see the docstring and MIN_LINEAGE_SCORE).
    mean_scores = scores.groupby(clusters).mean()
    out: Dict[str, str] = {}
    for cl, lin in maj.items():
        cl, lin = str(cl), str(lin)
        cl_score = float(mean_scores.loc[cl, lin])
        if cl_score <= MIN_LINEAGE_SCORE:
            logger.info(
                "[LINEAGE] cluster %s: best lineage %r scores only %s (<= %s); no panel fired — gate abstains ('Other') and defers to the voters.",
                cl,
                lin,
                format(cl_score, "+.4f"),
                MIN_LINEAGE_SCORE,
            )
            out[cl] = "Other"
        else:
            out[cl] = lin
    return out


# ===========================================================================
# Stage 7b — technical / activation STATE programmes (per cluster)
# ===========================================================================
# A cluster can be defined by a STATE rather than by an identity: dissociation
# heat shock, cell cycle, an interferon response, ambient haemoglobin. Such a
# cluster still gets marker genes, and every voter still names a cell type from
# them, so it ships with the same confidence as a cluster carrying real lineage
# evidence. Identity and state are different axes (see cell_hierarchy/spec/states.py);
# a cycling T cell is a T cell, and "cycling" is not an identity.
#
# Measured on the psoriasis run (11 clusters, 97,108 cells):
#   * cluster 1, 16,933 cells — top markers HSPA1B/HSPA1A/HSPA6/DNAJB1/HSPH1/
#     HSP90AA1 + NR4A1/NR4A2/JUND: 10 of the top 15 are stress/immediate-early.
#     It shipped as "T cell", tier HIGH, subtype "CD4-positive T cell" — with no
#     CD4, IL7R or CD40LG anywhere in its markers.
#   * cluster 10, 1,271 cells — 10 of the top 15 are cell cycle (MKI67/BIRC5/
#     CCNB2/CCNA2/AURKB/UBE2C/RRM2/CDCA5/GTSE1/DLGAP5).
# The nine identity-driven clusters score 0.00-0.13 on the same statistic, so the
# two populations separate by ~5x. See STATE_DOMINANCE_THRESHOLD.
#
# These panels are cell-state biology. Nothing here names a disease, a tissue or
# a cell type, so the disease-agnostic invariant is untouched.
STATE_PROGRAMMES: Dict[str, frozenset] = {
    "cell_cycle": frozenset(
        {
            "MKI67",
            "TOP2A",
            "BIRC5",
            "CCNB1",
            "CCNB2",
            "CCNA2",
            "CDK1",
            "AURKA",
            "AURKB",
            "UBE2C",
            "TYMS",
            "RRM2",
            "PCNA",
            "TUBB4B",
            "STMN1",
            "NUSAP1",
            "CENPF",
            "CENPE",
            "CDCA3",
            "CDCA5",
            "CDCA8",
            "GTSE1",
            "DLGAP5",
            "PLK1",
            "KIF11",
            "KIF20A",
            "KIF23",
            "SMC4",
            "ASPM",
            "HMGB2",
            "MCM2",
            "MCM3",
            "MCM4",
            "MCM5",
            "MCM6",
            "MCM7",
            "CLSPN",
            "GINS2",
            "FOXM1",
            "TPX2",
            "ANLN",
            "ECT2",
            "NDC80",
            "CKS1B",
            "CKS2",
            "ZWINT",
            "NCAPG",
            "RACGAP1",
            "HIST1H4C",
            "H2AFZ",
            "H4C3",
        }
    ),
    "stress_heat_shock": frozenset(
        {
            "HSPA1A",
            "HSPA1B",
            "HSPA1L",
            "HSPA6",
            "HSPA8",
            "HSPH1",
            "HSPB1",
            "HSPD1",
            "HSPE1",
            "HSP90AA1",
            "HSP90AB1",
            "DNAJA1",
            "DNAJB1",
            "DNAJB4",
            "DNAJB6",
            "BAG3",
            "AHSA1",
            "CHORDC1",
            "CACYBP",
            "ZFAND2A",
            "HSPA5",
            "DDIT3",
            "HERPUD1",
            "UBB",
            "UBC",
        }
    ),
    "immediate_early": frozenset(
        {
            "FOS",
            "FOSB",
            "FOSL1",
            "FOSL2",
            "JUN",
            "JUNB",
            "JUND",
            "EGR1",
            "EGR2",
            "EGR3",
            "NR4A1",
            "NR4A2",
            "NR4A3",
            "ATF3",
            "IER2",
            "IER3",
            "ZFP36",
            "ZFP36L1",
            "DUSP1",
            "DUSP2",
            "SOCS3",
            "BTG2",
            "PPP1R15A",
            "SGK1",
            # KLF2/KLF4/ZFP36L2 are deliberately EXCLUDED: they double as naive /
            # circulating T-cell identity markers (KLF2 with S1PR1/SELL), so counting
            # them as state genes penalised a correct naive-CD4 call. Same rule as
            # lineage_panels step 3 — a gene claimed by two programmes discriminates
            # neither.
        }
    ),
    "interferon": frozenset(
        {
            "ISG15",
            "IFI6",
            "IFI27",
            "IFI44",
            "IFI44L",
            "IFIT1",
            "IFIT2",
            "IFIT3",
            "MX1",
            "MX2",
            "OAS1",
            "OAS2",
            "OAS3",
            "OASL",
            "STAT1",
            "STAT2",
            "IRF7",
            "RSAD2",
            "XAF1",
            "EIF2AK2",
            "BST2",
            "SAMD9",
            "SAMD9L",
            "LY6E",
            "EPSTI1",
            "HERC5",
            "CMPK2",
            "PLSCR1",
        }
    ),
    "hemoglobin_ambient": frozenset(
        {
            "HBB",
            "HBA1",
            "HBA2",
            "HBD",
            "HBG1",
            "HBG2",
            "HBM",
            "ALAS2",
        }
    ),
}

# Ribosomal / mitochondrial genes are technical, not a biological programme, but a
# cluster defined by them is just as unusable for identity.
_TECHNICAL_GENE_RE = re.compile(
    r"^(MT-|MTRNR|RPL\d|RPS\d|MRPL\d|MRPS\d)", re.IGNORECASE
)

# Only the LEADING window of the ranked marker list is scored, so the statistic
# does not drift when `top_n_markers` changes (50 by default, 15 in the exported
# table). State genes concentrate at the top of a state-driven cluster.
STATE_PROFILE_TOP_N: int = 15

# Total state fraction above which a cluster is state-dominated and must not be
# promoted to High confidence. MEASURED, not assumed: on the psoriasis run the
# nine identity-driven clusters score 0.00-0.13 and the two state clusters both
# score 0.67. 0.40 sits an order of magnitude clear of both sides.
STATE_DOMINANCE_THRESHOLD: float = 0.40


def state_programme_profile(
    markers: List[str],
    *,
    top_n: int = STATE_PROFILE_TOP_N,
    threshold: float = STATE_DOMINANCE_THRESHOLD,
) -> Dict[str, object]:
    """How much of a cluster's marker evidence is cell STATE rather than identity.

    Pure list logic (no AnnData), so it is directly unit-testable. Scores only the
    leading ``top_n`` markers — see STATE_PROFILE_TOP_N.

    Returns ``{state_fraction, dominant_programme, dominant_fraction,
    state_dominated, n_markers_scored, per_programme}``. An empty marker list is
    NOT reported as state-dominated: absent evidence is handled by the existing
    ``markers_empty`` flag, and conflating the two would hide it.
    """
    genes = [str(g).strip().upper() for g in (markers or []) if str(g).strip()][
        : int(top_n)
    ]
    n = len(genes)
    if n == 0:
        return {
            "state_fraction": 0.0,
            "dominant_programme": "",
            "dominant_fraction": 0.0,
            "state_dominated": False,
            "n_markers_scored": 0,
            "per_programme": {},
        }

    per: Dict[str, float] = {}
    for prog, panel in STATE_PROGRAMMES.items():
        per[prog] = sum(1 for g in genes if g in panel) / n
    per["technical_ribo_mito"] = (
        sum(1 for g in genes if _TECHNICAL_GENE_RE.match(g)) / n
    )

    # Union, not sum: a gene must not be double-counted if two panels overlap.
    state_genes = {
        g
        for g in genes
        if any(g in p for p in STATE_PROGRAMMES.values()) or _TECHNICAL_GENE_RE.match(g)
    }
    state_fraction = len(state_genes) / n
    dominant = max(per, key=lambda k: (per[k], k))
    return {
        "state_fraction": round(float(state_fraction), 4),
        "dominant_programme": dominant if per[dominant] > 0 else "",
        "dominant_fraction": round(float(per[dominant]), 4),
        "state_dominated": bool(state_fraction >= float(threshold)),
        "n_markers_scored": n,
        "per_programme": {k: round(v, 4) for k, v in per.items() if v > 0},
    }


# ===========================================================================
# Stage 6b — marker evidence required by a SUBTYPE claim
# ===========================================================================
# The subtype layer takes the finest label offered by any single annotator whose
# coarse identity matches the consensus (see consensus.pick_subtype_with_source).
# Coarse-identity agreement does NOT check that the cluster actually shows the
# marker the subtype name asserts, so an annotator could name a specific,
# marker-defined distinction with nothing in the data behind it.
#
# Measured on the psoriasis run, where all 11 subtypes came from ONE voter:
#   * cluster 4 — subtype "CD8-positive T cell" with no CD8A and no CD8B in its
#     markers (CXCL13/CXCR6/ADGRG1; CXCL13 is classically a CD4 Tph programme).
#     CellTypist said `Th` (CD4) and was overridden.
#   * cluster 9 — subtype "CCR4-positive T cell (likely regulatory or Th2 ...)"
#     with no FOXP3/IL2RA/CTLA4 and no GATA3/IL4/IL13.
#   * cluster 10 — subtype "cycling B cell / plasmablast" with no MZB1/JCHAIN/
#     XBP1; it is a TCL1A+ IGHM+ naive B population that happens to be cycling.
#
# Corroboration by a SECOND VOTER was considered and rejected as the gate: the two
# open-vocabulary voters share one model and agree 89-100% of the time (measured),
# so their agreement is not independent evidence. Marker presence in the cluster
# is. Each rule therefore asks the DATA, not another voter.
#
# `requires_any` is used where the transcript is reliably detected in 10x.
# `contradicted_by` is used where it is NOT: CD4 mRNA drops out in a large
# fraction of real CD4 T cells, so demanding it would reject correct calls —
# instead a CD4 claim fails only when positive CD8 evidence is present.
_SubtypeRule = Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], str]

SUBTYPE_EVIDENCE_RULES: Tuple[_SubtypeRule, ...] = (
    # (claim tokens, requires_any, contradicted_by, note)
    (
        ("cd8", "cd8+", "cd8-positive", "cytotoxic t"),
        ("CD8A", "CD8B"),
        (),
        "a CD8 claim needs CD8A/CD8B, which are reliably detected",
    ),
    (
        ("cd4", "cd4+", "cd4-positive", "helper t", " th1", " th2", " th17"),
        (),
        ("CD8A", "CD8B"),
        "CD4 mRNA drops out; a CD4 claim fails only on positive CD8 evidence",
    ),
    (("treg", "regulatory t"), ("FOXP3", "IL2RA", "IKZF2", "CTLA4", "TIGIT"), (), ""),
    (("th17",), ("RORC", "IL17A", "IL17F", "CCR6", "IL23R"), (), ""),
    (("th2",), ("GATA3", "IL4", "IL5", "IL13", "IL1RL1"), (), ""),
    (("th1",), ("TBX21", "IFNG", "CXCR3"), (), ""),
    (("naive",), ("TCF7", "LEF1", "SELL", "CCR7", "MAL"), (), ""),
    (("central memory", "tcm"), ("TCF7", "LEF1", "SELL", "CCR7", "IL7R"), (), ""),
    (
        ("effector memory", "tem", "temra", "emra"),
        ("GZMK", "GZMH", "GZMA", "NKG7", "KLRG1", "CCL5"),
        (),
        "",
    ),
    (
        ("cytotoxic", "effector"),
        ("GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY"),
        (),
        "",
    ),
    (("exhausted",), ("PDCD1", "HAVCR2", "LAG3", "TOX", "TIGIT", "CTLA4"), (), ""),
    (("mait",), ("SLC4A10", "KLRB1", "TRAV1-2", "NCR3"), (), ""),
    (("gamma delta", "gd t"), ("TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2"), (), ""),
    (("cd16", "cd16+", "cd16-positive"), ("FCGR3A",), (), ""),
    (("cd56", "cd56+"), ("NCAM1",), (), ""),
    (
        ("plasma cell", "plasmablast"),
        ("MZB1", "JCHAIN", "XBP1", "PRDM1", "SDC1", "DERL3", "TNFRSF17"),
        (),
        "",
    ),
    (("memory b",), ("CD27", "TNFRSF13B", "AIM2"), (), ""),
    (("mregdc", "lamp3"), ("LAMP3", "CCL22", "CCL19", "IDO1", "FSCN1", "EBI3"), (), ""),
    (("pdc", "plasmacytoid"), ("LILRA4", "CLEC4C", "IL3RA", "GZMB", "IRF7"), (), ""),
    (("cdc1", "dc1"), ("CLEC9A", "XCR1", "CADM1", "BATF3"), (), ""),
    (("cdc2", "dc2"), ("CD1C", "FCER1A", "CLEC10A"), (), ""),
    (
        ("cycling", "proliferating", "proliferative"),
        tuple(sorted(STATE_PROGRAMMES["cell_cycle"])),
        (),
        "",
    ),
    (("interferon", "isg"), tuple(sorted(STATE_PROGRAMMES["interferon"])), (), ""),
    (("at1", "alveolar type 1"), ("AGER", "PDPN", "CAV1", "CLIC5"), (), ""),
    (
        ("at2", "alveolar type 2"),
        ("SFTPC", "SFTPB", "SFTPA1", "NAPSA", "LAMP3"),
        (),
        "",
    ),
    (("club", "secretory"), ("SCGB1A1", "SCGB3A2", "SCGB3A1", "MGP"), (), ""),
    (("ciliated",), ("FOXJ1", "TPPP3", "PIFO", "CAPS", "TUBA1A"), (), ""),
    (("goblet",), ("MUC5AC", "MUC5B", "TFF3", "SPDEF"), (), ""),
    (("basal",), ("KRT5", "KRT14", "TP63", "KRT15"), (), ""),
    (("myofibroblast", "caf"), ("ACTA2", "TAGLN", "POSTN", "FAP", "MYL9"), (), ""),
    (("pericyte",), ("RGS5", "PDGFRB", "NOTCH3", "KCNJ8"), (), ""),
    (("lymphatic",), ("PROX1", "LYVE1", "PDPN", "CCL21", "FLT4"), (), ""),
)


def subtype_marker_support(
    subtype: Optional[str],
    markers: Optional[List[str]],
) -> Tuple[bool, str]:
    """Is a subtype's marker-defined claim actually present in this cluster?

    Returns ``(supported, reason)``. A subtype that makes no recognised claim is
    ``(True, "")`` — there is nothing to check, and this function must never
    reject a label merely because it is unfamiliar.

    ALL recognised claims in the string must hold: "cycling B cell / plasmablast"
    asserts both, so absent plasma-cell evidence rejects it even though the
    cycling half is supported. A hedged subtype naming several alternatives is
    rejected if any named alternative is unsupported — a subtype is an assertion,
    not a shortlist.

    ``markers=None`` means NOT CHECKED and passes everything — for callers that have
    no marker context. ``markers=[]`` means the cluster HAS no markers, which is
    evidence of absence, so a claim-bearing subtype is rejected. The production path
    passes ``markers_by_cluster.get(cl, [])``, so an empty cluster fails closed.
    """
    s = f" {str(subtype or '').strip().lower().replace('_', ' ')} "
    if not s.strip():
        return True, ""
    if markers is None:
        return True, ""  # no marker context supplied; not checked
    present = {str(g).strip().upper() for g in markers if str(g).strip()}
    if not present:
        # The cluster has no markers, so nothing can corroborate a specific claim.
        # A claim-free subtype still passes.
        for tokens, req, _contra, _note in SUBTYPE_EVIDENCE_RULES:
            if any(t.strip() in s for t in tokens) and req:
                return False, "cluster has no marker genes to support the subtype claim"
        return True, ""

    for tokens, req, contra, note in SUBTYPE_EVIDENCE_RULES:
        if not any(t.strip() in s for t in tokens):
            continue
        hit = next((t.strip() for t in tokens if t.strip() in s), "")
        if req and not (present & set(req)):
            detail = f"; {note}" if note else ""
            return False, (
                f"claims {hit!r} but none of "
                f"{', '.join(sorted(set(req))[:6])} is a marker of this cluster{detail}"
            )
        if contra and (present & set(contra)):
            clash = ", ".join(sorted(present & set(contra)))
            detail = f"; {note}" if note else ""
            return False, f"claims {hit!r} but contradicted by {clash}{detail}"
    return True, ""


# ===========================================================================
# Stage 3b — observed lineage composition, for reference-model selection
# ===========================================================================
# The CellTypist model is picked from the study's ORGAN, inferred from GEO title
# text ("skin"), never from the data. A sorted-immune dataset that happens to come
# from skin therefore gets a skin model whose classes are mostly keratinocyte and
# stromal.
#
# Measured on the psoriasis run: tissue resolved to "skin", so
# Adult_Human_Skin.pkl was loaded — yet all 11 clusters are immune, with zero
# keratinocyte, fibroblast or endothelial clusters. Skin contains few B cells, so
# that model's B-cell coverage is thin, and cluster 10 (TCL1A+/CD79A+/IGHM+ B
# cells) was called `DC1` by CellTypist, which is what dragged it to Low/Review.
#
# WHY DETECTION FRACTION, not marker overlap or score_genes:
#   * marker overlap fails — `rank_genes_groups` returns cluster-DISCRIMINATING
#     genes, so in an all-T-cell dataset CD3D is nobody's marker. Measured: 4 of
#     11 clusters produced no lineage-panel hit at all.
#   * `score_genes` fails for the same reason the lineage gate abstains — it
#     subtracts a background of similarly-expressed genes, so in a homogeneous
#     dataset the dominant lineage scores ~0 BY CONSTRUCTION. Measured: the gate
#     abstained on 6 of 11 clusters (72.9% of cells) in pure immune tissue.
#   * panel DETECTION FRACTION is absolute, so it is unaffected by either.
#     Measured: 6/6 shipped labels correctly identified as Immune.
LINEAGE_PANEL_MIN_DETECTION: float = 0.02  # floor; observed range 0.087-0.140
OBSERVED_LINEAGE_MIN_CLUSTER_FRACTION: float = 0.90

#: Coarse lineage -> the pan-tissue CellTypist model specialising in it. Only
#: Immune has one in the catalog; the other lineages deliberately have no entry,
#: so no swap is offered where no specialist exists.
LINEAGE_SPECIALIST_MODELS: Dict[str, str] = {"Immune": "Immune_All_Low.pkl"}


def observed_lineage_profile(
    adata: AnnData,
    cluster_col: str = "leiden",
    *,
    min_detection: float = LINEAGE_PANEL_MIN_DETECTION,
) -> Dict[str, object]:
    """Which coarse lineages the DATA actually contains, per cluster.

    For each cluster, the mean per-cell fraction of each lineage panel that is
    DETECTED (non-zero). The winning panel must clear ``min_detection``, else the
    cluster is reported as "Other". ``adata`` is the log-normalized working copy.

    Returns ``{per_cluster, cluster_counts, dominant_lineage, dominant_fraction,
    panel_detection}``. Purely descriptive — the caller decides what to do.
    """
    import numpy as _np

    var_ns = adata.var_names.astype(str)
    clusters = adata.obs[cluster_col].astype(str)
    cols: Dict[str, "pd.Series"] = {}
    for lin, genes in lineage_markers().items():
        present = [g for g in genes if g in var_ns]
        if not present:
            continue
        X = adata[:, present].X
        nz = X > 0
        nz = (
            _np.asarray(nz.sum(axis=1)).ravel()
            if hasattr(nz, "sum")
            else _np.asarray(nz).sum(axis=1)
        )
        cols[lin] = pd.Series(nz / float(len(present)), index=adata.obs_names)
    if not cols:
        return {
            "per_cluster": {},
            "cluster_counts": {},
            "dominant_lineage": "Other",
            "dominant_fraction": 0.0,
            "panel_detection": {},
        }

    det = pd.DataFrame(cols).groupby(clusters.values).mean()
    per_cluster: Dict[str, str] = {}
    for cl, row in zip(det.index, det.to_dict("records"), strict=True):
        top = str(row.idxmax())
        per_cluster[str(cl)] = (
            top if float(row.max()) >= float(min_detection) else "Other"
        )

    counts = Counter(per_cluster.values())
    dom, dom_n = counts.most_common(1)[0] if counts else ("Other", 0)
    frac = dom_n / len(per_cluster) if per_cluster else 0.0
    logger.info(
        "[OBSERVED-LINEAGE] %s clusters -> %s; dominant %r at %s (panel detection is absolute, not score_genes).",
        len(per_cluster),
        dict(sorted(counts.items())),
        dom,
        format(frac, ".0%"),
    )
    return {
        "per_cluster": per_cluster,
        "cluster_counts": {k: int(v) for k, v in sorted(counts.items())},
        "dominant_lineage": str(dom),
        "dominant_fraction": round(float(frac), 4),
        "panel_detection": {
            str(c): {k: round(float(v), 4) for k, v in r.items()}
            for c, r in zip(det.index, det.to_dict("records"), strict=True)
        },
    }


def refine_celltypist_model_for_observed_lineage(
    model_name: str,
    profile: Dict[str, object],
    *,
    valid_models: Optional[frozenset] = None,
    min_cluster_fraction: float = OBSERVED_LINEAGE_MIN_CLUSTER_FRACTION,
) -> Tuple[str, str]:
    """Prefer a lineage specialist over an organ model when the data is one lineage.

    Returns ``(model, reason)``; ``reason`` is "" when nothing changed. Fires only
    when at least ``min_cluster_fraction`` of clusters share one coarse lineage AND
    a specialist exists for it AND it is not already the chosen model.

    Disease-blind by construction: the decision reads the observed expression
    profile, never a disease, condition or study term.
    """
    dom = str(profile.get("dominant_lineage") or "")
    frac = float(profile.get("dominant_fraction") or 0.0)
    specialist = LINEAGE_SPECIALIST_MODELS.get(dom)
    if not specialist or frac < float(min_cluster_fraction):
        return model_name, ""
    if specialist == model_name:
        return model_name, ""
    if valid_models is not None and specialist not in valid_models:
        return model_name, ""
    return specialist, (
        f"{frac:.0%} of clusters are {dom} by absolute panel detection, so the "
        f"organ model {model_name!r} is being used outside the compartment it "
        f"mostly describes; switched to the {dom.lower()} specialist {specialist!r}"
    )


# ===========================================================================
# Stage 8 — vote counting (logic only; adjudication lives in the agent layer)
# ===========================================================================
# Voters split into two kinds, and the difference decides how much a dissent is
# worth:
#
# CLOSED vocabulary (CellTypist, SingleR) can only return a label that exists in
#   the reference they were handed. They cannot abstain. Given a population the
#   reference does not contain, they are *forced* to emit something wrong.
# OPEN vocabulary (the LLM knowledge voter, PubMed) reason from the marker genes
#   and can name anything, including a cell type absent from every reference — and
#   can say "Unassigned" instead of guessing.
#
# Measured on GSE337706 (breast-cancer PBMC, tissue resolved to 'blood', so
# COVID19_HumanChallenge_Blood.pkl was loaded): cluster 0, 24,272 cells (27.8% of
# the run), markers PLP1 +15.1 / CRYAB +15.0 / APLP1 +14.2 / MAG +12.2 — an
# unambiguous myelinating-glia programme, with zero T-cell markers. A blood model
# has no Schwann-cell class, so CellTypist returned 'T CD4 Naive' spread over 17
# labels with only 0.536 of cells on top, and SingleR returned 'CD4+ T cells' at
# 0.553. The knowledge voter read the markers and said Schwann cell at 0.82. Plain
# vote counting made it 2-to-1 and the cluster shipped as "T cell", tier Medium.
# The same marker set is correctly called Oligodendrocyte at 0.997 by the *brain*
# model, which proves the failure is reference scope, not biology.
CLOSED_VOCABULARY_VOTERS: frozenset = frozenset({"celltypist", "singler"})
OPEN_VOCABULARY_VOTERS: frozenset = frozenset({"knowledge_based", "pubmed"})

# An open-vocabulary voter must be this confident (native scale) before it is
# allowed to overturn a closed-vocabulary majority. 0.80 sits above the LLM's
# hedging band and below its assertive calls (Schwann cell 0.82, Microglia 0.97).
OPEN_VOCAB_MIN_CONFIDENCE: float = 0.80

# A voter that returns the SAME label for nearly every cluster is not voting, it
# is saturated, and its dissent is noise. Measured on GSE157827 (Alzheimer
# prefrontal cortex): SingleR against HumanPrimaryCellAtlasData returned
# 'Astrocyte' for 18 of 20 clusters at 0.459-0.566 — including both
# oligodendrocyte clusters (PLP1/MAG/MOBP/OPALIN) and every neuronal cluster. Only
# one of those 18 is really an astrocyte, so it was ~6% accurate. Consensus
# outvoted it every time, but its permanent dissent set voters_disagree on 17 of 20
# clusters and dragged 83.5% of cells to Low/Review with ZERO clusters reaching
# High — making the tier column useless for triage.
DEGENERATE_VOTER_MODAL_FRACTION: float = 0.80
DEGENERATE_VOTER_MIN_CLUSTERS: int = 5


def degenerate_voters(
    labels_by_voter: Dict[str, Dict[str, str]],
    *,
    modal_fraction: float = DEGENERATE_VOTER_MODAL_FRACTION,
    min_clusters: int = DEGENERATE_VOTER_MIN_CLUSTERS,
) -> Dict[str, dict]:
    """Voters whose output is saturated on one label, so it carries no information.

    ``labels_by_voter`` is ``{voter: {cluster: harmonized_label}}``. A voter is
    reported when it produced a usable label for at least ``min_clusters`` clusters
    and at least ``modal_fraction`` of them are the SAME label.

    Returns ``{voter: {"modal_label", "modal_fraction", "n_clusters"}}`` for the
    saturated voters only. The caller decides what to do; nothing is discarded
    here, and the voter's raw calls must still be reported for transparency.

    The threshold is deliberately high (0.80). A blood cohort that genuinely is 75%
    T cells must not have its T-cell voter suppressed — this targets a voter with
    no discriminative power at all, not a skewed but real biological distribution.
    """
    out: Dict[str, dict] = {}
    for voter, per_cluster in (labels_by_voter or {}).items():
        usable = [str(v) for v in (per_cluster or {}).values() if v and v != UNASSIGNED]
        if len(usable) < min_clusters:
            continue
        counts = Counter(usable)
        modal_label, modal_count = counts.most_common(1)[0]
        frac = modal_count / len(usable)
        if frac >= modal_fraction:
            out[voter] = {
                "modal_label": modal_label,
                "modal_fraction": round(float(frac), 4),
                "n_clusters": len(usable),
            }
    return out


def out_of_domain_deference(
    method_labels: Dict[str, str],
    tally: Dict[str, object],
    *,
    open_confidences: Dict[str, float],
    celltypist_unreliable: bool,
) -> Optional[Tuple[str, str, str]]:
    """Should a confident marker-driven call overturn a forced closed-vocab majority?

    Returns ``(winning_label, winning_voter, reason)`` when every condition holds,
    otherwise ``None`` — the normal tally decides.

    All four must hold, so this fires on a reference-scope failure and not on
    ordinary voter disagreement:

    1. There IS a majority/unanimous call (a split already routes to the
       adjudicator, which reads the markers itself — no need to intervene).
    2. Every voter supporting it has a CLOSED vocabulary. If any open-vocabulary
       voter backs the majority, the markers already agree and nothing is forced.
    3. CellTypist's own call for the cluster is unreliable — its per-cell labels
       are scattered below the configured ``mixed_cluster_min_dominant_fraction``.
       A confident closed-vocabulary call is normal evidence and still wins.
    4. An open-vocabulary voter is confident at >= OPEN_VOCAB_MIN_CONFIDENCE and
       names something OTHER than the majority.

    The result must always be tiered for review, never promoted: preferring the
    marker-driven label is the better guess, not a validated identity.
    """
    if not (tally.get("has_majority") or tally.get("unanimous")):
        return None  # (1)
    majority = str(tally.get("majority_label") or "")
    if not majority or majority == UNASSIGNED:
        return None

    supporters = {m for m, label in method_labels.items() if label == majority}
    if not supporters or (supporters - CLOSED_VOCABULARY_VOTERS):
        return None  # (2)
    if not celltypist_unreliable:
        return None  # (3)

    best: Optional[Tuple[float, str, str]] = None  # (4)
    for voter in sorted(OPEN_VOCABULARY_VOTERS):
        label = method_labels.get(voter)
        if not label or label in (UNASSIGNED, majority):
            continue
        conf = open_confidences.get(voter)
        if conf is None or not isinstance(conf, (int, float)) or conf != conf:
            continue
        if float(conf) < OPEN_VOCAB_MIN_CONFIDENCE:
            continue
        if best is None or float(conf) > best[0]:
            best = (float(conf), voter, str(label))
    if best is None:
        return None

    conf, voter, label = best
    return (
        label,
        voter,
        f"closed-vocabulary voters ({'+'.join(sorted(supporters))}) were forced to "
        f"guess — CellTypist scattered below its dominant-fraction floor — while "
        f"{voter} read the markers and called {label!r} at {conf:.2f}; deferred to "
        f"the marker-driven call and flagged for review",
    )


def tally_votes(
    method_labels: Dict[str, str], method_conf: Optional[Dict[str, float]] = None
) -> Dict[str, object]:
    """Count harmonized votes with a principled, deterministic tie-break.

    method_labels: {method_name: harmonized_label} for methods that produced a
    usable (non-Unassigned) label.
    method_conf:   optional {method_name: confidence 0..1}. Used ONLY to break a
    tie for the top label — the label with the highest summed supporter confidence
    wins, alphabetical as the final deterministic fallback. This replaces the old
    ``Counter.most_common`` behavior, which broke ties by dict-insertion order and
    so silently favored whichever voter happened to be added first (CellTypist).

    Returns: majority_label, majority_count, n_methods, unanimous(bool),
    has_majority(bool), tied(bool), top_labels(list), pattern(str).
    """
    usable = {
        m: label for m, label in method_labels.items() if label and label != UNASSIGNED
    }
    n = len(usable)
    if n == 0:
        return {
            "majority_label": UNASSIGNED,
            "majority_count": 0,
            "n_methods": 0,
            "unanimous": False,
            "has_majority": False,
            "tied": False,
            "top_labels": [],
            "pattern": "none",
        }
    counts = Counter(usable.values())
    top_count = max(counts.values())
    top_labels = sorted(label for label, c in counts.items() if c == top_count)
    tied = len(top_labels) > 1

    def _sup_conf(label: str) -> float:
        if not method_conf:
            return 0.0
        s = 0.0
        for m, voter_label in usable.items():
            if voter_label != label:
                continue
            c = method_conf.get(m, 0.0)
            if isinstance(c, (int, float)) and c == c:  # exclude None/NaN
                s += float(c)
        return s

    # top label: highest summed confidence, then alphabetical -> fully deterministic
    top_label = sorted(top_labels, key=lambda label: (-_sup_conf(label), label))[0]
    unanimous = len(counts) == 1 and n >= 2
    # strict majority = strictly more than half of the usable voters, and only
    # meaningful with >= 2 voters. A single voter is NOT a "majority": there is
    # nothing to corroborate it, so it must fall through to the review tier
    # rather than be promoted to a Medium-confidence consensus.
    has_majority = n >= 2 and top_count * 2 > n and not unanimous
    pattern = "/".join(f"{m}={label}" for m, label in sorted(usable.items()))
    return {
        "majority_label": top_label,
        "majority_count": top_count,
        "n_methods": n,
        "unanimous": unanimous,
        "has_majority": has_majority,
        "tied": tied,
        "top_labels": top_labels,
        "pattern": pattern,
    }


def normalize_confidences(conf_by_cluster: Dict[str, float]) -> Dict[str, float]:
    """Rank-normalize ONE voter's per-cluster confidences into [0, 1].

    WHY: the voters report confidence on incompatible native scales — CellTypist =
    fraction of cells backing the label (~0.3-1.0), SingleR = max Spearman rho
    (~0.1-0.5 for scRNA vs a bulk reference), the LLM = a self-reported 0-1 that
    tends to sit near ~0.9, PubMed = an evidence score. Summing those raw numbers
    to break a tie (or handing them to the adjudicator) silently under-weights the
    voter whose native scale is smallest — in practice SingleR — even when it is
    right. Converting each voter's values to their within-run percentile rank makes
    only RELATIVE confidence matter, so the voters compare fairly.

    NaN/None values are dropped. A single finite value (or an all-equal set) maps to
    0.5 — there is nothing to rank, so it must not bias the tie-break either way.
    """
    vals = {
        str(c): float(v)
        for c, v in (conf_by_cluster or {}).items()
        if isinstance(v, (int, float)) and v == v
    }  # drop None / NaN
    if not vals:
        return {}
    if len(vals) == 1:
        return {c: 0.5 for c in vals}
    s = pd.Series(vals)
    if float(s.max() - s.min()) < 1e-12:  # all equal -> no ranking information
        return {c: 0.5 for c in vals}
    return {str(c): float(v) for c, v in s.rank(method="average", pct=True).items()}


# ===========================================================================
# Stage 9 — broadcast, validate, export
# ===========================================================================
def broadcast_and_validate(
    adata: AnnData,
    cluster_col: str,
    per_cluster_columns: Dict[str, Dict[str, str]],
) -> None:
    """Write per-cluster label dicts back to per-cell obs columns, in place.

    per_cluster_columns: {obs_column_name: {cluster: label}}.
    Asserts conservation of cells and that the final consensus column has no NaN.
    """
    n_in = adata.n_obs
    cl_str = adata.obs[cluster_col].astype(str)
    for obs_col, mapping in per_cluster_columns.items():
        adata.obs[obs_col] = cl_str.map(
            lambda c, _m=mapping: _m.get(c, UNASSIGNED)
        ).astype("category")
    if adata.n_obs != n_in:
        raise CellConservationError(
            f"cell count changed while writing annotation columns: "
            f"total-in ({n_in}) != total-out ({adata.n_obs})"
        )
    if "celltype_consensus" in adata.obs.columns:
        n_labeled = int(adata.obs["celltype_consensus"].notna().sum())
        if n_labeled != n_in:
            raise CellConservationError(
                f"{n_in - n_labeled} of {n_in} cells lack a consensus label"
            )


def write_consensus_table(rows: List[dict], out_dir: Path, analysis_name: str) -> Path:
    """Write the per-cluster consensus provenance table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / f"{analysis_name}_consensus_annotation.csv"
    atomic_to_csv(df, path, index=False)
    logger.info("[EXPORT] wrote consensus table: %s", path)
    return path


# ---------------------------------------------------------------------------
def _natural_key(s: str):
    """Sort '0','1','2','10' numerically when possible, else lexically."""
    try:
        return (0, int(s))
    except (TypeError, ValueError):
        return (1, str(s))
