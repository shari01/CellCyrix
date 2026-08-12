"""
Pan-lineage marker panels for the coarse lineage gate, derived from curated
reference data instead of a hand-typed list.

Why
---
The gate's original :data:`tools.LINEAGE_MARKERS` was five hand-written panels of
5-14 genes. It had **no mast-cell and no dendritic-cell coverage at all**, so those
clusters scored ~0 on every panel and the gate's ``idxmax`` handed them whichever
panel was least negative — in practice "Epithelial", producing a phantom epithelial
compartment with 0% EPCAM. Widening the panels by hand does not scale and is not
auditable.

How
---
Two references, each used for what it is authoritative about:

* ``TIS_CELL_markers_v3/master_celltype_markers_long.csv`` — 7,135 curated
  cell-type/marker rows carrying ``specificity_human`` and ``marker_score``.
  Supplies the genes.
* :mod:`cell_hierarchy` (subtype-ref) — supplies the lineage each of those cell
  types belongs to, so the mapping is a lookup rather than a guess.

Construction, in order:

1. Resolve every TIS_CELL ``cell_type`` through the hierarchy. Keep only
   ``confidence >= MIN_MAPPING_CONFIDENCE`` — fuzzy matches must not define a panel.
2. Map the hierarchy lineage onto the gate's five coarse lineages. Pericytes and
   vascular smooth muscle sit under *Stromal / mesenchymal* in the hierarchy but are
   **Mural** to the gate, so they are split out by ``main_cell_type``. The
   hierarchy's *Muscle cell* lineage (cardiomyocyte, skeletal myocyte) maps to
   **nothing**: a myocyte is not a mural cell, matching the long-standing note in
   :data:`tools.CANONICAL_TYPES`.
3. Drop any gene that appears under more than one gate lineage. A lineage panel is
   only meaningful if its genes discriminate *between* lineages.
4. Keep genes with ``specificity_human < MAX_SPECIFICITY`` (lower is more specific),
   rank by ``marker_score``, take the top :data:`PANEL_SIZE`.
5. Union with the original hand-written panel, so nothing the gate could previously
   detect is ever lost.

If the reference data is missing or unreadable the builder returns the original
hand-written panels unchanged and says so in the provenance — the gate degrades to
its previous behaviour rather than failing.

Nothing here is disease-aware; the inputs are cell-biology references only.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from .. import env_names

logger = logging.getLogger(__name__)

# Only mappings at or above this resolver confidence define a panel.
MIN_MAPPING_CONFIDENCE: float = 0.95
# TIS_CELL specificity_human: fraction of OTHER cell types also expressing the gene,
# so lower is more specific. 0.05 keeps genes seen in <5% of other types.
MAX_SPECIFICITY: float = 0.05
# Genes per lineage, ranked by marker_score.
PANEL_SIZE: int = 40

# hierarchy lineage -> gate lineage. "Muscle cell", "Neural cell", "Germ-cell and
# placental lineage" and "Unassigned" are deliberately absent: the gate has no such
# coarse bucket, and inventing one would let it make calls it cannot support.
_HIERARCHY_TO_GATE: Dict[str, str] = {
    "Haematopoietic cell": "Immune",
    "Epithelial cell": "Epithelial",
    "Endothelial cell": "Endothelial",
    "Stromal / mesenchymal cell": "Fibroblast",
}
# main_cell_type values that are Mural to the gate despite sitting under
# Stromal / mesenchymal in the hierarchy.
_MURAL_MAIN_TYPES = {"Pericyte", "Vascular smooth muscle cell", "Mural cell"}

_TIS_CELL_REL = (
    Path("celltype_markers_references")
    / "TIS_CELL_markers_v3"
    / "master_celltype_markers_long.csv"
)


def shared_reference_root() -> Path:
    """Root that CONTAINS the reference buckets. ``SCPIPE_SHARED_REFERENCE_ROOT`` wins."""
    env = (env_names.get_env(env_names.SHARED_REFERENCE_ROOT) or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # <pkg root>/shared_reference/single_cell_pipeline_agent_datasets
    pkg_root = Path(__file__).resolve().parents[4]
    return (pkg_root / "shared_reference").resolve()


def tis_cell_marker_table() -> Path:
    """Locate the TIS_CELL marker table.

    Checked in order: the ``shared_reference`` bucket (standalone package layout),
    then the legacy ``Celltype_Markers_References/`` folder that sits at the root of
    a source checkout. The same module then works from either tree without the
    28 MB reference set having to exist twice on disk.
    """
    root = shared_reference_root()
    candidates = [
        root / "single_cell_pipeline_agent_datasets" / _TIS_CELL_REL,
        root / _TIS_CELL_REL,
    ]
    # Legacy checkout: <repo root>/Celltype_Markers_References/TIS_CELL_markers_v3/...
    repo_root = Path(__file__).resolve().parents[4]
    legacy_rel = Path(*_TIS_CELL_REL.parts[1:])  # drop "celltype_markers_references"
    candidates += [
        repo_root / "Celltype_Markers_References" / legacy_rel,
        repo_root / _TIS_CELL_REL,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _gate_lineage_for(resolution) -> str | None:
    """Gate lineage for one resolver Resolution, or None to exclude it."""
    if resolution is None or not getattr(resolution, "node_id", None):
        return None
    if float(getattr(resolution, "confidence", 0.0) or 0.0) < MIN_MAPPING_CONFIDENCE:
        return None
    if str(getattr(resolution, "main_cell_type", "") or "") in _MURAL_MAIN_TYPES:
        return "Mural"
    return _HIERARCHY_TO_GATE.get(str(getattr(resolution, "lineage", "") or ""))


@lru_cache(maxsize=1)
def build_lineage_markers(
    fallback: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> Tuple[Dict[str, List[str]], Dict[str, object]]:
    """Return ``(panels, provenance)``.

    ``fallback`` is the hand-written panel set as a hashable tuple-of-tuples (the
    function is cached, so it cannot take a dict). Its genes are always unioned in.
    """
    base: Dict[str, List[str]] = {lin: list(genes) for lin, genes in fallback}
    prov: Dict[str, object] = {
        "source": "builtin",
        "reason": "",
        "table": None,
        "panel_sizes": {k: len(v) for k, v in base.items()},
        "min_mapping_confidence": MIN_MAPPING_CONFIDENCE,
        "max_specificity": MAX_SPECIFICITY,
        "panel_size": PANEL_SIZE,
        "cell_types_mapped": 0,
    }

    table = tis_cell_marker_table()
    if not table.is_file():
        prov["reason"] = f"TIS_CELL marker table not found at {table}"
        logger.info("[LINEAGE-PANELS] %s; using built-in panels.", prov["reason"])
        return base, prov

    try:
        import pandas as pd

        from .cell_hierarchy import CellHierarchy

        hierarchy = CellHierarchy.from_spec()
        df = pd.read_csv(table)
        needed = {"cell_type", "gene_symbol", "specificity_human", "marker_score"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")

        ct_to_gate: Dict[str, str] = {}
        for ct in df["cell_type"].dropna().unique():
            gate = _gate_lineage_for(hierarchy.resolve(str(ct), source="panglaodb"))
            if gate:
                ct_to_gate[str(ct)] = gate
        if not ct_to_gate:
            raise ValueError("no cell type resolved to a gate lineage")

        df = df[df["cell_type"].astype(str).isin(ct_to_gate)].copy()
        df["_gate"] = df["cell_type"].astype(str).map(ct_to_gate)

        # A gene claimed by two lineages cannot discriminate between them.
        per_gene = df.groupby("gene_symbol")["_gate"].nunique()
        df = df[df["gene_symbol"].isin(per_gene[per_gene == 1].index)]
        df = df[
            pd.to_numeric(df["specificity_human"], errors="coerce") < MAX_SPECIFICITY
        ]

        built: Dict[str, List[str]] = {}
        for gate, sub in df.groupby("_gate"):
            top = (
                sub.sort_values("marker_score", ascending=False)
                .drop_duplicates("gene_symbol")["gene_symbol"]
                .head(PANEL_SIZE)
                .astype(str)
                .tolist()
            )
            built[str(gate)] = top

        merged: Dict[str, List[str]] = {}
        for lin, hand in base.items():
            seen, out = set(), []
            for g in list(hand) + built.get(lin, []):  # hand-written genes first
                if g not in seen:
                    seen.add(g)
                    out.append(g)
            merged[lin] = out

        prov.update(
            {
                "source": "TIS_CELL_markers_v3 + cell_hierarchy",
                "table": str(table),
                "panel_sizes": {k: len(v) for k, v in merged.items()},
                "cell_types_mapped": len(ct_to_gate),
                "added_per_lineage": {k: len(merged[k]) - len(base[k]) for k in merged},
            }
        )
        logger.info(
            "[LINEAGE-PANELS] built from %s via cell_hierarchy: %s cell types mapped; panel sizes %s (was %s).",
            table.name,
            len(ct_to_gate),
            prov["panel_sizes"],
            {k: len(v) for k, v in base.items()},
        )
        return merged, prov

    except Exception as e:  # noqa: BLE001
        prov["reason"] = f"{type(e).__name__}: {e}"
        logger.warning(
            "[LINEAGE-PANELS] could not build panels from reference data (%s); using built-in panels.",
            prov["reason"],
        )
        return base, prov
