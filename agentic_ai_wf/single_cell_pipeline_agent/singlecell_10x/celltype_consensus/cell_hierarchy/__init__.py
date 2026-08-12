"""
Disease-agnostic cell-type hierarchy: lineage -> class -> main_cell_type ->
subtype -> fine_subtype, with a cross-vocabulary alias crosswalk and a pure-logic
resolver for multi-voter consensus annotation.

Design invariants
-----------------
1. No disease string appears anywhere in this package, in data or in logic.
2. Tissue is metadata only. It can break a tie among valid candidates; it can
   never gate or override a confident match.
3. Cell STATE (cycling, malignant, exhausted, doublet) is a separate axis from
   cell IDENTITY and lives in spec/states.py. States never enter the tree.
4. Everything in resolver.py is deterministic pure logic — no LLM, no network.
   LLM verification belongs in your agent layer, consuming Resolution objects.
5. total_in == total_out. Batch calls return one result per input; unresolvable
   labels are flagged, never dropped.

Quick start
-----------
    from cell_hierarchy import CellHierarchy

    h = CellHierarchy.from_spec()

    r = h.resolve("CD16+ NK cells", source="celltypist_immune")
    r.node_id          # 'cd56_dim_nk_cell'
    r.lineage          # 'Haematopoietic cell'
    r.main_cell_type   # 'Natural killer cell'

    c = h.consensus({
        "celltypist_immune": "CD16+ NK cells",
        "singler_blueprint": "NK cells",
        "azimuth_pbmc": "CD8 TEM",
    })
    c.consensus_label     # lowest common ancestor across voters
    c.agreement_score     # depth-weighted, abstention-penalised
"""

from .resolver import (
    CONFIDENCE,
    LEVEL_CREDIT,
    CellHierarchy,
    Consensus,
    NormalizedLabel,
    Resolution,
    normalize,
)
from .spec import LEVEL_NAMES, flat_nodes

__all__ = [
    "CellHierarchy",
    "Consensus",
    "Resolution",
    "NormalizedLabel",
    "normalize",
    "flat_nodes",
    "LEVEL_NAMES",
    "LEVEL_CREDIT",
    "CONFIDENCE",
]

__version__ = "1.0.0"
