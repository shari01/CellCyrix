"""Assembles the full hierarchy forest from the per-lineage spec modules."""

from __future__ import annotations

from typing import List

from . import epithelial, hematopoietic, mesenchymal, neural_germ_other
from ._dsl import LEVEL_NAMES, MAX_DEPTH, FlatNode, N, validate_tree, walk
from .aliases import ALIASES, COMPARTMENT_LABELS, SOURCES
from .states import PROTECTED_TOKENS, STATES, state_lookup

FOREST = [
    *hematopoietic.BRANCH,
    *epithelial.BRANCH,
    *mesenchymal.BRANCH,
    *neural_germ_other.BRANCH,
]


def flat_nodes() -> List[FlatNode]:
    """Validated, flattened node list. Raises on a malformed spec."""
    return validate_tree(FOREST)


__all__ = [
    "FOREST",
    "flat_nodes",
    "FlatNode",
    "LEVEL_NAMES",
    "MAX_DEPTH",
    "N",
    "walk",
    "validate_tree",
    "ALIASES",
    "COMPARTMENT_LABELS",
    "SOURCES",
    "STATES",
    "PROTECTED_TOKENS",
    "state_lookup",
]
