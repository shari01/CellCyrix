"""
Tiny DSL for declaring the cell-type hierarchy.

A node is declared as a 6-tuple:

    (node_id, canonical_label, cl_id, tissue_scope, markers, children)

    node_id       flat snake_case slug, globally unique, stable across renames
    canonical_label human-readable display name
    cl_id         Cell Ontology term (e.g. "CL:0000084") or "" when no clean 1:1 term
    tissue_scope  comma-separated tissue tokens, or "pan_tissue"
    markers       comma-separated canonical marker genes (indicative, not a validated panel)
    children      list of child node tuples (possibly empty)

Depth in the nested structure defines `level`:

    0 lineage        1 class        2 main_cell_type       3 subtype       4 fine_subtype

LEVEL_NAMES is the single source of truth for that mapping.

HARD RULES (enforced by validate_tree):
  * no disease names anywhere in this module or the spec modules
  * node_id must be unique across the whole forest
  * node_id must be lowercase snake_case
  * depth must not exceed len(LEVEL_NAMES) - 1
This module contains no I/O, no LLM calls, and no tissue- or disease-conditional logic.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Sequence, Tuple

LEVEL_NAMES: Tuple[str, ...] = (
    "lineage",
    "class",
    "main_cell_type",
    "subtype",
    "fine_subtype",
)

MAX_DEPTH = len(LEVEL_NAMES) - 1

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_CL_RE = re.compile(r"^CL:\d{7}$")

NodeTuple = Tuple[str, str, str, str, str, List]


def N(
    node_id: str,
    canonical_label: str,
    cl_id: str = "",
    tissue_scope: str = "pan_tissue",
    markers: str = "",
    children: Sequence[NodeTuple] | None = None,
) -> NodeTuple:
    """Construct a node tuple. Keyword-friendly wrapper over the raw 6-tuple."""
    return (
        node_id,
        canonical_label,
        cl_id,
        tissue_scope,
        markers,
        list(children) if children else [],
    )


class FlatNode:
    """Flattened node with resolved ancestry. Plain data container."""

    __slots__ = (
        "node_id",
        "canonical_label",
        "cl_id",
        "tissue_scope",
        "markers",
        "level",
        "level_name",
        "parent_id",
        "path_ids",
        "is_terminal",
    )

    def __init__(
        self,
        node_id: str,
        canonical_label: str,
        cl_id: str,
        tissue_scope: str,
        markers: str,
        level: int,
        parent_id: str,
        path_ids: Tuple[str, ...],
        is_terminal: bool,
    ) -> None:
        self.node_id = node_id
        self.canonical_label = canonical_label
        self.cl_id = cl_id
        self.tissue_scope = tissue_scope
        self.markers = markers
        self.level = level
        self.level_name = LEVEL_NAMES[level]
        self.parent_id = parent_id
        self.path_ids = path_ids
        self.is_terminal = is_terminal

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<FlatNode {self.node_id} L{self.level}>"


def walk(
    nodes: Sequence[NodeTuple],
    parent_id: str = "",
    path: Tuple[str, ...] = (),
) -> Iterator[FlatNode]:
    """Depth-first walk yielding FlatNode objects with resolved ancestry."""
    for node_id, label, cl_id, scope, markers, children in nodes:
        here = path + (node_id,)
        yield FlatNode(
            node_id=node_id,
            canonical_label=label,
            cl_id=cl_id,
            tissue_scope=scope,
            markers=markers,
            level=len(here) - 1,
            parent_id=parent_id,
            path_ids=here,
            is_terminal=not children,
        )
        if children:
            yield from walk(children, parent_id=node_id, path=here)


def validate_tree(nodes: Sequence[NodeTuple]) -> List[FlatNode]:
    """Flatten and validate. Raises ValueError on the first structural problem."""
    flat = list(walk(nodes))
    seen: dict[str, str] = {}
    problems: List[str] = []

    for fn in flat:
        if fn.node_id in seen:
            problems.append(
                f"duplicate node_id {fn.node_id!r} (first under {seen[fn.node_id]!r})"
            )
        seen[fn.node_id] = fn.parent_id or "<root>"

        if not _SLUG_RE.match(fn.node_id):
            problems.append(f"node_id {fn.node_id!r} is not lowercase snake_case")

        if fn.cl_id and not _CL_RE.match(fn.cl_id):
            problems.append(f"{fn.node_id}: malformed CL id {fn.cl_id!r}")

        if fn.level > MAX_DEPTH:
            problems.append(
                f"{fn.node_id}: depth {fn.level} exceeds MAX_DEPTH {MAX_DEPTH}"
            )

        if not fn.canonical_label.strip():
            problems.append(f"{fn.node_id}: empty canonical_label")

    if problems:
        raise ValueError(
            "hierarchy spec failed validation:\n  - " + "\n  - ".join(problems)
        )
    return flat
