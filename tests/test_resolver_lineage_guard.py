"""
test_resolver_lineage_guard.py — a fuzzy match must never cross a cell lineage.

Found on real data, not in any synthetic test. ``naive T cell`` had no alias, fell
through to the edit-distance fallback, and matched ``naive b cell`` at ratio 0.55 —
one character between two unrelated lineages. It resolved 11,386 cells in a single AIDA
run to ``naive_b_cell``, silently, with full confidence downstream.

Edit distance has no concept of B versus T. ``_lineage_compatible`` does, and these
tests pin it, because the failure mode is invisible: nothing errors, the labels simply
become wrong.
"""

from __future__ import annotations

import pytest

from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.cell_hierarchy import (  # noqa: E501
    CellHierarchy,
)
from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.cell_hierarchy.resolver import (  # noqa: E501
    _lineage_compatible,
    _lineage_of,
)


@pytest.fixture(scope="module")
def hierarchy():
    return CellHierarchy.from_spec()


# --- the specific regression -----------------------------------------------------


def test_naive_t_cell_is_not_a_b_cell(hierarchy):
    """The exact bug: 11,386 cells were mislabelled by this."""
    resolution = hierarchy.resolve("naive T cell")
    assert resolution.node_id != "naive_b_cell", (
        "'naive T cell' resolved to a B-cell node — the fuzzy matcher crossed a lineage"
    )
    assert "t_cell" in resolution.node_id


@pytest.mark.parametrize(
    "label,forbidden",
    [
        ("naive T cell", "naive_b_cell"),
        ("naive T cells", "naive_b_cell"),
        ("memory T cell", "memory_b_cell"),
        ("regulatory T cell", "regulatory_b_cell"),
    ],
)
def test_t_labels_never_resolve_to_b_nodes(hierarchy, label, forbidden):
    assert hierarchy.resolve(label).node_id != forbidden


def test_b_labels_still_resolve_correctly(hierarchy):
    """The guard must not break the labels that were already right."""
    assert hierarchy.resolve("naive B cell").node_id == "naive_b_cell"
    assert hierarchy.resolve("memory B cell").node_id == "memory_b_cell"


# --- the guard itself ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("naive t cell", "t"),
        ("naive b cell", "b"),
        ("nk cell", "nk"),
        ("natural killer cell", "nk"),
        ("classical monocyte", "myeloid"),
        ("plasmacytoid dendritic cell", "myeloid"),
        ("platelet", "erythroid"),
        ("fibroblast", None),
        ("t/nk cells", None),
    ],
)
def test_lineage_detection(text, expected):
    assert _lineage_of(text) == expected


def test_incompatible_lineages_are_rejected():
    assert not _lineage_compatible("naive t cell", "naive b cell")
    assert not _lineage_compatible("t cell", "nk cell")
    assert not _lineage_compatible("b cell", "monocyte")


def test_same_lineage_is_allowed():
    assert _lineage_compatible("naive t cell", "memory t cell")
    assert _lineage_compatible("cd8 t cell", "cd4 t cell")


def test_unmarked_labels_are_permissive():
    """The guard rejects known-wrong matches; it does not demand a declared lineage."""
    assert _lineage_compatible("fibroblast", "fibroblast like")
    assert _lineage_compatible("naive t cell", "some unmarked node")


# --- the pipeline's own subtype vocabulary ---------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("classical monocyte (CD14+)", "classical_monocyte"),
        ("cytotoxic effector CD8+ T cell", "cd8_effector_memory_t_cell"),
        ("CD1C-positive conventional dendritic cell (cDC2)", "conventional_dc2"),
        ("naive T cell", "t_cell"),
    ],
)
def test_pipeline_writes_labels_it_can_read_back(hierarchy, label, expected):
    """A tool must be able to parse its own output.

    These four are written into ``celltype_subtype`` by the pipeline itself, and the
    resolver could not place any of them — 32% of its own subtype calls.
    """
    resolution = hierarchy.resolve(label)
    assert resolution.resolved, f"{label!r} is unresolvable by its own resolver"
    assert resolution.node_id == expected
    assert resolution.confidence >= 0.95, "should be an exact alias, not a fuzzy guess"
