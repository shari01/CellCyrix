"""
test_ontology_export.py — the consensus call must carry a stable identifier, not just a name.

The resolver has always computed ``cl_id``, ``node_id`` and ``canonical_label`` for every
label it places, and the pipeline discarded all three: the consensus reached ``obs`` as a
display string with nothing stable behind it. "Natural killer cell" is a spelling;
``CL:0000623`` is an identity, and it is what lets an exported ``h5ad`` be joined against
CELLxGENE, Azimuth references or the HCA instead of matched on text.

These tests pin the property that matters most about the export — that it never invents a
term. An ontology column that guesses is worse than one that is empty, because a reader
has no way to tell a real mapping from a plausible one.
"""

from __future__ import annotations

import pytest

from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus import tools

ONTOLOGY_COLUMNS = (
    "celltype_cl_id",
    "celltype_ontology_node_id",
    "celltype_canonical_label",
)


def _hierarchy_available() -> bool:
    try:
        from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.cell_hierarchy import (  # noqa: E501
            CellHierarchy,
        )

        CellHierarchy.from_spec()
        return True
    except Exception:  # noqa: BLE001 - availability probe
        return False


requires_hierarchy = pytest.mark.skipif(
    not _hierarchy_available(), reason="cell hierarchy spec unavailable"
)


def test_returns_exactly_the_three_expected_columns():
    """The shape of the return value is part of the contract with `broadcast_and_validate`."""
    maps = tools.ontology_maps({"0": "T cell"})
    assert set(maps) == set(ONTOLOGY_COLUMNS)


def test_every_cluster_appears_in_every_map():
    """One value per cluster per column, so broadcasting cannot drop a cluster."""
    labels = {"0": "T cell", "1": "B cell", "2": "Fibroblast"}
    maps = tools.ontology_maps(labels)
    for column, mapping in maps.items():
        assert set(mapping) == set(labels), f"{column} does not cover every cluster"


@requires_hierarchy
def test_known_types_get_real_cell_ontology_ids():
    """Well-known types resolve to their actual CL terms."""
    maps = tools.ontology_maps(
        {"0": "T cell", "1": "Natural killer cell", "2": "Fibroblast"}
    )
    cl_ids = maps["celltype_cl_id"]
    assert cl_ids["0"] == "CL:0000084"
    assert cl_ids["1"] == "CL:0000623"
    assert cl_ids["2"] == "CL:0000057"


@requires_hierarchy
def test_synonyms_of_the_same_node_reach_the_same_identifier():
    """Different spellings that resolve to ONE node must share the id.

    This is the point of the export: "NK cells" and "Natural killer cell" are the same
    call written two ways, and both must land on ``CL:0000623``.
    """
    maps = tools.ontology_maps({"0": "NK cells", "1": "Natural killer cell"})
    assert set(maps["celltype_cl_id"].values()) == {"CL:0000623"}


@requires_hierarchy
def test_a_finer_node_without_a_cl_term_does_not_inherit_its_parents():
    """A subtype with no CL term of its own must report "", never the parent's id.

    ``CD16+ NK cells`` resolves to the ``cd56_dim_nk_cell`` node, which carries no CL
    term in the spec — only 202 of the hierarchy's 404 nodes do. Borrowing
    ``CL:0000623`` from the parent NK node would be an ontology claim the tree does not
    make, and would silently merge a subtype with its parent for any consumer joining on
    the id. Empty is the honest answer.
    """
    maps = tools.ontology_maps({"0": "CD16+ NK cells"})
    assert maps["celltype_ontology_node_id"]["0"] == "cd56_dim_nk_cell"
    assert maps["celltype_cl_id"]["0"] == tools.ONTOLOGY_UNRESOLVED
    # The node id and canonical label are still populated, so the call is fully traceable
    # even where no CL term exists.
    assert maps["celltype_canonical_label"]["0"]


@requires_hierarchy
def test_unresolvable_label_is_empty_never_guessed():
    """The property that makes the column trustworthy: no invented terms."""
    maps = tools.ontology_maps({"0": "definitely not a cell type xyzzy"})
    for column in ONTOLOGY_COLUMNS:
        assert maps[column]["0"] == tools.ONTOLOGY_UNRESOLVED, (
            f"{column} produced {maps[column]['0']!r} for an unresolvable label; an "
            f"ontology column must be empty rather than guess"
        )


@requires_hierarchy
def test_blank_and_none_labels_do_not_raise():
    """An empty consensus label is a real state (every voter abstained)."""
    maps = tools.ontology_maps({"0": "", "1": None, "2": "   "})
    for column in ONTOLOGY_COLUMNS:
        assert all(
            value == tools.ONTOLOGY_UNRESOLVED for value in maps[column].values()
        )


@requires_hierarchy
def test_canonical_label_is_the_resolver_spelling_not_the_input():
    """The canonical column normalises spelling; the original label is left alone."""
    maps = tools.ontology_maps({"0": "t cells"})
    assert maps["celltype_canonical_label"]["0"] == "T cell"


@requires_hierarchy
def test_node_id_is_populated_alongside_cl_id():
    """The internal node id travels too, so a run can be traced back to the tree."""
    maps = tools.ontology_maps({"0": "Fibroblast"})
    assert maps["celltype_ontology_node_id"]["0"] == "fibroblast"


def test_empty_input_is_empty_output():
    """No clusters is not an error."""
    maps = tools.ontology_maps({})
    assert set(maps) == set(ONTOLOGY_COLUMNS)
    assert all(mapping == {} for mapping in maps.values())
