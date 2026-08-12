"""Tests for the hierarchy spec and the pure-logic resolver."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cell_hierarchy import CellHierarchy, normalize  # noqa: E402
from cell_hierarchy.spec import (  # noqa: E402
    ALIASES,
    COMPARTMENT_LABELS,
    FOREST,  # noqa: E402
    LEVEL_NAMES,
    STATES,
    flat_nodes,
)
from cell_hierarchy.spec._dsl import validate_tree  # noqa: E402


@pytest.fixture(scope="module")
def h() -> CellHierarchy:
    return CellHierarchy.from_spec()


# --------------------------------------------------------------------------- #
# Spec integrity
# --------------------------------------------------------------------------- #


def test_spec_validates():
    nodes = validate_tree(FOREST)
    assert len(nodes) > 300, f"expected a substantial tree, got {len(nodes)}"


def test_node_ids_unique():
    ids = [n.node_id for n in flat_nodes()]
    assert len(ids) == len(set(ids))


def test_every_parent_exists():
    nodes = {n.node_id: n for n in flat_nodes()}
    for node in nodes.values():
        if node.parent_id:
            assert node.parent_id in nodes, f"{node.node_id} has orphan parent"


def test_level_matches_path_depth():
    for node in flat_nodes():
        assert node.level == len(node.path_ids) - 1


def test_aliases_reference_real_nodes():
    ids = {n.node_id for n in flat_nodes()}
    for node_id in ALIASES:
        assert node_id in ids, f"alias table references missing node {node_id!r}"
    for label, node_id in COMPARTMENT_LABELS.items():
        assert node_id in ids, (
            f"compartment {label!r} references missing node {node_id!r}"
        )


def test_no_disease_strings_in_spec():
    """Disease-agnosticism is a hard requirement, so assert it mechanically."""
    forbidden = [
        "cancer of",
        "carcinoma of",
        "ipf",
        "idiopathic pulmonary",
        "ulcerative colitis",
        "crohn",
        "lupus",
        "sle",
        "asthma",
        "copd",
        "diabetes",
        "alzheimer",
        "parkinson",
        "covid",
        "sepsis",
        "lusc",
        "luad",
        "melanoma",
        "leukemia",
        "leukaemia",
        "lymphoma",
        "aml",
        "ckd",
        "colitis",
        "fibrosis",
        "cirrhosis",
        "psoriasis",
        "arthritis",
    ]
    spec_dir = Path(__file__).resolve().parents[1] / "cell_hierarchy" / "spec"
    offences = []
    for path in sorted(spec_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for term in forbidden:
            for match in re.finditer(
                r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text
            ):
                line = text[: match.start()].count("\n") + 1
                offences.append(f"{path.name}:{line} contains {term!r}")
    assert not offences, "disease terms found in spec:\n" + "\n".join(offences)


def test_states_are_not_nodes():
    """A state must never also exist as an identity node."""
    ids = {n.node_id for n in flat_nodes()}
    for state_id in STATES:
        assert state_id not in ids, f"{state_id} is both a state and a node"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a,b",
    [
        ("CD4+ T cells", "cd4+ t cell"),
        ("CD4-positive T cells", "cd4+ t cell"),
        ("T-cells", "t cell"),
        ("T_cells", "t cell"),
        ("Tcells", "t cell"),
        ("NK cells", "nk cell"),
        ("Natural killer cells", "nk cell"),
        ("Fibroblasts", "fibroblast"),
        ("Macrophages", "macrophage"),
    ],
)
def test_normalisation_converges(a, b):
    assert normalize(a).normalized == b, normalize(a).normalized


def test_unicode_and_greek_handled():
    assert (
        normalize("γδ T cells").normalized
        == normalize("gamma-delta T cells").normalized
    )
    assert normalize("Müller cells").normalized == normalize("Muller cells").normalized
    assert (
        normalize("Naïve B cells").normalized == normalize("Naive B cells").normalized
    )


def test_state_extraction():
    n = normalize("Cycling T cells")
    assert "cycling" in n.states
    assert n.residual == "t cell"

    n2 = normalize("Malignant epithelial cells")
    assert "malignant" in n2.states
    assert "epithelial" in n2.residual


def test_protected_tokens_not_stripped():
    """'memory' and 'naive' are identity-bearing here, not states."""
    n = normalize("Memory B cells")
    assert "memory_state" not in n.states
    assert "memory" in n.residual


def test_state_only_labels_yield_state_and_no_identity():
    """'Cycling' and 'Doublet' name a state, not a cell. Say so; don't guess."""
    for label in ["Doublet", "Cycling", "Malignant", "Proliferating", "Low quality"]:
        n = normalize(label)
        assert n.states, f"{label!r} produced no state"
        assert n.residual == "", f"{label!r} left residual {n.residual!r}"


def test_state_only_labels_abstain_on_identity(h):
    from cell_hierarchy.resolver import MATCH_STATE_ONLY

    for label in ["Cycling", "Proliferating"]:
        res = h.resolve(label)
        assert res.match_method == MATCH_STATE_ONLY
        assert not res.resolved
        assert res.needs_review
        assert res.states

    doublet = h.resolve("Doublet")
    assert doublet.node_id == "technical_artefact"
    assert not doublet.resolved


def test_unassigned_subtree_never_counts_as_resolved(h):
    for label in ["Malignant cells", "Tumour cells", "Unknown", "Unassigned"]:
        res = h.resolve(label)
        assert not res.resolved, f"{label!r} wrongly counted as a resolved vote"
        assert res.needs_review


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,source,expected",
    [
        ("CD16+ NK cells", "celltypist_immune", "cd56_dim_nk_cell"),
        ("CD16- NK cells", "celltypist_immune", "cd56_bright_nk_cell"),
        ("CD14 Mono", "azimuth_pbmc", "classical_monocyte"),
        ("CD16 Mono", "azimuth_pbmc", "non_classical_monocyte"),
        ("Tregs", "singler_blueprint", "regulatory_t_cell"),
        ("AT2", "azimuth_lung", "alveolar_type_2_cell"),
        ("Alveolar Macrophages", "azimuth_lung", "alveolar_macrophage"),
        ("BEST4+ epithelial", "celltypist_intestine", "best4_epithelial_cell"),
        ("Paneth cells", "celltypist_intestine", "paneth_cell"),
        ("Podocytes", "panglaodb", "podocyte"),
        ("Kupffer cells", "panglaodb", "kupffer_cell"),
        ("pDC", "azimuth_pbmc", "plasmacytoid_dc"),
        ("cDC1", "azimuth_pbmc", "conventional_dc1"),
    ],
)
def test_known_aliases_resolve(h, label, source, expected):
    res = h.resolve(label, source=source)
    assert res.node_id == expected, f"{label!r} -> {res.node_id} ({res.match_method})"
    assert res.resolved


def test_resolution_by_node_id_and_cl_id(h):
    assert h.resolve("alveolar_type_2_cell").node_id == "alveolar_type_2_cell"
    assert h.resolve("CL:0000084").node_id == "t_cell"


def test_unseen_label_resolves_or_flags(h):
    res = h.resolve("Wibbly wobbly cells of Malik")
    assert res.node_id == "unknown_cell"
    assert not res.resolved
    assert res.needs_review
    assert res.confidence == 0.0


def test_compartment_label_resolves_coarse(h):
    res = h.resolve("Immune cells")
    assert res.node_id == "hematopoietic"
    assert res.needs_review, "coarse compartment matches should be flagged"


def test_malignant_label_does_not_become_a_lineage(h):
    """Malignancy is a state; it must not silently pick an identity."""
    res = h.resolve("Malignant cells")
    assert "malignant" in res.states
    assert res.lineage in {"", "Unassigned / not resolvable to a lineage"}


def test_state_travels_separately_from_identity(h):
    res = h.resolve("Cycling T cells")
    assert res.node_id in {"t_cell", "proliferating_t_cell"}
    assert "cycling" in res.states


def test_total_in_equals_total_out(h):
    labels = ["T cells", "", "NK cells", "utter nonsense", None, "AT2", "Doublet"]
    out = h.resolve_many(labels)
    assert len(out) == len(labels)
    assert all(r.node_id for r in out)


def test_resolution_is_deterministic(h):
    labels = ["T cells", "Fibroblasts", "Whatsit cells", "Goblet cells"]
    first = [r.node_id for r in h.resolve_many(labels)]
    second = [r.node_id for r in h.resolve_many(labels)]
    assert first == second


def test_tissue_only_breaks_ties_never_overrides(h):
    """A confident match must be identical with and without a tissue hint."""
    for label in ["AT2", "Paneth cells", "CD16+ NK cells", "Podocytes"]:
        assert h.resolve(label).node_id == h.resolve(label, tissue="lung").node_id


def test_ambiguous_goblet_label_is_stable(h):
    """'Goblet cells' exists in airway and intestine; tissue disambiguates."""
    plain = h.resolve("Goblet cells")
    lung = h.resolve("Goblet cells", tissue="lung")
    gut = h.resolve("Goblet cells", tissue="colon")
    assert plain.resolved
    assert lung.node_id == "goblet_cell_airway"
    assert gut.node_id == "goblet_cell_intestine"


# --------------------------------------------------------------------------- #
# Tree operations
# --------------------------------------------------------------------------- #


def test_rollup(h):
    assert h.rollup("cd8_effector_memory_t_cell", 0) == "hematopoietic"
    assert h.rollup("cd8_effector_memory_t_cell", 2) == "t_cell"
    assert h.rollup("cd8_effector_memory_t_cell", 3) == "cd8_t_cell"
    assert h.rollup("hematopoietic", 3) is None


def test_lca_within_lineage(h):
    assert (
        h.lowest_common_ancestor(["cd4_naive_t_cell", "cd8_naive_t_cell"]) == "t_cell"
    )
    assert h.lowest_common_ancestor(["t_cell", "nk_cell"]) == "lymphoid_cell"
    assert h.lowest_common_ancestor(["t_cell", "macrophage"]) == "hematopoietic"


def test_lca_across_lineages_is_none(h):
    assert h.lowest_common_ancestor(["t_cell", "alveolar_type_2_cell"]) is None


def test_lca_of_ancestor_and_descendant(h):
    assert h.lowest_common_ancestor(["t_cell", "cd8_temra_cell"]) == "t_cell"


def test_subtree_contains_self_and_descendants(h):
    sub = h.subtree("nk_cell")
    assert "nk_cell" in sub
    assert "cd56_dim_nk_cell" in sub
    assert "t_cell" not in sub


def test_level_labels_flatten_correctly(h):
    labels = h.level_labels("cd8_effector_memory_t_cell")
    assert labels[LEVEL_NAMES[0]] == "Haematopoietic cell"
    assert labels[LEVEL_NAMES[2]] == "T cell"
    assert labels[LEVEL_NAMES[3]] == "CD8-positive T cell"


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #


def test_consensus_exact_agreement(h):
    c = h.consensus(
        {
            "celltypist_immune": "Regulatory T cells",
            "singler_blueprint": "Tregs",
            "azimuth_pbmc": "Treg",
        }
    )
    assert c.consensus_node_id == "regulatory_t_cell"
    assert c.exact_agreement
    assert c.agreement_score > 0.8
    assert c.dissenting_sources == ()


def test_consensus_different_granularity_is_agreement(h):
    """Coarse and fine voters on one path agree at the coarse depth."""
    c = h.consensus(
        {
            "celltypist_immune": "CD16+ NK cells",
            "singler_blueprint": "NK cells",
        }
    )
    assert c.consensus_node_id == "nk_cell"
    assert not c.exact_agreement
    assert c.dissenting_sources == ()


def test_consensus_conflict_rolls_up(h):
    c = h.consensus(
        {
            "celltypist_immune": "CD8+ T cells",
            "singler_blueprint": "NK cells",
        }
    )
    assert c.consensus_node_id == "lymphoid_cell"
    assert c.agreement_score < 0.6


def test_consensus_disjoint_lineages_has_no_ancestor(h):
    c = h.consensus(
        {
            "celltypist_lung": "AT2",
            "singler_blueprint": "CD8+ T-cells",
        }
    )
    assert c.consensus_node_id == "unknown_cell"
    assert "disjoint" in c.note


def test_abstention_depresses_score(h):
    full = h.consensus(
        {
            "celltypist_immune": "Regulatory T cells",
            "singler_blueprint": "Tregs",
            "azimuth_pbmc": "Treg",
        }
    )
    partial = h.consensus(
        {
            "celltypist_immune": "Regulatory T cells",
            "singler_blueprint": "Tregs",
            "azimuth_pbmc": "gibberish label",
        }
    )
    assert partial.agreement_score < full.agreement_score
    assert partial.n_voters == 3
    assert partial.n_resolved == 2


def test_consensus_counts_all_voters(h):
    votes = {"a": "T cells", "b": "nonsense", "c": "NK cells"}
    c = h.consensus(votes)
    assert c.n_voters == len(votes)
    assert len(c.resolutions) == len(votes)


def test_consensus_below_min_voters(h):
    c = h.consensus({"a": "T cells", "b": "nonsense"}, min_voters=2)
    assert c.n_resolved == 1
    assert c.agreement_score == 0.0
    assert "min_voters" in c.note


def test_consensus_states_are_collected(h):
    c = h.consensus(
        {
            "celltypist_immune": "Cycling T cells",
            "singler_blueprint": "CD4+ T-cells",
        }
    )
    assert "cycling" in c.states


# --------------------------------------------------------------------------- #
# CSV round trip
# --------------------------------------------------------------------------- #


def test_csv_round_trip(tmp_path, h):
    from cell_hierarchy.build_reference import main as build_main

    build_main(["--outdir", str(tmp_path)])
    reloaded = CellHierarchy.from_csv(tmp_path)
    assert set(reloaded.nodes) == set(h.nodes)
    for label in ["CD16+ NK cells", "AT2", "Paneth cells", "Tregs", "Fibroblasts"]:
        assert reloaded.resolve(label).node_id == h.resolve(label).node_id
    assert reloaded.lowest_common_ancestor(["t_cell", "nk_cell"]) == "lymphoid_cell"
