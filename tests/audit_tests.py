"""
audit_tests.py — non-invasive scientific-correctness audit for the single-cell pipeline.

This suite does NOT import or modify the production run path. It imports the real
pipeline modules and drives their pure/offline functions against SMALL SYNTHETIC
AnnData objects with KNOWN biology, then asserts the results are biologically and
statistically sensible — and, importantly, that unsupported labels are rejected.

It is deliberately runnable with no network, no R, no CellTypist model, and no LLM
key. Tests whose subject genuinely requires those live dependencies are marked with
`skipUnless` so the suite stays green in a bare environment while still exercising
every offline-testable claim.

Run it with the sibling `run_audit_suite.py` (no CLI/argparse by convention):

    python tests/run_audit_suite.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Locate + import the real pipeline package (repo root holds agentic_ai_wf/)
# --------------------------------------------------------------------------- #
def _repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "agentic_ai_wf").is_dir():
            return p
    raise RuntimeError("repo root containing 'agentic_ai_wf' not found")


REPO_ROOT = _repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PKG = "agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x"


def _imp(mod: str):
    from importlib import import_module

    return import_module(f"{PKG}.{mod}")


# --------------------------------------------------------------------------- #
#  Synthetic AnnData with KNOWN, planted lineage biology
# --------------------------------------------------------------------------- #
# Canonical identity genes per synthetic cluster. These overlap the pipeline's
# LINEAGE_MARKERS so the lineage gate and marker detection have real signal.
_IDENTITY = {
    "0_T": ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "IL7R", "LCK", "PTPRC"],
    "1_B": ["MS4A1", "CD79A", "CD79B", "CD19", "CD74", "PTPRC"],
    "2_NK": ["NKG7", "GNLY", "KLRD1", "NCAM1", "PRF1", "PTPRC"],
    "3_Mono": ["LYZ", "CD14", "CD68", "FCGR3A", "AIF1", "PTPRC"],
    "4_Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7"],
    "5_Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "COL3A1"],
    "6_Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "FLT1"],
    "7_Doublet_TB": ["CD3D", "CD3E", "MS4A1", "CD79A", "PTPRC"],  # mixed T + B
    "8_LowQual_Mito": ["MT-CO1", "MT-ND1", "MT-ATP6", "MT-CYB"],  # dying cells
    "9_Housekeeping": ["ACTB", "GAPDH", "MALAT1", "RPL13", "RPS6"],  # no identity
}

# Extra genes so score_genes' expression binning has room to work.
_FILLER = [f"FILLER{i}" for i in range(20)]


def _all_genes():
    seen, genes = set(), []
    for gs in _IDENTITY.values():
        for g in gs:
            if g not in seen:
                seen.add(g)
                genes.append(g)
    for g in _FILLER:
        if g not in seen:
            seen.add(g)
            genes.append(g)
    return genes


def build_synth_adata(cells_per_cluster: int = 40, seed: int = 0):
    """Return an AnnData with integer counts in .X and layers['counts'], obs['leiden'],
    and obs['sample']/obs['group'] for pseudobulk tests. Each cluster strongly expresses
    only its identity genes over a low background."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    genes = _all_genes()
    g_index = {g: i for i, g in enumerate(genes)}
    clusters = list(_IDENTITY.keys())

    n = cells_per_cluster * len(clusters)
    X = rng.poisson(0.3, size=(n, len(genes))).astype(np.float64)  # low background

    leiden, samples, groups = [], [], []
    row = 0
    for ci, cl in enumerate(clusters):
        cl_id = cl.split("_")[0]  # "0".."9"
        for k in range(cells_per_cluster):
            for g in _IDENTITY[cl]:
                # strong, noisy identity signal
                X[row, g_index[g]] = float(rng.integers(25, 60))
            leiden.append(cl_id)
            # two samples per group so pseudobulk has replication; groups A/B
            grp = "CASE" if (ci % 2 == 0) else "CONTROL"
            smp = f"{grp}_s{(k % 2) + 1}"
            samples.append(smp)
            groups.append(grp)
            row += 1

    import pandas as pd

    obs = pd.DataFrame(
        {"leiden": leiden, "sample": samples, "group": groups},
        index=[f"cell{i}" for i in range(n)],
    )
    obs["leiden"] = obs["leiden"].astype("category")
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


# =========================================================================== #
#  1. Data-layer integrity / raw-count preservation
# =========================================================================== #
class TestDataLayerIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def test_counts_layer_is_integer_and_matches_X(self):
        a = build_synth_adata()
        c = a.layers["counts"]
        self.assertTrue(
            np.allclose(c, np.round(c)), "counts layer must be integer-valued"
        )
        self.assertTrue(np.array_equal(np.asarray(a.X), np.asarray(c)))

    def test_get_lognorm_recovers_lognorm_from_corrupted_raw(self):
        """Reproduce the scale-corruption trap: raw=adata then in-place scale.
        get_lognorm must re-derive log-norm from the untouched counts layer."""
        import scanpy as sc

        a = build_synth_adata()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.raw = a  # shares buffer with .X
        sc.pp.scale(a, max_value=10)  # corrupts .X (and raw.X via shared buffer)
        ln, src = self.tools.get_lognorm(a)
        self.assertEqual(src, "counts-layer")
        mx = float(np.nanmax(ln.X.toarray() if hasattr(ln.X, "toarray") else ln.X))
        self.assertLess(
            mx, 50.0, "recovered matrix should be log-normalized, not scaled"
        )


# =========================================================================== #
#  2. Lineage gating — positive + negative biological controls
# =========================================================================== #
class TestLineageGating(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")
        a = build_synth_adata()
        cls.ln, _ = cls.tools.get_lognorm(a)
        cls.gate = cls.tools.lineage_gate_per_cluster(cls.ln, cluster_col="leiden")

    def test_positive_controls(self):
        g = self.gate
        self.assertEqual(g["0"], "Immune")  # T cell
        self.assertEqual(g["1"], "Immune")  # B cell
        self.assertEqual(g["2"], "Immune")  # NK
        self.assertEqual(g["3"], "Immune")  # monocyte
        self.assertEqual(g["4"], "Epithelial")  # epithelial
        self.assertEqual(g["5"], "Fibroblast")  # fibroblast
        self.assertEqual(g["6"], "Endothelial")  # endothelial

    def test_ptprc_negative_epithelial_is_not_immune(self):
        # Epithelial cluster has no PTPRC — the CD45 gate must keep it non-immune.
        self.assertNotEqual(self.gate["4"], "Immune")

    # --- regression: the gate must ABSTAIN, never invent a lineage from noise ---
    #
    # Mast cells and dendritic cells have no panel in LINEAGE_MARKERS, and both
    # drop PTPRC in droplet data. Before the fix, such a cluster scored ~0 on every
    # panel, was ejected from Immune by the CD45 test, and then took the argmax of
    # five near-zero scores — landing on "Epithelial" with 0% EPCAM. Downstream that
    # became a phantom epithelial compartment in the Bisque export and in pseudobulk
    # DE. The gate must now return "Other" so the voters decide.
    def _gate_for(self, marker_genes):
        """Gate call for one cluster expressing only `marker_genes` (no PTPRC)."""
        import anndata as ad
        import pandas as pd

        rng = np.random.default_rng(0)
        genes = list(dict.fromkeys(_all_genes() + list(marker_genes)))
        idx = {g: i for i, g in enumerate(genes)}
        n = 60
        X = rng.poisson(0.3, size=(n, len(genes))).astype(np.float64)
        for g in marker_genes:
            X[:, idx[g]] = rng.integers(25, 60, size=n).astype(np.float64)
        if "PTPRC" in idx:
            X[:, idx["PTPRC"]] = 0.0  # CD45 dropout, as in real mast/DC data
        obs = pd.DataFrame({"leiden": ["0"] * n}, index=[f"c{i}" for i in range(n)])
        obs["leiden"] = obs["leiden"].astype("category")
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
        a.layers["counts"] = X.copy()
        ln, _ = self.tools.get_lognorm(a)
        return self.tools.lineage_gate_per_cluster(ln, cluster_col="leiden")["0"]

    # The gate is a SAFETY NET, not a primary annotator: it exists to catch voters
    # that contradict pan-lineage biology. Its contract is "confirm or abstain,
    # never invent" — a wrong non-immune call on an immune cluster is exactly the
    # failure that produced the phantom epithelial compartment.
    _NEVER_WRONG = ("Other", "Immune")

    def test_mast_cell_cluster_is_never_called_epithelial(self):
        call = self._gate_for(
            ["TPSAB1", "TPSB2", "TPSD1", "CPA3", "CTSG", "HDC", "HPGDS"]
        )
        self.assertIn(
            call,
            self._NEVER_WRONG,
            f"mast-cell markers must confirm Immune or abstain, got {call!r}",
        )

    def test_dendritic_cell_cluster_is_never_called_epithelial(self):
        call = self._gate_for(["LAMP3", "IDO1", "IL4I1", "CSF2RA", "MS4A7", "PILRA"])
        self.assertIn(
            call,
            self._NEVER_WRONG,
            f"dendritic-cell markers must confirm Immune or abstain, got {call!r}",
        )

    def test_canonical_myeloid_is_positively_confirmed(self):
        # Genes that ARE in the panel must still produce an active call — abstention
        # must not quietly become the answer to everything.
        self.assertEqual(self._gate_for(["LILRA4", "IRF8", "CD68", "LYZ"]), "Immune")

    def test_epithelial_markers_still_confirm_epithelial(self):
        self.assertEqual(
            self._gate_for(["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"]), "Epithelial"
        )

    def test_gate_abstains_when_nothing_is_expressed(self):
        # Pure background: no lineage has evidence, so the gate must not guess.
        self.assertEqual(self._gate_for([]), "Other")

    def test_real_lineages_still_score_far_above_the_floor(self):
        # The abstain floor must not silence genuine calls (regression guard for
        # MIN_LINEAGE_SCORE being set too high).
        self.assertEqual(self.gate["0"], "Immune")
        self.assertEqual(self.gate["4"], "Epithelial")
        self.assertEqual(self.tools.MIN_LINEAGE_SCORE, 0.1)


# =========================================================================== #
#  2b. Lineage panels are derived from the curated reference, not hand-typed
# =========================================================================== #
class TestLineagePanelProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def test_panels_are_built_from_reference_data(self):
        prov = self.tools.LINEAGE_MARKERS_PROVENANCE
        self.assertEqual(
            prov.get("source"),
            "TIS_CELL_markers_v3 + cell_hierarchy",
            f"panels fell back to built-ins: {prov.get('reason')!r}",
        )

    def test_immune_panel_covers_mast_and_dendritic_cells(self):
        # The exact gap that caused the bug: no mast/DC gene existed in any panel.
        immune = set(self.tools.LINEAGE_MARKERS["Immune"])
        self.assertTrue({"TPSAB1", "TPSB2", "CPA3"} <= immune, "mast markers missing")
        self.assertTrue(
            immune & {"LILRA4", "IRF8", "FLT3"}, "dendritic markers missing"
        )

    def test_mural_panel_excludes_cardiomyocyte_genes(self):
        # A myocyte is not a mural cell — only pericytes and vascular smooth muscle.
        mural = set(self.tools.LINEAGE_MARKERS["Mural"])
        self.assertFalse(mural & {"NPPA", "MYL2", "RYR2", "MYH7", "TNNT2"})
        self.assertTrue({"RGS5", "ACTA2"} <= mural, "pericyte/VSMC markers lost")

    def test_builtin_panels_are_never_lost(self):
        # Reference-derived panels must be a superset of the historical ones, so the
        # gate can never detect LESS than it did before.
        for lin, hand in self.tools.BUILTIN_LINEAGE_MARKERS.items():
            self.assertTrue(
                set(hand) <= set(self.tools.LINEAGE_MARKERS[lin]),
                f"{lin}: built-in markers dropped",
            )

    def test_no_gene_claims_two_lineages(self):
        seen = {}
        for lin, genes in self.tools.LINEAGE_MARKERS.items():
            for g in genes:
                if g in seen and seen[g] != lin:
                    # built-ins may legitimately overlap; derived genes must not
                    hand = self.tools.BUILTIN_LINEAGE_MARKERS
                    if not any(g in hand[k] for k in hand):
                        self.fail(f"{g!r} claimed by both {seen[g]} and {lin}")
                seen[g] = lin


# =========================================================================== #
#  3. Cluster-marker quality — planted markers must surface
# =========================================================================== #
class TestMarkerDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def test_planted_markers_rank_top(self):
        a = build_synth_adata()
        ln, _ = self.tools.get_lognorm(a)
        markers, empty = self.tools.compute_cluster_markers(
            ln, cluster_col="leiden", top_n=15
        )
        # T-cell cluster "0" should surface CD3D/CD3E; epithelial "4" EPCAM/keratins.
        self.assertTrue({"CD3D", "CD3E"} & set(markers["0"]), markers["0"])
        self.assertTrue(
            {"EPCAM", "KRT8", "KRT18", "KRT19"} & set(markers["4"]), markers["4"]
        )
        self.assertTrue({"COL1A1", "COL1A2", "DCN"} & set(markers["5"]), markers["5"])

    def test_housekeeping_cluster_has_no_identity_markers(self):
        # Cluster 9 expresses only ACTB/GAPDH/MALAT1 — it must not surface a lineage
        # identity gene as a top marker (guards against ambient-driven identity).
        a = build_synth_adata()
        ln, _ = self.tools.get_lognorm(a)
        markers, _ = self.tools.compute_cluster_markers(
            ln, cluster_col="leiden", top_n=10
        )
        identity = {"CD3D", "EPCAM", "MS4A1", "COL1A1", "PECAM1", "NKG7", "LYZ"}
        self.assertFalse(identity & set(markers["9"]), markers["9"])


# =========================================================================== #
#  4. Harmonization — parent/subtype, lineage compatibility, regex safety
# =========================================================================== #
class TestHarmonization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def test_subtypes_collapse_to_parent(self):
        # Labels now resolve through the cell_hierarchy, which is finer-grained than
        # the old keyword table: a monocyte and a macrophage are distinct cells and
        # no longer share one "Monocyte/Macrophage" node. They still meet at the
        # 'Myeloid cell' class, so hierarchy-level agreement is preserved.
        h = self.tools.harmonize_label
        self.assertEqual(h("Classical monocytes"), "Monocyte")
        self.assertEqual(h("CD8-positive, alpha-beta T cell"), "T cell")
        self.assertEqual(h("Cytotoxic T cell"), "T cell")  # NOT NK
        self.assertEqual(h("Memory B cells"), "B cell")

    def test_monocyte_and_macrophage_share_a_parent(self):
        # The precision above must not fragment the consensus: the two labels have
        # to remain reconcilable one level up.
        hier = self.tools._hierarchy()  # the same instance the code uses
        self.assertIsNotNone(hier, "cell_hierarchy resolver is not available")
        lca = hier.lowest_common_ancestor(
            [
                hier.resolve("Classical monocytes").node_id,
                hier.resolve("Macrophage").node_id,
            ]
        )
        self.assertIsNotNone(lca, "monocyte and macrophage must share an ancestor")

    def test_carcinoma_epithelial_but_generic_malignant_is_other(self):
        h = self.tools.harmonize_label
        self.assertEqual(h("carcinoma cell"), "Epithelial cell")
        self.assertEqual(h("lung adenocarcinoma"), "Epithelial cell")
        self.assertTrue(h("malignant cell").startswith("Other"))

    def test_numeric_marker_tokens_do_not_collide(self):
        h = self.tools.harmonize_label
        self.assertEqual(h("T cells"), "T cell")  # plural OK
        self.assertTrue(h("CD45+ immune cell").startswith("Other"))  # cd4 !~ cd45
        self.assertEqual(h("CD40+ B cell"), "B cell")  # cd4 !~ cd40

    def test_cardiomyocyte_is_not_mural(self):
        self.assertEqual(self.tools.coarse_lineage_of("cardiomyocyte"), "Other")

    def test_underscore_normalized_and_idempotent(self):
        h = self.tools.harmonize_label
        # A mast cell is now named as itself rather than collapsed into the
        # catch-all "Granulocyte" it shared with neutrophils/eosinophils/basophils.
        # This is the label that used to be overwritten with "Epithelial cell".
        self.assertEqual(h("Mast_cell"), "Mast cell")  # underscore -> "mast cell"
        self.assertEqual(h("NK_cell"), "Natural killer cell")
        # already-Othered label must not become "Other: Other: ..."
        self.assertEqual(h("Other: Mast_cell"), "Other: Mast_cell")

    def test_spelling_variants_collapse_to_one_node(self):
        # The failure that started this: three voters, three spellings, counted as
        # three different answers and therefore as a disagreement.
        h = self.tools.harmonize_label
        self.assertEqual(
            {h("Mast_cell"), h("Mast cell"), h("mast cell")}, {"Mast cell"}
        )
        self.assertEqual(
            {h("CD16+ NK cells"), h("NK cells"), h("NK")}, {"Natural killer cell"}
        )

    def test_celltypist_skin_model_labels_all_resolve(self):
        # Adult_Human_Skin.pkl is what the pipeline auto-selects for skin data.
        # An unresolved label abstains from the vote, so coverage gaps silently
        # discard a voter — 22 of these 34 were unresolved before the crosswalk
        # entry was added.
        h = self.tools.harmonize_label
        skin = [
            "DC1",
            "DC2",
            "Differentiated_KC",
            "F1",
            "F2",
            "F3",
            "ILC1_3",
            "ILC1_NK",
            "ILC2",
            "Inf_mac",
            "LC",
            "LE1",
            "LE2",
            "Macro_1",
            "Macro_2",
            "Mast_cell",
            "Melanocyte",
            "MigDC",
            "Mono_mac",
            "NK",
            "Pericyte_1",
            "Pericyte_2",
            "Plasma",
            "Schwann_1",
            "Schwann_2",
            "Tc",
            "Th",
            "Treg",
            "Undifferentiated_KC",
            "VE1",
            "VE2",
            "VE3",
            "migLC",
            "moDC",
        ]
        unresolved = [s for s in skin if h(s).startswith("Other:")]
        self.assertEqual(unresolved, [], f"unresolved skin labels: {unresolved}")

    def test_immune_labels_map_to_the_immune_lineage(self):
        # The gate compares maj_coarse against its own bucket names, so the
        # hierarchy lineages must land on those exact strings.
        for lbl in ("Mast cell", "Dendritic cell", "T cell", "Macrophage"):
            self.assertEqual(self.tools.coarse_lineage_of(lbl), "Immune", lbl)
        self.assertEqual(self.tools.coarse_lineage_of("Epithelial cell"), "Epithelial")
        self.assertEqual(self.tools.coarse_lineage_of("Pericyte"), "Mural")


# =========================================================================== #
#  4b. Subtype layer — finer label, but never contradicting the consensus
# =========================================================================== #
class TestSubtypeLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cons = _imp("celltype_consensus.consensus")

    def test_subtype_is_consensus_consistent(self):
        pick = self.cons._pick_subtype  # (final, *candidates) in priority order
        # LLM "Cytotoxic T cell" legitimately refines a "T cell" consensus
        self.assertEqual(
            pick("T cell", None, "Cytotoxic T cell", "Tc", None), "Cytotoxic T cell"
        )
        # a contradictory candidate ("DC1" under a "B cell" consensus) is rejected
        self.assertEqual(pick("B cell", None, None, "DC1", None), "B cell")
        # nothing finer than the coarse call -> return the coarse consensus
        self.assertEqual(pick("NK cell", None, "NK cell", None, None), "NK cell")


# =========================================================================== #
#  4c. Consensus plots — PubMed column present + per-method graphs written
# =========================================================================== #
class TestConsensusPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = _imp("celltype_consensus.consensus_plots")

    def _rows(self):
        return [
            {
                "cluster": "0",
                "consensus": "T cell",
                "celltype_subtype": "Cytotoxic T cell",
                "tier": "Medium",
                "celltypist": "Tc",
                "celltypist_conf": 0.8,
                "singler": "T_cells",
                "singler_conf": 0.3,
                "knowledge_based": "T cell",
                "knowledge_based_conf": 0.95,
                "pubmed": "T cell",
                "pubmed_conf": 0.6,
            },
            {
                "cluster": "1",
                "consensus": "B cell",
                "celltype_subtype": "B cell",
                "tier": "Low/Review",
                "celltypist": "DC1",
                "celltypist_conf": 0.5,
                "singler": "T_cells",
                "singler_conf": 0.2,
                "knowledge_based": "B cell",
                "knowledge_based_conf": 0.9,
                "pubmed": "B cell",
                "pubmed_conf": 0.55,
            },
        ]

    def test_agreement_plot_includes_pubmed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.cp.plot_method_agreement(
                self._rows(),
                root,
                "t",
                enable_singler=True,
                enable_llm=True,
                enable_pubmed=True,
            )
            self.assertTrue((root / "t_method_agreement.png").exists())
            self.assertTrue((root / "t_method_agreement.csv").exists())

    def test_per_method_graphs_written_for_each_voter(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.cp.plot_per_method_calls(
                self._rows(),
                root,
                "t",
                enable_celltypist=True,
                enable_singler=True,
                enable_llm=True,
                enable_pubmed=True,
            )
            for m in ("celltypist", "singler", "knowledge_based", "pubmed"):
                self.assertTrue((root / f"t_method_{m}_calls.png").exists(), m)


# =========================================================================== #
#  5. Consensus voting — replication, ties, abstain, mixed-lineage doublet
# =========================================================================== #
class TestConsensusVoting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def test_single_voter_is_not_a_majority(self):
        r = self.tools.tally_votes({"celltypist": "T cell"})
        self.assertFalse(r["has_majority"])  # nothing corroborates one voter
        self.assertFalse(r["unanimous"])

    def test_unanimous_and_majority(self):
        t = self.tools.harmonize_label("T cell")
        b = self.tools.harmonize_label("B cell")
        self.assertTrue(self.tools.tally_votes({"a": t, "b": t})["unanimous"])
        maj = self.tools.tally_votes({"a": t, "b": t, "c": b})
        self.assertTrue(maj["has_majority"])
        self.assertEqual(maj["majority_label"], t)

    def test_mixed_lineage_doublet_does_not_reach_consensus(self):
        # CellTypist says T, LLM says B: a 1-1 split must be tied / no majority
        # (a doublet-like cluster must never be promoted to a confident label).
        r = self.tools.tally_votes(
            {
                "celltypist": self.tools.harmonize_label("T cell"),
                "knowledge_based": self.tools.harmonize_label("B cell"),
            },
            {"celltypist": 0.6, "knowledge_based": 0.6},
        )
        self.assertTrue(r["tied"])
        self.assertFalse(r["has_majority"])
        self.assertFalse(r["unanimous"])

    def test_abstentions_do_not_vote(self):
        r = self.tools.tally_votes({"a": self.tools.UNASSIGNED})
        self.assertEqual(r["n_methods"], 0)
        self.assertEqual(r["majority_label"], self.tools.UNASSIGNED)


# =========================================================================== #
#  6. Confidence calibration + NaN/inf robustness
# =========================================================================== #
class TestConfidenceHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")
        cls.pa = _imp("pubmed_annotation")

    def test_rank_normalize_is_comparable_and_drops_nan(self):
        n = self.tools.normalize_confidences({"0": 0.9, "1": 0.5, "2": 0.7})
        self.assertAlmostEqual(n["0"], 1.0, places=3)
        self.assertNotIn(
            "1",
            self.tools.normalize_confidences({"0": 0.3, "1": float("nan"), "2": 0.9}),
        )
        self.assertEqual(self.tools.normalize_confidences({"5": 0.42}), {"5": 0.5})

    def test_pubmed_uncited_cannot_be_high_confidence(self):
        band, score, review = self.pa.compute_confidence(
            {
                "cell_type": "T cell",
                "supporting_markers": ["CD3D"],
                "pmids": [],
                "confidence": "high",
            },
            n_abstracts=3,
        )
        self.assertNotEqual(band, "high")
        self.assertTrue(review)
        self.assertLessEqual(score, 0.6)

    def test_pubmed_unknown_is_capped_low(self):
        band, score, review = self.pa.compute_confidence(
            {
                "cell_type": "Unknown",
                "supporting_markers": [],
                "pmids": [],
                "confidence": "high",
            },
            n_abstracts=0,
        )
        self.assertEqual(band, "low")
        self.assertTrue(review)
        self.assertLessEqual(score, 0.2)

    def test_pubmed_marker_cleaning_drops_low_info(self):
        cleaned = self.pa.clean_markers(
            ["CD3D", "MT-CO1", "RPL13", "MALAT1", "HBB", "CD3E"],
            species="human",
            top_n=30,
        )
        self.assertIn("CD3D", cleaned)
        for junk in ("MT-CO1", "RPL13", "MALAT1", "HBB"):
            self.assertNotIn(junk, cleaned)


# =========================================================================== #
#  7. Pseudobulk aggregation + sample-level replication + contrast direction
# =========================================================================== #
class TestPseudobulk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pb = _imp("pseudobulk_de")

    def test_sums_raw_counts_per_sample(self):
        a = build_synth_adata(cells_per_cluster=40)
        counts_df, meta_df = self.pb._build_pseudobulk(a, "sample", "group")
        self.assertIsNotNone(counts_df)
        # summed pseudobulk total == raw counts total (conservation)
        self.assertEqual(
            int(counts_df.values.sum()), int(np.asarray(a.layers["counts"]).sum())
        )
        self.assertTrue(np.issubdtype(counts_df.values.dtype, np.integer))
        # each pseudobulk row carries exactly one group label
        self.assertTrue(set(meta_df["group"]).issubset({"CASE", "CONTROL"}))

    def test_min_cells_per_pseudobulk_enforced(self):
        self.assertGreaterEqual(self.pb.MIN_CELLS_PER_PSEUDOBULK, 10)

    def test_contrast_requires_two_samples_per_group(self):
        import pandas as pd

        # one sample per group -> must be rejected, never faked
        counts = pd.DataFrame(
            {"GENE1": [10, 20], "GENE2": [5, 6]}, index=["s_case", "s_ctrl"]
        )
        meta = pd.DataFrame({"group": ["CASE", "CONTROL"]}, index=["s_case", "s_ctrl"])
        # `reference_selection` is keyword-only and required: contrasts.py resolves it
        # once per run and it is stamped onto every DE row, so the audit trail cannot
        # be omitted at the call site.
        res, status = self.pb._deseq_contrast(
            counts,
            meta,
            "CASE",
            "CONTROL",
            reference_selection="explicitly configured (test fixture)",
        )
        self.assertIsNone(res)
        self.assertIn("insufficient replicates", status)

    def test_run_skips_without_replication(self):
        # 2 groups but only 1 sample/group total -> compute_pseudobulk_de returns without error
        a = build_synth_adata(cells_per_cluster=6)
        a.obs["sample"] = [
            "CASE_s1" if g == "CASE" else "CONTROL_s1" for g in a.obs["group"]
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            # should log-and-return (n_samples < 2*MIN_SAMPLES_PER_GROUP), not raise
            self.pb.compute_pseudobulk_de(
                a,
                group_col="group",
                sample_col="sample",
                celltype_col=None,
                out_dir=Path(d),
            )


# =========================================================================== #
#  7b. Cell-level exploratory DE — deterministic direction + caveat stamping
# =========================================================================== #
class TestCellLevelDE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gd = _imp("group_de")

    def _adata_three_groups(self):
        import anndata as ad
        import pandas as pd

        rng = np.random.default_rng(0)
        genes = [f"GENE{i}" for i in range(20)]
        cts, groups, per = ["T cell", "B cell"], ["G1", "G2", "G3"], 6
        rows_ct, rows_grp, X = [], [], []
        for ct in cts:
            for g in groups:
                for _ in range(per):
                    X.append(rng.poisson(3.0, size=len(genes)).astype(float))
                    rows_ct.append(ct)
                    rows_grp.append(g)
        X = np.vstack(X)
        obs = pd.DataFrame(
            {"celltype": rows_ct, "group": rows_grp},
            index=[f"c{i}" for i in range(len(rows_ct))],
        )
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
        a.layers["counts"] = X.copy()
        return a

    def test_all_pairs_deterministic_and_stamped(self):
        import tempfile

        a = self._adata_three_groups()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.gd.compute_de_by_celltype(
                a, celltype_col="celltype", group_col="group", deg_root_dir=root
            )
            deg = root / "celltype_specific_deg"
            # caveat note is written so files can't be mistaken for cohort DE
            self.assertTrue((deg / "_README_cell_level_DE.txt").exists())
            # all 3 group pairs, deterministic focus_vs_ref (ref = alphabetically first)
            tdir = deg / "T_cell"
            names = {p.name for p in tdir.glob("*.csv")}
            for comp in ("G2_vs_G1", "G3_vs_G1", "G3_vs_G2"):
                self.assertTrue(
                    any(comp in n for n in names), f"missing {comp} in {names}"
                )
            # every row is stamped as exploratory / pseudoreplicated
            import pandas as pd

            one = pd.read_csv(sorted(tdir.glob("*.csv"))[0])
            self.assertIn("validity", one.columns)
            self.assertTrue(
                (one["validity"] == "cell_level_exploratory_pseudoreplicated").all()
            )


# =========================================================================== #
#  8. Species / reference safety (human vs mouse)
# =========================================================================== #
class TestReferenceSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cons = _imp("celltype_consensus.consensus")
        cls.cat = _imp("celltype_consensus.celltypist_catalog")

    def test_species_from_taxon(self):
        self.assertEqual(
            self.cons._species_from_taxon({"taxon": "Homo sapiens"}), "human"
        )
        self.assertEqual(
            self.cons._species_from_taxon({"taxon": "Mus musculus"}), "mouse"
        )
        self.assertIsNone(self.cons._species_from_taxon({"taxon": "Danio rerio"}))

    def test_singler_references_are_species_tagged(self):
        refs = {r["reference"]: r["species"] for r in self.cat.SINGLER_REFERENCES}
        self.assertEqual(refs["BlueprintEncodeData"], "human")
        self.assertEqual(
            self.cat.DEFAULT_SINGLER_REFERENCE, "BlueprintEncodeData"
        )  # human default
        # Every reference must carry a species tag, so a mouse reference can never
        # be handed a human run (or vice versa) by accident.
        self.assertTrue(all(r.get("species") for r in self.cat.SINGLER_REFERENCES))

    def test_singler_reference_list_is_human_only_by_design(self):
        """MouseRNAseqData is deliberately absent — see celltypist_catalog.

        SingleR is the only voter with a mouse reference: CellTypist has human
        models only and the lineage-gate panels are human symbols (PTPRC/EPCAM, not
        Ptprc/Epcam), so offering it would make one voter right and the rest of the
        stack wrong on the same run. This test previously asserted the row was
        present and went stale when it was removed.
        """
        refs = {r["reference"]: r["species"] for r in self.cat.SINGLER_REFERENCES}
        self.assertNotIn("MouseRNAseqData", refs)
        self.assertEqual(set(refs.values()), {"human"})
        # The R bridge may still name it; it is simply unreachable while absent here.
        self.assertTrue(set(refs) >= self.cat.VALID_SINGLER_REFERENCES)


# =========================================================================== #
#  9. Reproducibility
# =========================================================================== #
class TestReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rp = _imp("reproducibility")

    def test_seed_makes_numpy_deterministic(self):
        self.rp.set_global_seed(123)
        a = np.random.rand(5)
        self.rp.set_global_seed(123)
        b = np.random.rand(5)
        self.assertTrue(np.array_equal(a, b))


# =========================================================================== #
#  10. Gene-identifier heuristics
# =========================================================================== #
class TestGeneNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gn = _imp("gene_names")

    def test_ensembl_detection(self):
        # looks_like_ensembl takes a SEQUENCE of ids and votes >60% ENSG-prefixed.
        self.assertTrue(
            self.gn.looks_like_ensembl(
                ["ENSG00000141510", "ENSG00000012048", "ENSG00000155657"]
            )
        )
        self.assertFalse(self.gn.looks_like_ensembl(["TP53", "BRCA1", "TTN"]))


# =========================================================================== #
#  11. Pre-flight data validation — safe auto-fix + fail-fast on corruption
# =========================================================================== #
class TestDataValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dv = _imp("data_validation")

    @staticmethod
    def _status(rep, name):
        for c in rep.checks:
            if c.name == name:
                return c.status
        return None

    def _tiny(self, x):
        import anndata as ad
        import pandas as pd

        x = np.asarray(x, dtype=float)
        return ad.AnnData(
            X=x,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(x.shape[0])]),
            var=pd.DataFrame(index=[f"G{j}" for j in range(x.shape[1])]),
        )

    # --- FATAL: corrupt matrices are failed, never silently accepted ---
    def test_normalized_matrix_fails(self):
        a = self._tiny(
            [[0.5, 1.2, 0.0], [3.3, 0.0, 2.7]]
        )  # non-integer -> not raw counts
        _, rep = self.dv.validate_and_fix(a)
        self.assertEqual(self._status(rep, "raw_integer_counts"), "fail")
        self.assertFalse(rep.ok)

    def test_nan_and_negative_fail(self):
        a = self._tiny([[1, 2, np.nan], [3, 4, 5]])
        _, rep = self.dv.validate_and_fix(a)
        self.assertEqual(self._status(rep, "no_nan_inf"), "fail")
        b = self._tiny([[1, -2, 3], [4, 5, 6]])
        _, rep2 = self.dv.validate_and_fix(b)
        self.assertEqual(self._status(rep2, "non_negative"), "fail")

    def test_batch_equals_group_fails(self):
        _, rep = self.dv.validate_and_fix(
            build_synth_adata(cells_per_cluster=4), batch_key="group"
        )
        self.assertEqual(self._status(rep, "batch_not_confounded"), "fail")
        self.assertFalse(rep.ok)

    # --- SAFE FIXES: mechanical issues repaired losslessly ---
    def test_nonunique_barcodes_fixed(self):
        a = build_synth_adata(cells_per_cluster=6)
        names = list(a.obs_names)
        names[1] = names[0]  # inject a duplicate barcode
        a.obs_names = names
        a2, rep = self.dv.validate_and_fix(a)
        self.assertEqual(self._status(rep, "unique_barcodes"), "fixed")
        self.assertTrue(a2.obs_names.is_unique)

    def test_all_zero_gene_dropped(self):
        a = build_synth_adata(cells_per_cluster=6)
        a.X[:, 0] = 0.0
        a.layers["counts"][:, 0] = 0.0  # all-zero in the count source
        n_before = a.n_vars
        a2, rep = self.dv.validate_and_fix(a)
        self.assertEqual(self._status(rep, "drop_zero_genes"), "fixed")
        self.assertEqual(a2.n_vars, n_before - 1)

    def test_ensembl_var_names_renamed_to_symbols(self):
        import anndata as ad
        import pandas as pd

        x = np.random.default_rng(0).poisson(2, size=(8, 4)).astype(float)
        var = pd.DataFrame(
            {"gene_symbol": ["TP53", "EGFR", "CD3D", "MS4A1"]},
            index=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        )
        a = ad.AnnData(
            X=x, obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]), var=var
        )
        a2, rep = self.dv.validate_and_fix(a)
        self.assertEqual(self._status(rep, "gene_symbols"), "fixed")
        self.assertIn("TP53", list(a2.var_names))

    # --- WARN: real but non-corrupting limitations do not block ---
    def test_insufficient_replication_warns_not_fails(self):
        a = build_synth_adata(cells_per_cluster=6)
        a.obs["sample"] = [
            "CASE_s1" if g == "CASE" else "CONTROL_s1"
            for g in a.obs["group"].astype(str)
        ]
        _, rep = self.dv.validate_and_fix(a)
        self.assertEqual(self._status(rep, "pseudobulk_replication"), "warn")
        self.assertTrue(rep.ok)  # WARN must not fail the run

    def test_clean_data_passes(self):
        _, rep = self.dv.validate_and_fix(build_synth_adata(cells_per_cluster=8))
        self.assertTrue(rep.ok)
        self.assertEqual(self._status(rep, "raw_integer_counts"), "pass")


# =========================================================================== #
#  10. Clustering configuration — resolution is config-driven, default 0.5,
#      and the optional audit never clobbers the primary partition
# =========================================================================== #
def _synth_clustered(cells_per_cluster: int = 30, seed: int = 0):
    """Synthetic AnnData with PCA + a neighbour graph, ready for Leiden."""
    import scanpy as sc

    a = build_synth_adata(cells_per_cluster=cells_per_cluster, seed=seed)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=10, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=10, n_pcs=10, random_state=seed)
    return a


class TestClusteringConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = _imp("clustering")

    # --- requirement: default resolution stays 0.5 ---
    def test_default_resolution_is_point_five(self):
        self.assertEqual(self.cl.DEFAULT_LEIDEN_RESOLUTION, 0.5)
        # an absent/None/empty config value must resolve to the historical default
        for missing in (None, "", "   "):
            self.assertEqual(self.cl.resolve_leiden_resolution(missing), 0.5)

    # --- requirement: resolution is read from config, validated as positive numeric ---
    def test_resolution_read_from_config_and_validated(self):
        self.assertEqual(self.cl.resolve_leiden_resolution(1.2), 1.2)
        self.assertEqual(self.cl.resolve_leiden_resolution("0.8"), 0.8)  # YAML string
        for bad in (0, -1, "abc", float("nan"), float("inf"), True):
            with self.assertRaises(self.cl.ClusteringConfigError, msg=repr(bad)):
                self.cl.resolve_leiden_resolution(bad)

    def test_resolution_column_names(self):
        self.assertEqual(self.cl.resolution_column(0.2), "leiden_res_0_2")
        self.assertEqual(self.cl.resolution_column(0.5), "leiden_res_0_5")
        self.assertEqual(self.cl.resolution_column(1.0), "leiden_res_1_0")
        self.assertEqual(self.cl.resolution_column(0.25), "leiden_res_0_25")

    def test_candidates_always_include_primary_and_are_sorted(self):
        got = self.cl.resolve_resolution_candidates([0.8, 0.2, 0.2], primary=0.5)
        self.assertEqual(got, [0.2, 0.5, 0.8])  # de-duplicated + primary added

    # --- requirement: the audit must not overwrite the primary cluster field ---
    def test_evaluation_does_not_overwrite_primary_leiden(self):
        a = _synth_clustered()
        self.cl.run_leiden(a, resolution=0.5, seed=0)
        before = a.obs["leiden"].astype(str).tolist()

        df = self.cl.evaluate_leiden_resolutions(
            a,
            primary_resolution=0.5,
            candidates=[0.2, 0.5, 1.0],
            seed=0,
            out_dir=None,
            min_cluster_cells=5,
        )
        self.assertEqual(a.obs["leiden"].astype(str).tolist(), before)  # untouched
        for col in ("leiden_res_0_2", "leiden_res_0_5", "leiden_res_1_0"):
            self.assertIn(col, a.obs.columns)
        # the primary row reuses the primary labels rather than re-clustering
        self.assertEqual(a.obs["leiden_res_0_5"].astype(str).tolist(), before)
        # nothing is auto-selected
        self.assertTrue((df["selected_primary_resolution"] == 0.5).all())
        self.assertIn("user_configured", df["selection_rule"].iloc[0])

    def test_evaluation_csv_has_the_required_diagnostics(self):
        import tempfile

        a = _synth_clustered()
        self.cl.run_leiden(a, resolution=0.5, seed=0)
        with tempfile.TemporaryDirectory() as d:
            df = self.cl.evaluate_leiden_resolutions(
                a,
                primary_resolution=0.5,
                candidates=[0.4, 0.5],
                seed=0,
                out_dir=Path(d),
                min_cluster_cells=5,
            )
            self.assertTrue((Path(d) / self.cl.EVALUATION_CSV_NAME).exists())
        for col in (
            "resolution",
            "n_clusters",
            "min_cluster_size",
            "median_cluster_size",
            "max_cluster_size",
            "n_clusters_below_min_cells",
            "ari_vs_previous",
            "nmi_vs_previous",
            "silhouette_pca",
            "selected_primary_resolution",
        ):
            self.assertIn(col, df.columns, col)
        # adjacent-resolution agreement: first row has no predecessor, later rows do
        self.assertTrue(pd_isna(df["ari_vs_previous"].iloc[0]))
        self.assertFalse(pd_isna(df["ari_vs_previous"].iloc[1]))

    def test_higher_resolution_does_not_reduce_cluster_count(self):
        # sanity check that the sweep actually varies the partition
        a = _synth_clustered()
        self.cl.run_leiden(a, resolution=0.1, seed=0, key_added="lo")
        self.cl.run_leiden(a, resolution=1.5, seed=0, key_added="hi")
        self.assertGreaterEqual(a.obs["hi"].nunique(), a.obs["lo"].nunique())


def pd_isna(v) -> bool:
    import pandas as pd

    return bool(pd.isna(v))


# =========================================================================== #
#  11. QC thresholds — configurable, validated, defaults unchanged
# =========================================================================== #
class TestQCThresholds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qc = _imp("qc_filters")

    def test_defaults_are_the_historical_constants(self):
        t = cls_t = self.qc.resolve_qc_thresholds()
        self.assertEqual(
            (t.min_genes, t.max_genes, t.max_mito_percent), (200, 6000, 15.0)
        )
        self.assertEqual(cls_t.as_dict()["max_mito_percent"], 15.0)

    def test_config_values_are_applied_and_validated(self):
        t = self.qc.resolve_qc_thresholds(
            min_genes=300, max_genes=5000, max_mito_percent=10
        )
        self.assertEqual(
            (t.min_genes, t.max_genes, t.max_mito_percent), (300, 5000, 10.0)
        )
        with self.assertRaises(self.qc.QCConfigError):
            self.qc.resolve_qc_thresholds(min_genes=5000, max_genes=1000)  # max <= min
        with self.assertRaises(self.qc.QCConfigError):
            self.qc.resolve_qc_thresholds(max_mito_percent=0)  # out of range
        with self.assertRaises(self.qc.QCConfigError):
            self.qc.resolve_qc_thresholds(max_mito_percent=101)
        with self.assertRaises(self.qc.QCConfigError):
            self.qc.resolve_qc_thresholds(min_genes=-1)

    def _qc_adata(self):
        import anndata as ad
        import pandas as pd

        obs = pd.DataFrame(
            {
                # keep, fail-min, fail-max, fail-mito
                "n_genes_by_counts": [1000, 100, 9000, 1000],
                "pct_counts_mt": [1.0, 1.0, 1.0, 50.0],
            },
            index=[f"c{i}" for i in range(4)],
        )
        return ad.AnnData(
            X=np.ones((4, 3)), obs=obs, var=pd.DataFrame(index=list("abc"))
        )

    def test_per_rule_removal_counts_are_reported(self):
        a, t = self._qc_adata(), self.qc.resolve_qc_thresholds()
        out, rep = self.qc.apply_qc_filters(a, t)
        self.assertEqual(out.n_obs, 1)
        self.assertEqual(rep["removed_min_genes"], 1)
        self.assertEqual(rep["removed_max_genes"], 1)
        self.assertEqual(rep["removed_max_mito_percent"], 1)
        self.assertEqual(rep["removed_total"], 3)
        self.assertEqual(rep["n_cells_before"], 4)
        self.assertEqual(rep["n_cells_after"], 1)
        self.assertEqual(rep["thresholds_applied"]["min_genes"], 200)

    def test_removing_every_cell_raises_rather_than_returning_empty(self):
        a = self._qc_adata()
        t = self.qc.resolve_qc_thresholds(min_genes=100000, max_genes=200000)
        with self.assertRaises(self.qc.QCConfigError):
            self.qc.apply_qc_filters(a, t)


# =========================================================================== #
#  10c. Cell-type QC figures (per-donor proportions, dotplot, agreement, volcano)
# =========================================================================== #
class TestCelltypeQCPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")
        cls.Q = _imp("celltype_qc_plots")

    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="qcplots_test_"))

    def _adata(self, n_donors_per_group=5, groups=("ctrl", "case")):
        import anndata as ad
        import pandas as pd

        rng = np.random.default_rng(0)
        rows = []
        for g in groups:
            for d in range(n_donors_per_group):
                # Proportions are COMPOSITIONAL — they sum to 1 — so a cell type
                # cannot move on its own. 'Rare' trades against 'B cell' only, which
                # leaves 'T cell' genuinely flat. An earlier version of this fixture
                # let the fractions sum to >1, so normalizing pushed T cell down too
                # and it was (correctly) reported as significant.
                frac = (
                    {"T cell": 0.5, "B cell": 0.4, "Rare": 0.1}
                    if g == "ctrl"
                    else {"T cell": 0.5, "B cell": 0.1, "Rare": 0.4}
                )
                for ct, f in frac.items():
                    n = max(1, int(rng.normal(f, 0.02) * 200))
                    rows += [(f"{g}{d}", g, ct)] * n
        obs = pd.DataFrame(rows, columns=["sample", "group", "celltype"])
        obs.index = [f"c{i}" for i in range(len(obs))]
        return ad.AnnData(
            X=np.zeros((len(obs), 1), dtype="float32"),
            obs=obs,
            var=pd.DataFrame(index=["G1"]),
        )

    def test_per_donor_writes_values_and_a_test(self):
        """The unit of replication must be the donor, not the cell."""
        import pandas as pd

        p = self.Q.plot_per_donor_proportions(
            self._adata(),
            group_col="group",
            celltype_col="celltype",
            sample_col="sample",
            out_dir=self.tmp,
            analysis_name="t",
        )
        self.assertIsNotNone(p)
        stats = pd.read_csv(p)
        self.assertEqual(set(stats.celltype), {"T cell", "B cell", "Rare"})
        for col in ("mannwhitney_p", "bh_q", "significant_q05", "diff_pct"):
            self.assertIn(col, stats.columns)
        # one row per donor x cell type, not one per group
        long = pd.read_csv(self.tmp / "celltype_proportions_per_donor.csv")
        self.assertEqual(len(long), 10 * 3)
        self.assertTrue((self.tmp / "celltype_proportions_per_donor.png").exists())

    def test_per_donor_detects_a_real_shift_and_ignores_a_flat_one(self):
        import pandas as pd

        p = self.Q.plot_per_donor_proportions(
            self._adata(n_donors_per_group=8),
            group_col="group",
            celltype_col="celltype",
            sample_col="sample",
            out_dir=self.tmp,
        )
        s = pd.read_csv(p).set_index("celltype")
        self.assertTrue(s.loc["Rare", "significant_q05"], "0.10 vs 0.40 must be found")
        self.assertFalse(
            bool(s.loc["T cell", "significant_q05"]),
            "an unchanged cell type must not be called significant",
        )

    def test_too_few_donors_reports_values_but_no_verdict(self):
        """Two donors a side cannot support a p-value; write data, skip the test."""
        import pandas as pd

        p = self.Q.plot_per_donor_proportions(
            self._adata(n_donors_per_group=2),
            group_col="group",
            celltype_col="celltype",
            sample_col="sample",
            out_dir=self.tmp,
        )
        s = pd.read_csv(p)
        self.assertTrue(s["mannwhitney_p"].isna().all())
        self.assertFalse(s["significant_q05"].any())

    def test_bh_correction_is_applied(self):
        raw = [0.01, 0.02, 0.03, 0.04]
        q = self.Q._bh_fdr(raw)
        self.assertTrue(
            all(a <= b + 1e-12 for a, b in zip(raw, q, strict=True)),
            "q must never be below its raw p",
        )
        self.assertTrue(all(x is not None and x <= 1.0 for x in q))
        self.assertEqual(self.Q._bh_fdr([None, None]), [None, None])

    def test_degrades_instead_of_raising(self):
        a = self._adata()
        self.assertIsNone(
            self.Q.plot_per_donor_proportions(
                a,
                group_col="absent",
                celltype_col="celltype",
                sample_col="sample",
                out_dir=self.tmp,
            )
        )
        one = self._adata(groups=("only",))
        self.assertIsNone(
            self.Q.plot_per_donor_proportions(
                one,
                group_col="group",
                celltype_col="celltype",
                sample_col="sample",
                out_dir=self.tmp,
            ),
            "one group means nothing to compare",
        )
        self.assertIsNone(
            self.Q.plot_voter_agreement(
                self.tmp / "does_not_exist.csv", out_dir=self.tmp
            )
        )
        self.assertEqual(
            self.Q.plot_pseudobulk_volcanoes(self.tmp / "empty", out_dir=self.tmp), []
        )

    def test_output_filenames_are_windows_safe(self):
        """A colon makes an NTFS alternate data stream: the file exists but is
        invisible in Explorer and is silently dropped by copy/zip/upload. Seven
        pseudobulk DE files were lost this way on GSE157827."""
        for bad in ("Other: GABAergic interneuron", 'a/b\\c*d?e"f<g>h|i'):
            safe = self.Q._safe(bad)
            self.assertFalse(any(ch in safe for ch in ':\\/*?"<>|'), safe)
            self.assertTrue(safe)

    def test_volcano_written_per_celltype_table(self):
        import pandas as pd

        src = self.tmp / "per_celltype"
        src.mkdir()
        # Input names must already be sanitized — writing "Other: x_DE.csv"
        # directly would create an NTFS stream and the glob would not see it.
        # That the writers now sanitize is covered by TestSafeFilenames.
        for ct in ("T_cell", "Other_GABAergic_interneuron"):
            pd.DataFrame(
                {
                    "gene": [f"G{i}" for i in range(50)],
                    "log2FoldChange": np.linspace(-4, 4, 50),
                    "padj": np.linspace(1e-8, 0.9, 50),
                }
            ).to_csv(src / f"{ct}_pseudobulk_DE.csv", index=False)
        made = self.Q.plot_pseudobulk_volcanoes(src, out_dir=self.tmp / "v")
        self.assertEqual(len(made), 2)
        for p in made:
            self.assertTrue(p.exists() and p.stat().st_size > 0)
            self.assertNotIn(":", p.name)


# =========================================================================== #
#  10d. Filename sanitizing (silent data loss on Windows)
# =========================================================================== #
class TestSafeFilenames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sn = _imp("safe_names")

    def test_colon_is_removed(self):
        """The measured failure: seven pseudobulk DE tables (~30 MB) on GSE157827
        were written into NTFS alternate data streams on one 0-byte file named
        'Other', invisible to Explorer and silently dropped by copy/zip/upload."""
        out = self.sn.safe_filename("Other: GABAergic interneuron")
        self.assertNotIn(":", out)
        self.assertEqual(out, "Other_GABAergic_interneuron")

    def test_all_windows_reserved_characters(self):
        out = self.sn.safe_filename('a<b>c:d"e/f\\g|h?i*j')
        self.assertFalse(any(ch in out for ch in '<>:"/\\|?*'))

    def test_whitespace_collapses_and_trailing_dots_go(self):
        self.assertEqual(self.sn.safe_filename("  T   cell  "), "T_cell")
        # Windows silently drops a trailing dot, so "A." and "A" would collide.
        self.assertEqual(self.sn.safe_filename("Cell."), "Cell")

    def test_reserved_device_stems_are_escaped(self):
        for stem in ("CON", "nul", "COM1", "LPT9"):
            out = self.sn.safe_filename(stem)
            self.assertNotEqual(out.upper(), stem.upper())

    def test_empty_and_none_get_a_fallback(self):
        self.assertEqual(self.sn.safe_filename(""), "unnamed")
        self.assertEqual(self.sn.safe_filename(None), "unnamed")
        self.assertEqual(self.sn.safe_filename(":::"), "unnamed")

    def test_length_is_bounded(self):
        out = self.sn.safe_filename("x" * 500)
        self.assertLessEqual(len(out), self.sn.MAX_COMPONENT_CHARS)

    def test_ordinary_labels_are_unchanged_apart_from_spaces(self):
        """Existing outputs must keep their names, so nothing silently moves."""
        self.assertEqual(
            self.sn.safe_filename("Excitatory (glutamatergic) neuron"),
            "Excitatory_(glutamatergic)_neuron",
        )
        self.assertEqual(
            self.sn.safe_filename("Natural killer cell"), "Natural_killer_cell"
        )

    def test_de_writers_use_the_sanitizer(self):
        """Guard the wiring: both writers previously replaced only ' ' and '/'."""
        import inspect

        for mod in ("pseudobulk_de", "group_de"):
            src = inspect.getsource(_imp(mod))
            self.assertIn("safe_filename(ct)", src, f"{mod} must sanitize its paths")
            self.assertNotIn('str(ct).replace(" ", "_").replace("/", "_")', src)


# =========================================================================== #
#  11a. Reference scope, voter saturation and vocabulary consistency
#
#  Every case here is taken from a real run, with the observed numbers, so a
#  regression reproduces a mislabelled cluster rather than an abstract failure.
# =========================================================================== #
class TestReferenceScopeAndVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    # ---- vocabulary: the two routes must agree -----------------------------
    def test_microglia_is_not_collapsed_into_macrophage(self):
        """GSE157827 cluster 8, 6,785 cells (CSF1R/P2RY12/APBB1IP/TLR2/CD86).

        CellTypist, the knowledge voter and PubMed all said microglia; the cluster
        shipped as "Macrophage" because both the keyword table and the hierarchy's
        main_cell_type collapsed it. Microglia are yolk-sac-derived and resident;
        macrophage implies monocyte-derived and infiltrating.
        """
        h = self.tools.harmonize_label
        for raw in ("Microglia", "microglia", "Microglial cell", "MICROGLIA"):
            self.assertEqual(h(raw), "Microglia", f"{raw!r} must stay microglia")
        self.assertEqual(h("Macrophage"), "Macrophage")
        self.assertNotEqual(h("Microglia"), h("Macrophage"))

    def test_celltypist_fine_monocyte_aggregates_with_coarse_monocyte(self):
        """GSE337706 cluster 7: a real 3-vote monocyte agreement was split.

        ``Monocyte CD14+`` took the keyword route and became
        "Monocyte/Macrophage" while the other voters resolved to "Monocyte", so the
        tally read 2-vs-1-vs-1, lost its majority, and was adjudicated to
        "Dendritic cell" on a single dissent.
        """
        h = self.tools.harmonize_label
        self.assertEqual(h("Monocyte CD14+"), h("Monocyte"))
        self.assertEqual(h("Monocyte CD16+"), h("monocyte"))
        self.assertEqual(h("Monocyte"), "Monocyte")
        # and the 3-vote majority now survives the tally
        votes = {
            "celltypist": h("Monocyte CD14+"),
            "knowledge_based": h("Monocyte"),
            "pubmed": h("monocyte"),
            "singler": h("Dendritic cells"),
        }
        tally = self.tools.tally_votes(votes)
        self.assertTrue(tally["has_majority"])
        self.assertEqual(tally["majority_label"], "Monocyte")

    def test_monocyte_and_macrophage_remain_distinct_nodes(self):
        h = self.tools.harmonize_label
        self.assertNotEqual(h("Monocyte"), h("Macrophage"))
        self.assertEqual(self.tools.coarse_lineage_of(h("Monocyte")), "Immune")
        self.assertEqual(self.tools.coarse_lineage_of(h("Macrophage")), "Immune")

    def test_interneuron_labels_resolve_to_one_node(self):
        """GSE157827 clusters 6/9/11/13/16 were split across three names.

        ``\\bneuron`` cannot match inside "interneuron", so every interneuron label
        fell through to "Other: ...", fragmenting one population into
        "Other: GABAergic interneuron", "Other: Interneuron" and
        "Other: InN PVALB PDE3A" — which also weakened the per-cell-type DE.
        """
        h = self.tools.harmonize_label
        expected = "Inhibitory (GABAergic) neuron"
        for raw in (
            "GABAergic interneuron",
            "Interneuron",
            "interneuron",
            "Inhibitory neuron",
            "InN PVALB PDE3A",
            "InN VIP EXPH5",
        ):
            self.assertEqual(h(raw), expected, f"{raw!r} -> {h(raw)!r}")
            self.assertFalse(h(raw).startswith("Other:"))

    def test_brain_classifier_codes_resolve_without_other_prefix(self):
        """The brain CellTypist models emit "<class> <marker> <marker>" codes.

        The marker suffix makes each label unique, so no alias table can enumerate
        them; 9 clusters / 29,624 cells shipped as "Other: <raw code>", which also
        put a colon into every derived filename (see the NTFS-stream test below).
        """
        h = self.tools.harmonize_label
        cases = {
            "Astro AQP4 SLC1A2": "Astrocyte",
            "Oligo MOG OPALIN": "Oligodendrocyte",
            "OPC PDGFRA PCDH15": "Oligodendrocyte precursor cell",
            "COP GPR17 SOX4": "Oligodendrocyte precursor cell",
            "Micro P2RY12 APBB1IP": "Microglia",
            "L2-3 CUX2 ACVR1C THSD7A": "Excitatory (glutamatergic) neuron",
            "L5-6 FEZF2 NXPH2 CDH8": "Excitatory (glutamatergic) neuron",
            "L6 OPRK1 THEMIS RGS6": "Excitatory (glutamatergic) neuron",
            "InN SST FREM1": "Inhibitory (GABAergic) neuron",
        }
        for raw, want in cases.items():
            self.assertEqual(h(raw), want, f"{raw!r} -> {h(raw)!r}")

    def test_prefix_match_is_anchored_not_substring(self):
        """A bare substring rule would mis-map ordinary labels."""
        h = self.tools.harmonize_label
        # 'micro' must not swallow these. The hierarchy resolves microvascular EC to
        # 'Capillary endothelial cell' (correct — microvascular IS capillary), so
        # assert the lineage rather than an exact node name.
        self.assertNotEqual(h("Microvascular endothelial cell"), "Microglia")
        self.assertEqual(
            self.tools.coarse_lineage_of(h("Microvascular endothelial cell")),
            "Endothelial",
        )
        # cortical-layer prefixes must not fire on unrelated leading tokens
        self.assertNotEqual(
            h("L1CAM+ tumour cell"), "Excitatory (glutamatergic) neuron"
        )

    def test_celltypist_vascular_codes_stay_unresolved(self):
        """Deliberate: CellTypist's brain vascular calls proved unreliable.

        It called cluster 14 "Endo CLDN5 SLC7A5" at 0.977 and cluster 10 at 0.283 —
        both on unmistakable neurons (NEFL/NEFM/SNAP25/NRGN/RBFOX1, no
        CLDN5/PECAM1/FLT1). Endothelial/Mural/Fibroblast are already first-class
        nodes with lineage-gate panels, so admitting these codes only amplified
        CellTypist's errors; replay turned two correct calls into wrong ones.
        """
        h = self.tools.harmonize_label
        for raw in ("Endo CLDN5 SLC7A5", "PC P2RY14 GRM8", "VLMC ABCA6 FBLN1"):
            self.assertTrue(h(raw).startswith("Other:"), f"{raw!r} -> {h(raw)!r}")
        # the plain names still resolve, via the other voters
        self.assertEqual(h("Endothelial cell"), "Endothelial cell")

    # ---- voter saturation ---------------------------------------------------
    def test_saturated_voter_is_detected(self):
        """GSE157827: SingleR returned 'Astrocyte' for 18/20 clusters at 0.46-0.57.

        Only one of those is truly an astrocyte (~6% accurate). Consensus outvoted
        it every time, but its permanent dissent set voters_disagree on 17/20
        clusters and pushed 83.5% of cells to Low/Review with 0 clusters High.
        """
        labels = {f"c{i}": "Astrocyte" for i in range(18)}
        labels["c18"] = "Macrophage"
        labels["c19"] = "Endothelial cell"
        sat = self.tools.degenerate_voters({"singler": labels})
        self.assertIn("singler", sat)
        self.assertEqual(sat["singler"]["modal_label"], "Astrocyte")
        self.assertAlmostEqual(sat["singler"]["modal_fraction"], 0.9, places=3)

    def test_skewed_but_real_distribution_is_not_flagged(self):
        """A blood cohort that genuinely is ~70% T cells must keep its voter."""
        labels = {f"c{i}": "T cell" for i in range(7)}
        labels.update({"c7": "B cell", "c8": "Monocyte", "c9": "NK cell"})
        self.assertEqual(self.tools.degenerate_voters({"celltypist": labels}), {})

    def test_saturation_needs_enough_clusters_to_judge(self):
        """Three clusters cannot reveal saturation; do not guess from too little."""
        labels = {"c0": "Astrocyte", "c1": "Astrocyte", "c2": "Astrocyte"}
        self.assertEqual(self.tools.degenerate_voters({"singler": labels}), {})

    def test_unassigned_labels_do_not_count_toward_saturation(self):
        labels = {f"c{i}": self.tools.UNASSIGNED for i in range(10)}
        labels.update({"c10": "T cell", "c11": "B cell"})
        self.assertEqual(self.tools.degenerate_voters({"pubmed": labels}), {})

    # ---- reference scope ----------------------------------------------------
    def _breast_cluster0(self):
        """The real GSE337706 cluster 0 vote: 24,272 cells (27.8% of the run).

        Markers PLP1 +15.1 / CRYAB +15.0 / APLP1 +14.2 / MAG +12.2 (padj 0) — a
        myelinating-glia programme with zero T-cell markers. Tissue resolved to
        'blood', so COVID19_HumanChallenge_Blood.pkl was loaded; it has no
        Schwann-cell class and cannot abstain.
        """
        h = self.tools.harmonize_label
        return {
            "celltypist": h("T CD4 Naive"),
            "singler": h("CD4+ T cells"),
            "knowledge_based": h("Schwann cell"),
        }

    def test_forced_closed_vocab_majority_defers_to_confident_markers(self):
        votes = self._breast_cluster0()
        self.assertEqual(votes["celltypist"], "T cell")  # both forced to guess
        self.assertEqual(votes["singler"], "T cell")
        tally = self.tools.tally_votes(votes)
        self.assertTrue(tally["has_majority"])
        self.assertEqual(tally["majority_label"], "T cell")  # 2-vs-1 the old way

        ood = self.tools.out_of_domain_deference(
            votes,
            tally,
            open_confidences={"knowledge_based": 0.82},  # observed value
            celltypist_unreliable=True,
        )  # observed 0.536 < 0.70
        self.assertIsNotNone(ood, "reference-scope check must fire on cluster 0")
        label, voter, reason = ood
        self.assertEqual(label, "Schwann cell")
        self.assertEqual(voter, "knowledge_based")
        self.assertIn("forced to", reason)

    def test_reliable_closed_vocab_call_is_not_overturned(self):
        """A confident CellTypist call is ordinary evidence and must still win."""
        votes = self._breast_cluster0()
        ood = self.tools.out_of_domain_deference(
            votes,
            self.tools.tally_votes(votes),
            open_confidences={"knowledge_based": 0.82},
            celltypist_unreliable=False,
        )  # dominant fraction above floor
        self.assertIsNone(ood)

    def test_hedging_open_voter_cannot_overturn(self):
        """Below OPEN_VOCAB_MIN_CONFIDENCE the marker voter is guessing too."""
        votes = self._breast_cluster0()
        ood = self.tools.out_of_domain_deference(
            votes,
            self.tools.tally_votes(votes),
            open_confidences={"knowledge_based": 0.55},
            celltypist_unreliable=True,
        )
        self.assertIsNone(ood)

    def test_open_voter_backing_the_majority_blocks_deference(self):
        """If the markers already agree, nothing was forced — leave it alone."""
        h = self.tools.harmonize_label
        votes = {
            "celltypist": h("T CD4 Naive"),
            "singler": h("CD4+ T cells"),
            "knowledge_based": h("T cell"),
        }
        ood = self.tools.out_of_domain_deference(
            votes,
            self.tools.tally_votes(votes),
            open_confidences={"knowledge_based": 0.97},
            celltypist_unreliable=True,
        )
        self.assertIsNone(ood)

    def test_split_vote_is_left_to_the_adjudicator(self):
        """No majority means the adjudicator already reads the markers itself."""
        h = self.tools.harmonize_label
        votes = {
            "celltypist": h("Endothelial cell"),
            "singler": h("Astrocyte"),
            "knowledge_based": h("Neuron"),
        }
        tally = self.tools.tally_votes(votes)
        self.assertFalse(tally["has_majority"])
        self.assertIsNone(
            self.tools.out_of_domain_deference(
                votes,
                tally,
                open_confidences={"knowledge_based": 0.95},
                celltypist_unreliable=True,
            )
        )

    def test_brain_cluster8_becomes_unanimous_microglia(self):
        """End-to-end on the real cluster-8 vote, with SingleR withheld.

        Saturated SingleR ('Macrophage' here, 'Astrocyte' on 18/20 clusters) drops
        out; the three remaining voters now agree because the vocabulary keeps
        microglia distinct and resolves CellTypist's compact code.
        """
        h = self.tools.harmonize_label
        votes = {
            "celltypist": h("Micro P2RY12 APBB1IP"),
            "knowledge_based": h("Microglia"),
            "pubmed": h("microglia"),
        }
        tally = self.tools.tally_votes(votes)
        self.assertTrue(tally["unanimous"])
        self.assertEqual(tally["majority_label"], "Microglia")


# =========================================================================== #
#  11b. HVG selection survives a library too small for a per-batch LOESS fit
# =========================================================================== #
class TestHVGSelection(unittest.TestCase):
    """Regression cover for the GSE157827 crash.

    ``seurat_v3`` fits one LOESS per batch; a 353-nucleus library had too few
    distinct mean values for a local quadratic and skmisc aborted the entire
    158,084-cell run with ``There are other near singularities as well``. The
    synthetic 5-cell batch below reproduces that exact error.
    """

    @classmethod
    def setUpClass(cls):
        cls.hvg = _imp("hvg_selection")

    def _adata(self, n_big=300, n_tiny=0, n_genes=400, seed=0):
        import anndata as ad
        import pandas as pd
        import scanpy as sc
        from scipy import sparse

        rng = np.random.default_rng(seed)
        blocks = [rng.negative_binomial(3, 0.3, size=(n_big, n_genes))]
        names = ["big"] * n_big
        if n_tiny:
            # 0/1 counts only -> means take just n_tiny+1 distinct values, the
            # degenerate shape that makes a local quadratic singular.
            blocks.append((rng.random((n_tiny, n_genes)) < 0.15).astype(int))
            names += ["tiny"] * n_tiny
        X = np.vstack(blocks).astype(np.float32)
        a = ad.AnnData(
            X=sparse.csr_matrix(X),
            obs=pd.DataFrame(
                {"sample": names}, index=[f"c{i}" for i in range(X.shape[0])]
            ),
            var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
        )
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        return a

    def test_healthy_cohort_takes_the_unchanged_first_rung(self):
        """No fallback may trigger on data that already works."""
        a = self._adata()
        a.obs["sample"] = (["s1"] * 150) + (["s2"] * 150)
        rep = self.hvg.select_hvgs(a, n_top_genes=100, batch_key="sample")
        self.assertEqual(rep["method"], "seurat_v3 (per-batch)")
        self.assertEqual(rep["flavor"], "seurat_v3")
        self.assertEqual(rep["batch_key"], "sample")
        self.assertEqual(rep["excluded_batches"], [])
        self.assertEqual(rep["fallback_reason"], "")
        self.assertEqual(rep["n_hvg"], 100)

    def test_first_rung_really_does_raise_on_a_degenerate_batch(self):
        """Guard the premise: if scanpy stops raising, the fallback is dead code."""
        import scanpy as sc

        a = self._adata(n_tiny=5)
        with self.assertRaises(ValueError):
            sc.pp.highly_variable_genes(
                a,
                flavor="seurat_v3",
                n_top_genes=100,
                layer="counts",
                batch_key="sample",
            )

    def test_degenerate_batch_is_excluded_from_ranking_but_keeps_its_cells(self):
        a = self._adata(n_tiny=5)
        n_before = a.n_obs
        rep = self.hvg.select_hvgs(a, n_top_genes=100, batch_key="sample")
        self.assertIn("degenerate batches excluded", rep["method"])
        self.assertEqual(rep["excluded_batches"], ["tiny"])
        self.assertTrue(rep["fallback_reason"])  # the failure is recorded
        self.assertEqual(rep["n_hvg"], 100)
        # the whole point: only the gene *vote* is dropped, never the cells
        self.assertEqual(a.n_obs, n_before)
        self.assertEqual(int((a.obs["sample"] == "tiny").sum()), 5)

    def test_hvg_columns_written_back_are_usable_downstream(self):
        """A subset-computed flag column must still be a clean bool over ALL genes."""
        import scanpy as sc

        a = self._adata(n_tiny=5)
        self.hvg.select_hvgs(a, n_top_genes=100, batch_key="sample")
        hv = a.var["highly_variable"]
        self.assertEqual(hv.dtype, bool)
        self.assertEqual(len(hv), a.n_vars)
        self.assertFalse(hv.isna().any())
        sc.pp.scale(a, max_value=10)
        sc.tl.pca(
            a, n_comps=10, svd_solver="arpack", use_highly_variable=True, random_state=0
        )
        self.assertEqual(a.obsm["X_pca"].shape, (a.n_obs, 10))

    def test_every_rung_leaves_uns_hvg_for_the_plot(self):
        """``sc.pl.highly_variable_genes`` reads ``uns['hvg']['flavor']``.

        Rung 2 computes on a copy; forgetting to carry ``uns`` back left the real run
        dying with ``KeyError: 'hvg'`` on the very next line of the pipeline.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc

        cases = {
            "rung1_healthy": self._adata(),
            "rung2_degenerate_batch": self._adata(n_tiny=5),
        }
        cases["rung1_healthy"].obs["sample"] = (["s1"] * 150) + (["s2"] * 150)
        for name, a in cases.items():
            with self.subTest(rung=name):
                rep = self.hvg.select_hvgs(a, n_top_genes=100, batch_key="sample")
                self.assertIn("hvg", a.uns, f"{name}: uns['hvg'] missing")
                self.assertEqual(a.uns["hvg"]["flavor"], rep["flavor"])
                sc.pl.highly_variable_genes(a, show=False)  # the call that crashed
                plt.close("all")

    def test_pooled_and_dispersion_rungs_also_stamp_uns(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc

        a = self._adata(n_tiny=5)
        orig = self.hvg._batch_fits
        self.hvg._batch_fits = lambda *args, **kw: False  # force past rung 2
        try:
            rep = self.hvg.select_hvgs(a, n_top_genes=100, batch_key="sample")
        finally:
            self.hvg._batch_fits = orig
        self.assertIn("pooled", rep["method"])
        self.assertEqual(a.uns["hvg"]["flavor"], "seurat_v3")
        sc.pl.highly_variable_genes(a, show=False)
        plt.close("all")

        # rung 4: dispersion binning, reached only if seurat_v3 is unusable
        b = self._adata()
        self.hvg._stamp_uns(b, "stale_value_from_a_failed_attempt")
        sc.pp.highly_variable_genes(b, flavor="seurat", n_top_genes=100)
        self.hvg._stamp_uns(b, "seurat")
        self.assertEqual(b.uns["hvg"]["flavor"], "seurat")
        sc.pl.highly_variable_genes(b, show=False)
        plt.close("all")

    def test_batch_probe_separates_fittable_from_degenerate(self):
        a = self._adata(n_tiny=5)
        self.assertTrue(self.hvg._batch_fits(a, "big", "sample", "counts"))
        self.assertFalse(self.hvg._batch_fits(a, "tiny", "sample", "counts"))
        self.assertEqual(self.hvg._degenerate_batches(a, "sample", "counts"), ["tiny"])

    def test_pooled_rung_used_when_every_batch_is_degenerate(self):
        """Rung 2 cannot help if there is nothing left to rank; rung 3 must."""
        a = self._adata(n_big=300, n_tiny=5)
        # force rung 2 to be inapplicable by declaring both batches unfittable
        orig = self.hvg._batch_fits
        self.hvg._batch_fits = lambda *args, **kw: False
        try:
            rep = self.hvg.select_hvgs(a, n_top_genes=100, batch_key="sample")
        finally:
            self.hvg._batch_fits = orig
        self.assertIn("pooled", rep["method"])
        self.assertIsNone(rep["batch_key"])
        self.assertEqual(rep["n_hvg"], 100)

    def test_single_sample_run_has_no_batch_key_and_still_works(self):
        a = self._adata()
        rep = self.hvg.select_hvgs(a, n_top_genes=100, batch_key=None)
        self.assertEqual(rep["method"], "seurat_v3")
        self.assertIsNone(rep["batch_key"])
        self.assertEqual(rep["n_hvg"], 100)


# =========================================================================== #
#  12. CellTypist mixed / heterogeneous cluster metrics
# =========================================================================== #
class TestMixedClusterMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def _series(self, per_cell, clusters):
        import pandas as pd

        idx = [f"c{i}" for i in range(len(per_cell))]
        return pd.Series(per_cell, index=idx), pd.Series(clusters, index=idx)

    def test_pure_cluster_is_not_mixed(self):
        pc, cl = self._series(["T cell"] * 10, ["0"] * 10)
        votes, m = self.tools.summarize_celltypist_by_cluster(pc, cl)
        self.assertEqual(votes["0"], ("T cell", 1.0))
        self.assertFalse(m["0"]["mixed_cluster_flag"])
        self.assertEqual(m["0"]["celltypist_dominant_fraction"], 1.0)
        self.assertEqual(m["0"]["celltypist_second_label"], "")
        self.assertEqual(m["0"]["celltypist_second_fraction"], 0.0)
        self.assertEqual(m["0"]["celltypist_label_entropy"], 0.0)
        self.assertEqual(m["0"]["celltypist_unique_label_count"], 1)

    def test_split_cluster_is_flagged_mixed_with_correct_metrics(self):
        # 55 % T / 45 % NK -> dominant below 0.70 AND runner-up above 0.20
        pc, cl = self._series(["T cell"] * 11 + ["NK cell"] * 9, ["0"] * 20)
        votes, m = self.tools.summarize_celltypist_by_cluster(pc, cl)
        self.assertEqual(votes["0"][0], "T cell")  # dominant vote unchanged
        self.assertAlmostEqual(m["0"]["celltypist_dominant_fraction"], 0.55, places=4)
        self.assertEqual(m["0"]["celltypist_second_label"], "NK cell")
        self.assertAlmostEqual(m["0"]["celltypist_second_fraction"], 0.45, places=4)
        self.assertEqual(m["0"]["celltypist_unique_label_count"], 2)
        self.assertTrue(m["0"]["mixed_cluster_flag"])
        # entropy of a 0.55/0.45 split, in bits
        self.assertAlmostEqual(m["0"]["celltypist_label_entropy"], 0.9928, places=3)

    def test_second_label_rule_alone_can_flag_mixed(self):
        # dominant 0.75 (>= 0.70, passes rule 1) but runner-up 0.25 (>= 0.20)
        pc, cl = self._series(["T cell"] * 15 + ["NK cell"] * 5, ["0"] * 20)
        _, m = self.tools.summarize_celltypist_by_cluster(pc, cl)
        self.assertGreaterEqual(m["0"]["celltypist_dominant_fraction"], 0.70)
        self.assertTrue(m["0"]["mixed_cluster_flag"])

    def test_thresholds_are_configurable(self):
        pc, cl = self._series(["T cell"] * 15 + ["NK cell"] * 5, ["0"] * 20)
        _, m = self.tools.summarize_celltypist_by_cluster(
            pc, cl, min_dominant_fraction=0.5, second_label_fraction=0.5
        )
        self.assertFalse(m["0"]["mixed_cluster_flag"])  # loosened -> not mixed

    def test_metrics_are_per_cluster(self):
        pc, cl = self._series(
            ["T cell"] * 10 + ["T cell"] * 5 + ["B cell"] * 5,
            ["0"] * 10 + ["1"] * 10,
        )
        _, m = self.tools.summarize_celltypist_by_cluster(pc, cl)
        self.assertFalse(m["0"]["mixed_cluster_flag"])
        self.assertTrue(m["1"]["mixed_cluster_flag"])
        self.assertEqual(m["0"]["celltypist_n_cells"], 10)
        self.assertEqual(m["1"]["celltypist_n_cells"], 10)

    def test_entropy_helper(self):
        import pandas as pd

        self.assertEqual(self.tools.label_entropy(pd.Series([10])), 0.0)
        self.assertAlmostEqual(
            self.tools.label_entropy(pd.Series([5, 5])), 1.0, places=6
        )
        self.assertAlmostEqual(
            self.tools.label_entropy(pd.Series([1, 1, 1, 1])), 2.0, places=6
        )

    # --- CellTypist must not become the pipeline's clustering ---
    def test_celltypist_wrapper_keeps_the_historical_return_shape(self):
        import inspect

        sig = inspect.signature(self.tools.annotate_celltypist)
        self.assertEqual(
            list(sig.parameters)[:3], ["adata", "cluster_col", "model_name"]
        )
        # the wrapper returns only {cluster: (label, frac)} — no metrics leak in
        src = inspect.getsource(self.tools.annotate_celltypist)
        self.assertIn("votes, _ = annotate_celltypist_detailed", src)


# =========================================================================== #
#  13. Marker ranking — effect size beats a merely smaller p-value
# =========================================================================== #
class TestMarkerRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def _frame(self, rows):
        import pandas as pd

        return pd.DataFrame(rows)

    def test_strong_effect_outranks_smaller_pvalue(self):
        # BIG has a much larger logFC; TINY only has a smaller adjusted p-value.
        df = self._frame(
            [
                {
                    "names": "TINY",
                    "logfoldchanges": 0.2,
                    "scores": 3.0,
                    "pvals_adj": 1e-30,
                },
                {
                    "names": "BIG",
                    "logfoldchanges": 6.0,
                    "scores": 20.0,
                    "pvals_adj": 1e-10,
                },
            ]
        )
        got = self.tools.rank_cluster_marker_frame(
            df, top_n=2, min_detection_fraction=0.0
        )
        self.assertEqual(got["names"].tolist(), ["BIG", "TINY"])

    def test_padj_ties_are_broken_by_effect_size_not_input_order(self):
        # The realistic failure case: many genes underflow to padj == 0.0.
        df = self._frame(
            [
                {
                    "names": "WEAK",
                    "logfoldchanges": 0.5,
                    "scores": 4.0,
                    "pvals_adj": 0.0,
                },
                {
                    "names": "MID",
                    "logfoldchanges": 2.5,
                    "scores": 9.0,
                    "pvals_adj": 0.0,
                },
                {
                    "names": "STRONG",
                    "logfoldchanges": 8.0,
                    "scores": 30.0,
                    "pvals_adj": 0.0,
                },
            ]
        )
        got = self.tools.rank_cluster_marker_frame(
            df, top_n=3, min_detection_fraction=0.0
        )
        self.assertEqual(got["names"].tolist(), ["STRONG", "MID", "WEAK"])

    def test_non_positive_and_invalid_markers_are_excluded(self):
        df = self._frame(
            [
                {
                    "names": "UP",
                    "logfoldchanges": 3.0,
                    "scores": 10.0,
                    "pvals_adj": 0.01,
                },
                {
                    "names": "DOWN",
                    "logfoldchanges": -3.0,
                    "scores": -10.0,
                    "pvals_adj": 0.001,
                },
                {
                    "names": "ZERO",
                    "logfoldchanges": 0.0,
                    "scores": 0.0,
                    "pvals_adj": 0.001,
                },
                {
                    "names": "INF",
                    "logfoldchanges": np.inf,
                    "scores": 5.0,
                    "pvals_adj": 0.001,
                },
                {
                    "names": "NAN",
                    "logfoldchanges": np.nan,
                    "scores": 5.0,
                    "pvals_adj": 0.001,
                },
                {
                    "names": None,
                    "logfoldchanges": 4.0,
                    "scores": 5.0,
                    "pvals_adj": 0.001,
                },
            ]
        )
        got = self.tools.rank_cluster_marker_frame(
            df, top_n=10, min_detection_fraction=0.0
        )
        self.assertEqual(got["names"].tolist(), ["UP"])

    def test_detection_fraction_floor_filters_barely_detected_genes(self):
        df = self._frame(
            [
                {
                    "names": "RARE",
                    "logfoldchanges": 9.0,
                    "scores": 30.0,
                    "pvals_adj": 0.0,
                    "pct_nz_group": 0.02,
                },
                {
                    "names": "COMMON",
                    "logfoldchanges": 3.0,
                    "scores": 12.0,
                    "pvals_adj": 0.0,
                    "pct_nz_group": 0.80,
                },
            ]
        )
        got = self.tools.rank_cluster_marker_frame(
            df, top_n=5, min_detection_fraction=0.10
        )
        self.assertEqual(
            got["names"].tolist(), ["COMMON"]
        )  # RARE dropped despite big logFC

    def test_detection_floor_relaxes_instead_of_emptying_a_cluster(self):
        df = self._frame(
            [
                {
                    "names": "SPARSE",
                    "logfoldchanges": 5.0,
                    "scores": 15.0,
                    "pvals_adj": 0.0,
                    "pct_nz_group": 0.01,
                },
            ]
        )
        got = self.tools.rank_cluster_marker_frame(
            df, top_n=5, min_detection_fraction=0.50
        )
        self.assertEqual(got["names"].tolist(), ["SPARSE"])

    def test_top_n_is_respected(self):
        df = self._frame(
            [
                {
                    "names": f"G{i}",
                    "logfoldchanges": float(i),
                    "scores": float(i),
                    "pvals_adj": 0.0,
                }
                for i in range(1, 21)
            ]
        )
        got = self.tools.rank_cluster_marker_frame(
            df, top_n=5, min_detection_fraction=0.0
        )
        self.assertEqual(len(got), 5)
        self.assertEqual(got["names"].tolist(), ["G20", "G19", "G18", "G17", "G16"])

    def test_ranking_method_is_recorded_for_provenance(self):
        self.assertIn("logfoldchanges", self.tools.MARKER_RANKING_METHOD)
        self.assertIn("pvals_adj", self.tools.MARKER_RANKING_METHOD)
        self.assertEqual(self.tools.MARKER_SORT_COLUMNS[0], "logfoldchanges")

    def test_planted_markers_still_rank_top_end_to_end(self):
        # the real function must still find the planted identity genes
        a = build_synth_adata(cells_per_cluster=20)
        ln, _ = self.tools.get_lognorm(a)
        markers, empty = self.tools.compute_cluster_markers(
            ln, cluster_col="leiden", top_n=8
        )
        self.assertNotIn("0", empty)
        self.assertTrue(
            {"CD3D", "CD3E"} & set(markers["0"]),
            f"T-cell markers missing from {markers['0']}",
        )
        self.assertTrue(
            {"EPCAM", "KRT8", "KRT18"} & set(markers["4"]),
            f"epithelial markers missing from {markers['4']}",
        )


# =========================================================================== #
#  14. Downstream gating — Low/Review excluded ONLY when configured,
#      and never deleted from the object
# =========================================================================== #
class TestDownstreamGating(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = _imp("downstream_gating")

    def _tiered(self, cells_per_cluster: int = 20):
        """Synthetic object where clusters 8 and 9 are Low/Review."""
        a = build_synth_adata(cells_per_cluster=cells_per_cluster)
        tier = {str(i): "High" for i in range(8)}
        tier["8"] = "Low/Review"
        tier["9"] = "Low/Review"
        a.obs["consensus_tier"] = (
            a.obs["leiden"].astype(str).map(tier).astype("category")
        )
        return a

    def test_flag_is_all_true_when_filtering_is_disabled(self):
        a = self._tiered()
        rep = self.g.annotate_downstream_inclusion(a, exclude_low_confidence=False)
        self.assertTrue(a.obs[self.g.INCLUDE_COL].all())
        self.assertEqual(rep["n_cells_excluded"], 0)
        self.assertIn("filtering disabled", rep["reason"])

    def test_low_review_excluded_only_when_configured(self):
        a = self._tiered()
        rep = self.g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        n_low = int((a.obs["consensus_tier"].astype(str) == "Low/Review").sum())
        self.assertEqual(rep["n_cells_excluded"], n_low)
        self.assertEqual(sorted(rep["excluded_clusters"]), ["8", "9"])
        self.assertEqual(rep["n_clusters_excluded"], 2)
        # cells are MARKED, not removed
        self.assertEqual(a.n_obs, self._tiered().n_obs)
        self.assertEqual(int((~a.obs[self.g.INCLUDE_COL]).sum()), n_low)

    def test_all_cells_remain_available_for_audit_after_subsetting(self):
        a = self._tiered()
        self.g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        n_before = a.n_obs
        sub, rep = self.g.subset_for_downstream(a)
        self.assertTrue(rep["filtered"])
        self.assertLess(sub.n_obs, n_before)
        self.assertEqual(a.n_obs, n_before)  # source object untouched
        self.assertIn(self.g.INCLUDE_COL, a.obs.columns)
        # excluded cells still carry their labels for audit
        excl = a[~a.obs[self.g.INCLUDE_COL]]
        self.assertGreater(excl.n_obs, 0)
        self.assertTrue((excl.obs["consensus_tier"].astype(str) == "Low/Review").all())

    def test_custom_excluded_tiers_are_honoured(self):
        a = self._tiered()
        a.obs["consensus_tier"] = a.obs["consensus_tier"].astype(str)
        a.obs.loc[a.obs["leiden"].astype(str) == "0", "consensus_tier"] = "Medium"
        rep = self.g.annotate_downstream_inclusion(
            a, exclude_low_confidence=True, excluded_tiers=["Medium"]
        )
        self.assertEqual(rep["excluded_clusters"], ["0"])

    def test_missing_tier_column_includes_everything_with_a_reason(self):
        a = build_synth_adata(cells_per_cluster=5)
        rep = self.g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        self.assertTrue(a.obs[self.g.INCLUDE_COL].all())
        self.assertIn("absent", rep["reason"])

    def test_excluding_everything_falls_back_with_a_warning(self):
        a = build_synth_adata(cells_per_cluster=5)
        a.obs["consensus_tier"] = "Low/Review"
        rep = self.g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        self.assertTrue(a.obs[self.g.INCLUDE_COL].all())
        self.assertIn("every cell", rep["reason"])

    # --- requirement: skip with a clear reason rather than break replication ---
    def test_filtering_that_destroys_replication_is_skipped_with_a_reason(self):
        a = self._tiered()
        # collapse to one sample per group so any exclusion breaks the design
        a.obs["sample"] = [
            "CASE_s1" if g == "CASE" else "CONTROL_s1"
            for g in a.obs["group"].astype(str)
        ]
        self.g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        sub, rep = self.g.subset_for_downstream(a, require_replication=True)
        self.assertFalse(rep["filtered"])
        self.assertEqual(sub.n_obs, a.n_obs)  # analysis keeps its replication
        self.assertIn("SKIPPED tier filtering", rep["reason"])

    def test_check_replication(self):
        a = self._tiered()
        ok, why = self.g.check_replication(a)
        self.assertTrue(ok, why)
        a.obs["sample"] = "only_one"
        ok, why = self.g.check_replication(a)
        self.assertFalse(ok)
        self.assertIn("group", why)

    def test_resolve_excluded_tiers_defaults(self):
        self.assertEqual(self.g.resolve_excluded_tiers(None), ["Low/Review"])
        self.assertEqual(self.g.resolve_excluded_tiers([]), ["Low/Review"])
        self.assertEqual(self.g.resolve_excluded_tiers("Medium"), ["Medium"])
        self.assertEqual(self.g.resolve_excluded_tiers(["A", " B "]), ["A", "B"])

    def test_consensus_table_inclusion_is_stamped_from_obs(self):
        import tempfile

        import pandas as pd

        a = self._tiered(cells_per_cluster=5)
        self.g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "x_consensus_annotation.csv"
            pd.DataFrame(
                {
                    "cluster": [str(i) for i in range(10)],
                    "included_in_downstream_analysis": [""] * 10,
                }
            ).to_csv(csv, index=False)
            self.assertTrue(self.g.stamp_consensus_table_inclusion(csv, a))
            out = pd.read_csv(csv)
            flags = dict(
                zip(
                    out["cluster"].astype(str),
                    out["included_in_downstream_analysis"].astype(bool),
                    strict=True,
                )
            )
        self.assertFalse(flags["8"])
        self.assertFalse(flags["9"])
        self.assertTrue(flags["0"])


# =========================================================================== #
#  15. Re-run safety — a single-voter column must never become the final label
# =========================================================================== #
class TestAnnotationReuseGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = _imp("pipeline")

    def _with_voter_cols(self):
        a = build_synth_adata(cells_per_cluster=5)
        a.obs["celltype_celltypist"] = "T cell"
        a.obs["celltype_singler"] = "T_cells"
        return a

    # --- requirement: consensus is NOT skipped because celltype_celltypist exists ---
    def test_celltypist_column_does_not_skip_consensus(self):
        a = self._with_voter_cols()
        info = self.pipe._resolve_annotation_reuse(
            a,
            reuse_existing_final_annotation=False,
            final_annotation_column=None,
            analysis_name="t",
        )
        self.assertIsNone(info["column"])  # -> recompute consensus
        self.assertIn("recomputing", info["reason"])

    def test_voter_columns_are_excluded_from_the_reuse_whitelist(self):
        for col in (
            "celltype_celltypist",
            "celltype_singler",
            "celltype_knowledge_based",
            "celltype_pubmed",
        ):
            self.assertNotIn(col, self.pipe.FINAL_ANNOTATION_COLUMNS, col)
            self.assertIn(col, self.pipe._VOTER_ONLY_COLUMNS, col)

    def test_reuse_true_still_refuses_a_voter_only_column(self):
        a = self._with_voter_cols()
        info = self.pipe._resolve_annotation_reuse(
            a,
            reuse_existing_final_annotation=True,
            final_annotation_column="celltype_celltypist",
            analysis_name="t",
        )
        self.assertIsNone(info["column"])
        self.assertIn("single-voter", info["reason"])

    def test_reuse_true_accepts_a_complete_final_column(self):
        a = self._with_voter_cols()
        a.obs["celltype_consensus"] = "T cell"
        info = self.pipe._resolve_annotation_reuse(
            a,
            reuse_existing_final_annotation=True,
            final_annotation_column=None,
            analysis_name="t",
        )
        self.assertEqual(info["column"], "celltype_consensus")
        self.assertIn("reusing", info["reason"])

    def test_reuse_rejects_an_incomplete_final_column(self):
        a = self._with_voter_cols()
        vals = ["T cell"] * a.n_obs
        vals[0] = "Unassigned"
        a.obs["celltype_consensus"] = vals
        info = self.pipe._resolve_annotation_reuse(
            a,
            reuse_existing_final_annotation=True,
            final_annotation_column=None,
            analysis_name="t",
        )
        self.assertIsNone(info["column"])  # incomplete -> recompute
        self.assertIn("incomplete", info["reason"])

    def test_completeness_check_detects_nan_and_empty(self):
        import pandas as pd

        a = build_synth_adata(cells_per_cluster=5)
        a.obs["c"] = "T cell"
        ok, _ = self.pipe._annotation_column_is_complete(a, "c")
        self.assertTrue(ok)
        a.obs["c"] = pd.Series([None] * a.n_obs, index=a.obs_names)
        ok, why = self.pipe._annotation_column_is_complete(a, "c")
        self.assertFalse(ok)
        self.assertIn("incomplete", why)
        ok, why = self.pipe._annotation_column_is_complete(a, "does_not_exist")
        self.assertFalse(ok)
        self.assertIn("absent", why)


# =========================================================================== #
#  16. Provenance — resolved (not "auto") annotation resources reach the manifest
# =========================================================================== #
class TestResolvedProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = _imp("pipeline")
        cls.cons = _imp("celltype_consensus.consensus")

    def _adata_with_resolved_uns(self):
        a = build_synth_adata(cells_per_cluster=5)
        a.uns[self.cons.CONSENSUS_UNS_KEY] = {
            "resolved_celltypist_model": "Human_Lung_Atlas.pkl",
            "resolved_singler_reference": "HumanPrimaryCellAtlasData",
            "annotation_tissue": "lung",
            "annotation_species": "human",
            "celltypist_enabled": True,
            "singler_enabled": True,
            "knowledge_based_enabled": True,
            "pubmed_enabled": False,
            "marker_ranking_method": "logfoldchanges desc ...",
            "openrouter_model": "some/model",
            "tier_counts": {"High": 5, "Medium": 3, "Low/Review": 2},
            "n_mixed_clusters": 1,
        }
        return a

    def test_manifest_records_requested_and_resolved(self):
        a = self._adata_with_resolved_uns()
        prov = self.pipe._annotation_provenance(
            a,
            enable_celltypist=True,
            enable_llm=True,
            enable_singler=True,
            enable_pubmed=False,
            tissue="auto",
            species="auto",
            celltypist_model="auto",
            singler_reference="auto",
            celltype_source="Consensus (...)",
            reuse_info={"column": None, "reason": "r"},
            use_subtypes_for_downstream=False,
        )
        # requested keeps "auto"; resolved carries what actually ran
        self.assertEqual(prov["requested_celltypist_model"], "auto")
        self.assertEqual(prov["resolved_celltypist_model"], "Human_Lung_Atlas.pkl")
        self.assertEqual(prov["requested_singler_reference"], "auto")
        self.assertEqual(
            prov["resolved_singler_reference"], "HumanPrimaryCellAtlasData"
        )
        self.assertEqual(prov["annotation_tissue"], "lung")
        self.assertEqual(prov["annotation_species"], "human")
        self.assertNotEqual(prov["resolved_celltypist_model"], "auto")
        # voter enablement is explicit
        for k in (
            "celltypist_enabled",
            "singler_enabled",
            "knowledge_based_enabled",
            "pubmed_enabled",
        ):
            self.assertIn(k, prov)
        self.assertTrue(prov["celltypist_enabled"])
        self.assertFalse(prov["pubmed_enabled"])
        # marker ranking + tier accounting
        self.assertIn("marker_ranking_method", prov)
        self.assertEqual(prov["tier_counts"]["Low/Review"], 2)
        self.assertEqual(prov["n_mixed_clusters"], 1)
        # legacy keys preserved for older manifest readers
        for legacy in (
            "enable_llm",
            "enable_singler",
            "tissue",
            "species",
            "celltypist_model",
            "singler_reference",
        ):
            self.assertIn(legacy, prov)

    def test_resolved_fields_are_none_when_annotation_did_not_run(self):
        a = build_synth_adata(cells_per_cluster=5)  # no uns key
        prov = self.pipe._annotation_provenance(
            a,
            enable_celltypist=True,
            enable_llm=False,
            enable_singler=None,
            enable_pubmed=False,
            tissue=None,
            species=None,
            celltypist_model="auto",
            singler_reference="auto",
            celltype_source=None,
            reuse_info={"column": None, "reason": "why"},
            use_subtypes_for_downstream=False,
        )
        self.assertIsNone(prov["resolved_celltypist_model"])
        self.assertIsNone(prov["resolved_singler_reference"])
        self.assertEqual(prov["reuse_decision"], "why")


# =========================================================================== #
#  17. Subtype layer — attributed, never promoted to the main label
# =========================================================================== #
class TestStateProgrammes(unittest.TestCase):
    """A cluster defined by a STATE must not be promoted to a confident identity.

    All fixtures are the VERBATIM top-15 marker lists from the psoriasis run
    (11 clusters, 97,108 cells), so these are regression tests against measured
    output, not synthetic ones.
    """

    # cluster -> (top-15 markers, shipped label, was it state-driven?)
    PSORIASIS = {
        "0": (
            [
                "IL7R",
                "PCSK1N",
                "KLF2",
                "SESN3",
                "PABPC1",
                "PASK",
                "LEF1",
                "FXYD7",
                "ZFP36L2",
                "MAL",
                "S1PR1",
                "ICAM2",
                "IMPDH2",
                "EEF1G",
                "TCF7",
            ],
            False,
        ),
        "1": (
            [
                "HSPA1B",
                "HSPA1A",
                "HSPA6",
                "NR4A2",
                "DNAJB1",
                "IFNG",
                "MTRNR2L12",
                "NR4A1",
                "AL138963.3",
                "CD69",
                "MT1X",
                "MT1E",
                "JUND",
                "HSPH1",
                "HSP90AA1",
            ],
            True,
        ),
        "2": (
            [
                "CCL5",
                "NKG7",
                "GZMK",
                "EOMES",
                "GZMH",
                "CMC1",
                "CD8A",
                "GZMA",
                "PRF1",
                "SLA2",
                "C1orf21",
                "SAMD3",
                "GZMB",
                "CCL4",
                "KLRD1",
            ],
            False,
        ),
        "3": (
            [
                "FOXP3",
                "MIR4435-2HG",
                "AC017002.3",
                "LAYN",
                "LAIR2",
                "RTKN2",
                "MAGEH1",
                "ZC2HC1A",
                "TIGIT",
                "GPR55",
                "CTLA4",
                "ADTRP",
                "BATF",
                "CARD16",
                "HTATIP2",
            ],
            False,
        ),
        "5": (
            [
                "CXCL3",
                "PID1",
                "CD300E",
                "CXCL2",
                "IL1B",
                "CXCL8",
                "TLR2",
                "CD14",
                "EREG",
                "CXCL1",
                "CD93",
                "CXCL5",
                "MMP14",
                "MAFB",
                "FCGR2A",
            ],
            False,
        ),
        "6": (
            [
                "SH2D1B",
                "ADGRG3",
                "LINC00996",
                "KLRF1",
                "KRT86",
                "NCR1",
                "GNLY",
                "IGFBP7",
                "FCGR3A",
                "KIR2DL4",
                "CLIC3",
                "FGFBP2",
                "TNFSF4",
                "S1PR5",
                "TXK",
            ],
            False,
        ),
        "8": (
            [
                "TPSB2",
                "TPSAB1",
                "TPSD1",
                "CPA3",
                "GCSAML",
                "CTSG",
                "MLPH",
                "HDC",
                "MS4A2",
                "GATA2",
                "CALB2",
                "CMA1",
                "MAOB",
                "CAVIN2",
                "SLC24A3",
            ],
            False,
        ),
        "10": (
            [
                "TCL1A",
                "CD79A",
                "DLGAP5",
                "BIRC5",
                "LINC00926",
                "GTSE1",
                "IGHA1",
                "IGHM",
                "RRM2",
                "CDCA5",
                "CCNB2",
                "UBE2C",
                "AURKB",
                "CCNA2",
                "MKI67",
            ],
            True,
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")

    def test_measured_clusters_are_classified_correctly(self):
        """Every one of the 8 fixtures must land on the right side of the threshold."""
        for cl, (markers, expect_state) in self.PSORIASIS.items():
            p = self.tools.state_programme_profile(markers)
            self.assertEqual(
                p["state_dominated"],
                expect_state,
                f"cluster {cl}: state_fraction={p['state_fraction']} "
                f"dominant={p['dominant_programme']}",
            )

    def test_threshold_sits_clear_of_both_populations(self):
        """Measured separation: identity clusters <=0.13, state clusters 0.67.

        If a future panel edit narrows that gap, this fails rather than silently
        degrading the tier cap into a coin flip.
        """
        ident = [
            self.tools.state_programme_profile(m)["state_fraction"]
            for m, st in self.PSORIASIS.values()
            if not st
        ]
        state = [
            self.tools.state_programme_profile(m)["state_fraction"]
            for m, st in self.PSORIASIS.values()
            if st
        ]
        self.assertLess(max(ident), self.tools.STATE_DOMINANCE_THRESHOLD)
        self.assertGreater(min(state), self.tools.STATE_DOMINANCE_THRESHOLD)
        self.assertGreater(min(state), 2 * max(max(ident), 0.05))  # >=2x margin

    def test_dominant_programme_is_named(self):
        self.assertEqual(
            self.tools.state_programme_profile(self.PSORIASIS["1"][0])[
                "dominant_programme"
            ],
            "stress_heat_shock",
        )
        self.assertEqual(
            self.tools.state_programme_profile(self.PSORIASIS["10"][0])[
                "dominant_programme"
            ],
            "cell_cycle",
        )

    def test_naive_t_identity_genes_are_not_counted_as_state(self):
        """KLF2/ZFP36L2 double as naive/circulating T markers, so they are excluded
        from the immediate-early panel. Cluster 0 must score 0.0, not 0.13."""
        p = self.tools.state_programme_profile(self.PSORIASIS["0"][0])
        self.assertEqual(p["state_fraction"], 0.0)

    def test_empty_markers_are_not_reported_as_state(self):
        """Absent evidence is `markers_empty`, a different failure. Conflating them
        would hide it."""
        p = self.tools.state_programme_profile([])
        self.assertFalse(p["state_dominated"])
        self.assertEqual(p["n_markers_scored"], 0)

    def test_scoring_window_is_stable_when_top_n_changes(self):
        """Only the leading window is scored, so the statistic does not drift when
        `top_n_markers` moves between 15 and 50."""
        state = self.PSORIASIS["10"][0]
        padded = state + [f"FILLER{i}" for i in range(35)]  # simulate top_n=50
        self.assertEqual(
            self.tools.state_programme_profile(state)["state_fraction"],
            self.tools.state_programme_profile(padded)["state_fraction"],
        )

    def test_no_gene_is_claimed_by_two_state_programmes(self):
        """A gene in two panels discriminates neither (same rule as lineage_panels)."""
        seen, clash = {}, []
        for prog, panel in self.tools.STATE_PROGRAMMES.items():
            for g in panel:
                if g in seen:
                    clash.append(f"{g}: {seen[g]} + {prog}")
                seen[g] = prog
        self.assertEqual(clash, [])

    def test_state_panels_name_no_cell_type_or_disease(self):
        """Disease-agnostic invariant: these are programmes, not identities."""
        forbidden = ("CD3", "CD19", "EPCAM", "PTPRC", "COL1A1", "PECAM1")
        allgenes = {g for p in self.tools.STATE_PROGRAMMES.values() for g in p}
        for f in forbidden:
            self.assertFalse([g for g in allgenes if g.startswith(f)], f)


# =========================================================================== #
#  17b. Reference-model selection must match the OBSERVED compartment
# =========================================================================== #
class TestModelLineageRefinement(unittest.TestCase):
    """The organ comes from GEO title text; the compartment must come from the data.

    Measured on the psoriasis run: tissue resolved to "skin" so Adult_Human_Skin.pkl
    was loaded, yet all 11 clusters are immune (zero keratinocyte/fibroblast/
    endothelial). Skin has few B cells, so cluster 10 — TCL1A+/CD79A+/IGHM+ B cells —
    was called `DC1` by CellTypist, which is what dragged it to Low/Review.
    """

    @classmethod
    def setUpClass(cls):
        cls.tools = _imp("celltype_consensus.tools")
        cls.cat = _imp("celltype_consensus.celltypist_catalog")

    def _profile(self, dominant, fraction):
        return {"dominant_lineage": dominant, "dominant_fraction": fraction}

    def test_all_immune_dataset_switches_off_the_organ_model(self):
        model, why = self.tools.refine_celltypist_model_for_observed_lineage(
            "Adult_Human_Skin.pkl",
            self._profile("Immune", 1.0),
            valid_models=frozenset(self.cat.VALID_MODEL_NAMES),
        )
        self.assertEqual(model, "Immune_All_Low.pkl")
        self.assertIn("Immune", why)

    def test_mixed_tissue_keeps_the_organ_model(self):
        """A real solid-tissue cohort must keep its organ model — this rule exists to
        catch sorted/enriched data, not to funnel everything to the immune model."""
        for frac in (0.5, 0.72, 0.89):
            model, why = self.tools.refine_celltypist_model_for_observed_lineage(
                "Adult_Human_Skin.pkl",
                self._profile("Immune", frac),
                valid_models=frozenset(self.cat.VALID_MODEL_NAMES),
            )
            self.assertEqual(model, "Adult_Human_Skin.pkl", f"frac={frac}")
            self.assertEqual(why, "")

    def test_no_specialist_means_no_switch(self):
        """Epithelial/Fibroblast/Endothelial have no pan-tissue specialist in the
        catalog, so nothing is offered rather than something wrong."""
        for lin in ("Epithelial", "Fibroblast", "Endothelial", "Mural", "Other"):
            model, why = self.tools.refine_celltypist_model_for_observed_lineage(
                "Human_Lung_Atlas.pkl",
                self._profile(lin, 1.0),
                valid_models=frozenset(self.cat.VALID_MODEL_NAMES),
            )
            self.assertEqual(model, "Human_Lung_Atlas.pkl", lin)
            self.assertEqual(why, "")

    def test_already_correct_model_is_left_alone(self):
        model, why = self.tools.refine_celltypist_model_for_observed_lineage(
            "Immune_All_Low.pkl",
            self._profile("Immune", 1.0),
            valid_models=frozenset(self.cat.VALID_MODEL_NAMES),
        )
        self.assertEqual(model, "Immune_All_Low.pkl")
        self.assertEqual(why, "")

    def test_specialists_exist_in_the_catalog(self):
        for lin, m in self.tools.LINEAGE_SPECIALIST_MODELS.items():
            self.assertIn(m, self.cat.VALID_MODEL_NAMES, f"{lin} -> {m}")

    def test_decision_reads_no_disease_term(self):
        """Disease-blind by construction: the input is an expression profile."""
        _m, why = self.tools.refine_celltypist_model_for_observed_lineage(
            "Adult_Human_Skin.pkl",
            self._profile("Immune", 1.0),
            valid_models=frozenset(self.cat.VALID_MODEL_NAMES),
        )
        for term in ("psoriasis", "cancer", "tumor", "tumour", "disease", "fibrosis"):
            self.assertNotIn(term, why.lower())


class TestSubtypeAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cons = _imp("celltype_consensus.consensus")

    def test_subtype_records_its_source_and_confidence(self):
        sub, src, conf, rej = self.cons.pick_subtype_with_source(
            "T cell",
            [
                (None, "pubmed", None),
                ("Cytotoxic T cell", "knowledge_based", 0.91),
                ("Tc", "celltypist", 0.8),
            ],
        )
        self.assertEqual(sub, "Cytotoxic T cell")
        self.assertEqual(src, "knowledge_based")
        self.assertAlmostEqual(conf, 0.91)
        self.assertEqual(rej, [])

    def test_contradictory_subtype_is_rejected_and_falls_back_to_coarse(self):
        sub, src, conf, _rej = self.cons.pick_subtype_with_source(
            "B cell", [("DC1", "celltypist", 0.99)]
        )
        self.assertEqual(sub, "B cell")  # NOT promoted
        self.assertEqual(src, self.cons.SUBTYPE_SOURCE_COARSE)
        self.assertIsNone(conf)

    def test_priority_order_is_respected(self):
        sub, src, _c, _rej = self.cons.pick_subtype_with_source(
            "NK cell",
            [
                ("CD16+ NK cell", "pubmed", 0.7),
                ("Cytotoxic NK cell", "knowledge_based", 0.9),
            ],
        )
        self.assertEqual(sub, "CD16+ NK cell")  # first valid candidate wins
        self.assertEqual(src, "pubmed")

    # --- markers must back a subtype's claim (measured psoriasis failures) ---
    def test_cd8_subtype_needs_cd8a_in_the_cluster(self):
        """Psoriasis cluster 4, 7,416 cells: subtype "CD8-positive T cell" shipped
        on CXCL13/CXCR6/ADGRG1 markers with NO CD8A and NO CD8B. CXCL13 is a CD4
        Tph programme; CellTypist said `Th` (CD4) and was overridden."""
        cl4 = [
            "CXCL13",
            "PTPN13",
            "ADGRG1",
            "CXCR6",
            "SLC1A4",
            "LMO4",
            "NBAS",
            "JAML",
            "CTSH",
            "TMEM173",
            "FURIN",
            "TSHZ2",
            "RBPJ",
            "LINC01871",
            "ARL3",
        ]
        sub, src, _c, rej = self.cons.pick_subtype_with_source(
            "T cell", [("CD8-positive T cell", "pubmed", 0.9)], markers=cl4
        )
        self.assertEqual(sub, "T cell")  # claim withdrawn
        self.assertEqual(src, self.cons.SUBTYPE_SOURCE_COARSE)
        self.assertTrue(rej and "CD8A" in rej[0][1])
        # the same claim on a cluster that DOES show CD8A survives (cluster 2)
        cl2 = [
            "CCL5",
            "NKG7",
            "GZMK",
            "EOMES",
            "GZMH",
            "CMC1",
            "CD8A",
            "GZMA",
            "PRF1",
            "SLA2",
            "C1orf21",
            "SAMD3",
            "GZMB",
            "CCL4",
            "KLRD1",
        ]
        sub2, src2, _c2, _r2 = self.cons.pick_subtype_with_source(
            "T cell", [("cytotoxic CD8+ T cell", "pubmed", 0.9)], markers=cl2
        )
        self.assertEqual(sub2, "cytotoxic CD8+ T cell")
        self.assertEqual(src2, "pubmed")

    def test_hedged_treg_th2_subtype_without_foxp3_is_rejected(self):
        """Psoriasis cluster 9, 2,260 cells: subtype "CCR4-positive T cell (likely
        regulatory or Th2/skin-homing T cell)" with no FOXP3/IL2RA/CTLA4 and no
        GATA3/IL4/IL13. A subtype is an assertion, not a shortlist."""
        cl9 = [
            "CCR4",
            "SYNE2",
            "XIST",
            "PTPRC",
            "LNPEP",
            "PDE3B",
            "KIAA1109",
            "ATP2B4",
            "SLFN5",
            "PAG1",
            "KIAA1551",
            "RNF213",
            "MACF1",
            "GLG1",
            "ADAM10",
        ]
        sub, _s, _c, rej = self.cons.pick_subtype_with_source(
            "T cell",
            [
                (
                    "CCR4-positive T cell (likely regulatory or Th2/skin-homing T cell)",
                    "pubmed",
                    0.9,
                )
            ],
            markers=cl9,
        )
        self.assertEqual(sub, "T cell")
        self.assertTrue(rej)

    def test_plasmablast_claim_needs_plasma_cell_evidence(self):
        """Psoriasis cluster 10, 1,271 cells: subtype "cycling B cell / plasmablast"
        on a TCL1A+/CD79A+/IGHM+ naive B population — no MZB1/JCHAIN/XBP1. The
        cycling half is supported; the plasmablast half is not, so the whole
        assertion fails."""
        cl10 = [
            "TCL1A",
            "CD79A",
            "DLGAP5",
            "BIRC5",
            "LINC00926",
            "GTSE1",
            "IGHA1",
            "IGHM",
            "RRM2",
            "CDCA5",
            "CCNB2",
            "UBE2C",
            "AURKB",
            "CCNA2",
            "MKI67",
        ]
        sub, _s, _c, rej = self.cons.pick_subtype_with_source(
            "B cell", [("cycling B cell / plasmablast", "pubmed", 0.8)], markers=cl10
        )
        self.assertEqual(sub, "B cell")
        self.assertTrue(rej and "MZB1" in rej[0][1])

    def test_correct_naive_cd4_subtype_survives_cd4_dropout(self):
        """Psoriasis cluster 0, 24,875 cells: "naive/central memory CD4-positive T
        cell" is CORRECT (LEF1/TCF7/MAL/IL7R/S1PR1) and must not be rejected just
        because CD4 mRNA drops out in 10x. CD4 claims are gated on the ABSENCE of
        CD8 evidence, not the presence of CD4."""
        cl0 = [
            "IL7R",
            "PCSK1N",
            "KLF2",
            "SESN3",
            "PABPC1",
            "PASK",
            "LEF1",
            "FXYD7",
            "ZFP36L2",
            "MAL",
            "S1PR1",
            "ICAM2",
            "IMPDH2",
            "EEF1G",
            "TCF7",
        ]
        sub, src, _c, rej = self.cons.pick_subtype_with_source(
            "T cell",
            [("naive/central memory CD4-positive T cell", "pubmed", 0.9)],
            markers=cl0,
        )
        self.assertEqual(sub, "naive/central memory CD4-positive T cell")
        self.assertEqual(src, "pubmed")
        self.assertEqual(rej, [])

    def test_state_dominated_cluster_gets_no_subtype(self):
        """A cluster defined by a state programme has no identity evidence for a
        finer call to rest on, so every candidate is refused (psoriasis cluster 1)."""
        sub, src, conf, rej = self.cons.pick_subtype_with_source(
            "T cell",
            [("CD4-positive T cell", "pubmed", 0.9)],
            markers=["HSPA1A", "HSPA1B", "DNAJB1"],
            suppress=True,
        )
        self.assertEqual(sub, "T cell")
        self.assertEqual(src, self.cons.SUBTYPE_SOURCE_SUPPRESSED)
        self.assertIsNone(conf)
        self.assertTrue(rej)

    def test_unrecognised_subtype_makes_no_claim_and_is_kept(self):
        """The marker gate must never reject a label merely for being unfamiliar.

        "T cell subset A" asserts no marker-defined distinction, so there is nothing
        to verify and it must pass through untouched.
        """
        sub, src, _c, rej = self.cons.pick_subtype_with_source(
            "T cell",
            [("T cell subset A", "celltypist", 0.9)],
            markers=["CD3D", "CD3E", "IL7R"],
        )
        self.assertEqual(sub, "T cell subset A")
        self.assertEqual(src, "celltypist")
        self.assertEqual(rej, [])

    def test_legacy_positional_wrapper_still_works(self):
        pick = self.cons._pick_subtype
        self.assertEqual(
            pick("T cell", None, "Cytotoxic T cell", "Tc", None), "Cytotoxic T cell"
        )
        self.assertEqual(pick("B cell", None, None, "DC1", None), "B cell")
        self.assertEqual(pick("NK cell", None, "NK cell", None, None), "NK cell")


# =========================================================================== #
#  18. Backward compatibility — old configs run, columns survive, Bisque label
# =========================================================================== #
class TestBackwardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = _imp("pipeline")
        cls.bisq = _imp("sc_to_bisq")
        cls.cons = _imp("celltype_consensus.consensus")
        import importlib

        cls.main = importlib.import_module("main")

    # --- requirement: existing configs still run without modification ---
    def test_old_flat_config_produces_valid_kwargs(self):
        import inspect

        old_common = {
            "do_pathway_clustering": False,
            "do_dpt": True,
            "dpt_root_group": "auto",
            "geo_json_path": None,
            "logos_dir": None,
            "generate_report": True,
            "prepare_for_bisque": True,
            "enable_celltypist": True,
            "enable_knowledge_based": True,
            "enable_singler": True,
            "enable_pubmed": True,
            "tissue": "auto",
            "species": "auto",
            "celltypist_model": "auto",
            "singler_reference": "auto",
            "skip_tsne": True,
        }
        cfg = {
            "mode": "multi",
            "common": dict(old_common),
            "multi": {
                "multi_base_dir": "/x",
                "out_name": "o",
                "batch_key": "sample",
                "integration_method": "bbknn",
                "do_groupwise_de": True,
                "run_per_sample": False,
            },
        }
        # no nested sections at all -> flatten yields nothing, nothing breaks
        self.assertEqual(self.main.flatten_sections(cfg), {})
        from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x import (
            run_pipeline_multi,
        )

        kwargs = self.main._build_kwargs(
            run_pipeline_multi, {}, cfg["common"], cfg["multi"]
        )
        accepted = set(inspect.signature(run_pipeline_multi).parameters)
        self.assertTrue(set(kwargs) <= accepted)
        self.assertEqual(kwargs["out_name"], "o")
        # unset clustering/QC keys are simply absent -> pipeline defaults apply
        self.assertNotIn("leiden_resolution", kwargs)
        self.assertNotIn("qc_min_genes", kwargs)

    # --- requirement: new nested config actually reaches the drivers ---
    def test_new_nested_config_reaches_the_driver_signature(self):
        import inspect

        cfg = {
            "clustering": {
                "leiden_resolution": 0.8,
                "evaluate_resolutions": True,
                "resolution_candidates": [0.4, 0.8],
                "min_cluster_cells": 15,
            },
            "qc": {
                "min_genes": 250,
                "max_genes": 5500,
                "max_mito_percent": 12.0,
                "remove_doublets": False,
            },
            "annotation": {
                "mixed_cluster_min_dominant_fraction": 0.6,
                "use_subtypes_for_downstream": False,
                "reuse_existing_final_annotation": False,
            },
            "downstream": {
                "exclude_low_confidence_de": True,
                "excluded_consensus_tiers": ["Low/Review"],
            },
        }
        flat = self.main.flatten_sections(cfg)
        self.assertEqual(flat["leiden_resolution"], 0.8)
        self.assertEqual(flat["qc_min_genes"], 250)  # renamed to avoid collisions
        self.assertEqual(flat["qc_max_mito_percent"], 12.0)
        self.assertFalse(flat["remove_doublets"])
        self.assertTrue(flat["exclude_low_confidence_de"])
        self.assertEqual(flat["mixed_cluster_min_dominant_fraction"], 0.6)

        # every flattened key must be a real parameter of BOTH drivers and the pipeline
        from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x import (
            run_pipeline,
            run_pipeline_multi,
        )

        for fn in (run_pipeline, run_pipeline_multi, self.pipe.run_scanpy_pipeline):
            params = set(inspect.signature(fn).parameters)
            missing = sorted(set(flat) - params)
            self.assertEqual(missing, [], f"{fn.__name__} cannot receive {missing}")

    def test_sections_are_also_read_from_common(self):
        flat = self.main.flatten_sections(
            {"common": {"clustering": {"leiden_resolution": 0.3}}}
        )
        self.assertEqual(flat["leiden_resolution"], 0.3)

    def test_unknown_section_keys_are_reported_not_silently_applied(self):
        flat = self.main.flatten_sections({"clustering": {"not_a_real_key": 1}})
        self.assertEqual(flat, {})

    # --- requirement: existing output columns remain present ---
    def test_existing_obs_columns_are_still_produced(self):
        import inspect

        src = inspect.getsource(self.cons.run_consensus_annotation)
        for col in (
            "lineage_coarse",
            "celltype_celltypist",
            "celltype_knowledge_based",
            "consensus_tier",
            "annotation_provenance",
            "celltype_singler",
            "celltype_pubmed",
        ):
            self.assertIn(col, src, col)
        self.assertEqual(self.cons.CELLTYPE_CONSENSUS_COL, "celltype_consensus")

    def test_new_consensus_table_keeps_the_legacy_column_names(self):
        import inspect

        src = inspect.getsource(self.cons.run_consensus_annotation)
        for legacy in (
            '"cluster"',
            '"consensus"',
            '"tier"',
            '"lineage_gate"',
            '"celltypist"',
            '"celltypist_conf"',
            '"singler"',
            '"singler_conf"',
            '"knowledge_based"',
            '"pubmed"',
            '"top_markers"',
            '"provenance"',
        ):
            self.assertIn(legacy, src, legacy)
        for added in (
            '"leiden"',
            '"final_celltype"',
            '"consensus_tier"',
            '"celltypist_label"',
            '"celltypist_confidence"',
            '"mixed_cluster_flag"',
            '"decision_reason"',
            '"included_in_downstream_analysis"',
            '"n_cells"',
        ):
            self.assertIn(added, src, added)

    # --- requirement: Bisque uses the intended final label ---
    def test_bisque_keeps_celltype_and_sample_and_drops_leiden(self):
        import anndata as ad
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 12
        obs = pd.DataFrame(
            {
                "celltype": ["T cell"] * 6 + ["B cell"] * 6,
                "celltype_consensus": ["T cell"] * 6 + ["B cell"] * 6,
                "celltype_subtype": ["Cytotoxic T cell"] * 6 + ["B cell"] * 6,
                "sample": ["s1", "s2"] * 6,
                "leiden": ["0"] * 6 + ["1"] * 6,
                "consensus_tier": ["High"] * 6 + ["Low/Review"] * 6,
                "include_in_downstream_analysis": [True] * 6 + [False] * 6,
                "total_counts": rng.integers(100, 200, n).astype(float),
                "pct_counts_mt": rng.random(n) * 5,
            },
            index=[f"c{i}" for i in range(n)],
        )
        X = rng.poisson(3, size=(n, 4)).astype(float)
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=list("abcd")))
        a.layers["counts"] = X.copy()

        out = self.bisq.prepare_for_bisque(a)
        self.assertIn("celltype", out.obs.columns)  # the final consensus label
        self.assertIn("sample", out.obs.columns)
        self.assertNotIn("leiden", out.obs.columns)  # historical behaviour
        self.assertEqual(out.n_obs, n)  # no cells dropped
        # counts (not log-norm) are what Bisque receives
        self.assertTrue(
            np.allclose(
                np.asarray(out.X.todense() if hasattr(out.X, "todense") else out.X), X
            )
        )
        # low-confidence cells survive the export for audit
        self.assertEqual(int(out.obs["celltype"].value_counts().sum()), n)

    # --- requirement: low-confidence cells remain in the exported h5ad ---
    def test_low_confidence_cells_survive_an_h5ad_roundtrip(self):
        import tempfile

        import scanpy as sc

        a = build_synth_adata(cells_per_cluster=6)
        tier = {str(i): ("Low/Review" if i >= 8 else "High") for i in range(10)}
        a.obs["consensus_tier"] = (
            a.obs["leiden"].astype(str).map(tier).astype("category")
        )
        a.obs["celltype_consensus"] = "T cell"
        a.obs["celltype"] = "T cell"
        g = _imp("downstream_gating")
        g.annotate_downstream_inclusion(a, exclude_low_confidence=True)
        n_excluded = int((~a.obs[g.INCLUDE_COL]).sum())
        self.assertGreater(n_excluded, 0)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "processed.h5ad"
            a.write_h5ad(p)
            back = sc.read_h5ad(p)
        self.assertEqual(back.n_obs, a.n_obs)  # nothing dropped on export
        self.assertIn(g.INCLUDE_COL, back.obs.columns)
        self.assertEqual(int((~back.obs[g.INCLUDE_COL].astype(bool)).sum()), n_excluded)
        self.assertTrue(
            (
                back.obs.loc[
                    ~back.obs[g.INCLUDE_COL].astype(bool), "consensus_tier"
                ].astype(str)
                == "Low/Review"
            ).all()
        )


# =========================================================================== #
#  19. Report — the annotation-confidence section reads the consensus CSV
# =========================================================================== #
class TestAnnotationReportSection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = _imp("singlecell_sc_report_generation")

    def _write_csv(self, d: Path):
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "cluster": "0",
                    "leiden": "0",
                    "n_cells": 100,
                    "consensus": "T cell",
                    "final_celltype": "T cell",
                    "tier": "High",
                    "consensus_tier": "High",
                    "lineage_coarse": "Immune",
                    "celltypist_label": "Tc",
                    "singler_label": "T_cells",
                    "knowledge_based_label": "T cell",
                    "pubmed_label": "T cell",
                    "voters_disagree": False,
                    "mixed_cluster_flag": False,
                    "celltype_subtype": "Cytotoxic T cell",
                    "included_in_downstream_analysis": True,
                    "decision_reason": "unanimous",
                },
                {
                    "cluster": "1",
                    "leiden": "1",
                    "n_cells": 40,
                    "consensus": "B cell",
                    "final_celltype": "B cell",
                    "tier": "Low/Review",
                    "consensus_tier": "Low/Review",
                    "lineage_coarse": "Immune",
                    "celltypist_label": "DC1",
                    "singler_label": "T_cells",
                    "knowledge_based_label": "B cell",
                    "pubmed_label": "B cell",
                    "voters_disagree": True,
                    "mixed_cluster_flag": True,
                    "celltype_subtype": "B cell",
                    "included_in_downstream_analysis": False,
                    "decision_reason": "adjudicated",
                },
                {
                    "cluster": "2",
                    "leiden": "2",
                    "n_cells": 10,
                    "consensus": "Unassigned",
                    "final_celltype": "Unassigned",
                    "tier": "Low/Review",
                    "consensus_tier": "Low/Review",
                    "lineage_coarse": "Other",
                    "celltypist_label": "Unassigned",
                    "singler_label": "Unassigned",
                    "knowledge_based_label": "Unassigned",
                    "pubmed_label": "Unassigned",
                    "voters_disagree": True,
                    "mixed_cluster_flag": False,
                    "celltype_subtype": "Unassigned",
                    "included_in_downstream_analysis": False,
                    "decision_reason": "no voters",
                },
            ]
        ).to_csv(d / "t_consensus_annotation.csv", index=False)

    def test_summary_counts_every_required_category(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_csv(Path(d))
            df, s = self.rep.read_consensus_annotation(Path(d))
        self.assertIsNotNone(df)
        self.assertEqual(s["n_clusters"], 3)
        self.assertEqual(s["high"], 1)
        self.assertEqual(s["medium"], 0)
        self.assertEqual(s["low_review"], 2)
        self.assertEqual(s["mixed"], 1)
        self.assertEqual(s["excluded"], 2)
        self.assertEqual(s["unassigned"], 1)
        self.assertEqual(s["n_cells"], 150)
        self.assertEqual(s["n_cells_excluded"], 50)

    def test_display_table_shows_all_voters_and_inclusion(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_csv(Path(d))
            df, _ = self.rep.read_consensus_annotation(Path(d))
        for col in (
            "Cluster",
            "Cells",
            "Final cell type",
            "Tier",
            "Lineage gate",
            "CellTypist",
            "SingleR",
            "Knowledge-based",
            "PubMed",
            "Voters disagree",
            "Mixed cluster",
            "Subtype",
            "In downstream",
        ):
            self.assertIn(col, df.columns, col)
        self.assertEqual(df["In downstream"].tolist(), ["yes", "EXCLUDED", "EXCLUDED"])
        self.assertEqual(df["Mixed cluster"].tolist(), ["no", "yes", "no"])

    def test_missing_table_omits_the_section(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            df, s = self.rep.read_consensus_annotation(Path(d))
        self.assertIsNone(df)
        self.assertEqual(s, {})

    def test_resource_rows_expose_resolved_values(self):
        rows = self.rep.annotation_resource_rows(
            {
                "params": {
                    "clustering": {"leiden_resolution": 0.8, "n_clusters": 12},
                    "integration_method": "bbknn",
                    "integration_method_used": "bbknn",
                    "annotation": {
                        "requested_celltypist_model": "auto",
                        "resolved_celltypist_model": "Human_Lung_Atlas.pkl",
                        "requested_singler_reference": "auto",
                        "resolved_singler_reference": "BlueprintEncodeData",
                        "annotation_tissue": "lung",
                        "annotation_species": "human",
                        "celltypist_enabled": True,
                        "singler_enabled": True,
                        "knowledge_based_enabled": True,
                        "pubmed_enabled": False,
                    },
                    "qc": {"applied": {"min_genes": 200}},
                    "confidence_filtering": {
                        "exclude_low_confidence_de": True,
                        "excluded_consensus_tiers": ["Low/Review"],
                    },
                }
            }
        )
        d = dict(rows)
        self.assertEqual(d["CellTypist model used"], "Human_Lung_Atlas.pkl")
        self.assertEqual(d["SingleR reference used"], "BlueprintEncodeData")
        self.assertEqual(d["Leiden resolution (used)"], "0.8")
        self.assertNotIn("PubMed", d["Voters enabled"])
        self.assertIn("CellTypist", d["Voters enabled"])

    def test_report_template_renders_the_new_section(self):
        html = self.rep.HTML.render(
            consensus_summary={
                "high": 1,
                "medium": 0,
                "low_review": 2,
                "mixed": 1,
                "excluded": 2,
                "unassigned": 1,
                "n_clusters": 3,
            },
            consensus_table="<table></table>",
            consensus_warnings=["a caveat"],
            annotation_resources=[("CellTypist model used", "X.pkl")],
        )
        for probe in (
            "Annotation Confidence",
            "High-confidence clusters",
            "Low/Review clusters",
            "Mixed clusters",
            "Excluded clusters",
            "Unassigned clusters",
            "a caveat",
            "X.pkl",
        ):
            self.assertIn(probe, html, probe)

    def test_section_absent_without_consensus_data(self):
        self.assertNotIn("Annotation Confidence", self.rep.HTML.render())


if __name__ == "__main__":
    unittest.main(verbosity=2)
