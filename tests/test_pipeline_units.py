#!/usr/bin/env python3
"""
test_pipeline_units.py — focused unit tests for the single-cell pipeline core.

Written with the stdlib :mod:`unittest` so they run with no extra dependency:

    python tests/test_pipeline_units.py

These tests are finer-grained than the benchmark harness: they pin the behavior of
individual functions — the 10x loader, QC helpers, seed discipline, pseudobulk
aggregation, consensus vote counting/harmonization, the PubMed voter's marker
cleaning / query ladder / confidence scoring, gene-name heuristics, and the
provenance manifest — as well as smoke-checking that both static auditors import
and run. Everything is deterministic (fixed seeds); no network, no credentials.
"""

from __future__ import annotations

import gzip
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Locate + import the real pipeline package
# --------------------------------------------------------------------------- #
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "agentic_ai_wf").is_dir():
            return p
    raise RuntimeError("repo root with 'agentic_ai_wf' not found")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PKG = "agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x"


def _write_10x(dir_path: Path, mat_genes_by_cells, gene_ids, gene_syms, barcodes):
    from scipy import sparse as sp
    from scipy.io import mmwrite

    dir_path.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    mmwrite(buf, sp.coo_matrix(mat_genes_by_cells), field="integer")
    with gzip.open(dir_path / "matrix.mtx.gz", "wb") as f:
        f.write(buf.getvalue())
    with gzip.open(dir_path / "barcodes.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(barcodes) + "\n")
    with gzip.open(dir_path / "features.tsv.gz", "wt", encoding="utf-8") as f:
        f.write(
            "\n".join(
                f"{i}\t{s}\tGene Expression"
                for i, s in zip(gene_ids, gene_syms, strict=True)
            )
            + "\n"
        )


# --------------------------------------------------------------------------- #
#  loader_10x
# --------------------------------------------------------------------------- #
class TestLoader10x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module

        cls.load = staticmethod(
            import_module(f"{PKG}.loader_10x").load_10x_feature_barcode_matrix
        )

    def test_roundtrip_shape_and_symbols(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            rng = np.random.default_rng(0)
            mat = rng.poisson(1.0, size=(12, 8))  # genes x cells
            _write_10x(
                d,
                mat,
                [f"ENSG{i}" for i in range(12)],
                [f"SYM{i}" for i in range(12)],
                [f"BC{j}" for j in range(8)],
            )
            a = self.load(d)
            self.assertEqual(a.shape, (8, 12))  # transposed to cells x genes
            self.assertEqual(str(a.var["gene_symbol"].iloc[0]), "SYM0")
            self.assertIn("feature_id", a.var.columns)
            self.assertEqual(int(round(a.X.toarray().sum())), int(mat.sum()))

    def test_missing_files_raise_filenotfound(self):
        with (
            tempfile.TemporaryDirectory() as td,
            self.assertRaises(FileNotFoundError),
        ):
            self.load(Path(td))

    def test_corrupt_matrix_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            with gzip.open(d / "matrix.mtx.gz", "wt", encoding="utf-8") as f:
                f.write("not a matrix market file\n")
            with gzip.open(d / "barcodes.tsv.gz", "wt", encoding="utf-8") as f:
                f.write("BC0\n")
            with gzip.open(d / "features.tsv.gz", "wt", encoding="utf-8") as f:
                f.write("ENSG0\tSYM0\tGene Expression\n")
            with self.assertRaises(ValueError):
                self.load(d)


# --------------------------------------------------------------------------- #
#  reproducibility
# --------------------------------------------------------------------------- #
class TestReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module

        cls.rp = import_module(f"{PKG}.reproducibility")

    def test_set_global_seed_makes_numpy_deterministic(self):
        self.rp.set_global_seed(0)
        a = np.random.rand(4)
        self.rp.set_global_seed(0)
        b = np.random.rand(4)
        self.assertTrue(np.allclose(a, b))

    def test_capture_versions_has_core_packages(self):
        v = self.rp.capture_versions()
        for pkg in ("python", "scanpy", "numpy", "pandas"):
            self.assertIn(pkg, v)

    def test_manifest_is_complete(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            p = self.rp.write_run_manifest(
                Path(td), analysis_name="t", seed=0, params={"a": 1}, n_obs=10, n_vars=5
            )
            man = json.loads(Path(p).read_text(encoding="utf-8"))
            for k in (
                "analysis_name",
                "seed",
                "dataset",
                "params",
                "package_versions",
                "timestamp_utc",
            ):
                self.assertIn(k, man)
            self.assertEqual(man["dataset"], {"n_obs": 10, "n_vars": 5})
            self.assertEqual(man["seed"], 0)


# --------------------------------------------------------------------------- #
#  pseudobulk_de
# --------------------------------------------------------------------------- #
class TestPseudobulk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module

        cls.pb = import_module(f"{PKG}.pseudobulk_de")

    def _adata(self):
        import anndata as ad

        rng = np.random.default_rng(1)
        samples, groups, rows = [], [], []
        for s, (g, n) in {
            "s1": ("A", 20),
            "s2": ("A", 20),
            "s3": ("B", 20),
            "s4": ("B", 4),
        }.items():
            rows.append(rng.poisson(3.0, size=(n, 6)))
            samples += [s] * n
            groups += [g] * n
        X = np.vstack(rows).astype(float)
        a = ad.AnnData(X=X)
        a.var_names = [f"G{i}" for i in range(6)]
        a.obs["sample"] = samples
        a.obs["group"] = groups
        a.layers["counts"] = X.copy()
        return a, X, np.array(samples)

    def test_aggregation_sums_match(self):
        a, X, samples = self._adata()
        counts_df, meta_df = self.pb._build_pseudobulk(a, "sample", "group")
        for s in ("s1", "s2", "s3"):
            expected = X[samples == s].sum(axis=0).round().astype(int).tolist()
            self.assertEqual(counts_df.loc[s].astype(int).tolist(), expected)

    def test_min_cells_gate_excludes_small_sample(self):
        a, _, _ = self._adata()
        counts_df, _ = self.pb._build_pseudobulk(a, "sample", "group")
        self.assertNotIn("s4", counts_df.index)  # 4 cells < MIN_CELLS_PER_PSEUDOBULK

    def test_group_mapping(self):
        a, _, _ = self._adata()
        _, meta_df = self.pb._build_pseudobulk(a, "sample", "group")
        self.assertEqual(meta_df.loc["s1", "group"], "A")
        self.assertEqual(meta_df.loc["s3", "group"], "B")


# --------------------------------------------------------------------------- #
#  celltype_consensus.tools
# --------------------------------------------------------------------------- #
class TestConsensusVoting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module

        cls.tools = import_module(f"{PKG}.celltype_consensus.tools")

    def test_harmonize_synonyms(self):
        self.assertEqual(
            self.tools.harmonize_label("T cells"), self.tools.harmonize_label("T_cells")
        )

    def test_harmonize_unknown(self):
        self.assertEqual(self.tools.harmonize_label(None), self.tools.UNASSIGNED)

    def test_unanimous(self):
        t = self.tools.harmonize_label("T cell")
        r = self.tools.tally_votes({"a": t, "b": t, "c": t})
        self.assertTrue(r["unanimous"])
        self.assertEqual(r["majority_label"], t)

    def test_majority(self):
        t = self.tools.harmonize_label("T cell")
        b = self.tools.harmonize_label("B cell")
        r = self.tools.tally_votes({"a": t, "b": t, "c": b})
        self.assertEqual(r["majority_label"], t)
        self.assertEqual(r["majority_count"], 2)

    def test_tie_broken_by_confidence(self):
        t = self.tools.harmonize_label("T cell")
        b = self.tools.harmonize_label("B cell")
        r = self.tools.tally_votes({"a": t, "b": b}, {"a": 0.9, "b": 0.1})
        self.assertTrue(r["tied"])
        self.assertEqual(r["majority_label"], t)

    def test_all_abstain(self):
        r = self.tools.tally_votes({"a": self.tools.UNASSIGNED})
        self.assertEqual(r["n_methods"], 0)
        self.assertEqual(r["majority_label"], self.tools.UNASSIGNED)


# --------------------------------------------------------------------------- #
#  pubmed_annotation
# --------------------------------------------------------------------------- #
class TestPubMedVoter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module

        cls.pa = import_module(f"{PKG}.pubmed_annotation")

    def test_clean_markers_removes_lowinfo(self):
        cleaned = self.pa.clean_markers(
            ["CD3D", "RPL13", "MT-CO1", "MALAT1", "cd8a", "CD3D", "XIST"],
            species="human",
            top_n=30,
        )
        self.assertNotIn("RPL13", cleaned)
        self.assertNotIn("MT-CO1", cleaned)
        self.assertNotIn("MALAT1", cleaned)
        self.assertIn("CD3D", cleaned)
        self.assertIn("CD8A", cleaned)  # uppercased
        self.assertEqual(cleaned.count("CD3D"), 1)  # deduped

    def test_clean_markers_caps_top_n(self):
        genes = [f"GENE{i}" for i in range(100)]
        self.assertEqual(len(self.pa.clean_markers(genes, top_n=10)), 10)

    def test_query_ladder_broadens(self):
        ladder = self.pa.build_query_ladder(
            ["CD3D", "CD8A"],
            disease="cervical cancer",
            biosample="cervix",
            species="human",
        )
        self.assertEqual(
            [lvl for lvl, _ in ladder], ["tissue+concept", "concept", "gene-only"]
        )

    def test_confidence_uncited_is_reviewed_and_not_high(self):
        band, score, review = self.pa.compute_confidence(
            {
                "cell_type": "CD8-positive T cell",
                "confidence": "high",
                "supporting_markers": ["CD3D", "CD8A", "GZMK"],
                "pmids": [],
                "contradicting_markers": [],
                "review_required": False,
            },
            n_abstracts=6,
        )
        self.assertNotEqual(band, "high")  # no PMIDs -> cannot be high
        self.assertTrue(review)

    def test_confidence_unknown_capped_low(self):
        band, score, review = self.pa.compute_confidence(
            {
                "cell_type": "Unknown",
                "confidence": "low",
                "supporting_markers": [],
                "pmids": [],
                "contradicting_markers": [],
                "review_required": True,
            },
            n_abstracts=0,
        )
        self.assertEqual(band, "low")
        self.assertLessEqual(score, 0.25)
        self.assertTrue(review)

    def test_confidence_strong_beats_uncited(self):
        strong = self.pa.compute_confidence(
            {
                "cell_type": "CD8-positive T cell",
                "confidence": "high",
                "supporting_markers": ["CD3D", "CD8A", "GZMK"],
                "pmids": ["1", "2"],
                "contradicting_markers": [],
                "review_required": False,
            },
            n_abstracts=6,
        )[1]
        uncited = self.pa.compute_confidence(
            {
                "cell_type": "CD8-positive T cell",
                "confidence": "high",
                "supporting_markers": ["CD3D", "CD8A", "GZMK"],
                "pmids": [],
                "contradicting_markers": [],
                "review_required": False,
            },
            n_abstracts=6,
        )[1]
        self.assertGreater(strong, uncited)


# --------------------------------------------------------------------------- #
#  gene_names
# --------------------------------------------------------------------------- #
class TestGeneNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module

        cls.gn = import_module(f"{PKG}.gene_names")

    def test_looks_like_ensembl_true(self):
        self.assertTrue(
            self.gn.looks_like_ensembl([f"ENSG{i:011d}" for i in range(20)])
        )

    def test_looks_like_ensembl_false(self):
        self.assertFalse(self.gn.looks_like_ensembl(["CD3D", "CD8A", "MS4A1", "LYZ"]))

    def test_empty_is_false(self):
        self.assertFalse(self.gn.looks_like_ensembl([]))


# --------------------------------------------------------------------------- #
#  Both static auditors import and run
# --------------------------------------------------------------------------- #
class TestAuditorsRun(unittest.TestCase):
    """The two STATIC auditor scripts, when they are present.

    ``audit/module_auditor.py`` and ``audit/pipeline_auditor.py`` are development
    tools that are not part of this package (see STANDALONE_LAYOUT.md). Without them
    these two tests used to ERROR with ``ModuleNotFoundError``, which reads as a
    pipeline defect in every test run. They now skip with the reason, so a red result
    means something is actually broken.
    """

    @classmethod
    def setUpClass(cls):
        if not (REPO_ROOT / "audit").is_dir():
            raise unittest.SkipTest(
                f"static auditors not shipped in this package "
                f"(no {REPO_ROOT / 'audit'}); see STANDALONE_LAYOUT.md"
            )

    def test_module_auditor_scores_the_pipeline(self):
        sys.path.insert(0, str(REPO_ROOT / "audit"))
        import importlib

        ma = importlib.import_module("module_auditor")
        target = (
            REPO_ROOT
            / "agentic_ai_wf"
            / "single_cell_pipeline_agent"
            / "singlecell_10x"
        )
        rep = ma.audit(target, include_consensus=True)
        self.assertGreaterEqual(rep["overall_score"], 95.0)
        self.assertGreater(rep["modules_scored"], 10)

    def test_pipeline_auditor_scores_the_pipeline(self):
        sys.path.insert(0, str(REPO_ROOT / "audit"))
        import importlib

        pa = importlib.import_module("pipeline_auditor")
        target = (
            REPO_ROOT
            / "agentic_ai_wf"
            / "single_cell_pipeline_agent"
            / "singlecell_10x"
        )
        rep = pa.audit(target, include_consensus=False)
        self.assertGreaterEqual(rep["overall_score"], 95.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
