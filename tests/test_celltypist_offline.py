"""test_celltypist_offline.py — does the CellTypist voter survive an air-gapped machine?

The shipped package carries the CellTypist model set under
``shared_reference/celltypist_models/data/models/``. Without it, a recipient with
no network gets no CellTypist vote at all: the tool tries to fetch its .pkl over
HTTP, fails, and the voter abstains on every cluster — a silent loss of one of the
four opinions the consensus is built on.

This test asserts the guarantee rather than the plumbing. It makes the machine look
brand new (``CELLTYPIST_FOLDER`` pointed at an empty directory, so the developer's
own ``~/.celltypist`` cache cannot mask a failure) and then hard-blocks every
outbound HTTP call before the pipeline is imported. Under those conditions:

  1. WITH the bundle       -> real per-cell labels, and not one network call.
  2. WITHOUT the bundle    -> a clean abstention (``None``), never an exception.

Case 2 matters as much as case 1: the voter is allowed to be unavailable, it is not
allowed to take the run down with it.

Run directly (no pytest required):

    python tests/test_celltypist_offline.py

Skips, rather than fails, in a source checkout that carries no model bundle.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Both must happen before celltypist is imported anywhere: `models_path` is
# resolved from the environment at import time and then frozen for the process.
_FRESH_CACHE = tempfile.mkdtemp(prefix="fresh_celltypist_")
os.environ["CELLTYPIST_FOLDER"] = _FRESH_CACHE

MODEL = "Adult_Human_Skin.pkl"  # the tissue of the bundled demo cohort
N_CELLS = 240
N_GENES = 800


class NetworkAccessAttempted(RuntimeError):
    """Raised in place of any HTTP call, so a download shows up as a failure."""


def _block_http() -> None:
    """Make every outbound request raise. Covers the requests entry points that
    ``celltypist.models`` reaches through (``_requests_get`` -> ``requests.get``)."""
    import requests

    def blocked(*args, **kwargs):
        raise NetworkAccessAttempted(f"network access attempted: {args[:1]}")

    requests.get = blocked
    requests.Session.get = blocked
    requests.Session.request = blocked


def _lognorm_adata(features):
    """A small log1p(CP10k) AnnData over real model features, as the voter expects."""
    import anndata as ad
    import numpy as np
    import scanpy as sc

    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(N_CELLS, len(features))).astype("float32"))
    a.var_names = list(features)
    a.obs_names = [f"cell{i}" for i in range(a.n_obs)]
    a.obs["leiden"] = [str(i % 4) for i in range(a.n_obs)]
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


class TestCellTypistOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _block_http()
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

        from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.celltype_consensus import (
            tools,
        )

        cls.tools = tools
        cls.bundled = tools._bundled_celltypist_model(MODEL)
        if not cls.bundled:
            raise unittest.SkipTest(
                f"no bundled CellTypist model set (looked for {MODEL} under "
                "shared_reference/celltypist_models/) — expected in a source checkout"
            )

        import celltypist

        cls.adata = _lognorm_adata(
            list(celltypist.models.Model.load(cls.bundled).features)[:N_GENES]
        )

    def test_01_fresh_cache_is_actually_empty(self):
        """Guard the premise: a populated cache would make the test vacuous."""
        from celltypist import models as ctm

        self.assertEqual(
            len(list(Path(ctm.models_path).glob("*.pkl"))),
            0,
            f"CELLTYPIST_FOLDER {ctm.models_path} is not empty; the home cache "
            "could be answering instead of the bundle",
        )

    def test_02_bundled_path_uses_forward_slashes(self):
        """A backslash path sends Model.load down its bare-name branch, which calls
        get_all_models() -> download_if_required() and pulls the whole repertoire."""
        self.assertIn("/", self.bundled)
        self.assertNotIn("\\", self.bundled)
        self.assertTrue(Path(self.bundled).is_file())

    def test_03_http_is_really_blocked(self):
        import requests

        with self.assertRaises(NetworkAccessAttempted):
            requests.get("https://example.invalid")

    def test_04_annotates_with_no_network_and_no_cache(self):
        labels = self.tools._celltypist_per_cell_labels(self.adata, "leiden", MODEL)
        self.assertIsNotNone(labels, "voter abstained on a fresh air-gapped machine")
        self.assertEqual(len(labels), self.adata.n_obs)
        self.assertGreater(labels.nunique(), 0)
        self.assertListEqual(list(labels.index), list(self.adata.obs_names))

    def test_05_abstains_cleanly_when_bundle_is_absent(self):
        """Negative control: no bundle and no network must degrade, not raise."""
        self.tools._bundled_celltypist_model.cache_clear()
        prev = os.environ.get("SHARED_REFERENCE_ROOT")
        os.environ["SHARED_REFERENCE_ROOT"] = _FRESH_CACHE
        try:
            self.assertIsNone(self.tools._bundled_celltypist_model(MODEL))
            self.assertIsNone(
                self.tools._celltypist_per_cell_labels(self.adata, "leiden", MODEL)
            )
        finally:
            if prev is None:
                os.environ.pop("SHARED_REFERENCE_ROOT", None)
            else:
                os.environ["SHARED_REFERENCE_ROOT"] = prev
            self.tools._bundled_celltypist_model.cache_clear()

    def test_06_every_catalogued_model_is_bundled(self):
        """The selector may return any name in the catalog, so all of them must be
        present — otherwise auto-selection silently loses the voter on some tissues."""
        from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.celltypist_catalog import (
            HUMAN_CELLTYPIST_MODELS,
        )

        missing = [
            m["model"]
            for m in HUMAN_CELLTYPIST_MODELS
            if not self.tools._bundled_celltypist_model(m["model"])
        ]
        self.assertListEqual(
            missing, [], f"{len(missing)} catalogued models not bundled"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
