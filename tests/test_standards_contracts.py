#!/usr/bin/env python3
"""
test_standards_contracts.py — tests for the guarantees the standards fixes introduced.

Each test pins one contract that used to be unenforced, so a regression fails here
instead of silently in a run:

* output directories are never resolved against the process CWD (`output_paths`)
* label-derived filenames are always sanitised (`safe_names`)
* table writes are atomic and never leave a truncated file behind (`atomic_io`)
* every emitted table uses one column vocabulary (`column_names`)
* both drivers accept the same option set, keyword-only (`pipeline_options`)
* a CellTypist model is checksum-verified before it can be unpickled
  (`celltype_consensus.model_integrity`)

Deterministic, offline, and all file output goes to a temp directory.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x import (  # noqa: E402
    run_pipeline,
    run_pipeline_multi,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.atomic_io import (  # noqa: E402
    atomic_write,
    read_table,
    write_table,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.celltype_consensus import (  # noqa: E402
    model_integrity,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.model_integrity import (  # noqa: E402
    ModelIntegrityError,
    ensure_model_file,
    load_manifest,
    verify_model_file,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.column_names import (  # noqa: E402
    to_canonical_columns,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.output_paths import (  # noqa: E402
    OutputPathError,
    resolve_output_dir,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.pipeline_options import (  # noqa: E402
    MULTI_ONLY_PARAMS,
    SHARED_PARAMS,
    SINGLE_ONLY_PARAMS,
    PipelineOptions,
    UnknownPipelineOption,
    check_parameter_contract,
)
from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.safe_names import (  # noqa: E402
    safe_filename,
)


class TestOutputPathResolution(unittest.TestCase):
    """A relative out_name must never be resolved against the working directory."""

    def test_relative_out_name_without_root_raises(self):
        with self.assertRaises(OutputPathError) as ctx:
            resolve_output_dir("SC_RESULTS", None, create=False)
        self.assertIn("current working directory", str(ctx.exception))

    def test_relative_out_name_resolves_under_the_given_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = resolve_output_dir("run_a", root)
            self.assertEqual(result, (root / "run_a").resolve())
            self.assertTrue(result.is_dir())

    def test_absolute_out_name_needs_no_root(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "explicit"
            self.assertEqual(resolve_output_dir(target, None), target.resolve())

    def test_out_name_escaping_the_root_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "inside"
            root.mkdir()
            with self.assertRaises(OutputPathError):
                resolve_output_dir(Path("..") / "escaped", root, create=False)


class TestSafeFilename(unittest.TestCase):
    """Label-to-filename sanitisation must strip every Windows-reserved character."""

    def test_colon_is_removed(self):
        # The measured failure: "Other: X" wrote into an NTFS alternate data stream.
        self.assertNotIn(":", safe_filename("Other: GABAergic interneuron"))

    def test_every_reserved_character_is_removed(self):
        produced = safe_filename('a<b>c:d"e/f\\g|h?i*j')
        for reserved in '<>:"/\\|?*':
            self.assertNotIn(reserved, produced)

    def test_unusable_label_falls_back(self):
        self.assertEqual(safe_filename(":::"), "unnamed")

    def test_reserved_device_stem_is_escaped(self):
        self.assertNotEqual(safe_filename("CON").upper(), "CON")


class TestAtomicWrites(unittest.TestCase):
    """A failed write must leave neither a partial file nor a stray temp file."""

    def test_failed_write_leaves_nothing_behind(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "table.csv"

            def failing_write(path: Path) -> None:
                path.write_text("partial", encoding="utf-8")
                raise OSError("disk full")

            with self.assertRaises(OSError):
                atomic_write(target, failing_write)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_successful_write_leaves_no_temp_file(self):
        with TemporaryDirectory() as tmp:
            target = write_table(pd.DataFrame({"gene": ["A"]}), Path(tmp) / "t.csv")
            self.assertTrue(target.is_file())
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_write_does_not_add_an_index_column(self):
        with TemporaryDirectory() as tmp:
            path = write_table(pd.DataFrame({"gene": ["A", "B"]}), Path(tmp) / "t.csv")
            self.assertEqual(list(pd.read_csv(path).columns), ["gene"])


class TestCanonicalColumns(unittest.TestCase):
    """Every producer's column names must converge on one output vocabulary."""

    def test_deseq2_names_are_canonicalised(self):
        frame = to_canonical_columns(
            pd.DataFrame(
                {
                    "baseMean": [1.0],
                    "log2FoldChange": [2.0],
                    "lfcSE": [0.1],
                    "pvalue": [0.01],
                    "padj": [0.02],
                }
            )
        )
        self.assertEqual(
            list(frame.columns),
            ["base_mean", "log2_fold_change", "lfc_se", "p_value", "p_value_adj"],
        )

    def test_scanpy_and_deseq2_agree_on_the_fold_change_column(self):
        scanpy = to_canonical_columns(pd.DataFrame({"logfoldchanges": [1.0]}))
        deseq2 = to_canonical_columns(pd.DataFrame({"log2FoldChange": [1.0]}))
        self.assertEqual(list(scanpy.columns), list(deseq2.columns))

    def test_gseapy_names_with_spaces_are_canonicalised(self):
        frame = to_canonical_columns(
            pd.DataFrame({"Adjusted P-value": [0.01], "Combined Score": [9.0]})
        )
        self.assertEqual(list(frame.columns), ["p_value_adj", "combined_score"])
        self.assertFalse(any(" " in c for c in frame.columns))

    def test_existing_canonical_column_is_not_overwritten(self):
        frame = to_canonical_columns(
            pd.DataFrame({"padj": [0.1], "p_value_adj": [0.2]})
        )
        self.assertIn("padj", frame.columns)
        self.assertIn("p_value_adj", frame.columns)

    def test_pre_rename_file_reads_back_canonical(self):
        with TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy.csv"
            pd.DataFrame({"names": ["A"], "pvals_adj": [0.01]}).to_csv(
                legacy, index=False
            )
            self.assertEqual(list(read_table(legacy).columns), ["gene", "p_value_adj"])


class TestDriverParameterContract(unittest.TestCase):
    """The two drivers must keep offering the same options, keyword-only."""

    def test_single_driver_matches_the_declared_contract(self):
        drift = check_parameter_contract(run_pipeline, mode_only=SINGLE_ONLY_PARAMS)
        self.assertTrue(drift.ok, drift.describe())

    def test_multi_driver_matches_the_declared_contract(self):
        drift = check_parameter_contract(
            run_pipeline_multi, mode_only=MULTI_ONLY_PARAMS
        )
        self.assertTrue(drift.ok, drift.describe())

    def test_shared_options_are_declared_exactly_once(self):
        # The point of PipelineOptions: the option set exists in one place, so it
        # cannot drift between the two drivers.
        self.assertEqual(PipelineOptions.field_names(), SHARED_PARAMS)

    def test_only_four_defaults_differ_between_the_modes(self):
        # A cohort has groups to contrast and per-sample batch structure to correct;
        # one sample has neither. Everything else must be identical.
        single = PipelineOptions.for_single()
        multi = PipelineOptions.for_multi()
        differing = {
            name
            for name in PipelineOptions.field_names()
            if getattr(single, name) != getattr(multi, name)
        }
        self.assertEqual(
            differing,
            {"out_name", "do_groupwise_de", "batch_key", "integration_method"},
        )

    def test_options_are_keyword_only(self):
        for driver in (run_pipeline, run_pipeline_multi):
            positional = [
                name
                for name, param in inspect.signature(driver).parameters.items()
                if param.kind is param.POSITIONAL_OR_KEYWORD
            ]
            self.assertEqual(
                len(positional),
                1,
                f"{driver.__name__} should take exactly one positional argument "
                f"(the input directory), got {positional}",
            )

    def test_unknown_option_raises_and_suggests_the_real_name(self):
        with self.assertRaises(UnknownPipelineOption) as ctx:
            PipelineOptions.for_single(qc_max_mito=10)
        message = str(ctx.exception)
        self.assertIn("qc_max_mito", message)
        self.assertIn("qc_max_mito_percent", message)

    def test_driver_rejects_an_unknown_keyword(self):
        # Reaches the driver's **overrides, so the guard has to hold there too. The
        # input directory does not need to exist: validation happens first.
        with self.assertRaises(UnknownPipelineOption):
            run_pipeline("nonexistent", skip_tnse=True)

    def test_run_scanpy_pipeline_takes_the_options_object(self):
        # The whole point of the refactor: the core function no longer restates the
        # option list. 6 parameters, one of which is the options object.
        from agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.pipeline import (
            run_scanpy_pipeline,
        )

        params = inspect.signature(run_scanpy_pipeline).parameters
        self.assertIn("options", params)
        self.assertLessEqual(len(params), 6, sorted(params))
        overlap = set(params) & SHARED_PARAMS
        self.assertEqual(overlap, set(), f"options restated as parameters: {overlap}")

    def test_knowledge_based_flag_is_a_declared_option(self):
        # The pipeline's internal spelling is `enable_llm`; the option is the clearer
        # `enable_knowledge_based`. The mapping is inside run_scanpy_pipeline now, so
        # what matters here is that the option exists and toggles.
        self.assertFalse(
            PipelineOptions.for_single(
                enable_knowledge_based=False
            ).enable_knowledge_based
        )

    def test_unset_seed_stays_none_for_the_pipeline_default(self):
        # None means "not configured"; run_scanpy_pipeline resolves it to DEFAULT_SEED.
        self.assertIsNone(PipelineOptions.for_single().seed)
        self.assertEqual(PipelineOptions.for_single(seed=7).seed, 7)

    def test_merged_returns_a_new_object(self):
        base = PipelineOptions.for_single()
        changed = base.merged(skip_tsne=False)
        self.assertTrue(base.skip_tsne)
        self.assertFalse(changed.skip_tsne)


class TestModelIntegrity(unittest.TestCase):
    """A CellTypist pickle must be checksum-verified before it is loaded."""

    MODELS_DIR = (
        REPO_ROOT / "shared_reference" / "celltypist_models" / "data" / "models"
    )

    def test_manifest_parses_with_digest_and_size(self):
        if not (self.MODELS_DIR / "SHA256SUMS.txt").is_file():
            self.skipTest(
                "model bundle not fetched (scripts/fetch_celltypist_models.py)"
            )
        sums = load_manifest(self.MODELS_DIR)
        self.assertGreater(len(sums), 0)
        digest, size = next(iter(sums.values()))
        self.assertEqual(len(digest), 64)
        self.assertIsInstance(size, int)

    def test_bundled_models_present_on_disk_all_verify(self):
        if not (self.MODELS_DIR / "SHA256SUMS.txt").is_file():
            self.skipTest(
                "model bundle not fetched (scripts/fetch_celltypist_models.py)"
            )
        models = sorted(self.MODELS_DIR.glob("*.pkl"))
        if not models:
            self.skipTest(
                "model bundle not fetched (scripts/fetch_celltypist_models.py)"
            )
        for model in models:
            with self.subTest(model=model.name):
                self.assertTrue(verify_model_file(model, required=True))

    def test_tampered_file_raises_before_load(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            model = models_dir / "Fake_Model.pkl"
            model.write_bytes(b"tampered")
            # A manifest whose digest belongs to different content.
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Fake_Model.pkl  {len(b'tampered')}\n", encoding="utf-8"
            )
            with self.assertRaises(ModelIntegrityError) as ctx:
                verify_model_file(model, required=True)
            self.assertIn("Checksum mismatch", str(ctx.exception))

    def test_unlisted_file_raises_when_verification_is_required(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            model = models_dir / "Unlisted.pkl"
            model.write_bytes(b"x")
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Other.pkl  1\n", encoding="utf-8"
            )
            with self.assertRaises(ModelIntegrityError):
                verify_model_file(model, required=True)

    def test_present_model_is_verified_without_touching_the_network(self):
        if not (self.MODELS_DIR / "SHA256SUMS.txt").is_file():
            self.skipTest("model bundle manifest absent")
        present = sorted(self.MODELS_DIR.glob("*.pkl"))
        if not present:
            self.skipTest("no models on disk")
        with mock.patch.object(
            model_integrity.urllib.request, "urlopen", side_effect=AssertionError
        ):
            result = ensure_model_file(self.MODELS_DIR, present[0].name)
        self.assertEqual(result, present[0])

    def test_missing_model_reports_failure_offline_and_leaves_no_partial(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Wanted.pkl  10\n", encoding="utf-8"
            )
            (models_dir / "models.json").write_text(
                '{"models": [{"filename": "Wanted.pkl", "url": "https://example.invalid/x"}]}',
                encoding="utf-8",
            )
            with mock.patch.object(
                model_integrity.urllib.request,
                "urlopen",
                side_effect=model_integrity.urllib.error.URLError("offline"),
            ):
                self.assertIsNone(ensure_model_file(models_dir, "Wanted.pkl"))
            self.assertEqual(list(models_dir.glob("*.part")), [])
            self.assertEqual(list(models_dir.glob("*.pkl")), [])

    def test_unlisted_model_is_never_fetched(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Listed.pkl  10\n", encoding="utf-8"
            )
            with mock.patch.object(
                model_integrity.urllib.request, "urlopen", side_effect=AssertionError
            ):
                self.assertIsNone(ensure_model_file(models_dir, "Unlisted.pkl"))

    def test_fetch_disabled_never_downloads(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Wanted.pkl  10\n", encoding="utf-8"
            )
            with mock.patch.object(
                model_integrity.urllib.request, "urlopen", side_effect=AssertionError
            ):
                self.assertIsNone(
                    ensure_model_file(models_dir, "Wanted.pkl", allow_fetch=False)
                )

    def test_fetched_file_failing_verification_is_deleted(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Wanted.pkl  4\n", encoding="utf-8"
            )
            (models_dir / "models.json").write_text(
                '{"models": [{"filename": "Wanted.pkl", "url": "https://example.invalid/x"}]}',
                encoding="utf-8",
            )

            class FakeResponse:
                def read(self):
                    return b"junk"  # right size, wrong digest

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with mock.patch.object(
                model_integrity.urllib.request,
                "urlopen",
                return_value=FakeResponse(),
            ):
                self.assertIsNone(ensure_model_file(models_dir, "Wanted.pkl"))
            # A pickle that failed verification must not survive anywhere on disk.
            self.assertEqual(list(models_dir.glob("*.pkl")), [])
            self.assertEqual(list(models_dir.glob("*.part")), [])

    def test_truncated_file_is_caught_by_the_size_precheck(self):
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            model = models_dir / "Truncated.pkl"
            model.write_bytes(b"short")
            (models_dir / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  Truncated.pkl  999999\n", encoding="utf-8"
            )
            with self.assertRaises(ModelIntegrityError) as ctx:
                verify_model_file(model, required=True)
            self.assertIn("Size mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
