"""
test_benchmarks.py — the benchmark harness must be correct before its numbers are.

A benchmark is measurement apparatus, and an apparatus that has not been checked against
known answers cannot support a claim. These tests feed the metrics inputs whose correct
output is known by construction — a perfect classifier, a constant classifier, a
perfectly calibrated one, an oracle-ranked one — and assert the reported numbers.

They also pin the two properties that make the comparison fair rather than flattering:

* Harmonisation is applied by ONE function with no per-method branch, so truth and every
  prediction are transformed identically.
* A method's own failures never shrink the denominator. Unresolved predictions are scored
  as errors; only unresolvable GROUND TRUTH is dropped. Otherwise a method could raise
  its score by abstaining.

No network and no reference data: labels are literal strings, and the resolver-backed
tests skip if the hierarchy spec is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.harmonise import (
    UNRESOLVED,
    evaluable_mask,
    harmonise,
    is_abstention,
    mapping_table,
    resolution_report,
)
from benchmarks.metrics import (
    area_under_risk_coverage,
    calibration,
    confusion_matrix,
    correlate,
    macro_f1_with_ci,
    per_class_f1,
    risk_at_coverage,
    risk_coverage_curve,
    voter_entropy,
)

# --------------------------------------------------------------------------------------
# macro-F1: known answers
# --------------------------------------------------------------------------------------


def test_perfect_classifier_scores_one():
    truth = ["T cell", "B cell", "T cell", "Monocyte"]
    result = macro_f1_with_ci(truth, truth, n_bootstrap=50)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n"] == 4
    assert result["n_classes"] == 3


def test_constant_classifier_is_punished_by_macro_not_accuracy():
    """The reason macro-F1 is the headline metric rather than accuracy.

    90 T cells and 10 B cells: always answering "T cell" is 90% accurate and useless.
    """
    truth = ["T cell"] * 90 + ["B cell"] * 10
    predicted = ["T cell"] * 100
    result = macro_f1_with_ci(truth, predicted, n_bootstrap=0)
    assert result["accuracy"] == pytest.approx(0.90)
    # F1 for T cell = 2*.9/(1+.9) ~= 0.947; for B cell = 0. Macro ~= 0.474.
    assert result["macro_f1"] == pytest.approx(0.4737, abs=1e-3)
    assert result["macro_f1"] < result["accuracy"]


def test_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    truth = rng.choice(["A", "B", "C"], size=300).tolist()
    predicted = [
        label if rng.random() < 0.8 else rng.choice(["A", "B", "C"]) for label in truth
    ]
    result = macro_f1_with_ci(truth, predicted, n_bootstrap=200, seed=0)
    assert result["ci_low"] <= result["macro_f1"] <= result["ci_high"]
    assert result["ci_high"] - result["ci_low"] > 0


def test_ci_is_reproducible_under_the_same_seed():
    truth = ["A", "B"] * 50
    predicted = ["A"] * 100
    first = macro_f1_with_ci(truth, predicted, n_bootstrap=100, seed=7)
    second = macro_f1_with_ci(truth, predicted, n_bootstrap=100, seed=7)
    assert first == second


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        macro_f1_with_ci(["A", "B"], ["A"])


def test_unresolved_prediction_counts_as_wrong():
    """A method cannot improve its score by abstaining."""
    truth = ["T cell"] * 10
    abstaining = [UNRESOLVED] * 10
    assert macro_f1_with_ci(truth, abstaining, n_bootstrap=0)["macro_f1"] == 0.0


def test_per_class_f1_matches_hand_computation():
    truth = ["A", "A", "B", "B"]
    predicted = ["A", "B", "B", "B"]
    frame = per_class_f1(truth, predicted).set_index("cell_type")
    # A: tp=1, predicted=1 -> precision 1.0; truth=2 -> recall 0.5; f1 = 2/3
    assert frame.loc["A", "precision"] == pytest.approx(1.0)
    assert frame.loc["A", "recall"] == pytest.approx(0.5)
    assert frame.loc["A", "f1"] == pytest.approx(2 / 3)
    # B: tp=2, predicted=3 -> precision 2/3; truth=2 -> recall 1.0; f1 = 0.8
    assert frame.loc["B", "f1"] == pytest.approx(0.8)
    assert frame.loc["B", "support"] == 2


# --------------------------------------------------------------------------------------
# risk-coverage: the headline figure
# --------------------------------------------------------------------------------------


def test_oracle_confidence_gives_zero_error_at_partial_coverage():
    """Perfect ranking: all errors sit at the bottom, so early coverage is error-free."""
    correct = [True] * 80 + [False] * 20
    confidence = [0.99] * 80 + [0.01] * 20
    curve = risk_coverage_curve(correct, confidence, seed=0)

    assert risk_at_coverage(curve, 0.8) == pytest.approx(0.0)
    assert risk_at_coverage(curve, 1.0) == pytest.approx(0.20)
    assert area_under_risk_coverage(curve) < 0.05


def test_useless_confidence_gives_flat_curve():
    """Constant confidence cannot rank, so error is ~flat and AURC ~ the base rate."""
    rng = np.random.default_rng(1)
    correct = (rng.random(400) < 0.75).tolist()
    curve = risk_coverage_curve(correct, [0.5] * 400, seed=0)
    assert risk_at_coverage(curve, 1.0) == pytest.approx(1 - np.mean(correct))
    # Flat means the AURC is close to the overall error rate.
    assert area_under_risk_coverage(curve) == pytest.approx(
        1 - np.mean(correct), abs=0.06
    )


def test_ties_are_broken_by_seed_not_input_order():
    """A table sorted by class must not look like it has a meaningful ranking.

    All-equal confidence with the errors grouped at the front: without a shuffle the
    curve would start at error 1.0 and look like informative ranking, reversed.
    """
    correct = [False] * 50 + [True] * 50
    curve = risk_coverage_curve(correct, [0.5] * 100, seed=0)
    early_error = float(curve["error"].iloc[9])  # first 10% covered
    assert early_error < 0.95, "input order leaked into the ranking"


def test_curve_is_reproducible_under_the_same_seed():
    correct = [True, False] * 50
    first = risk_coverage_curve(correct, [0.5] * 100, seed=3)
    second = risk_coverage_curve(correct, [0.5] * 100, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_nan_confidence_is_ranked_last():
    """A cell with no score must not be treated as maximally confident."""
    correct = [False, True, True]
    confidence = [float("nan"), 0.9, 0.8]
    curve = risk_coverage_curve(correct, confidence, seed=0)
    # The first two covered cells are the scored, correct ones.
    assert curve["error"].iloc[0] == pytest.approx(0.0)
    assert curve["error"].iloc[1] == pytest.approx(0.0)
    assert curve["error"].iloc[2] == pytest.approx(1 / 3)


def test_empty_input_is_empty_not_an_error():
    curve = risk_coverage_curve([], [], seed=0)
    assert curve.empty
    assert np.isnan(risk_at_coverage(curve, 0.9))
    assert np.isnan(area_under_risk_coverage(curve))


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------


def test_perfectly_calibrated_has_near_zero_ece():
    """Confidence p, correct exactly p of the time, in each of four groups."""
    correct: list[bool] = []
    confidence: list[float] = []
    for probability in (0.1, 0.35, 0.65, 0.9):
        n = 1000
        n_correct = int(round(probability * n))
        correct += [True] * n_correct + [False] * (n - n_correct)
        confidence += [probability] * n
    _, summary = calibration(correct, confidence, n_bins=10)
    assert summary["ece"] < 0.01
    assert abs(summary["overconfidence"]) < 0.01


def test_overconfident_method_is_detected():
    """Always claims 0.99, right half the time."""
    correct = [True, False] * 500
    confidence = [0.99] * 1000
    _, summary = calibration(correct, confidence, n_bins=10)
    assert summary["ece"] == pytest.approx(0.49, abs=0.02)
    assert summary["overconfidence"] == pytest.approx(0.49, abs=0.02)


def test_underconfident_method_has_negative_overconfidence():
    correct = [True] * 1000
    confidence = [0.5] * 1000
    _, summary = calibration(correct, confidence, n_bins=10)
    assert summary["overconfidence"] == pytest.approx(-0.5, abs=0.01)


def test_calibration_ignores_unscored_cells():
    correct = [True, False, True]
    confidence = [0.9, float("nan"), 0.9]
    _, summary = calibration(correct, confidence)
    assert summary["n_scored"] == 2
    assert summary["accuracy"] == pytest.approx(1.0)


def test_calibration_of_nothing_is_empty():
    curve, summary = calibration([], [])
    assert curve.empty
    assert summary["n_scored"] == 0


def test_confidence_of_zero_lands_in_a_bin():
    """Boundary case: 0.0 must be binned, not dropped."""
    curve, summary = calibration([False] * 10, [0.0] * 10, n_bins=10)
    assert summary["n_scored"] == 10
    assert int(curve["n"].sum()) == 10


# --------------------------------------------------------------------------------------
# voter disagreement
# --------------------------------------------------------------------------------------


def test_unanimous_voters_have_zero_entropy():
    frame = pd.DataFrame(
        {"a": ["T cell"] * 3, "b": ["T cell"] * 3, "c": ["T cell"] * 3}
    )
    assert voter_entropy(frame, ["a", "b", "c"]).tolist() == [0.0, 0.0, 0.0]


def test_two_way_split_is_one_bit():
    frame = pd.DataFrame({"a": ["T cell"], "b": ["B cell"]})
    assert voter_entropy(frame, ["a", "b"]).iloc[0] == pytest.approx(1.0)


def test_abstentions_do_not_manufacture_disagreement():
    """Two agreeing voters plus two abstentions is unanimity, not a four-way split."""
    frame = pd.DataFrame(
        {
            "a": ["T cell"],
            "b": ["T cell"],
            "c": [UNRESOLVED],
            "d": [UNRESOLVED],
        }
    )
    assert voter_entropy(frame, ["a", "b", "c", "d"]).iloc[0] == pytest.approx(0.0)


def test_all_abstaining_is_nan_not_zero():
    """No information is not the same as agreement."""
    frame = pd.DataFrame({"a": [UNRESOLVED], "b": [UNRESOLVED]})
    assert np.isnan(voter_entropy(frame, ["a", "b"]).iloc[0])


def test_missing_voter_columns_give_all_nan():
    frame = pd.DataFrame({"x": ["T cell"]})
    assert voter_entropy(frame, ["nope"]).isna().all()


def test_correlate_detects_a_known_relationship():
    left = pd.Series([1.0, 2, 3, 4, 5])
    right = pd.Series([2.0, 4, 6, 8, 10])
    result = correlate(left, right)
    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 5


def test_correlate_of_a_constant_is_nan_not_an_error():
    result = correlate(pd.Series([1.0] * 5), pd.Series([1.0, 2, 3, 4, 5]))
    assert np.isnan(result["rho"])


# --------------------------------------------------------------------------------------
# confusion matrix
# --------------------------------------------------------------------------------------


def test_row_normalised_confusion_rows_sum_to_one():
    truth = ["A", "A", "B", "B"]
    predicted = ["A", "B", "B", "B"]
    matrix = confusion_matrix(truth, predicted, normalise="truth")
    np.testing.assert_allclose(matrix.sum(axis=1).to_numpy(), [1.0, 1.0])
    assert matrix.loc["A", "A"] == pytest.approx(0.5)


def test_unnormalised_confusion_is_counts():
    matrix = confusion_matrix(["A", "A"], ["A", "B"])
    assert int(matrix.loc["A", "A"]) == 1
    assert int(matrix.loc["A", "B"]) == 1


# --------------------------------------------------------------------------------------
# harmonisation and the fairness guards
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["", "NA", "nan", "None", "unknown", "Unassigned", "Unknown cell", "doublet"],
)
def test_abstention_tokens_are_recognised(token):
    assert is_abstention(token)


def test_real_label_is_not_an_abstention():
    assert not is_abstention("T cell")
    assert not is_abstention("Natural killer cell")


def test_evaluable_mask_drops_only_unresolvable_truth():
    truth = pd.Series(["T cell", UNRESOLVED, "B cell"])
    mask = evaluable_mask(truth)
    assert mask.tolist() == [True, False, True]


def test_evaluable_mask_ignores_predictions():
    """Passing predictions must not change the denominator."""
    truth = pd.Series(["T cell", "B cell", "Monocyte"])
    predictions = pd.Series([UNRESOLVED, UNRESOLVED, UNRESOLVED])
    assert evaluable_mask(truth, predictions).tolist() == [True, True, True]


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


@requires_hierarchy
def test_synonyms_collapse_to_one_label():
    """The whole point: spelling differences must stop reading as disagreement."""
    harmonised = harmonise(
        ["NK cells", "CD16+ NK cells", "Natural killer cell"], level="main_cell_type"
    )
    assert harmonised.nunique() == 1, f"expected one label, got {set(harmonised)}"


@requires_hierarchy
def test_distinct_lineages_stay_distinct():
    """Harmonisation must not collapse genuinely different types."""
    harmonised = harmonise(
        ["T cells", "Fibroblast", "Epithelial cell"], level="lineage"
    )
    assert harmonised.nunique() == 3


@requires_hierarchy
def test_unresolvable_label_becomes_unresolved():
    assert harmonise(["definitely not a cell type xyzzy"]).iloc[0] == UNRESOLVED


@requires_hierarchy
def test_harmonise_preserves_index_and_length():
    series = pd.Series(["T cells", "B cells"], index=["cell_a", "cell_b"])
    harmonised = harmonise(series)
    assert list(harmonised.index) == ["cell_a", "cell_b"]
    assert len(harmonised) == 2


@requires_hierarchy
def test_resolution_report_exposes_a_vocabulary_gap():
    """The confound guard: a method whose labels do not resolve is visible, not hidden."""
    frame = pd.DataFrame(
        {
            "truth": ["T cells", "B cells", "Monocyte"],
            "good_method": ["T cell", "B cell", "Monocyte"],
            "bad_method": ["xyzzy1", "xyzzy2", "xyzzy3"],
        }
    )
    report = resolution_report(frame, ["truth", "good_method", "bad_method"]).set_index(
        "column"
    )
    assert report.loc["bad_method", "unresolved_fraction"] == pytest.approx(1.0)
    assert report.loc["truth", "unresolved_fraction"] < 0.5


@requires_hierarchy
def test_resolution_report_skips_absent_columns():
    frame = pd.DataFrame({"truth": ["T cells"]})
    report = resolution_report(frame, ["truth", "not_a_column"])
    assert list(report["column"]) == ["truth"]


@requires_hierarchy
def test_mapping_table_is_publishable():
    """Every distinct raw label and its destination, for the supplement."""
    frame = pd.DataFrame({"m": ["T cells", "T cells", "B cells"]})
    table = mapping_table(frame, ["m"])
    assert set(table.columns) == {
        "source_column",
        "raw_label",
        "harmonised_label",
        "n_cells",
        "resolved",
    }
    assert int(table.loc[table["raw_label"] == "T cells", "n_cells"].iloc[0]) == 2


# --------------------------------------------------------------------------------------
# driver: ablation, majority vote, and the end-to-end contract
# --------------------------------------------------------------------------------------


@requires_hierarchy
def test_majority_vote_ignores_abstentions_and_breaks_ties_by_order():
    from benchmarks.run_annotation_benchmark import _majority_vote

    frame = pd.DataFrame(
        {
            "v1": ["T cell", "T cell", UNRESOLVED],
            "v2": ["T cell", "B cell", UNRESOLVED],
            "v3": [UNRESOLVED, UNRESOLVED, UNRESOLVED],
        }
    )
    result = _majority_vote(frame, ["v1", "v2", "v3"])
    assert result.iloc[0] == "T cell"  # 2-0 with one abstention
    assert result.iloc[1] == "T cell"  # 1-1 tie -> first listed voter
    assert result.iloc[2] == UNRESOLVED  # nobody called


@requires_hierarchy
def test_end_to_end_on_a_synthetic_h5ad(tmp_path):
    """The driver runs and writes every table, on data with a known correct answer."""
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_annotation_benchmark import run_benchmark

    n = 120
    rng = np.random.default_rng(0)
    types = ["T cells", "B cells", "Monocyte"]
    truth = rng.choice(types, size=n)

    # A strong voter, a weak voter, and a consensus that tracks truth closely.
    strong = np.where(rng.random(n) < 0.9, truth, rng.choice(types, size=n))
    weak = np.where(rng.random(n) < 0.5, truth, rng.choice(types, size=n))
    consensus = np.where(rng.random(n) < 0.95, truth, rng.choice(types, size=n))

    obs = pd.DataFrame(
        {
            "cell_type": truth,
            "celltype_celltypist": strong,
            "celltype_knowledge_based": weak,
            "celltype_consensus": consensus,
            "celltype_subtype_confidence": np.where(consensus == truth, 0.95, 0.30),
            "consensus_tier": np.where(consensus == truth, "high", "low"),
            "doublet_score": rng.random(n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = anndata.AnnData(X=rng.random((n, 5)).astype("float32"), obs=obs)
    h5ad = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(h5ad)

    out_dir = tmp_path / "results"
    manifest = run_benchmark(h5ad, "cell_type", out_dir=out_dir, n_bootstrap=25, seed=0)

    assert manifest["n_cells_total"] == n
    assert "celltype_consensus" in manifest["methods_scored"]

    for expected in (
        "00_resolution_report.csv",
        "00_label_mapping.csv",
        "01_method_comparison.csv",
        "02_ablation.csv",
        "03_per_class_f1.csv",
        "04_risk_coverage.csv",
        "05_risk_coverage_summary.csv",
        "06_calibration.csv",
        "07_calibration_summary.csv",
        "08_disagreement.csv",
        "manifest.json",
    ):
        assert (out_dir / expected).is_file(), f"{expected} was not written"

    # The strong voter must outscore the weak one, or the harness is not measuring.
    comparison = pd.read_csv(out_dir / "01_method_comparison.csv").set_index("method")
    assert (
        comparison.loc["celltype_celltypist", "macro_f1"]
        > comparison.loc["celltype_knowledge_based", "macro_f1"]
    )
    # Confidence here is oracle-derived, so partial coverage must be error-free.
    rc = pd.read_csv(out_dir / "05_risk_coverage_summary.csv").set_index("method")
    assert rc.loc["celltype_consensus", "error_at_70pct_coverage"] == pytest.approx(0.0)


@requires_hierarchy
def test_driver_rejects_a_missing_truth_column(tmp_path):
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_annotation_benchmark import run_benchmark

    adata = anndata.AnnData(
        X=np.zeros((3, 2), dtype="float32"),
        obs=pd.DataFrame(
            {"celltype_consensus": ["T cells"] * 3}, index=["c0", "c1", "c2"]
        ),
    )
    h5ad = tmp_path / "no_truth.h5ad"
    adata.write_h5ad(h5ad)

    with pytest.raises(SystemExit, match="not in obs"):
        run_benchmark(h5ad, "cell_type", out_dir=tmp_path / "out", n_bootstrap=0)


@requires_hierarchy
def test_driver_rejects_a_file_with_no_method_columns(tmp_path):
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_annotation_benchmark import run_benchmark

    adata = anndata.AnnData(
        X=np.zeros((3, 2), dtype="float32"),
        obs=pd.DataFrame({"cell_type": ["T cells"] * 3}, index=["c0", "c1", "c2"]),
    )
    h5ad = tmp_path / "no_methods.h5ad"
    adata.write_h5ad(h5ad)

    with pytest.raises(SystemExit, match="No method columns"):
        run_benchmark(h5ad, "cell_type", out_dir=tmp_path / "out", n_bootstrap=0)


# --------------------------------------------------------------------------------------
# held-out cell type: abstention vs confident mislabeling
# --------------------------------------------------------------------------------------


def test_censoring_replaces_only_the_held_out_label():
    """Censoring must remove exactly one label and leave every other call untouched."""
    from benchmarks.run_heldout_celltype import _censor

    predictions = pd.Series(["T cell", "B cell", "T cell", UNRESOLVED])
    censored = _censor(predictions, "T cell")
    assert censored.tolist() == [UNRESOLVED, "B cell", UNRESOLVED, UNRESOLVED]


def test_abstention_detects_all_three_signals():
    """Unresolved label, an abstaining tier, or low confidence each count as declining."""
    from benchmarks.run_heldout_celltype import _is_abstaining

    predictions = pd.Series(["T cell", "T cell", "T cell", "T cell"])
    confidence = pd.Series([0.95, 0.10, 0.95, float("nan")])
    tier = pd.Series(["high", "high", "low", "high"])

    result = _is_abstaining(predictions, confidence, tier)
    assert result.tolist() == [False, True, True, False]


def test_nan_confidence_is_not_treated_as_abstention():
    """ "No score" is not "low score" — otherwise every scoreless method looks cautious."""
    from benchmarks.run_heldout_celltype import _is_abstaining

    predictions = pd.Series(["T cell", "B cell"])
    confidence = pd.Series([float("nan"), float("nan")])
    assert _is_abstaining(predictions, confidence, None).tolist() == [False, False]


def test_unresolved_prediction_counts_as_abstention_without_a_tier():
    from benchmarks.run_heldout_celltype import _is_abstaining

    predictions = pd.Series([UNRESOLVED, "T cell"])
    confidence = pd.Series([float("nan"), float("nan")])
    assert _is_abstaining(predictions, confidence, None).tolist() == [True, False]


def test_rerun_plan_names_every_held_out_type():
    """The emitted reference-ablation plan must mention each type it covers."""
    from benchmarks.run_heldout_celltype import rerun_plan

    plan = rerun_plan(["T cell", "Natural killer cell"])
    assert "T cell" in plan
    assert "Natural killer cell" in plan
    assert "SCPIPE_SHARED_REFERENCE_ROOT" in plan


@requires_hierarchy
def test_heldout_end_to_end_rewards_abstention_over_confident_error(tmp_path):
    """A method that abstains on the held-out type must beat one that mislabels it.

    Built so the answer is known: the consensus is given a low tier exactly on the
    held-out type's cells, while the voter confidently calls them something else. The
    consensus must therefore show a higher abstention rate and a lower confident error
    rate — if the harness cannot detect that, it cannot detect it on real data either.
    """
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_heldout_celltype import run_heldout

    n = 150
    rng = np.random.default_rng(0)
    truth = np.array(["T cells"] * 50 + ["B cells"] * 50 + ["Monocyte"] * 50)
    held_out = "B cell"  # harmonised form of "B cells"

    # The voter always commits, and is wrong on the B cells.
    voter = np.where(truth == "B cells", "T cells", truth)
    # The consensus agrees with the voter but flags those cells as low confidence.
    consensus = voter.copy()
    tier = np.where(truth == "B cells", "low", "high")
    confidence = np.where(truth == "B cells", 0.10, 0.95)

    obs = pd.DataFrame(
        {
            "cell_type": truth,
            "celltype_celltypist": voter,
            "celltype_consensus": consensus,
            "consensus_tier": tier,
            "celltype_subtype_confidence": confidence,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = anndata.AnnData(X=rng.random((n, 4)).astype("float32"), obs=obs)
    h5ad = tmp_path / "heldout.h5ad"
    adata.write_h5ad(h5ad)

    out_dir = tmp_path / "results"
    manifest = run_heldout(
        h5ad, "cell_type", out_dir=out_dir, n_types=None, min_cells=10
    )
    assert held_out in manifest["held_out_types"]

    for expected in (
        "10_heldout_detail.csv",
        "10_heldout_summary.csv",
        "10_heldout_manifest.json",
    ):
        assert (out_dir / expected).is_file(), f"{expected} not written"

    detail = pd.read_csv(out_dir / "10_heldout_detail.csv")
    row = detail[detail["held_out_type"] == held_out].set_index("method")

    assert row.loc["celltype_consensus", "abstention_rate"] == pytest.approx(1.0)
    assert row.loc["celltype_consensus", "confident_error_rate"] == pytest.approx(0.0)
    assert row.loc["celltype_celltypist", "abstention_rate"] == pytest.approx(0.0)
    assert row.loc["celltype_celltypist", "confident_error_rate"] == pytest.approx(1.0)
    assert row.loc["celltype_celltypist", "top_confusion"] == "T cell"


@requires_hierarchy
def test_heldout_rejects_a_cohort_with_no_abundant_type(tmp_path):
    """Too few cells of any type means the experiment would be noise; say so."""
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_heldout_celltype import run_heldout

    obs = pd.DataFrame(
        {"cell_type": ["T cells"] * 3, "celltype_consensus": ["T cells"] * 3},
        index=["c0", "c1", "c2"],
    )
    adata = anndata.AnnData(X=np.zeros((3, 2), dtype="float32"), obs=obs)
    h5ad = tmp_path / "tiny.h5ad"
    adata.write_h5ad(h5ad)

    with pytest.raises(SystemExit, match="nothing worth holding out"):
        run_heldout(h5ad, "cell_type", out_dir=tmp_path / "out", min_cells=100)


# --------------------------------------------------------------------------------------
# categorical obs columns — the real-data shape the synthetic tests missed
# --------------------------------------------------------------------------------------


@requires_hierarchy
def test_categorical_columns_are_handled():
    """AnnData stores obs as Categorical, and mapping one returns a Categorical.

    That result refuses ordinary reductions, so `resolution_report` raised
    `TypeError: 'Categorical' ... does not support reduction 'sum'` on the first real
    dataset while every string-column test passed. Both entry points now cast first.
    """
    frame = pd.DataFrame(
        {
            "truth": pd.Categorical(["T cells", "B cells", "unknown"]),
            "pred": pd.Categorical(["T cell", "B cell", "T cell"]),
        }
    )
    report = resolution_report(frame, ["truth", "pred"]).set_index("column")
    assert int(report.loc["truth", "n_abstained"]) == 1
    assert int(report.loc["pred", "n_abstained"]) == 0

    harmonised = harmonise(frame["truth"])
    assert harmonised.dtype == object, "harmonise must not return a Categorical"
    assert int((harmonised == UNRESOLVED).sum()) == 1


@requires_hierarchy
def test_categorical_survives_the_full_metric_path():
    """A categorical column must flow through scoring without a dtype error."""
    truth = harmonise(pd.Categorical(["T cells", "B cells", "T cells", "B cells"]))
    pred = harmonise(pd.Categorical(["T cell", "B cell", "B cell", "B cell"]))
    result = macro_f1_with_ci(truth.to_numpy(), pred.to_numpy(), n_bootstrap=20)
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["n"] == 4


@requires_hierarchy
def test_a_disabled_voter_is_excluded_not_scored_as_zero(tmp_path):
    """A voter that was switched off must not appear as a method scoring 0.0.

    Its obs column survives the run filled with a placeholder, so scoring it reports
    macro_f1 0.0 and a flat error-1.0 risk curve — indistinguishable in the tables from
    a method that ran and failed completely.
    """
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_annotation_benchmark import run_benchmark

    n = 60
    rng = np.random.default_rng(0)
    truth = rng.choice(["T cells", "B cells"], size=n)
    obs = pd.DataFrame(
        {
            "cell_type": truth,
            "celltype_celltypist": truth,
            # Disabled voter: one placeholder for every cell.
            "celltype_knowledge_based": ["Unassigned"] * n,
            "celltype_consensus": truth,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = anndata.AnnData(X=rng.random((n, 3)).astype("float32"), obs=obs)
    h5ad = tmp_path / "disabled_voter.h5ad"
    adata.write_h5ad(h5ad)

    out_dir = tmp_path / "res"
    manifest = run_benchmark(h5ad, "cell_type", out_dir=out_dir, n_bootstrap=10)

    assert "celltype_knowledge_based" in manifest["methods_excluded_no_calls"]
    assert "celltype_knowledge_based" not in manifest["methods_scored"]

    comparison = pd.read_csv(out_dir / "01_method_comparison.csv")
    assert "celltype_knowledge_based" not in set(comparison["method"])
    assert set(comparison["method"]) == {"celltype_celltypist", "celltype_consensus"}


@requires_hierarchy
def test_all_methods_empty_is_an_error(tmp_path):
    """If no voter made a call there is nothing to score; say so rather than emit zeros."""
    anndata = pytest.importorskip("anndata")
    from benchmarks.run_annotation_benchmark import run_benchmark

    n = 20
    obs = pd.DataFrame(
        {
            "cell_type": ["T cells"] * n,
            "celltype_consensus": ["Unassigned"] * n,
        },
        index=[f"c{i}" for i in range(n)],
    )
    adata = anndata.AnnData(X=np.zeros((n, 2), dtype="float32"), obs=obs)
    h5ad = tmp_path / "all_empty.h5ad"
    adata.write_h5ad(h5ad)

    with pytest.raises(SystemExit, match="Every method column is empty"):
        run_benchmark(h5ad, "cell_type", out_dir=tmp_path / "out", n_bootstrap=0)
