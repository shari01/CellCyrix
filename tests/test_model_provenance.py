"""
test_model_provenance.py — the scikit-learn version-skew guard.

The bundled CellTypist models were not pickled by the scikit-learn that runs them: 30
of the 41 carry ``_sklearn_version == "0.24.1"``, 8 carry ``"1.2.2"``, 3 carry
``"1.0.1"``. sklearn therefore warns that results "might be invalid" on every load.

These tests are what turns that warning from an open question into a checked property.
A CellTypist model is a ``LogisticRegression`` plus a ``StandardScaler``, so inference
is fully determined by ``mean_``, ``scale_``, ``coef_`` and ``intercept_``. If the
installed scikit-learn still reproduces the model's own arithmetic from those arrays,
the skew is cosmetic. If it ever stops doing so, these tests fail — which is the
outcome that matters, because the alternative is every annotation silently shifting.

There is also a golden test: labels and probabilities for a fixed, seeded input are
compared against values recomputed from the persisted arrays with NumPy alone. That
pins behaviour across scikit-learn upgrades without storing a large fixture.

All tests here skip cleanly when the model bundle is not present, so a fresh clone that
has not fetched the 54 MB of pickles still runs the suite.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.model_provenance import (  # noqa: E501
    MODEL_KEY,
    REQUIRED_CLASSIFIER_ATTRS,
    REQUIRED_SCALER_ATTRS,
    SCALER_KEY,
    ModelNumericsError,
    collect_model_provenance,
    model_provenance_record,
    pickled_sklearn_versions,
    record_validated_model,
    reset_run_provenance,
    run_provenance,
    validate_model_numerics,
)

# Probe size for the golden check. Small enough to stay fast, wide enough that a
# systematic arithmetic change cannot hide in rounding.
GOLDEN_ROWS = 16
GOLDEN_SEED = 12345

# The reconstruction is the same arithmetic in the same dtype, so the observed
# difference is exactly 0.0. Kept as a tolerance only to absorb BLAS reordering.
EXACTNESS_TOL = 1e-9


def _models_dir() -> Path | None:
    """The bundled CellTypist model directory, or None when it is not present."""
    from cellcyrix.single_cell_pipeline_agent.singlecell_10x.celltype_consensus.lineage_panels import (  # noqa: E501
        shared_reference_root,
    )

    candidate = shared_reference_root() / "celltypist_models" / "data" / "models"
    return candidate if candidate.is_dir() else None


def _available_models(limit: int | None = None) -> list[Path]:
    """Bundled ``.pkl`` paths, optionally capped for the slower whole-bundle tests."""
    directory = _models_dir()
    if directory is None:
        return []
    found = sorted(directory.glob("*.pkl"))
    return found[:limit] if limit else found


def _load(path: Path) -> dict:
    """Unpickle a model, silencing the very warning under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as handle:
            return pickle.load(handle)


requires_models = pytest.mark.skipif(
    not _available_models(),
    reason="CellTypist model bundle not present; run fetch-celltypist-models",
)


# --------------------------------------------------------------------------------------
# Version reading (no unpickling)
# --------------------------------------------------------------------------------------


@requires_models
def test_versions_are_readable_without_unpickling():
    """Every bundled model states which scikit-learn wrote it, readable from bytes."""
    missing = [p.name for p in _available_models() if not pickled_sklearn_versions(p)]
    assert not missing, (
        f"{len(missing)} model(s) carry no _sklearn_version, so the skew cannot be "
        f"audited: {missing[:5]}"
    )


@requires_models
def test_version_skew_is_present_and_documented():
    """The skew this module exists for is real — if it ever vanishes, simplify.

    Not a failure condition so much as a tripwire: if upstream re-exports the zoo under
    a current scikit-learn, this test tells us the validation machinery is no longer
    load-bearing and the module can be retired.
    """
    import sklearn

    versions = {v for p in _available_models() for v in pickled_sklearn_versions(p)}
    assert versions, "no version strings found at all"
    if versions == {sklearn.__version__}:
        pytest.skip(
            "the bundle now matches the runtime scikit-learn exactly; "
            "model_provenance validation is no longer load-bearing"
        )
    assert versions - {sklearn.__version__}, "expected at least one skewed model"


# --------------------------------------------------------------------------------------
# The core claim: the runtime reproduces each model's own arithmetic
# --------------------------------------------------------------------------------------


@requires_models
@pytest.mark.parametrize("path", _available_models(limit=6), ids=lambda p: p.name)
def test_runtime_reproduces_model_arithmetic(path: Path):
    """predict_proba matches a NumPy reconstruction from the persisted arrays."""
    result = validate_model_numerics(_load(path))
    assert result["decision_rule"] in {"multinomial", "one_vs_rest"}
    assert result["max_abs_diff"] <= EXACTNESS_TOL, (
        f"{path.name}: predict_proba diverges from its own persisted arrays by "
        f"{result['max_abs_diff']:.3e}. The scikit-learn version skew has stopped "
        f"being cosmetic; annotations from this model are not trustworthy."
    )
    assert result["n_classes"] >= 2


@requires_models
def test_every_bundled_model_validates():
    """The whole bundle passes, not just the sample the parametrised test covers."""
    report = collect_model_provenance(_models_dir(), validate=True)
    failed = [r for r in report["models"] if not r.get("numerics_validated")]
    assert not failed, (
        f"{len(failed)} of {len(report['models'])} models failed numeric validation: "
        f"{[(r.get('model'), r.get('error')) for r in failed[:3]]}"
    )
    assert report["all_validated"] is True


@requires_models
@pytest.mark.parametrize("path", _available_models(limit=6), ids=lambda p: p.name)
def test_required_state_survived_unpickling(path: Path):
    """The attributes inference depends on are all present after the version jump."""
    model = _load(path)
    classifier, scaler = model[MODEL_KEY], model[SCALER_KEY]
    for attr in REQUIRED_CLASSIFIER_ATTRS:
        assert hasattr(classifier, attr), f"{path.name}: classifier lost {attr}"
    for attr in REQUIRED_SCALER_ATTRS:
        assert hasattr(scaler, attr), f"{path.name}: scaler lost {attr}"
    assert int(classifier.n_features_in_) == int(scaler.n_features_in_)
    assert classifier.coef_.shape[1] == int(classifier.n_features_in_)


# --------------------------------------------------------------------------------------
# Golden output: pins behaviour across scikit-learn upgrades
# --------------------------------------------------------------------------------------


@requires_models
@pytest.mark.parametrize("path", _available_models(limit=3), ids=lambda p: p.name)
def test_golden_predictions_are_stable(path: Path):
    """Labels and probabilities for a fixed seeded input match the array reconstruction.

    The expected values are recomputed rather than stored, which keeps the fixture out
    of the repo while still failing if scikit-learn changes how it reads this state.
    """
    model = _load(path)
    classifier, scaler = model[MODEL_KEY], model[SCALER_KEY]
    n_features = int(classifier.n_features_in_)

    rng = np.random.default_rng(GOLDEN_SEED)
    features = rng.normal(size=(GOLDEN_ROWS, n_features)).astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual_proba = np.asarray(classifier.predict_proba(scaler.transform(features)))
        actual_labels = np.asarray(classifier.predict(scaler.transform(features)))

    scaled = (features - scaler.mean_) / scaler.scale_
    logits = scaled @ classifier.coef_.T + classifier.intercept_
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected_proba = shifted / shifted.sum(axis=1, keepdims=True)
    expected_labels = np.asarray(classifier.classes_)[logits.argmax(axis=1)]

    np.testing.assert_allclose(
        actual_proba,
        expected_proba,
        atol=EXACTNESS_TOL,
        rtol=0,
        err_msg=f"{path.name}: probabilities drifted from the persisted arrays",
    )
    assert list(actual_labels) == list(expected_labels), (
        f"{path.name}: argmax label assignment drifted"
    )
    # Probabilities must still be a distribution after the version jump.
    np.testing.assert_allclose(
        actual_proba.sum(axis=1), np.ones(GOLDEN_ROWS), atol=1e-9, rtol=0
    )


# --------------------------------------------------------------------------------------
# The guard fails when it should
# --------------------------------------------------------------------------------------


def test_missing_estimator_raises():
    """A dict without the expected keys is rejected rather than silently skipped."""
    with pytest.raises(ModelNumericsError, match="missing"):
        validate_model_numerics({"description": {}})


def test_stripped_state_raises():
    """An estimator missing inference state is rejected, not used."""

    class Hollow:
        """A LogisticRegression-shaped object that lost its fitted arrays."""

    with pytest.raises(ModelNumericsError, match="missing state required"):
        validate_model_numerics({MODEL_KEY: Hollow(), SCALER_KEY: Hollow()})


def test_feature_count_mismatch_raises():
    """A classifier and scaler that disagree on width cannot be used together."""

    class Clf:
        coef_ = np.zeros((3, 5))
        intercept_ = np.zeros(3)
        classes_ = np.arange(3)
        n_features_in_ = 5

    class Scaler:
        mean_ = np.zeros(7)
        scale_ = np.ones(7)
        n_features_in_ = 7

    with pytest.raises(ModelNumericsError, match="Feature-count mismatch"):
        validate_model_numerics({MODEL_KEY: Clf(), SCALER_KEY: Scaler()})


def test_divergent_arithmetic_raises():
    """A predict_proba that contradicts the persisted arrays is fatal, not a warning."""

    class Divergent:
        coef_ = np.zeros((3, 4))
        intercept_ = np.zeros(3)
        classes_ = np.arange(3)
        n_features_in_ = 4

        def predict_proba(self, features):
            # Deliberately unrelated to coef_/intercept_: this is what a real
            # scikit-learn regression on pickled state would look like.
            rows = np.asarray(features).shape[0]
            out = np.zeros((rows, 3))
            out[:, 0] = 1.0
            return out

    class Scaler:
        mean_ = np.zeros(4)
        scale_ = np.ones(4)
        n_features_in_ = 4

        def transform(self, features):
            return np.asarray(features)

    with pytest.raises(ModelNumericsError, match="does not reproduce"):
        validate_model_numerics({MODEL_KEY: Divergent(), SCALER_KEY: Scaler()})


# --------------------------------------------------------------------------------------
# Run-scoped accumulator feeding the provenance manifest
# --------------------------------------------------------------------------------------


@requires_models
def test_run_provenance_accumulates_and_caches():
    """Repeated loads validate once and land in the manifest block."""
    reset_run_provenance()
    try:
        assert run_provenance()["models"] == []

        path = _available_models(limit=1)[0]
        first = record_validated_model(path)
        second = record_validated_model(path)
        assert first is second, "a model should be validated once per run, not per call"

        report = run_provenance()
        assert [r["model"] for r in report["models"]] == [path.name]
        assert report["all_validated"] is True
        assert report["runtime_sklearn"]
        assert report["models"][0]["numerics_validated"] is True
    finally:
        reset_run_provenance()


@requires_models
def test_provenance_record_reports_skew_honestly():
    """The record states both versions and whether they differ."""
    import sklearn

    path = _available_models(limit=1)[0]
    record = model_provenance_record(path, validate=True)
    assert record["model"] == path.name
    assert record["runtime_sklearn"] == sklearn.__version__
    assert record["pickled_with_sklearn"]
    expected_skew = sklearn.__version__ not in record["pickled_with_sklearn"]
    assert record["version_skew"] is expected_skew
    assert record["numerics_validated"] is True


def test_absent_model_is_recorded_not_raised(tmp_path):
    """A named-but-missing model is reported, so one gap cannot abort a whole run."""
    report = collect_model_provenance(tmp_path, ["NotThere.pkl"], validate=True)
    assert report["models"] == [{"model": "NotThere.pkl", "error": "not present"}]
    assert report["all_validated"] is False


def test_reset_clears_accumulator():
    """reset_run_provenance leaves no state behind between runs in one process."""
    reset_run_provenance()
    assert run_provenance()["models"] == []
