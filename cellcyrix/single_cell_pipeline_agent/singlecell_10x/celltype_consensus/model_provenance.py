"""
model_provenance.py — record, and numerically validate, the scikit-learn version
that each CellTypist model was pickled with.

Why this exists
---------------
The 41 CellTypist models in the bundle were not all pickled by the same scikit-learn:
30 carry ``_sklearn_version == "0.24.1"``, 8 carry ``"1.2.2"``, 3 carry ``"1.0.1"``.
The pipeline runs a much newer scikit-learn, so loading any of them emits::

    InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from
    version 0.24.1 when using version 1.8.0. This might lead to breaking code or
    invalid results.

That warning is upstream's model zoo, not a defect in this pipeline — but "might lead
to invalid results" is not something an annotation benchmark can leave unresolved, and
silencing the warning would only hide it.

Re-pickling the models under the current scikit-learn was rejected deliberately. It
would invalidate ``SHA256SUMS.txt`` — the control in :mod:`model_integrity` that stops
a tampered pickle from being unpickled — and destroy the upstream provenance, while
changing no number, because what is persisted is plain arrays.

So this module proves the skew is harmless instead of assuming it. A CellTypist model
is a ``LogisticRegression`` plus a ``StandardScaler``, and inference is fully
determined by four arrays::

    z = ((X - scaler.mean_) / scaler.scale_) @ clf.coef_.T + clf.intercept_
    P = softmax(z)                      # multinomial
    P = normalise(sigmoid(z))           # one-vs-rest

:func:`validate_model_numerics` recomputes ``P`` from those arrays with NumPy alone and
asserts it matches what the installed scikit-learn returns. If a future scikit-learn
ever changes how these two estimators interpret their own state, that assertion fails
loudly at the point of load rather than silently shifting every annotation.

The version each model was pickled with is read from the pickle *bytes*, never by
unpickling, so it is available before the file is trusted, and is recorded into
``provenance/manifest.json`` beside the runtime version so any run can be audited.
"""

from __future__ import annotations

import logging
import pickle  # noqa: S403 - load path is gated by model_integrity's checksum verify
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Attributes a CellTypist classifier must expose for inference to be well-defined.
REQUIRED_CLASSIFIER_ATTRS = ("coef_", "intercept_", "classes_", "n_features_in_")

#: Attributes the paired feature scaler must expose.
REQUIRED_SCALER_ATTRS = ("mean_", "scale_", "n_features_in_")

#: Keys CellTypist uses inside the pickled dict.
MODEL_KEY = "Model"
SCALER_KEY = "Scaler_"

#: ``_sklearn_version`` is stored as a short string in the estimator's pickled state.
#: Matching the bytes avoids unpickling just to learn which version wrote the file.
_VERSION_RE = re.compile(rb"_sklearn_version[\x00-\xff]{0,6}?(\d+\.\d+(?:\.\d+)?)")

#: Rows of random probe data used for the numeric equivalence check. Enough to exercise
#: every class boundary without making the check a measurable cost at load time.
_PROBE_ROWS = 32

#: The reconstruction is the same arithmetic in the same dtype, so agreement is exact
#: in practice (observed max|diff| == 0.0). The tolerance only absorbs BLAS reordering.
_NUMERIC_TOL = 1e-9


class ModelNumericsError(RuntimeError):
    """The installed scikit-learn does not reproduce the model's own arithmetic.

    Raised when a model's ``predict_proba`` disagrees with a direct NumPy
    reconstruction from its persisted arrays. That means the version skew has stopped
    being cosmetic, and annotations from this model are not trustworthy.
    """


def pickled_sklearn_versions(model_path: str | Path) -> list[str]:
    """Read the scikit-learn version(s) stamped into a model pickle, without loading it.

    Args:
        model_path: Path to a CellTypist ``.pkl``.

    Returns:
        Sorted unique version strings found in the pickle's estimator state. Empty when
        the file carries no ``_sklearn_version`` (very old or non-sklearn pickles).
    """
    raw = Path(model_path).read_bytes()
    return sorted({match.decode() for match in _VERSION_RE.findall(raw)})


def _missing_attrs(obj: Any, required: tuple[str, ...]) -> list[str]:
    """Names in `required` that `obj` does not carry."""
    return [name for name in required if not hasattr(obj, name)]


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Logistic sigmoid that does not overflow on large-magnitude logits.

    ``1/(1+exp(-z))`` overflows for very negative ``z``; the algebraically identical
    ``exp(z)/(1+exp(z))`` overflows for very positive ``z``. Selecting per element by
    sign keeps every ``exp`` argument non-positive, so the result is warning-free and
    numerically identical to what sklearn computes.

    Args:
        logits: Raw decision values of any shape.

    Returns:
        Elementwise sigmoid, same shape.
    """
    out = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponentiated = np.exp(logits[~positive])
    out[~positive] = exponentiated / (1.0 + exponentiated)
    return out


def _reconstruct_proba(
    classifier: Any, scaler: Any, features: np.ndarray
) -> np.ndarray:
    """Recompute class probabilities from persisted arrays using NumPy only.

    Mirrors what ``LogisticRegression.predict_proba`` does on top of a
    ``StandardScaler``, reading only ``mean_``/``scale_``/``coef_``/``intercept_`` so
    the result depends on the stored state and not on library behaviour.

    Args:
        classifier: The pickled ``LogisticRegression``.
        scaler: The pickled ``StandardScaler``.
        features: Raw (unscaled) probe matrix, shape ``(n_rows, n_features_in_)``.

    Returns:
        Probability matrix, shape ``(n_rows, n_classes)``.
    """
    scaled = (features - scaler.mean_) / scaler.scale_
    logits = scaled @ classifier.coef_.T + classifier.intercept_

    if classifier.coef_.shape[0] == 1:
        # Binary: sklearn returns [P(neg), P(pos)] from a single logit row.
        positive = _stable_sigmoid(logits)
        return np.hstack([1.0 - positive, positive])

    # Multinomial: softmax, computed shifted for numerical stability.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    softmax = exponentiated / exponentiated.sum(axis=1, keepdims=True)

    # One-vs-rest models normalise independent sigmoids instead of a softmax. Both
    # shapes are (n_rows, n_classes), so the caller cannot distinguish them by shape —
    # return whichever the estimator actually agrees with, decided in the caller.
    return softmax


def _reconstruct_proba_ovr(
    classifier: Any, scaler: Any, features: np.ndarray
) -> np.ndarray:
    """One-vs-rest variant of :func:`_reconstruct_proba`: normalised sigmoids."""
    scaled = (features - scaler.mean_) / scaler.scale_
    logits = scaled @ classifier.coef_.T + classifier.intercept_
    sigmoid = _stable_sigmoid(logits)
    totals = sigmoid.sum(axis=1, keepdims=True)
    # A row of all-zero sigmoids cannot be normalised; leave it, the comparison will
    # reject the ovr hypothesis rather than divide by zero.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, sigmoid / totals, sigmoid)


def validate_model_numerics(
    model: dict[str, Any],
    *,
    probe_rows: int = _PROBE_ROWS,
    seed: int = 0,
    tol: float = _NUMERIC_TOL,
) -> dict[str, Any]:
    """Assert the installed scikit-learn reproduces the model's persisted arithmetic.

    Args:
        model: An unpickled CellTypist model dict (``{"Model": ..., "Scaler_": ...}``).
        probe_rows: Rows of seeded random probe data to score.
        seed: Seed for the probe matrix, so the check is deterministic.
        tol: Maximum tolerated absolute probability difference.

    Returns:
        A record with the decision rule that matched (``"multinomial"`` or
        ``"one_vs_rest"``), the observed ``max_abs_diff``, and the probe shape.

    Raises:
        ModelNumericsError: If required attributes are absent, or if neither decision
            rule reproduces ``predict_proba`` within `tol`.
    """
    classifier = model.get(MODEL_KEY)
    scaler = model.get(SCALER_KEY)
    if classifier is None or scaler is None:
        raise ModelNumericsError(
            f"Model dict is missing {MODEL_KEY!r} and/or {SCALER_KEY!r}; keys present: "
            f"{sorted(model)}."
        )

    missing = _missing_attrs(classifier, REQUIRED_CLASSIFIER_ATTRS)
    missing += [
        f"Scaler_.{name}" for name in _missing_attrs(scaler, REQUIRED_SCALER_ATTRS)
    ]
    if missing:
        raise ModelNumericsError(
            "Unpickled estimator is missing state required for inference: "
            f"{', '.join(missing)}. The pickle was written by an incompatible "
            "scikit-learn and cannot be used."
        )

    n_features = int(classifier.n_features_in_)
    if int(scaler.n_features_in_) != n_features:
        raise ModelNumericsError(
            f"Feature-count mismatch: classifier expects {n_features}, scaler expects "
            f"{int(scaler.n_features_in_)}."
        )

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(probe_rows, n_features)).astype(np.float64)

    # predict_proba is what the pipeline actually relies on; scaler.transform is used
    # here (rather than the manual scaling) so the reference path is entirely sklearn's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = np.asarray(classifier.predict_proba(scaler.transform(features)))

    candidates = {
        "multinomial": _reconstruct_proba(classifier, scaler, features),
        "one_vs_rest": _reconstruct_proba_ovr(classifier, scaler, features),
    }

    best_rule, best_diff = None, np.inf
    for rule, reconstructed in candidates.items():
        if reconstructed.shape != reference.shape:
            continue
        diff = float(np.abs(reconstructed - reference).max())
        if diff < best_diff:
            best_rule, best_diff = rule, diff

    if best_rule is None or best_diff > tol:
        raise ModelNumericsError(
            "The installed scikit-learn does not reproduce this model's own "
            f"arithmetic (best rule {best_rule!r}, max|diff| {best_diff:.3e} > "
            f"tol {tol:.1e}). The pickled-vs-runtime scikit-learn skew has stopped "
            "being cosmetic; annotations from this model would be invalid. Pin "
            "scikit-learn to a version that reproduces it, or re-export the model."
        )

    return {
        "decision_rule": best_rule,
        "max_abs_diff": best_diff,
        "probe_shape": [int(probe_rows), n_features],
        "n_classes": int(np.asarray(classifier.classes_).size),
    }


def model_provenance_record(
    model_path: str | Path, *, validate: bool = True
) -> dict[str, Any]:
    """Build the provenance record for one model file.

    Args:
        model_path: Path to a CellTypist ``.pkl``. Should already have passed
            :func:`~.model_integrity.verify_model_file`, since this unpickles it when
            `validate` is True.
        validate: Run the numeric equivalence check. Set False to record versions only.

    Returns:
        A JSON-serialisable record: filename, the version(s) the pickle was written
        with, the runtime version, whether they differ, and the numeric-check outcome.

    Raises:
        ModelNumericsError: If `validate` is True and the check fails.
    """
    import sklearn  # local import: keeps module import cost off the non-CellTypist path

    path = Path(model_path)
    pickled = pickled_sklearn_versions(path)
    runtime = sklearn.__version__

    record: dict[str, Any] = {
        "model": path.name,
        "pickled_with_sklearn": pickled or None,
        "runtime_sklearn": runtime,
        "version_skew": bool(pickled) and runtime not in pickled,
        "numerics_validated": False,
    }

    if not validate:
        return record

    with warnings.catch_warnings():
        # InconsistentVersionWarning is the very condition being validated; suppressing
        # it here keeps the log readable, and the validation below is what makes that
        # suppression legitimate.
        warnings.simplefilter("ignore")
        with open(path, "rb") as handle:
            model = pickle.load(handle)  # noqa: S301 - checksum-verified upstream

    record.update(validate_model_numerics(model))
    record["numerics_validated"] = True

    if record["version_skew"]:
        logger.info(
            "[MODEL-PROVENANCE] %s was pickled with scikit-learn %s, running %s; "
            "predict_proba reproduced exactly (max|diff| %.1e, %s rule).",
            path.name,
            ", ".join(pickled),
            runtime,
            record["max_abs_diff"],
            record["decision_rule"],
        )
    return record


#: Records for the models this run actually loaded, keyed by filename. Populated by
#: :func:`record_validated_model` at the load site and drained into the run manifest.
#: Keyed so a model loaded once per cluster is validated once, not once per call —
#: unpickling a 50 MB model is the expensive part, not the 32-row probe.
_RUN_RECORDS: dict[str, dict[str, Any]] = {}


def record_validated_model(model_path: str | Path) -> dict[str, Any]:
    """Validate a model's numerics once per run and remember the outcome.

    Called at the CellTypist load site, immediately after the checksum verify. The
    first call for a given filename unpickles and validates; later calls return the
    cached record.

    Args:
        model_path: Path to a checksum-verified CellTypist ``.pkl``.

    Returns:
        The provenance record for this model.

    Raises:
        ModelNumericsError: If the installed scikit-learn does not reproduce the
            model's persisted arithmetic. This is deliberately fatal: an annotation
            run on a model whose inference cannot be reproduced is not salvageable.
    """
    name = Path(model_path).name
    cached = _RUN_RECORDS.get(name)
    if cached is not None:
        return cached
    record = model_provenance_record(model_path, validate=True)
    _RUN_RECORDS[name] = record
    return record


def run_provenance() -> dict[str, Any]:
    """Provenance for every model validated so far this run, for the manifest.

    Returns:
        ``{"runtime_sklearn": str, "models": [record, ...], "all_validated": bool}``.
        ``models`` is empty when no CellTypist model was loaded (an offline run with
        the voter disabled, for instance).
    """
    import sklearn

    records = [_RUN_RECORDS[name] for name in sorted(_RUN_RECORDS)]
    return {
        "runtime_sklearn": sklearn.__version__,
        "models": records,
        "all_validated": all(r.get("numerics_validated") for r in records),
    }


def reset_run_provenance() -> None:
    """Clear the accumulated records. Call between runs in the same process."""
    _RUN_RECORDS.clear()


def collect_model_provenance(
    models_dir: str | Path,
    model_names: Optional[list[str]] = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Provenance for the models a run actually used, for ``provenance/manifest.json``.

    A model that cannot be validated is recorded with its error rather than raising, so
    one bad file in a 41-model directory does not abort a run that never used it. The
    load path itself still raises, via :func:`model_provenance_record`.

    Args:
        models_dir: Directory holding the ``.pkl`` files.
        model_names: Filenames to record. None records every ``.pkl`` in the directory.
        validate: Run the numeric equivalence check on each.

    Returns:
        ``{"runtime_sklearn": str, "models": [record, ...], "all_validated": bool}``.
    """
    import sklearn

    directory = Path(models_dir)
    if model_names is None:
        names = sorted(p.name for p in directory.glob("*.pkl"))
    else:
        names = list(model_names)

    records: list[dict[str, Any]] = []
    for name in names:
        path = directory / name
        if not path.is_file():
            records.append({"model": name, "error": "not present"})
            continue
        try:
            records.append(model_provenance_record(path, validate=validate))
        except (ModelNumericsError, OSError, pickle.UnpicklingError) as exc:
            logger.warning("[MODEL-PROVENANCE] %s: %s", name, exc)
            records.append({"model": name, "error": str(exc)})

    return {
        "runtime_sklearn": sklearn.__version__,
        "models": records,
        "all_validated": bool(records)
        and all(r.get("numerics_validated") for r in records),
    }


__all__ = [
    "ModelNumericsError",
    "MODEL_KEY",
    "SCALER_KEY",
    "REQUIRED_CLASSIFIER_ATTRS",
    "REQUIRED_SCALER_ATTRS",
    "pickled_sklearn_versions",
    "validate_model_numerics",
    "model_provenance_record",
    "collect_model_provenance",
    "record_validated_model",
    "run_provenance",
    "reset_run_provenance",
]
