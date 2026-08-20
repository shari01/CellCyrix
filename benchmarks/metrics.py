"""
metrics.py — the four measurements the annotation benchmark reports.

1. :func:`macro_f1_with_ci` — accuracy with a bootstrap confidence interval. Macro-F1
   rather than accuracy because cell types are severely imbalanced: a classifier that
   calls everything "T cell" scores well on a PBMC accuracy metric and badly here,
   which is the correct verdict. A point estimate without an interval cannot support
   "method A beats method B", so the interval is not optional.

2. :func:`risk_coverage_curve` — error rate as a function of how many cells a method
   is willing to call. This is the benchmark's headline: a method that abstains on its
   hardest 10% and is more accurate on the remaining 90% than a competitor is on 100%
   is genuinely better, and no conventional accuracy table can show that.

3. :func:`calibration` — do stated confidences mean anything? Expected calibration
   error compares predicted confidence against observed accuracy in bins. Almost
   nothing in the cell-annotation literature reports this, which is precisely why it
   differentiates.

4. :func:`voter_entropy` — disagreement among voters as a per-cell quantity, so it can
   be correlated against doublet scores, transitional position, or malignancy. This is
   what turns an engineering feature into a biological claim.

Every function takes plain pandas/NumPy and returns a DataFrame or dict, so results are
written to CSV and re-read rather than recomputed inside a plotting script.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Bootstrap resamples for confidence intervals. 1000 is enough for a stable 95%
#: interval at the cell counts these datasets have.
DEFAULT_BOOTSTRAP = 1000

#: Bins for the calibration curve / expected calibration error.
DEFAULT_CALIBRATION_BINS = 10


def _f1_macro(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Macro-averaged F1 over the classes present in `truth`.

    Averaged over truth classes rather than the union of truth and prediction classes:
    a method that invents labels absent from the truth set should be penalised through
    recall on the real classes, not by having its spurious classes averaged in as
    additional zeros, which would scale the penalty with how many junk labels it emits.

    Args:
        truth: Ground-truth labels.
        predicted: Predicted labels, same length.

    Returns:
        Macro-F1 in [0, 1].
    """
    classes = np.unique(truth)
    scores = np.empty(classes.size, dtype=float)
    for index, label in enumerate(classes):
        truth_positive = truth == label
        predicted_positive = predicted == label
        true_positives = float(np.count_nonzero(truth_positive & predicted_positive))
        if true_positives == 0.0:
            scores[index] = 0.0
            continue
        precision = true_positives / float(np.count_nonzero(predicted_positive))
        recall = true_positives / float(np.count_nonzero(truth_positive))
        scores[index] = 2.0 * precision * recall / (precision + recall)
    return float(scores.mean()) if scores.size else 0.0


def macro_f1_with_ci(
    truth: Sequence[str],
    predicted: Sequence[str],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Macro-F1, accuracy, and a bootstrap confidence interval.

    Resamples cells with replacement. Cells are the resampling unit because that is what
    the datasets vary in; for a claim about generalisation across *datasets*, bootstrap
    the datasets instead and report both.

    Args:
        truth: Ground-truth labels.
        predicted: Predicted labels, same length.
        n_bootstrap: Resamples. 0 skips the interval.
        seed: Seed, so the interval is reproducible.
        alpha: 0.05 gives a 95% interval.

    Returns:
        ``macro_f1``, ``accuracy``, ``ci_low``, ``ci_high``, ``n``, ``n_classes``.

    Raises:
        ValueError: If the two label sequences differ in length.
    """
    truth_array = np.asarray(truth, dtype=object)
    predicted_array = np.asarray(predicted, dtype=object)
    if truth_array.shape != predicted_array.shape:
        raise ValueError(
            f"truth and predicted differ in length: {truth_array.shape} vs "
            f"{predicted_array.shape}"
        )

    n = int(truth_array.size)
    point = _f1_macro(truth_array, predicted_array)
    accuracy = float(np.mean(truth_array == predicted_array)) if n else 0.0

    result = {
        "macro_f1": point,
        "accuracy": accuracy,
        "n": n,
        "n_classes": int(np.unique(truth_array).size),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if n_bootstrap <= 0 or n == 0:
        return result

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        picks = rng.integers(0, n, n)
        samples[i] = _f1_macro(truth_array[picks], predicted_array[picks])
    result["ci_low"] = float(np.percentile(samples, 100 * alpha / 2))
    result["ci_high"] = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return result


def per_class_f1(truth: Sequence[str], predicted: Sequence[str]) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support.

    Report this alongside macro-F1: a macro score hides which cell types a method fails
    on, and "fails on rare types" versus "fails on the dominant type" are different
    findings.

    Args:
        truth: Ground-truth labels.
        predicted: Predicted labels.

    Returns:
        One row per truth class, sorted by descending support.
    """
    truth_array = np.asarray(truth, dtype=object)
    predicted_array = np.asarray(predicted, dtype=object)
    rows = []
    for label in np.unique(truth_array):
        truth_positive = truth_array == label
        predicted_positive = predicted_array == label
        true_positives = float(np.count_nonzero(truth_positive & predicted_positive))
        n_predicted = float(np.count_nonzero(predicted_positive))
        n_truth = float(np.count_nonzero(truth_positive))
        precision = true_positives / n_predicted if n_predicted else 0.0
        recall = true_positives / n_truth if n_truth else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "cell_type": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(n_truth),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("support", ascending=False).reset_index(drop=True)


def risk_coverage_curve(
    correct: Sequence[bool],
    confidence: Sequence[float],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Error rate as a function of coverage, ordered by confidence.

    Ties in `confidence` are broken by a seeded shuffle rather than by input order.
    Without that, a table sorted by cell type would make a constant-confidence method
    look like it had meaningful ranking — the curve would trace the class ordering.

    Args:
        correct: Per-cell correctness of the method's call.
        confidence: Per-cell confidence used to rank which calls to keep. Higher means
            more confident.
        seed: Seed for tie-breaking.

    Returns:
        Columns ``coverage``, ``n_covered``, ``error``, ``accuracy``,
        ``confidence_threshold`` — one row per prefix of the confidence ordering.
    """
    correct_array = np.asarray(correct, dtype=bool)
    confidence_array = np.asarray(confidence, dtype=float)
    n = correct_array.size
    if n == 0:
        return pd.DataFrame(
            columns=[
                "coverage",
                "n_covered",
                "error",
                "accuracy",
                "confidence_threshold",
            ]
        )

    # NaN confidence means "no score"; rank those last rather than letting NaN sort
    # unpredictably.
    filled = np.where(np.isnan(confidence_array), -np.inf, confidence_array)
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(n)
    order = np.lexsort((tiebreak, -filled))

    ordered_correct = correct_array[order]
    cumulative_correct = np.cumsum(ordered_correct)
    n_covered = np.arange(1, n + 1)
    accuracy = cumulative_correct / n_covered

    return pd.DataFrame(
        {
            "coverage": n_covered / n,
            "n_covered": n_covered,
            "error": 1.0 - accuracy,
            "accuracy": accuracy,
            "confidence_threshold": filled[order],
        }
    )


def risk_at_coverage(curve: pd.DataFrame, coverage: float) -> float:
    """Error rate at a target coverage, read off a curve from :func:`risk_coverage_curve`.

    Args:
        curve: Output of :func:`risk_coverage_curve`.
        coverage: Target coverage in (0, 1], e.g. 0.9.

    Returns:
        The error rate at the smallest coverage that is >= `coverage`, or NaN when the
        curve is empty.
    """
    if curve.empty:
        return float("nan")
    eligible = curve[curve["coverage"] >= coverage]
    if eligible.empty:
        return float(curve["error"].iloc[-1])
    return float(eligible["error"].iloc[0])


def area_under_risk_coverage(curve: pd.DataFrame) -> float:
    """Area under the risk-coverage curve. Lower is better.

    A single number summarising selective-prediction quality across all operating
    points, so methods can be ranked without picking a coverage target first.

    Args:
        curve: Output of :func:`risk_coverage_curve`.

    Returns:
        AURC, or NaN for an empty curve.
    """
    if curve.empty:
        return float("nan")
    return float(np.trapezoid(curve["error"].to_numpy(), curve["coverage"].to_numpy()))


def calibration(
    correct: Sequence[bool],
    confidence: Sequence[float],
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reliability curve and expected calibration error.

    A method is calibrated when cells it calls with confidence 0.8 are right about 80%
    of the time. Equal-width bins over [0, 1] are used so the curve is comparable across
    methods whose confidence distributions differ in shape.

    Args:
        correct: Per-cell correctness.
        confidence: Per-cell confidence in [0, 1].
        n_bins: Number of bins.

    Returns:
        ``(curve, summary)``. `curve` has one row per non-empty bin with
        ``bin_lower``, ``bin_upper``, ``n``, ``mean_confidence``, ``accuracy``, ``gap``.
        `summary` carries ``ece`` (expected calibration error, support-weighted),
        ``mce`` (maximum calibration error), ``mean_confidence``, ``accuracy``,
        ``overconfidence`` (mean confidence minus accuracy; positive means the method
        overstates itself) and ``n_scored``.
    """
    correct_array = np.asarray(correct, dtype=bool)
    confidence_array = np.asarray(confidence, dtype=float)

    usable = ~np.isnan(confidence_array)
    correct_array = correct_array[usable]
    confidence_array = confidence_array[usable]
    n = correct_array.size

    empty_summary = {
        "ece": float("nan"),
        "mce": float("nan"),
        "mean_confidence": float("nan"),
        "accuracy": float("nan"),
        "overconfidence": float("nan"),
        "n_scored": 0,
    }
    if n == 0:
        return pd.DataFrame(), empty_summary

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts a confidence of exactly 0.0 in bin 0, which
    # would otherwise fall outside; clip keeps every value in a real bin.
    assignments = np.clip(
        np.digitize(confidence_array, edges[1:-1], right=True), 0, n_bins - 1
    )

    rows = []
    weighted_gap = 0.0
    max_gap = 0.0
    for index in range(n_bins):
        in_bin = assignments == index
        count = int(np.count_nonzero(in_bin))
        if count == 0:
            continue
        bin_confidence = float(confidence_array[in_bin].mean())
        bin_accuracy = float(correct_array[in_bin].mean())
        gap = abs(bin_confidence - bin_accuracy)
        weighted_gap += (count / n) * gap
        max_gap = max(max_gap, gap)
        rows.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "n": count,
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "gap": gap,
            }
        )

    mean_confidence = float(confidence_array.mean())
    accuracy = float(correct_array.mean())
    summary = {
        "ece": weighted_gap,
        "mce": max_gap,
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
        "overconfidence": mean_confidence - accuracy,
        "n_scored": n,
    }
    return pd.DataFrame(rows), summary


def voter_entropy(frame: pd.DataFrame, voter_columns: Iterable[str]) -> pd.Series:
    """Shannon entropy (bits) of the voters' harmonised calls, per cell.

    0.0 means unanimity; higher means the voters disagree. Correlate the result against
    the doublet score, diffusion pseudotime, or malignant status to test whether
    disagreement is picking up real biology rather than noise.

    Abstentions are excluded from each cell's distribution, so a cell where two voters
    agree and two abstain reads as unanimous rather than as a four-way split.

    Args:
        frame: Table of per-cell voter labels (already harmonised).
        voter_columns: Columns holding one voter's call each.

    Returns:
        Per-cell entropy in bits, indexed like `frame`. NaN where no voter called.
    """
    from benchmarks.harmonise import UNRESOLVED

    columns = [column for column in voter_columns if column in frame.columns]
    if not columns:
        logger.warning("[BENCH] no voter columns present; entropy is all-NaN")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    votes = frame[columns].astype(str)

    def _entropy(row: pd.Series) -> float:
        values = [value for value in row.to_numpy() if value != UNRESOLVED]
        if not values:
            return float("nan")
        _, counts = np.unique(values, return_counts=True)
        if counts.size <= 1:
            return 0.0
        proportions = counts / counts.sum()
        return float(-(proportions * np.log2(proportions)).sum())

    return votes.apply(_entropy, axis=1)


def correlate(
    left: pd.Series, right: pd.Series, *, method: str = "spearman"
) -> dict[str, float]:
    """Correlation between two per-cell quantities, on their shared non-null cells.

    Args:
        left: First quantity, e.g. voter entropy.
        right: Second quantity, e.g. doublet score.
        method: ``spearman`` (default, rank-based) or ``pearson``.

    Returns:
        ``rho``, ``n``, and ``method``. ``rho`` is NaN when fewer than three cells are
        shared or either side is constant.
    """
    joined = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(joined) < 3 or joined["left"].nunique() < 2 or joined["right"].nunique() < 2:
        return {"rho": float("nan"), "n": int(len(joined)), "method": method}
    rho = float(joined["left"].corr(joined["right"], method=method))
    return {"rho": rho, "n": int(len(joined)), "method": method}


def confusion_matrix(
    truth: Sequence[str], predicted: Sequence[str], *, normalise: Optional[str] = None
) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame.

    Args:
        truth: Ground-truth labels (rows).
        predicted: Predicted labels (columns).
        normalise: None for counts, ``"truth"`` for row-normalised (recall per class),
            ``"predicted"`` for column-normalised (precision per class).

    Returns:
        Matrix indexed by truth label, columns by predicted label.
    """
    table = pd.crosstab(
        pd.Series(list(truth), name="truth"),
        pd.Series(list(predicted), name="predicted"),
    )
    if normalise == "truth":
        return table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if normalise == "predicted":
        return table.div(table.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    return table


__all__ = [
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_CALIBRATION_BINS",
    "macro_f1_with_ci",
    "per_class_f1",
    "risk_coverage_curve",
    "risk_at_coverage",
    "area_under_risk_coverage",
    "calibration",
    "voter_entropy",
    "correlate",
    "confusion_matrix",
]
