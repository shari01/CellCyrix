"""
run_annotation_benchmark.py — the annotation benchmark, end to end, from one h5ad.

Produces every table the paper needs from a pipeline output that already carries a
ground-truth column:

    benchmarks/results/<name>/
      00_resolution_report.csv      per-column resolution rates — READ THIS FIRST
      00_label_mapping.csv          every raw label and where it mapped (supplement)
      01_method_comparison.csv      macro-F1 + bootstrap CI, single voters vs consensus
      02_ablation.csv               every voter subset, so "why four voters?" is answered
      03_per_class_f1.csv           where each method fails, by cell type
      04_risk_coverage.csv          the headline curve, per method with a confidence
      05_risk_coverage_summary.csv  AURC + error at 100/95/90/80/70% coverage
      06_calibration.csv            reliability curve per method
      07_calibration_summary.csv    ECE, MCE, overconfidence
      08_disagreement.csv           per-cell voter entropy + correlations
      09_confusion_<method>.csv     row-normalised confusion matrix per method
      manifest.json                 inputs, columns used, seed, versions

Why single-voter baselines are free: the pipeline already writes every voter's own call
into ``.obs`` (``celltype_celltypist``, ``celltype_singler``,
``celltype_knowledge_based``), so CellTypist-alone and SingleR-alone need no extra runs
and the ablations are arithmetic over columns rather than five more pipelines.

Azimuth and GPTCelltype are NOT computed here — they need R. Run them separately, join
their per-cell labels onto the same ``obs`` index as extra columns, and pass them with
``--extra-method``; they then flow through the identical harmonisation and scoring.

Usage::

    python benchmarks/run_annotation_benchmark.py \\
        --h5ad outputs/tabula_sapiens/ts_processed_scanpy_output.h5ad \\
        --truth-column cell_type \\
        --name tabula_sapiens

    # with externally-computed baselines joined into obs beforehand
    python benchmarks/run_annotation_benchmark.py --h5ad ... --truth-column cell_type \\
        --extra-method celltype_azimuth --extra-method celltype_gptcelltype
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Import as a package so `benchmarks.harmonise` resolves whether this is run as a
# script from the repo root or imported by a test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harmonise import (  # noqa: E402
    DEFAULT_LEVEL,
    UNRESOLVED,
    evaluable_mask,
    harmonise,
    mapping_table,
    resolution_report,
)
from benchmarks.metrics import (  # noqa: E402
    DEFAULT_BOOTSTRAP,
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

logger = logging.getLogger("benchmarks")

#: The pipeline's per-voter obs columns. Each is a complete standalone baseline.
VOTER_COLUMNS = (
    "celltype_celltypist",
    "celltype_singler",
    "celltype_knowledge_based",
    # The PubMed literature voter. Omitted from this tuple originally, which meant it
    # ran on every cohort and was then silently absent from the comparison, the
    # ablation, and the disagreement entropy — the pipeline's four-voter design was
    # being scored as three.
    "celltype_pubmed",
)

#: The pipeline's own consensus call. Deliberately COARSE — the lineage gate collapses
#: to a conservative identity, so at fine granularity this column is not the pipeline's
#: answer and scoring it there understates the pipeline badly.
CONSENSUS_COLUMN = "celltype_consensus"

#: The pipeline's fine-grained call, carried in its own column so a single-annotator
#: subtype is never promoted to the main label. It IS the pipeline's answer at subtype
#: granularity, so it must be scored alongside the consensus or a subtype-level
#: comparison comes out meaningless: `celltype_consensus` harmonises to one label at that
#: level and scores ~0, which reads as failure rather than as "wrong column".
SUBTYPE_COLUMN = "celltype_subtype"

#: Per-cell confidence columns, by method, used to rank calls for the risk-coverage
#: curve. Only the consensus carries one natively; a baseline without a confidence
#: gets a flat curve, which is the honest representation of "it cannot abstain".
CONFIDENCE_COLUMNS = {
    CONSENSUS_COLUMN: "celltype_subtype_confidence",
    SUBTYPE_COLUMN: "celltype_subtype_confidence",
}

#: Ordered tiers, mapped to a numeric confidence when the float column is absent.
TIER_COLUMN = "consensus_tier"
TIER_SCORES = {"high": 1.0, "medium": 0.5, "low": 0.0}

#: Coverage points reported in the risk-coverage summary.
COVERAGE_TARGETS = (1.0, 0.95, 0.9, 0.8, 0.7)

#: Per-cell columns that voter disagreement is correlated against, when present.
CORRELATE_AGAINST = (
    "doublet_score",
    "predicted_doublet",
    "dpt_pseudotime",
    "celltypist_label_entropy",
    "n_genes_by_counts",
    "pct_counts_mt",
)


def _majority_vote(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Per-cell modal label across `columns`, ignoring unresolved calls.

    Ties are broken by the column order given, which makes the result deterministic:
    with two columns disagreeing, the earlier column wins. That is arbitrary but fixed,
    and it is reported in the ablation table so the reader knows the rule.

    Args:
        frame: Harmonised label columns.
        columns: Voter columns to combine.

    Returns:
        The combined call per cell, :data:`UNRESOLVED` where every voter abstained.
    """
    votes = frame[list(columns)]

    def _mode(row: pd.Series) -> str:
        values = [value for value in row.to_numpy() if value != UNRESOLVED]
        if not values:
            return UNRESOLVED
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        best = max(counts.values())
        # Preserve `columns` order among tied winners rather than dict order.
        for value in values:
            if counts[value] == best:
                return value
        return values[0]

    return votes.apply(_mode, axis=1)


def _confidence_for(
    frame: pd.DataFrame, method: str, n_cells: int, index: pd.Index
) -> tuple[pd.Series, str]:
    """Per-cell confidence for a method, and a note on where it came from.

    Args:
        frame: The obs table.
        method: Method column name.
        n_cells: Row count, for building a flat fallback.
        index: Index to align to.

    Returns:
        ``(confidence, source)``. A method with no confidence signal gets all-NaN and
        source ``"none"``, which produces a flat risk-coverage curve — the correct
        depiction of a method that cannot rank its own calls.
    """
    column = CONFIDENCE_COLUMNS.get(method)
    if column and column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            return values, column

    # Fall back to the ordered tier, which every consensus run writes.
    if method == CONSENSUS_COLUMN and TIER_COLUMN in frame.columns:
        mapped = frame[TIER_COLUMN].astype(str).str.strip().str.lower().map(TIER_SCORES)
        if mapped.notna().any():
            return pd.to_numeric(mapped, errors="coerce"), TIER_COLUMN

    return pd.Series(np.nan, index=index, dtype=float), "none"


def run_benchmark(
    h5ad_path: Path,
    truth_column: str,
    *,
    out_dir: Path,
    level: str = DEFAULT_LEVEL,
    extra_methods: Sequence[str] = (),
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Run every measurement and write the result tables.

    Args:
        h5ad_path: Annotated pipeline output carrying `truth_column` in ``obs``.
        truth_column: Ground-truth label column.
        out_dir: Directory for the result tables. Created if absent.
        level: Hierarchy level to compare at.
        extra_methods: Additional prediction columns (Azimuth, GPTCelltype, ...),
            already joined into ``obs``.
        n_bootstrap: Bootstrap resamples for the confidence intervals.
        seed: Seed for bootstrap and tie-breaking.

    Returns:
        The manifest dict that was written.

    Raises:
        SystemExit: If the file has no such truth column, or no method columns at all —
            both are setup errors that should stop the run rather than emit empty tables.
    """
    import anndata

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reading %s", h5ad_path)
    # backed='r' keeps a large atlas off the heap: only obs is needed here.
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    logger.info("obs: %d cells x %d columns", len(obs), obs.shape[1])

    if truth_column not in obs.columns:
        raise SystemExit(
            f"Truth column {truth_column!r} not in obs. Available columns include: "
            f"{sorted(obs.columns)[:25]}"
        )

    voters = [column for column in VOTER_COLUMNS if column in obs.columns]
    methods = list(voters)
    if CONSENSUS_COLUMN in obs.columns:
        methods.append(CONSENSUS_COLUMN)
    if SUBTYPE_COLUMN in obs.columns:
        methods.append(SUBTYPE_COLUMN)
    methods += [column for column in extra_methods if column in obs.columns]

    missing_extra = [column for column in extra_methods if column not in obs.columns]
    if missing_extra:
        logger.warning(
            "Requested method column(s) absent from obs, skipping: %s", missing_extra
        )
    if not methods:
        raise SystemExit(
            "No method columns found in obs. Expected some of "
            f"{list(VOTER_COLUMNS) + [CONSENSUS_COLUMN]}."
        )
    logger.info("Voters: %s", voters or "none")
    logger.info("Methods scored: %s", methods)

    label_columns = [truth_column] + methods

    # --- 00: resolution rates and the mapping table. Reported BEFORE any accuracy
    # number, because a low resolution rate makes an accuracy comparison confounded.
    resolution = resolution_report(obs, label_columns, level=level)
    resolution.to_csv(out_dir / "00_resolution_report.csv", index=False)
    mapping_table(obs, label_columns, level=level).to_csv(
        out_dir / "00_label_mapping.csv", index=False
    )
    logger.info("Resolution rates:\n%s", resolution.to_string(index=False))

    truth_unresolved = float(
        resolution.loc[
            resolution["column"] == truth_column, "unresolved_fraction"
        ].iloc[0]
    )
    confounded = []
    for _, row in resolution.iterrows():
        if row["column"] == truth_column:
            continue
        # A method whose labels resolve much worse than the truth is being scored partly
        # on vocabulary coverage. Flag it rather than presenting the gap as accuracy.
        if row["unresolved_fraction"] > truth_unresolved + 0.10:
            confounded.append(
                {
                    "column": row["column"],
                    "unresolved_fraction": float(row["unresolved_fraction"]),
                    "truth_unresolved_fraction": truth_unresolved,
                }
            )
    if confounded:
        logger.warning(
            "These methods resolve >10pp worse than the truth column; their scores "
            "partly measure vocabulary coverage, not biology: %s",
            [item["column"] for item in confounded],
        )

    # --- harmonise everything through the identical function
    harmonised = pd.DataFrame(index=obs.index)
    for column in label_columns:
        harmonised[column] = harmonise(obs[column], level=level).to_numpy()

    # A voter that was switched off still leaves its obs column behind, filled with a
    # single placeholder. Scoring it produces macro_f1 0.0 and a flat error-1.0 curve in
    # every table, which reads as "this method is terrible" rather than "this method did
    # not run". Drop those, and record it, so the tables describe methods that actually
    # made calls.
    inactive = [
        method for method in methods if bool((harmonised[method] == UNRESOLVED).all())
    ]
    if inactive:
        logger.warning(
            "Excluding %d method(s) that produced no resolvable call for ANY cell "
            "(voter disabled or unavailable): %s",
            len(inactive),
            inactive,
        )
        methods = [method for method in methods if method not in inactive]
        voters = [voter for voter in voters if voter not in inactive]
        if not methods:
            raise SystemExit(
                "Every method column is empty. No voter produced a call, so there is "
                "nothing to score. Check that at least one voter was enabled."
            )

    keep = evaluable_mask(harmonised[truth_column])
    n_dropped = int((~keep).sum())
    logger.info(
        "Evaluable cells: %d of %d (%d dropped for unresolvable ground truth)",
        int(keep.sum()),
        len(keep),
        n_dropped,
    )
    if not keep.any():
        raise SystemExit(
            f"No cell's ground truth resolved at level {level!r}. Check that "
            f"{truth_column!r} holds cell-type names."
        )

    scored = harmonised[keep]
    scored_obs = obs[keep]
    truth = scored[truth_column].to_numpy()

    # --- 01: method comparison
    comparison_rows = []
    for method in methods:
        stats = macro_f1_with_ci(
            truth, scored[method].to_numpy(), n_bootstrap=n_bootstrap, seed=seed
        )
        comparison_rows.append(
            {
                "method": method,
                "kind": (
                    "consensus"
                    if method == CONSENSUS_COLUMN
                    else "consensus_subtype"
                    if method == SUBTYPE_COLUMN
                    else "single_voter"
                ),
                **stats,
                "unresolved_predictions": int((scored[method] == UNRESOLVED).sum()),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(out_dir / "01_method_comparison.csv", index=False)
    logger.info("Method comparison:\n%s", comparison.to_string(index=False))

    # --- 02: ablation over every voter subset
    ablation_rows = []
    for size in range(1, len(voters) + 1):
        for subset in combinations(voters, size):
            predicted = (
                scored[list(subset)[0]]
                if size == 1
                else _majority_vote(scored, list(subset))
            )
            stats = macro_f1_with_ci(
                truth, predicted.to_numpy(), n_bootstrap=n_bootstrap, seed=seed
            )
            ablation_rows.append(
                {
                    "voters": "+".join(subset),
                    "n_voters": size,
                    **stats,
                    "tie_rule": "first listed voter wins" if size > 1 else "n/a",
                }
            )
    if CONSENSUS_COLUMN in scored.columns:
        stats = macro_f1_with_ci(
            truth,
            scored[CONSENSUS_COLUMN].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        ablation_rows.append(
            {
                "voters": "pipeline_consensus (hierarchy + lineage gate)",
                "n_voters": len(voters),
                **stats,
                "tie_rule": "pipeline consensus logic",
            }
        )
    ablation = pd.DataFrame(ablation_rows).sort_values("macro_f1", ascending=False)
    ablation.to_csv(out_dir / "02_ablation.csv", index=False)

    # --- 03: per-class F1
    per_class = []
    for method in methods:
        frame = per_class_f1(truth, scored[method].to_numpy())
        frame.insert(0, "method", method)
        per_class.append(frame)
    if per_class:
        pd.concat(per_class, ignore_index=True).to_csv(
            out_dir / "03_per_class_f1.csv", index=False
        )

    # --- 04/05: risk-coverage
    curves, summary_rows = [], []
    for method in methods:
        confidence, source = _confidence_for(
            scored_obs, method, len(scored_obs), scored_obs.index
        )
        correct = scored[method].to_numpy() == truth
        curve = risk_coverage_curve(correct, confidence.to_numpy(), seed=seed)
        if curve.empty:
            continue
        curve.insert(0, "method", method)
        curve.insert(1, "confidence_source", source)
        curves.append(curve)
        summary_rows.append(
            {
                "method": method,
                "confidence_source": source,
                "aurc": area_under_risk_coverage(curve),
                **{
                    f"error_at_{int(target * 100)}pct_coverage": risk_at_coverage(
                        curve, target
                    )
                    for target in COVERAGE_TARGETS
                },
            }
        )
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            out_dir / "04_risk_coverage.csv", index=False
        )
        rc_summary = pd.DataFrame(summary_rows).sort_values("aurc")
        rc_summary.to_csv(out_dir / "05_risk_coverage_summary.csv", index=False)
        logger.info("Risk-coverage summary:\n%s", rc_summary.to_string(index=False))

    # --- 06/07: calibration
    calib_curves, calib_summary = [], []
    for method in methods:
        confidence, source = _confidence_for(
            scored_obs, method, len(scored_obs), scored_obs.index
        )
        if confidence.isna().all():
            continue
        correct = scored[method].to_numpy() == truth
        curve, stats = calibration(correct, confidence.to_numpy())
        if not curve.empty:
            curve.insert(0, "method", method)
            calib_curves.append(curve)
        calib_summary.append({"method": method, "confidence_source": source, **stats})
    if calib_curves:
        pd.concat(calib_curves, ignore_index=True).to_csv(
            out_dir / "06_calibration.csv", index=False
        )
    if calib_summary:
        pd.DataFrame(calib_summary).to_csv(
            out_dir / "07_calibration_summary.csv", index=False
        )

    # --- 08: disagreement entropy + correlations
    correlations = []
    if len(voters) >= 2:
        entropy = voter_entropy(scored, voters)
        disagreement = pd.DataFrame(
            {
                "voter_entropy_bits": entropy,
                "consensus_correct": (
                    scored[CONSENSUS_COLUMN].to_numpy() == truth
                    if CONSENSUS_COLUMN in scored.columns
                    else np.nan
                ),
                "truth": truth,
            }
        )
        for column in CORRELATE_AGAINST:
            if column not in scored_obs.columns:
                continue
            values = pd.to_numeric(scored_obs[column], errors="coerce")
            if values.notna().sum() < 3:
                continue
            disagreement[column] = values.to_numpy()
            result = correlate(entropy, values)
            correlations.append({"against": column, **result})
        # Does disagreement predict the consensus being wrong? If yes, entropy is a
        # usable abstention signal on its own, which is a claim worth making.
        if CONSENSUS_COLUMN in scored.columns:
            correlations.append(
                {
                    "against": "consensus_incorrect",
                    **correlate(
                        entropy,
                        pd.Series(
                            (scored[CONSENSUS_COLUMN].to_numpy() != truth).astype(
                                float
                            ),
                            index=scored.index,
                        ),
                    ),
                }
            )
        disagreement.to_csv(out_dir / "08_disagreement.csv", index=False)
        if correlations:
            pd.DataFrame(correlations).to_csv(
                out_dir / "08_disagreement_correlations.csv", index=False
            )
            logger.info(
                "Disagreement correlations:\n%s",
                pd.DataFrame(correlations).to_string(index=False),
            )

    # --- 09: confusion matrices
    for method in methods:
        confusion_matrix(truth, scored[method].to_numpy(), normalise="truth").to_csv(
            out_dir / f"09_confusion_{method}.csv"
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "h5ad": str(Path(h5ad_path).resolve()),
        "truth_column": truth_column,
        "comparison_level": level,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_cells_total": int(len(obs)),
        "n_cells_evaluable": int(keep.sum()),
        "n_cells_dropped_unresolvable_truth": n_dropped,
        "voters_present": voters,
        "methods_scored": methods,
        "extra_methods_missing": missing_extra,
        "methods_excluded_no_calls": inactive,
        "confounded_by_resolution_gap": confounded,
        "package_versions": _versions(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote results to %s", out_dir)
    return manifest


def _versions() -> dict[str, str]:
    """Versions of the packages whose behaviour affects these numbers."""
    import importlib.metadata as metadata

    out = {"python": sys.version.split()[0]}
    for package in (
        "anndata",
        "numpy",
        "pandas",
        "scikit-learn",
        "celltypist",
        "scanpy",
    ):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return out


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_annotation_benchmark",
        description=(
            "Score the pipeline's consensus annotation against ground truth, alongside "
            "every single-voter baseline and voter-subset ablation, with bootstrap "
            "confidence intervals, risk-coverage curves, calibration, and voter "
            "disagreement."
        ),
    )
    parser.add_argument(
        "--h5ad", type=Path, required=True, help="Annotated pipeline output (.h5ad)."
    )
    parser.add_argument(
        "--truth-column",
        required=True,
        help="obs column holding ground-truth cell-type labels.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Result subdirectory name (default: the h5ad's stem).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Root for result directories (default: benchmarks/results).",
    )
    parser.add_argument(
        "--level",
        default=DEFAULT_LEVEL,
        choices=["lineage", "class", "main_cell_type", "subtype", "fine_subtype"],
        help=(
            "Hierarchy level to compare at. Coarser is fairer across methods with "
            f"different granularity (default: {DEFAULT_LEVEL})."
        ),
    )
    parser.add_argument(
        "--extra-method",
        action="append",
        default=[],
        dest="extra_methods",
        metavar="OBS_COLUMN",
        help=(
            "Additional prediction column already joined into obs (e.g. "
            "celltype_azimuth, celltype_gptcelltype). Repeatable."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=f"Bootstrap resamples for CIs; 0 to skip (default: {DEFAULT_BOOTSTRAP}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed (default: 0).")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    name = args.name or Path(args.h5ad).stem
    run_benchmark(
        args.h5ad,
        args.truth_column,
        out_dir=args.out_dir / name,
        level=args.level,
        extra_methods=args.extra_methods,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
