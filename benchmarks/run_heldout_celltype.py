"""
run_heldout_celltype.py — does the pipeline abstain on a cell type it was never told about?

This is the experiment that tests the *disease-agnostic* claim, and it is the one a
reviewer cannot argue with. Every conventional annotator always emits a label: remove a
cell type from its references and it confidently returns the nearest neighbour instead.
A method with a calibrated abstention should decline.

Method
------
For each held-out cell type T:

1. Take the annotated ``h5ad`` (which already carries every voter's call and ground
   truth) and identify the cells whose truth is T.
2. Simulate T's absence from the reference by **censoring** it: any prediction that
   resolves to T is replaced with the method's next-best available answer. Since the
   voters' raw per-cell calls are already in ``obs``, this is done post hoc from the
   existing run rather than by re-running the pipeline 5x per dataset.
3. Measure, on T's cells only:
   * ``abstention_rate``  — fraction the method declines to call (low tier / veto / no
     resolvable label). Higher is better: T is genuinely not in the reference.
   * ``confident_error_rate`` — fraction given a *confident* wrong label. Lower is
     better. This is the number that matters, because a confident wrong call is what
     propagates into someone's downstream biology.
   * ``top_confusion``   — which type T is mistaken for, so the failure is legible.

The comparison is between methods, on the same cells, under the same censoring.

Why post-hoc censoring rather than re-running with edited references
--------------------------------------------------------------------
Editing ``master_celltype_markers_long.csv`` and the CellTypist model list, then
re-running, is the stronger experiment and you should do it for the final paper — this
script emits the exact commands for it with ``--emit-rerun-plan``. But it costs one full
pipeline run per held-out type per dataset, and it changes two things at once (the
marker panel AND the model's label space), which makes the result harder to attribute.
Censoring isolates "the label is unavailable" cleanly and runs in seconds, so it is the
right first pass and a useful cross-check on the re-run.

Usage::

    python benchmarks/run_heldout_celltype.py \\
        --h5ad outputs/pbmc/pbmc_processed_scanpy_output.h5ad \\
        --truth-column cell_type \\
        --n-types 5

    # print the commands for the full reference-ablation version, then exit
    python benchmarks/run_heldout_celltype.py --h5ad ... --truth-column cell_type \\
        --emit-rerun-plan
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harmonise import (  # noqa: E402
    DEFAULT_LEVEL,
    UNRESOLVED,
    evaluable_mask,
    harmonise,
)
from benchmarks.run_annotation_benchmark import (  # noqa: E402
    CONSENSUS_COLUMN,
    TIER_COLUMN,
    VOTER_COLUMNS,
    _confidence_for,
)

logger = logging.getLogger("benchmarks.heldout")

#: A call at or above this confidence counts as "confident" for
#: ``confident_error_rate``. 0.5 is the midpoint of the tier scale, so Medium and High
#: are confident and Low is not.
CONFIDENT_THRESHOLD = 0.5

#: Tiers that count as the pipeline declining to commit, independent of any numeric
#: confidence. These are what the lineage gate's veto produces.
ABSTAINING_TIERS = frozenset({"low", "review", "unassigned"})

#: Minimum cells of a type for its held-out result to mean anything.
MIN_CELLS_PER_TYPE = 20


def _is_abstaining(
    predictions: pd.Series,
    confidence: pd.Series,
    tier: Optional[pd.Series],
) -> pd.Series:
    """Per-cell mask of "the method declined to commit".

    Three ways a method can decline, any of which counts:
      * it emitted no resolvable label at all;
      * its tier is one the pipeline uses to mean "not confident" (the veto path);
      * it carries a numeric confidence below :data:`CONFIDENT_THRESHOLD`.

    A method with none of these signals can never abstain, and will score 0.0 here —
    which is the correct, and damning, result for it.

    Args:
        predictions: Harmonised per-cell labels.
        confidence: Per-cell numeric confidence, NaN where unavailable.
        tier: Per-cell tier strings, or None when the method has no tier.

    Returns:
        Boolean mask aligned to `predictions`.
    """
    declined = predictions == UNRESOLVED
    if tier is not None:
        declined = declined | tier.astype(str).str.strip().str.lower().isin(
            ABSTAINING_TIERS
        )
    # NaN confidence is "no score", not "low score" — do not count it as abstention.
    low_confidence = confidence.notna() & (confidence < CONFIDENT_THRESHOLD)
    return declined | low_confidence


def _censor(predictions: pd.Series, held_out: str) -> pd.Series:
    """Replace any prediction of `held_out` with abstention.

    Simulates the label being unavailable to the method. Applied identically to every
    method, so no method is advantaged by the simulation.

    Args:
        predictions: Harmonised per-cell labels.
        held_out: The label being withheld.

    Returns:
        Predictions with `held_out` replaced by :data:`UNRESOLVED`.
    """
    return predictions.where(predictions != held_out, UNRESOLVED)


def run_heldout(
    h5ad_path: Path,
    truth_column: str,
    *,
    out_dir: Path,
    level: str = DEFAULT_LEVEL,
    n_types: Optional[int] = None,
    extra_methods: Sequence[str] = (),
    min_cells: int = MIN_CELLS_PER_TYPE,
) -> dict:
    """Run the held-out cell-type experiment and write the result tables.

    Args:
        h5ad_path: Annotated pipeline output with ground truth in ``obs``.
        truth_column: Ground-truth label column.
        out_dir: Directory for results. Created if absent.
        level: Hierarchy level to compare at.
        n_types: Hold out only the N most abundant eligible types. None does all.
        extra_methods: Additional prediction columns already joined into ``obs``.
        min_cells: Types with fewer cells than this are skipped as uninformative.

    Returns:
        The manifest dict that was written.

    Raises:
        SystemExit: If the truth column is missing, no methods are present, or no type
            has enough cells to be worth holding out.
    """
    import anndata

    out_dir.mkdir(parents=True, exist_ok=True)
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()

    if truth_column not in obs.columns:
        raise SystemExit(
            f"Truth column {truth_column!r} not in obs. Available: "
            f"{sorted(obs.columns)[:25]}"
        )

    methods = [column for column in VOTER_COLUMNS if column in obs.columns]
    if CONSENSUS_COLUMN in obs.columns:
        methods.append(CONSENSUS_COLUMN)
    methods += [column for column in extra_methods if column in obs.columns]
    if not methods:
        raise SystemExit("No method columns found in obs.")

    harmonised = pd.DataFrame(index=obs.index)
    harmonised[truth_column] = harmonise(obs[truth_column], level=level).to_numpy()
    for method in methods:
        harmonised[method] = harmonise(obs[method], level=level).to_numpy()

    keep = evaluable_mask(harmonised[truth_column])
    harmonised, obs = harmonised[keep], obs[keep]
    truth = harmonised[truth_column]

    counts = truth[truth != UNRESOLVED].value_counts()
    eligible = counts[counts >= min_cells]
    if eligible.empty:
        raise SystemExit(
            f"No cell type has >= {min_cells} cells; nothing worth holding out. "
            f"Largest type has {int(counts.max()) if len(counts) else 0}."
        )
    held_out_types = list(eligible.index[:n_types] if n_types else eligible.index)
    logger.info(
        "Holding out %d type(s): %s", len(held_out_types), ", ".join(held_out_types)
    )

    tier = obs[TIER_COLUMN] if TIER_COLUMN in obs.columns else None

    rows, confusions = [], []
    for held_out in held_out_types:
        target = truth == held_out
        n_target = int(target.sum())

        for method in methods:
            confidence, confidence_source = _confidence_for(
                obs, method, len(obs), obs.index
            )
            censored = _censor(harmonised[method], held_out)

            method_tier = tier if method == CONSENSUS_COLUMN else None
            abstaining = _is_abstaining(censored, confidence, method_tier)

            on_target = censored[target]
            abstained_here = abstaining[target]
            confident_wrong = (~abstained_here) & (on_target != UNRESOLVED)

            wrong_labels = on_target[confident_wrong]
            top_confusion, top_confusion_n = "", 0
            if not wrong_labels.empty:
                top = wrong_labels.value_counts()
                top_confusion, top_confusion_n = str(top.index[0]), int(top.iloc[0])
                for label, count in top.items():
                    confusions.append(
                        {
                            "held_out_type": held_out,
                            "method": method,
                            "mistaken_for": label,
                            "n_cells": int(count),
                            "fraction": float(count) / n_target,
                        }
                    )

            rows.append(
                {
                    "held_out_type": held_out,
                    "method": method,
                    "confidence_source": confidence_source,
                    "n_cells_of_type": n_target,
                    "abstention_rate": float(abstained_here.mean()),
                    "confident_error_rate": float(confident_wrong.mean()),
                    "n_confident_errors": int(confident_wrong.sum()),
                    "top_confusion": top_confusion,
                    "top_confusion_n": top_confusion_n,
                    "can_abstain": bool(
                        method_tier is not None or confidence.notna().any()
                    ),
                }
            )

    detail = pd.DataFrame(rows).sort_values(["held_out_type", "confident_error_rate"])
    detail.to_csv(out_dir / "10_heldout_detail.csv", index=False)

    # Per-method summary, averaged over held-out types. This is the table for the paper.
    summary = (
        detail.groupby("method")
        .agg(
            n_types_held_out=("held_out_type", "nunique"),
            mean_abstention_rate=("abstention_rate", "mean"),
            mean_confident_error_rate=("confident_error_rate", "mean"),
            total_confident_errors=("n_confident_errors", "sum"),
            can_abstain=("can_abstain", "first"),
        )
        .sort_values("mean_confident_error_rate")
        .reset_index()
    )
    summary.to_csv(out_dir / "10_heldout_summary.csv", index=False)
    logger.info("Held-out summary:\n%s", summary.to_string(index=False))

    if confusions:
        pd.DataFrame(confusions).sort_values(
            ["held_out_type", "n_cells"], ascending=[True, False]
        ).to_csv(out_dir / "10_heldout_confusions.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "h5ad": str(Path(h5ad_path).resolve()),
        "truth_column": truth_column,
        "comparison_level": level,
        "method": "post-hoc label censoring (see --emit-rerun-plan for the full "
        "reference-ablation version)",
        "confident_threshold": CONFIDENT_THRESHOLD,
        "abstaining_tiers": sorted(ABSTAINING_TIERS),
        "min_cells_per_type": min_cells,
        "held_out_types": held_out_types,
        "methods": methods,
        "n_cells_evaluable": int(len(harmonised)),
    }
    (out_dir / "10_heldout_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote results to %s", out_dir)
    return manifest


def rerun_plan(held_out_types: Sequence[str]) -> str:
    """The commands for the full reference-ablation version of this experiment.

    Post-hoc censoring isolates "the label is unavailable" but leaves the underlying
    references intact. For the paper, also run the version that actually removes the
    type from the reference data, which is what a reviewer will ask about.

    Args:
        held_out_types: Types to generate a plan for.

    Returns:
        A shell-script-shaped plan, for the reader to adapt to their paths.
    """
    lines = [
        "# Full reference-ablation held-out experiment.",
        "# One pipeline run per held-out type. Costs real compute; run it once, late.",
        "#",
        "# For each type T:",
        "#   1. Copy the marker table and drop T's rows:",
        "#        shared_reference/single_cell_pipeline_agent_datasets/",
        "#          celltype_markers_references/TIS_CELL_markers_v3/",
        "#          master_celltype_markers_long.csv",
        "#   2. Point SCPIPE_SHARED_REFERENCE_ROOT at the edited copy.",
        "#   3. Pin celltypist_model to a model whose label space excludes T",
        "#      (or accept that CellTypist still knows T and report that caveat).",
        "#   4. Run, then score the run's own cells of type T.",
        "",
    ]
    for held_out in held_out_types:
        slug = (
            "".join(character if character.isalnum() else "_" for character in held_out)
            .strip("_")
            .lower()
        )
        lines += [
            f"# --- held out: {held_out} ---",
            f'python benchmarks/make_ablated_reference.py --drop "{held_out}" \\',
            f"    --out /tmp/ref_without_{slug}",
            f"SCPIPE_SHARED_REFERENCE_ROOT=/tmp/ref_without_{slug} \\",
            f"    cellcyrix --config config.yaml --output-root outputs/heldout_{slug}",
            "",
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_heldout_celltype",
        description=(
            "Held-out cell-type experiment: when a type is not in the reference, does "
            "the method abstain or confidently mislabel it?"
        ),
    )
    parser.add_argument("--h5ad", type=Path, required=True, help="Annotated .h5ad.")
    parser.add_argument(
        "--truth-column", required=True, help="obs column with ground-truth labels."
    )
    parser.add_argument(
        "--name", default=None, help="Result subdirectory (default: the h5ad's stem)."
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
        help=f"Hierarchy level to compare at (default: {DEFAULT_LEVEL}).",
    )
    parser.add_argument(
        "--n-types",
        type=int,
        default=5,
        help="Hold out the N most abundant types. 0 for all (default: 5).",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=MIN_CELLS_PER_TYPE,
        help=f"Skip types with fewer cells (default: {MIN_CELLS_PER_TYPE}).",
    )
    parser.add_argument(
        "--extra-method",
        action="append",
        default=[],
        dest="extra_methods",
        metavar="OBS_COLUMN",
        help="Additional prediction column already in obs. Repeatable.",
    )
    parser.add_argument(
        "--emit-rerun-plan",
        action="store_true",
        help="Print the full reference-ablation commands and exit without scoring.",
    )
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

    if args.emit_rerun_plan:
        import anndata

        adata = anndata.read_h5ad(args.h5ad, backed="r")
        truth = harmonise(adata.obs[args.truth_column], level=args.level)
        counts = truth[truth != UNRESOLVED].value_counts()
        types = list(counts.index[: args.n_types or None])
        logger.info("Reference-ablation plan:\n%s", rerun_plan(types))
        return 0

    name = args.name or Path(args.h5ad).stem
    run_heldout(
        args.h5ad,
        args.truth_column,
        out_dir=args.out_dir / name,
        level=args.level,
        n_types=args.n_types or None,
        extra_methods=args.extra_methods,
        min_cells=args.min_cells,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
