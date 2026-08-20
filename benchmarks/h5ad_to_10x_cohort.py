"""
h5ad_to_10x_cohort.py — turn a labelled .h5ad into the 10x cohort layout the pipeline reads.

The problem this solves
----------------------
Every benchmark dataset worth using arrives as an ``.h5ad`` — CELLxGENE Census returns
one, and so do most atlas downloads. The pipeline's loader reads only the Cell Ranger
trio (``matrix.mtx`` + ``barcodes.tsv`` + ``features.tsv``) from a directory per sample,
so there is no way to feed it a Census dataset directly.

This writes that layout::

    <out>/
      group_map.csv
      <GROUP>/<SAMPLE>/matrix.mtx.gz      genes x cells, integer counts
                       barcodes.tsv.gz
                       features.tsv.gz    feature_id, feature_name, "Gene Expression"

and, separately, a ``truth_labels.csv`` mapping every cell to its ground-truth label.
That sidecar is the part people get wrong: the pipeline does not carry an arbitrary
``obs`` column through a 10x round trip, so the truth has to be re-attached afterwards or
there is nothing to score against. ``attach-truth`` does that, joining on
``sample`` + ``barcode`` — both of which the pipeline preserves — rather than on
``obs_names``, which it regenerates.

Two subcommands::

    # 1. before the run
    python benchmarks/h5ad_to_10x_cohort.py convert \\
        --h5ad input_data/ts_blood.h5ad \\
        --sample-column donor_id \\
        --truth-column cell_type \\
        --out input_data/ts_blood_cohort

    # 2. after the run, so the benchmark has something to score
    python benchmarks/h5ad_to_10x_cohort.py attach-truth \\
        --h5ad outputs/ts_blood/combined_all_samples_processed_scanpy_output.h5ad \\
        --truth-csv input_data/ts_blood_cohort/truth_labels.csv

Counts must be raw integers. ``_validate_raw_counts`` in the loader rejects anything
else, which is correct — a normalised matrix silently breaks DESeq2 — so this refuses to
write non-integer data rather than letting the failure surface three stages later.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmarks.convert")

#: Written into the third column of features.tsv. The loader tolerates its absence but
#: real Cell Ranger output has it, and matching the real format keeps the code path
#: identical to a production run.
FEATURE_TYPE = "Gene Expression"

#: Group used when no group column is supplied. A single-group cohort still runs — QC,
#: clustering and annotation all work — but group contrasts are skipped.
DEFAULT_GROUP = "ALL"

#: Samples with fewer cells than this are dropped: a donor with a handful of cells
#: contributes noise to a pseudobulk column and can destabilise dispersion estimation.
MIN_CELLS_PER_SAMPLE = 50


def _safe_name(value: object) -> str:
    """Filesystem-safe directory segment for a sample or group id."""
    text = str(value).strip()
    cleaned = "".join(
        char if (char.isalnum() or char in "-_.") else "_" for char in text
    )
    return cleaned.strip("_.") or "UNKNOWN"


def _integer_valued(matrix) -> bool:
    """Whether a sparse matrix's stored values are whole numbers.

    Samples the first 100k non-zeros rather than the whole array: an atlas has billions
    of them, and normalised data is non-integer from its very first value.
    """
    data = getattr(matrix, "data", None)
    if data is None or data.size == 0:
        return True
    head = data[: min(data.size, 100_000)]
    return bool(np.allclose(head, np.rint(head)))


def _counts_matrix(adata):
    """Raw integer count matrix as CSR, cells x genes, with the var table that matches it.

    Three places counts can live, checked in order:

    1. ``layers["counts"]`` — where a locally-processed file usually keeps them.
    2. ``X`` — raw for a Census API pull.
    3. ``raw.X`` — where CELLxGENE's **downloadable .h5ad files** keep them. Their ``X``
       holds normalised, log1p-transformed data, so a portal download hits this branch.
       ``raw`` carries its own ``var`` (usually more genes, since ``X`` is filtered), so
       the feature table has to travel with the matrix or the two disagree on width.

    Each candidate is checked for integer values and skipped if it fails, rather than
    trusting the convention — a normalised matrix silently invalidates DESeq2, which is
    the one error that must not reach the output.

    Args:
        adata: Source AnnData.

    Returns:
        ``(matrix, var)`` — CSR counts, cells x genes, and the matching ``var`` frame.

    Raises:
        SystemExit: If no candidate holds integer counts.
    """
    import scipy.sparse as sp

    candidates = []
    if "counts" in adata.layers:
        candidates.append(("layers['counts']", adata.layers["counts"], adata.var))
    candidates.append(("X", adata.X, adata.var))
    if adata.raw is not None:
        candidates.append(("raw.X", adata.raw.X, adata.raw.var))

    rejected = []
    for source, matrix, var in candidates:
        matrix = sp.csr_matrix(matrix) if not sp.issparse(matrix) else matrix.tocsr()
        if _integer_valued(matrix):
            logger.info(
                "using %s as the count matrix (%d genes)", source, matrix.shape[1]
            )
            return matrix, var
        example = matrix.data[:3].tolist() if matrix.data.size else []
        rejected.append(f"{source} (non-integer, e.g. {example})")

    raise SystemExit(
        "No raw integer counts found. Checked: "
        + "; ".join(rejected)
        + ". Every candidate holds normalised or log-transformed data, and pseudobulk "
        "DESeq2 is invalid on anything but raw counts. Re-export the dataset with its "
        "count matrix in X, layers['counts'], or raw.X."
    )


def _write_gzip_tsv(path: Path, lines: Sequence[str]) -> None:
    """Write newline-terminated lines to a gzipped text file."""
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def _write_mtx_gz(path: Path, matrix) -> None:
    """Write a genes x cells Matrix Market file, gzipped.

    ``scipy.io.mmwrite`` cannot write into a gzip handle directly, so this writes the
    header and triplets itself. Values are emitted as integers because the loader
    validates raw counts, and a ``1.0`` would read as float data.

    Args:
        path: Destination ``.mtx.gz``.
        matrix: Genes x cells sparse matrix.
    """
    coo = matrix.tocoo()
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("%%MatrixMarket matrix coordinate integer general\n")
        handle.write(f"{coo.shape[0]} {coo.shape[1]} {coo.nnz}\n")
        if coo.nnz == 0:
            return
        # Written through pandas rather than a Python loop over the triplets: an
        # atlas sample carries ~10^8 non-zeros, where a per-line loop runs for
        # hours. Same output via a C writer, chunked so the formatted strings
        # never all exist at once.
        chunk = 5_000_000
        for start in range(0, coo.nnz, chunk):
            stop = min(start + chunk, coo.nnz)
            pd.DataFrame(
                {
                    # 1-based indices, per the Matrix Market spec.
                    "row": coo.row[start:stop].astype(np.int64) + 1,
                    "col": coo.col[start:stop].astype(np.int64) + 1,
                    "val": coo.data[start:stop].astype(np.int64),
                }
            ).to_csv(handle, sep=" ", header=False, index=False, lineterminator="\n")


def convert(
    h5ad_path: Path,
    *,
    out_dir: Path,
    sample_column: str,
    truth_column: Optional[str],
    group_column: Optional[str],
    min_cells: int = MIN_CELLS_PER_SAMPLE,
    max_samples: Optional[int] = None,
    max_cells: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Write a 10x cohort plus a truth sidecar from an annotated ``.h5ad``.

    Args:
        h5ad_path: Source file.
        out_dir: Cohort root to create.
        sample_column: ``obs`` column identifying the donor. This becomes the unit of
            replication for pseudobulk DE, so it must be the DONOR, not the cell.
        truth_column: ``obs`` column with ground-truth labels, written to the sidecar.
            None skips it (a cohort you only want to run, not score).
        group_column: ``obs`` column defining the experimental arms. None puts every
            sample in one group, which runs but skips group contrasts.
        min_cells: Drop samples smaller than this.
        max_samples: Keep only the `max_samples` largest donors, with ALL of their cells.
            Prefer this over `max_cells` for a small trial run: an atlas with hundreds of
            donors subsampled by cell leaves a few cells per donor, which `min_cells`
            then discards entirely. Applied before `max_cells`.
        max_cells: Subsample to at most this many cells before writing, for a fast first
            run. None keeps everything. Applied after `max_samples`.
        seed: Seed for subsampling.

    Returns:
        A summary dict: cells and genes written, samples, groups, and paths.

    Raises:
        SystemExit: If a named column is absent, or no sample survives `min_cells`.
    """
    import anndata

    # Backed when a subset was requested: an atlas .h5ad can be tens of GB in memory,
    # and for a trial run only a few donors are ever needed. The subset is materialised
    # below, so everything after that point works on an ordinary in-memory AnnData.
    wants_subset = (max_samples is not None) or (max_cells is not None)
    logger.info("reading %s%s", h5ad_path, " (backed)" if wants_subset else "")
    adata = anndata.read_h5ad(h5ad_path, backed="r" if wants_subset else None)
    logger.info("source: %d cells x %d genes", adata.n_obs, adata.n_vars)

    for name, column in (
        ("--sample-column", sample_column),
        ("--truth-column", truth_column),
        ("--group-column", group_column),
    ):
        if column and column not in adata.obs.columns:
            raise SystemExit(
                f"{name} {column!r} is not in obs. Available: "
                f"{sorted(adata.obs.columns)[:30]}"
            )

    if max_samples is not None:
        # Largest donors first, so a trial cohort has donors that actually survive
        # `min_cells`. Ties broken by donor id so the choice is reproducible.
        counts = adata.obs[sample_column].astype(str).value_counts()
        ordered = sorted(counts.index, key=lambda name: (-int(counts[name]), str(name)))
        keep_samples = set(ordered[:max_samples])
        mask = adata.obs[sample_column].astype(str).isin(keep_samples).to_numpy()
        adata = adata[mask]
        adata = adata.to_memory() if adata.isbacked else adata.copy()
        logger.info(
            "kept the %d largest donor(s) (--max-samples): %d cells",
            len(keep_samples),
            adata.n_obs,
        )

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        picked = rng.choice(adata.n_obs, size=max_cells, replace=False)
        # Sorted so the written order matches the source order, which keeps the output
        # reproducible and the truth sidecar easy to read.
        adata = adata[np.sort(picked)]
        adata = adata.to_memory() if adata.isbacked else adata.copy()
        logger.info("subsampled to %d cells (--max-cells)", adata.n_obs)

    matrix, var = _counts_matrix(adata)

    # Feature table, taken from whichever var matches the chosen matrix — `raw.var` is
    # usually wider than `var`, so using adata.var here would mismatch the width.
    # Census keeps Ensembl ids in the index and symbols in feature_name; fall back to
    # the index for both when there is no symbol column, since the loader needs two
    # columns and downstream code matches on the symbol.
    feature_ids = [str(value) for value in var.index]
    symbol_column = next(
        (
            name
            for name in ("feature_name", "gene_symbols", "gene_name", "symbol")
            if name in var.columns
        ),
        None,
    )
    if symbol_column:
        feature_names = [str(value) for value in var[symbol_column]]
        logger.info("gene symbols from var[%r]", symbol_column)
    else:
        feature_names = list(feature_ids)
        logger.warning(
            "no gene-symbol column in var; using var_names for both columns. "
            "Mitochondrial QC keys on 'MT-' symbols, so it will find nothing if these "
            "are Ensembl ids."
        )

    feature_lines = [
        f"{feature_id}\t{name}\t{FEATURE_TYPE}"
        for feature_id, name in zip(feature_ids, feature_names, strict=True)
    ]

    samples = adata.obs[sample_column].astype(str)
    groups = (
        adata.obs[group_column].astype(str)
        if group_column
        else pd.Series(DEFAULT_GROUP, index=adata.obs.index)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    truth_rows, group_rows = [], []
    n_cells_written = 0
    skipped = []

    for sample in sorted(samples.unique()):
        mask = (samples == sample).to_numpy()
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            skipped.append((sample, n_cells))
            continue

        # One sample must sit in exactly one group; a donor spanning arms is a design
        # error, so take the mode and say so rather than splitting them silently.
        sample_groups = groups[mask].unique()
        group = str(sample_groups[0])
        if len(sample_groups) > 1:
            group = str(groups[mask].mode().iloc[0])
            logger.warning(
                "sample %s spans groups %s; assigning it to %s",
                sample,
                list(sample_groups),
                group,
            )

        safe_sample, safe_group = _safe_name(sample), _safe_name(group)
        sample_dir = out_dir / safe_group / safe_sample
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Barcodes must be unique WITHIN a sample. Prefixing with the sample keeps them
        # unique across the cohort too, which is what makes the truth join unambiguous.
        barcodes = [
            f"{safe_sample}_{_safe_name(name)}" for name in adata.obs_names[mask]
        ]

        _write_mtx_gz(sample_dir / "matrix.mtx.gz", matrix[mask].T)
        _write_gzip_tsv(sample_dir / "barcodes.tsv.gz", barcodes)
        _write_gzip_tsv(sample_dir / "features.tsv.gz", feature_lines)

        if truth_column:
            truth_rows += [
                {"sample": safe_sample, "barcode": barcode, "truth": str(label)}
                for barcode, label in zip(
                    barcodes, adata.obs[truth_column][mask], strict=True
                )
            ]

        group_rows.append({"sample": safe_sample, "group": safe_group})
        n_cells_written += n_cells
        logger.info("wrote %s/%s (%d cells)", safe_group, safe_sample, n_cells)

    if not group_rows:
        raise SystemExit(
            f"No sample had >= {min_cells} cells. Largest was "
            f"{int(samples.value_counts().max())}; lower --min-cells or check "
            f"--sample-column {sample_column!r}."
        )

    group_map = pd.DataFrame(group_rows)
    group_map.to_csv(out_dir / "group_map.csv", index=False)

    truth_path = None
    if truth_column:
        truth_path = out_dir / "truth_labels.csv"
        pd.DataFrame(truth_rows).to_csv(truth_path, index=False)

    if skipped:
        logger.warning(
            "skipped %d sample(s) below --min-cells: %s",
            len(skipped),
            skipped[:10],
        )

    summary = {
        "cohort_dir": str(out_dir.resolve()),
        "n_cells": n_cells_written,
        "n_genes": len(feature_ids),
        "n_samples": len(group_rows),
        "groups": sorted(group_map["group"].unique().tolist()),
        "truth_csv": str(truth_path) if truth_path else None,
        "skipped_samples": skipped,
    }
    logger.info(
        "cohort ready: %d cells x %d genes, %d samples, groups %s",
        summary["n_cells"],
        summary["n_genes"],
        summary["n_samples"],
        summary["groups"],
    )
    if len(summary["groups"]) < 2:
        logger.warning(
            "only one group, so group contrasts and pseudobulk DE will be skipped. "
            "Pass --group-column to define arms (e.g. disease)."
        )
    return summary


def attach_truth(h5ad_path: Path, truth_csv: Path, *, column: str = "truth") -> dict:
    """Join the truth sidecar onto a processed ``.h5ad``, in place.

    Joins on ``sample`` + ``barcode``. Both survive the pipeline; ``obs_names`` does not,
    so joining on the index would silently mismatch.

    Args:
        h5ad_path: Processed pipeline output to modify.
        truth_csv: Sidecar written by :func:`convert`.
        column: ``obs`` column name to write.

    Returns:
        A summary: cells matched, cells unmatched, and the column written.

    Raises:
        SystemExit: If the h5ad lacks ``sample``/``barcode``, or nothing matches.
    """
    import anndata

    adata = anndata.read_h5ad(h5ad_path)
    missing = [name for name in ("sample", "barcode") if name not in adata.obs.columns]
    if missing:
        raise SystemExit(
            f"processed h5ad has no {missing} column(s) in obs, so the truth cannot be "
            f"joined. Available: {sorted(adata.obs.columns)[:30]}"
        )

    truth = pd.read_csv(truth_csv, dtype=str)
    keyed = truth.set_index(
        truth["sample"].astype(str) + "|" + truth["barcode"].astype(str)
    )["truth"]

    keys = adata.obs["sample"].astype(str) + "|" + adata.obs["barcode"].astype(str)
    mapped = keys.map(keyed)

    n_matched = int(mapped.notna().sum())
    if n_matched == 0:
        raise SystemExit(
            "no cell matched the truth sidecar on sample+barcode. Check that this h5ad "
            "came from the cohort this CSV was written for. Example h5ad key: "
            f"{keys.iloc[0]!r}; example CSV key: {keyed.index[0]!r}"
        )

    adata.obs[column] = mapped.fillna("").astype(str)
    adata.write_h5ad(h5ad_path)

    n_unmatched = int(len(mapped) - n_matched)
    logger.info(
        "attached obs[%r]: %d/%d cells matched (%d unmatched, written as empty)",
        column,
        n_matched,
        len(mapped),
        n_unmatched,
    )
    if n_unmatched:
        logger.warning(
            "%d cells have no truth label — expected, since QC drops cells; they are "
            "excluded from scoring automatically.",
            n_unmatched,
        )
    logger.info("now run: --truth-column %s", column)
    return {
        "h5ad": str(Path(h5ad_path).resolve()),
        "column": column,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="h5ad_to_10x_cohort",
        description=(
            "Convert a labelled .h5ad into the 10x cohort layout the pipeline reads, "
            "and re-attach the ground-truth labels afterwards."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert_parser = sub.add_parser(
        "convert", help="Write a 10x cohort + truth sidecar from an .h5ad."
    )
    convert_parser.add_argument("--h5ad", type=Path, required=True)
    convert_parser.add_argument("--out", type=Path, required=True, dest="out_dir")
    convert_parser.add_argument(
        "--sample-column",
        required=True,
        help="obs column identifying the DONOR (e.g. donor_id). This is the unit of "
        "replication for pseudobulk DE.",
    )
    convert_parser.add_argument(
        "--truth-column",
        default=None,
        help="obs column with ground-truth labels (e.g. cell_type), written to "
        "truth_labels.csv for re-attachment after the run.",
    )
    convert_parser.add_argument(
        "--group-column",
        default=None,
        help="obs column defining experimental arms (e.g. disease). Omit for a "
        "single-group cohort; group contrasts are then skipped.",
    )
    convert_parser.add_argument(
        "--min-cells",
        type=int,
        default=MIN_CELLS_PER_SAMPLE,
        help=f"Drop samples smaller than this (default: {MIN_CELLS_PER_SAMPLE}).",
    )
    convert_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Keep only the N largest donors, with all their cells. Use this rather "
        "than --max-cells for a trial run on an atlas with many donors.",
    )
    convert_parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Subsample to at most this many cells, for a fast first run.",
    )
    convert_parser.add_argument("--seed", type=int, default=0)

    attach_parser = sub.add_parser(
        "attach-truth", help="Join truth_labels.csv onto a processed .h5ad, in place."
    )
    attach_parser.add_argument("--h5ad", type=Path, required=True)
    attach_parser.add_argument("--truth-csv", type=Path, required=True)
    attach_parser.add_argument(
        "--column", default="truth", help="obs column to write (default: truth)."
    )

    parser.add_argument("-v", "--verbose", action="store_true")
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
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    if args.command == "convert":
        convert(
            args.h5ad,
            out_dir=args.out_dir,
            sample_column=args.sample_column,
            truth_column=args.truth_column,
            group_column=args.group_column,
            min_cells=args.min_cells,
            max_samples=args.max_samples,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    else:
        attach_truth(args.h5ad, args.truth_csv, column=args.column)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
