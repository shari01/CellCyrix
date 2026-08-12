"""
data_validation.py — pre-flight data validation with SAFE auto-repair.

Two responsibilities:
  1. VALIDATE the input AnnData against every invariant the pipeline assumes, and
  2. AUTO-FIX only the mechanical, lossless problems — never the scientific ones.

Severity model (deliberate — silently "repairing" science is how a pipeline
produces plausible-but-wrong results):

  * FAIL  — would CORRUPT or MISLEAD every downstream result if the run continued.
            The pre-flight hook raises `DataValidationError` and stops the run.
            e.g. matrix is normalized/scaled (not raw counts), NaN/Inf, negative
            values, empty matrix, integration batch == biological condition, or a
            sample carrying more than one group label.
  * FIXED — a mechanical, information-preserving defect the tool repairs in place
            and LOGS. e.g. non-unique barcodes, duplicate gene names, all-zero
            genes/cells, float-but-integer dtype, Ensembl IDs when a symbol column
            exists.
  * WARN  — a genuine scientific limitation the pipeline already handles safely; a
            human should note it but it does not corrupt anything, so it does NOT
            block. e.g. <2 samples per group (pseudobulk DE is skipped, correctly),
            no mitochondrial genes detectable, `sample` column missing.
  * PASS  — the check passed.

The pipeline calls `run_preflight_validation(...)` at the very start of a run
(fail-fast, before hours of compute). It is also usable standalone:

    from ...data_validation import validate_and_fix, validate_path
    adata, report = validate_and_fix(adata, batch_key="sample")
    report.write("out/00_data_validation")            # CSV + text
    # or straight from disk:
    adata, report = validate_path("path/to/10x_dir")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp

from .atomic_io import atomic_to_csv
from .exceptions import PipelineComputationError

if TYPE_CHECKING:  # annotation only — anndata is not imported at module load
    from anndata import AnnData

logger = logging.getLogger(__name__)

PASS, WARN, FAIL, FIXED = "pass", "warn", "fail", "fixed"

# candidate columns that may hold a gene SYMBOL when var_names are Ensembl IDs
_SYMBOL_COLS = (
    "gene_symbol",
    "feature_name",
    "symbol",
    "SYMBOL",
    "gene_name",
    "GeneName",
)


class DataValidationError(PipelineComputationError):
    """Raised by the pre-flight hook when a FATAL (fail-severity) data problem exists."""


@dataclass
class Check:
    """One validation check and its outcome.

    Attributes:
        name: Short identifier for the check, used as the CSV row key.
        status: One of ``pass``, ``warn``, ``fail``, ``fixed``.
        message: Human-readable detail, including any numbers that justify the status.
    """

    name: str
    status: str  # pass | warn | fail | fixed
    message: str


@dataclass
class ValidationReport:
    """The full set of validation checks for one run, plus the shape it changed.

    Attributes:
        checks: Every check performed, in the order it ran.
        n_obs_before: Cells before validation dropped or fixed anything.
        n_vars_before: Genes before validation dropped or fixed anything.
        n_obs_after: Cells after validation.
        n_vars_after: Genes after validation.
    """

    checks: List[Check] = field(default_factory=list)
    n_obs_before: int = 0
    n_vars_before: int = 0
    n_obs_after: int = 0
    n_vars_after: int = 0

    def add(self, name: str, status: str, message: str) -> None:
        """Record one check and log it at the level its status implies.

        Args:
            name: Short check identifier, e.g. ``"raw_counts"``.
            status: One of :data:`PASS`, :data:`WARN`, :data:`FAIL`, :data:`FIXED`.
            message: Operator-facing explanation, including the offending values.
        """
        self.checks.append(Check(name, status, message))
        lvl = {FAIL: logging.ERROR, WARN: logging.WARNING}.get(status, logging.INFO)
        logger.log(lvl, "[VALIDATE] %-5s %s: %s", status.upper(), name, message)

    @property
    def fails(self) -> List[Check]:
        """Checks at FAIL severity — these block the run."""
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warns(self) -> List[Check]:
        """Checks at WARN severity — recorded, but the run continues."""
        return [c for c in self.checks if c.status == WARN]

    @property
    def fixes(self) -> List[Check]:
        """Checks where the problem was found and repaired automatically."""
        return [c for c in self.checks if c.status == FIXED]

    @property
    def ok(self) -> bool:
        """True when there are no FAIL-severity checks (WARN/FIXED are acceptable)."""
        return not self.fails

    def to_frame(self) -> pd.DataFrame:
        """Return the checks as a ``check`` / ``status`` / ``message`` DataFrame."""
        return pd.DataFrame(
            [
                {"check": c.name, "status": c.status, "message": c.message}
                for c in self.checks
            ]
        )

    def write(self, out_dir: str | Path, analysis_name: str = "data") -> Path:
        """Write the report as both CSV and human-readable text.

        Args:
            out_dir: Directory to write into; created if absent.
            analysis_name: Prefix for both output filenames.

        Returns:
            Path to the CSV that was written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{analysis_name}_data_validation.csv"
        atomic_to_csv(self.to_frame(), csv_path, index=False)
        lines = [
            f"Data validation — {analysis_name}",
            f"cells {self.n_obs_before} -> {self.n_obs_after} | "
            f"genes {self.n_vars_before} -> {self.n_vars_after}",
            f"FAIL={len(self.fails)}  WARN={len(self.warns)}  FIXED={len(self.fixes)}  "
            f"overall={'OK' if self.ok else 'FAILED'}",
            "",
        ]
        for c in self.checks:
            lines.append(f"[{c.status.upper():5s}] {c.name}: {c.message}")
        (out_dir / f"{analysis_name}_data_validation.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        return csv_path


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _stored_values(X) -> np.ndarray:
    """Stored (explicit) values of X — sparse .data or dense ravel. Implicit zeros
    in a sparse matrix are non-negative/finite, so omitting them is safe for the
    finite/negative/integer checks."""
    if sp.issparse(X):
        return np.asarray(X.data)
    return np.asarray(X).ravel()


def _looks_ensembl(names, prefix: str = "ENSG", frac: float = 0.6) -> bool:
    sample = [str(x).upper() for x in list(names)[:50]]
    if not sample:
        return False
    return float(np.mean([s.startswith(prefix) for s in sample])) > frac


# --------------------------------------------------------------------------- #
#  core
# --------------------------------------------------------------------------- #
def validate_and_fix(
    adata: AnnData,
    *,
    group_col: str = "group",
    sample_col: str = "sample",
    batch_key: Optional[str] = None,
    fix: bool = True,
    min_samples_per_group: int = 2,
) -> Tuple["object", ValidationReport]:
    """Validate `adata`; auto-fix mechanical issues when `fix=True`.

    Returns `(adata, report)`. `report.ok` is False iff a FAIL check fired. The
    returned adata may be a subset copy (all-zero genes/cells dropped) or have
    unique-ified names; the original is not relied upon by the caller.
    """
    rep = ValidationReport(
        n_obs_before=int(adata.n_obs), n_vars_before=int(adata.n_vars)
    )

    # ---- empty matrix (fatal, and nothing else is meaningful) ----
    if adata.n_obs == 0 or adata.n_vars == 0:
        rep.add(
            "nonempty",
            FAIL,
            f"empty matrix ({adata.n_obs} cells x {adata.n_vars} genes).",
        )
        rep.n_obs_after, rep.n_vars_after = int(adata.n_obs), int(adata.n_vars)
        return adata, rep
    rep.add("nonempty", PASS, f"{adata.n_obs} cells x {adata.n_vars} genes.")

    # ---- raw-count integrity (FATAL): the counts layer if present, else X ----
    src = (
        adata.layers["counts"] if "counts" in getattr(adata, "layers", {}) else adata.X
    )
    vals = _stored_values(src)
    finite_ok = True
    if vals.size and not np.all(np.isfinite(vals)):
        finite_ok = False
        rep.add("no_nan_inf", FAIL, "matrix contains NaN/Inf values.")
    else:
        rep.add("no_nan_inf", PASS, "no NaN/Inf values.")

    neg_ok = True
    if vals.size and finite_ok and float(vals.min()) < 0:
        neg_ok = False
        rep.add(
            "non_negative",
            FAIL,
            f"matrix has negative values (min={float(vals.min()):.3g}); "
            "this looks like normalized/scaled data, not raw counts.",
        )
    else:
        rep.add("non_negative", PASS, "no negative values.")

    if vals.size and finite_ok and neg_ok:
        if np.allclose(vals, np.rint(vals), rtol=0, atol=1e-8):
            rep.add(
                "raw_integer_counts", PASS, "values are integer-valued (raw counts)."
            )
        else:
            frac = float(np.mean(~np.isclose(vals, np.rint(vals), rtol=0, atol=1e-8)))
            rep.add(
                "raw_integer_counts",
                FAIL,
                f"{frac:.1%} of values are non-integer; this looks like "
                "normalized/log/scaled data, not raw counts. Load raw counts.",
            )

    # ---- mechanical FIXES (lossless) ----
    # unique barcodes
    if not adata.obs_names.is_unique:
        n_dup = int(pd.Index(adata.obs_names).duplicated().sum())
        if fix:
            adata.obs_names_make_unique()
            rep.add(
                "unique_barcodes", FIXED, f"made {n_dup} duplicate barcode(s) unique."
            )
        else:
            rep.add("unique_barcodes", WARN, f"{n_dup} duplicate barcode(s).")
    else:
        rep.add("unique_barcodes", PASS, "barcodes are unique.")

    # unique gene names
    if not adata.var_names.is_unique:
        n_dup = int(pd.Index(adata.var_names).duplicated().sum())
        if fix:
            adata.var_names_make_unique()
            rep.add(
                "unique_genes", FIXED, f"made {n_dup} duplicate gene name(s) unique."
            )
        else:
            rep.add("unique_genes", WARN, f"{n_dup} duplicate gene name(s).")
    else:
        rep.add("unique_genes", PASS, "gene names are unique.")

    # Ensembl var_names -> symbols (only when a usable symbol column exists)
    if _looks_ensembl(adata.var_names):
        sym_col = next((c for c in _SYMBOL_COLS if c in adata.var.columns), None)
        syms = adata.var[sym_col].astype(str) if sym_col else None
        usable = sym_col is not None and float((syms.str.lower() != "nan").mean()) > 0.5
        if usable and fix:
            adata.var_names = syms.values
            adata.var_names_make_unique()
            rep.add(
                "gene_symbols",
                FIXED,
                f"var_names looked like Ensembl IDs; renamed to symbols from var['{sym_col}'].",
            )
        else:
            rep.add(
                "gene_symbols",
                WARN,
                "var_names look like Ensembl IDs and no usable symbol column is present; "
                "SingleR/CellTypist gene matching needs symbols (map Ensembl->symbol first).",
            )
    else:
        rep.add("gene_symbols", PASS, "var_names look like gene symbols.")

    # drop all-zero genes / cells (safe: they carry no information; QC would remove
    # them anyway). Only when the matrix passed integrity, so we never mask a
    # corrupt matrix by trimming it.
    if fix and not rep.fails:
        X = (
            adata.layers["counts"]
            if "counts" in getattr(adata, "layers", {})
            else adata.X
        )
        gene_tot = (
            np.asarray(X.sum(axis=0)).ravel()
            if sp.issparse(X)
            else np.asarray(X).sum(0)
        )
        cell_tot = (
            np.asarray(X.sum(axis=1)).ravel()
            if sp.issparse(X)
            else np.asarray(X).sum(1)
        )
        n_zg = int((gene_tot == 0).sum())
        n_zc = int((cell_tot == 0).sum())
        if n_zg and n_zg < adata.n_vars:
            adata = adata[:, gene_tot > 0].copy()
            rep.add("drop_zero_genes", FIXED, f"dropped {n_zg} all-zero gene(s).")
        else:
            rep.add(
                "drop_zero_genes",
                PASS,
                "no all-zero genes." if not n_zg else "all genes zero (left as-is).",
            )
        if n_zc and n_zc < adata.n_obs:
            keep = (
                np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
                if "counts" in getattr(adata, "layers", {})
                and sp.issparse(adata.layers["counts"])
                else (
                    np.asarray(adata.X).sum(1)
                    if not sp.issparse(adata.X)
                    else np.asarray(adata.X.sum(axis=1)).ravel()
                )
            ) > 0
            adata = adata[keep].copy()
            rep.add("drop_zero_cells", FIXED, f"dropped {n_zc} all-zero cell(s).")
        else:
            rep.add(
                "drop_zero_cells",
                PASS,
                "no all-zero cells." if not n_zc else "all cells zero (left as-is).",
            )

    # ---- biology / QC readiness (WARN — real but non-corrupting) ----
    mt = adata.var_names.str.upper().str.startswith("MT-")
    if int(mt.sum()) == 0:
        rep.add(
            "mito_genes",
            WARN,
            "no MT- genes found; pct_counts_mt will be 0 and the mito QC filter is inert.",
        )
    else:
        rep.add("mito_genes", PASS, f"{int(mt.sum())} mitochondrial gene(s) present.")

    # ---- one group label per sample (FATAL) ----
    # A donor that carries two group labels breaks the unit of replication: the
    # pseudobulk aggregator would assign the whole sample to one arm, so its cells
    # would silently count as evidence for a condition they are not from. Same class
    # of invariant as cells-in == cells-out — cheap to assert, corrupting if violated.
    if group_col in adata.obs.columns and sample_col in adata.obs.columns:
        _sg = pd.DataFrame(
            {
                "_g": adata.obs[group_col].astype(str).values,
                "_s": adata.obs[sample_col].astype(str).values,
            }
        )
        _per_sample = _sg.groupby("_s")["_g"].nunique()
        _multi = _per_sample[_per_sample > 1]
        if len(_multi):
            detail = "; ".join(
                f"{s}={sorted(_sg.loc[_sg['_s'] == s, '_g'].unique().tolist())}"
                for s in list(_multi.index)[:5]
            )
            more = f" (+{len(_multi) - 5} more)" if len(_multi) > 5 else ""
            rep.add(
                "sample_group_unique",
                FAIL,
                f"{len(_multi)} of {len(_per_sample)} sample(s) carry more than one "
                f"'{group_col}' label: {detail}{more}. Each sample must belong to "
                f"exactly one group — donor-level DE cannot split a sample across arms.",
            )
        else:
            rep.add(
                "sample_group_unique",
                PASS,
                f"all {len(_per_sample)} sample(s) carry exactly one '{group_col}' label.",
            )

    # ---- design / replication (WARN) + batch confound (FAIL) ----
    if group_col in adata.obs.columns:
        n_groups = int(adata.obs[group_col].astype(str).nunique())
        if n_groups >= 2:
            if sample_col in adata.obs.columns:
                tmp = pd.DataFrame(
                    {
                        "_g": adata.obs[group_col].astype(str).values,
                        "_s": adata.obs[sample_col].astype(str).values,
                    }
                )
                per = tmp.groupby("_g")["_s"].nunique()
                bad = per[per < min_samples_per_group]
                if len(bad):
                    rep.add(
                        "pseudobulk_replication",
                        WARN,
                        f"groups with <{min_samples_per_group} samples: {bad.to_dict()} — "
                        "pseudobulk (cohort) DE will be skipped for these; cell-level DE "
                        "would pseudoreplicate, so treat those results as exploratory only.",
                    )
                else:
                    rep.add(
                        "pseudobulk_replication",
                        PASS,
                        f"all {n_groups} groups have >= {min_samples_per_group} samples.",
                    )
            else:
                rep.add(
                    "pseudobulk_replication",
                    WARN,
                    f"'{sample_col}' column missing with {n_groups} groups; cannot form "
                    "donor pseudobulk — cohort DE will be unavailable.",
                )

    if batch_key and group_col and str(batch_key) == str(group_col):
        rep.add(
            "batch_not_confounded",
            FAIL,
            f"batch_key == group_col ('{batch_key}'): integrating on the biological "
            "condition erases the very signal you test. Use a technical batch or none.",
        )
    elif (
        batch_key and batch_key in adata.obs.columns and group_col in adata.obs.columns
    ):
        # Batch NESTED inside condition (every sample in exactly one arm) is the normal
        # cohort design, not a defect — so this WARNs and the run proceeds. It matters
        # because integration then cannot separate technical from biological structure:
        # clustering/UMAP/annotation may under-separate the arms. DE is unaffected (it
        # reads raw counts per donor, never the corrected graph) — see integration.py.
        _bd = pd.DataFrame(
            {
                "_b": adata.obs[batch_key].astype(str).values,
                "_g": adata.obs[group_col].astype(str).values,
            }
        )
        _span = _bd.groupby("_b")["_g"].nunique()
        _n_groups = int(_bd["_g"].nunique())
        _spanning = sorted(_span[_span > 1].index.tolist())
        if _n_groups >= 2 and not _spanning:
            rep.add(
                "batch_not_confounded",
                WARN,
                f"integration batch_key '{batch_key}' is NESTED inside '{group_col}' "
                f"({int(_span.size)} batches, each in exactly one of {_n_groups} groups) — "
                f"the standard cohort design, so the run proceeds. Batch and condition "
                f"are not separable: clustering/UMAP/annotation may under-separate the "
                f"conditions. Differential expression is unaffected (raw counts per "
                f"donor). Compare against integration_method=null if the UMAP looks "
                f"implausibly intermixed.",
            )
        else:
            rep.add(
                "batch_not_confounded",
                PASS,
                f"integration batch_key '{batch_key}' is distinct from group_col "
                f"'{group_col}'"
                + (
                    f" and {len(_spanning)} batch(es) span groups, so batch and "
                    f"condition are separable."
                    if _spanning
                    else "."
                ),
            )
    elif batch_key:
        rep.add(
            "batch_not_confounded",
            PASS,
            f"integration batch_key '{batch_key}' is distinct from group_col '{group_col}'.",
        )

    rep.n_obs_after, rep.n_vars_after = int(adata.n_obs), int(adata.n_vars)
    return adata, rep


def run_preflight_validation(
    adata: AnnData,
    *,
    out_dir: str | Path,
    analysis_name: str,
    group_col: str = "group",
    sample_col: str = "sample",
    batch_key: Optional[str] = None,
    block_on_fail: bool = True,
) -> AnnData:
    """Pipeline hook: validate + safe-fix, write the report, and (by default) raise
    on any FATAL issue so the run stops before hours of compute. Returns the
    (possibly repaired) adata. A bug INSIDE the validator never sinks a run — only a
    genuine data FAIL does."""
    try:
        adata, rep = validate_and_fix(
            adata,
            group_col=group_col,
            sample_col=sample_col,
            batch_key=batch_key,
            fix=True,
        )
        try:
            rep.write(Path(out_dir), analysis_name)
        except Exception as e:  # report I/O must not block a run
            logger.warning(
                "[VALIDATE] could not write validation report (%s).", e, exc_info=True
            )
        logger.info(
            "[VALIDATE] %s: %s auto-fix(es), %s warning(s), %s fatal issue(s).",
            analysis_name,
            len(rep.fixes),
            len(rep.warns),
            len(rep.fails),
        )
        if rep.fails and block_on_fail:
            msgs = "; ".join(f"{c.name} ({c.message})" for c in rep.fails)
            raise DataValidationError(
                f"Pre-flight data validation FAILED with {len(rep.fails)} fatal issue(s): "
                f"{msgs}. Fix the input and re-run "
                f"(details in {analysis_name}_data_validation.txt)."
            )
        return adata
    except DataValidationError:
        raise
    except Exception as e:  # never let a validator bug abort an otherwise-fine run
        logger.exception(
            "[VALIDATE] pre-flight validation skipped (internal error: %s).", e
        )
        return adata


def validate_path(path: str | Path, **kwargs) -> Tuple["object", ValidationReport]:
    """Standalone: load a 10x directory or an .h5ad and validate it. `kwargs` pass
    through to `validate_and_fix`."""
    import scanpy as sc

    from .loader_10x import load_10x_feature_barcode_matrix

    p = Path(path)
    adata = load_10x_feature_barcode_matrix(p) if p.is_dir() else sc.read_h5ad(p)
    return validate_and_fix(adata, **kwargs)
