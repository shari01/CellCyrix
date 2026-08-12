"""
pseudobulk_de.py — donor/sample-level pseudobulk differential expression.

The statistically correct CASE-vs-CONTROL test for a cohort: aggregate raw
counts per SAMPLE (the unit of replication) — optionally within each cell type —
and run DESeq2 (via pyDESeq2). This avoids the pseudoreplication of cell-level
Wilcoxon, which treats thousands of cells from one donor as independent and
massively inflates significance.

Aggregation is a SUM of raw integer counts per (sample x cell type). Never a mean
of normalized values: DESeq2's negative-binomial model and its size-factor
normalization are defined on counts, and averaging CP10k values destroys both the
count scale and the mean-variance relationship the dispersion estimate depends on.

Three properties this module guarantees:

* **Direction is biological, not alphabetical.** The reference level comes from
  :mod:`contrasts`, and the literal contrast (``focus_vs_reference``) plus a
  plain-language direction note are stamped onto every output row.
* **Effect size is tested, not filtered.** ``lfc_threshold`` is passed to DESeq2 as
  a formal null (``H0: |LFC| <= threshold``, Wald test), instead of computing
  p-values against zero and then filtering ``|LFC| > 1`` post hoc — which inflates
  the false-positive rate because the filter is not part of the test.
* **Fold changes are shrunk.** apeGLM shrinkage (``lfc_shrink``) pulls the estimate
  toward zero where the evidence is thin, so low-count genes stop producing
  enormous fold changes at n=2-3 donors per arm. p-values are unaffected by
  shrinkage; both the shrunken and the raw MLE estimate are reported.

Requires >=2 samples per group for a contrast (real replication). Contrasts with
fewer are skipped and reported, never silently faked. If pyDESeq2 is not
installed the whole step is skipped with a clear log line (non-fatal).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse as sp

from .atomic_io import atomic_to_csv, write_table
from .config_cli import logger
from .contrasts import ordered_contrasts, stamp_contrast
from .exceptions import PipelineInputError
from .safe_names import safe_filename

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

MIN_SAMPLES_PER_GROUP = 2  # real biological replication
MIN_CELLS_PER_PSEUDOBULK = 10  # a pseudobulk sample needs enough cells to be meaningful

# Formal effect-size null: H0 is |log2FC| <= LFC_THRESHOLD, tested by DESeq2 itself.
DEFAULT_LFC_THRESHOLD = 1.0
DEFAULT_ALPHA = 0.05


class PseudobulkInputError(PipelineInputError):
    """Raised when the matrix handed to pseudobulk aggregation is not raw counts."""


def _counts_matrix(adata):
    """Raw counts for aggregation.

    FATAL when ``layers['counts']`` is absent. By the time this runs in the pipeline
    ``adata.X`` has been normalized, log1p'd and z-scaled — summing that and casting
    to int produces negative "counts" and silently meaningless DESeq2 results. The
    previous fallback warned and continued; this refuses.
    """
    if "counts" in adata.layers:
        return adata.layers["counts"]
    raise PseudobulkInputError(
        "pseudobulk DE requires raw counts in adata.layers['counts']; the layer is "
        "absent and adata.X is normalized/scaled at this point in the pipeline. "
        "Refusing to aggregate non-count data."
    )


def _build_pseudobulk(adata, sample_col, group_col, cell_mask=None):
    """Sum raw counts per sample -> (counts_df [samples x genes, int], meta_df).

    Each sample must carry exactly ONE group label; a sample spanning two arms is a
    metadata error that would silently assign the whole donor to one side of the
    contrast, so it raises rather than taking the first cell's label.
    """
    X = _counts_matrix(adata)
    obs = adata.obs
    if cell_mask is not None:
        X = X[cell_mask]
        obs = obs[cell_mask]

    samples = obs[sample_col].astype(str).values
    genes = adata.var_names.astype(str)

    rows, groups, ncells = {}, {}, {}
    for s in pd.unique(samples):
        m = samples == s
        if m.sum() < MIN_CELLS_PER_PSEUDOBULK:
            continue
        sub = X[m]
        # SUM of raw counts — not a mean, and not of normalized values.
        vec = (
            np.asarray(sub.sum(axis=0)).ravel()
            if sp.issparse(sub)
            else np.asarray(sub).sum(axis=0).ravel()
        )
        labels = pd.unique(obs.loc[m, group_col].astype(str))
        if len(labels) != 1:
            raise PseudobulkInputError(
                f"sample {s!r} carries {len(labels)} group labels {sorted(labels)}; "
                f"each sample must belong to exactly one group for a donor-level "
                f"contrast. Fix obs['{group_col}'] / the sample->group map."
            )
        rows[s] = vec
        groups[s] = str(labels[0])
        ncells[s] = int(m.sum())

    if len(rows) < 2:
        return None, None
    counts_df = pd.DataFrame(rows, index=genes).T

    # Guard the count invariant explicitly: a non-integer or negative aggregate means
    # the wrong matrix was summed, and DESeq2 would model it without complaint.
    vals = counts_df.to_numpy()
    if not np.all(np.isfinite(vals)):
        raise PseudobulkInputError(
            "pseudobulk aggregate contains NaN/Inf; source is not raw counts."
        )
    if float(vals.min()) < 0:
        raise PseudobulkInputError(
            f"pseudobulk aggregate has negative values (min={float(vals.min()):.3g}); "
            "the summed matrix is normalized/scaled data, not raw counts."
        )
    if not np.allclose(vals, np.rint(vals), rtol=0, atol=1e-6):
        raise PseudobulkInputError(
            "pseudobulk aggregate is not integer-valued; the summed matrix is "
            "normalized data, not raw counts."
        )

    counts_df = counts_df.round().astype(int)
    meta_df = pd.DataFrame({"group": pd.Series(groups), "n_cells": pd.Series(ncells)})
    meta_df = meta_df.loc[counts_df.index]
    return counts_df, meta_df


def _shrinkage_coeff(stat, focus: str) -> Optional[str]:
    """The LFC coefficient name to shrink, resolved from what pyDESeq2 actually built.

    The formula backend names it ``group[T.<focus>]``, but that spelling is a
    pyDESeq2/formulaic implementation detail — resolve it from the fitted object
    instead of hard-coding, so a version bump degrades to "no shrinkage" rather
    than raising.
    """
    lfc = getattr(stat, "LFC", None)
    if lfc is None:
        return None
    cols = [str(c) for c in lfc.columns]
    cands = [c for c in cols if c.lower() not in {"intercept"}]
    if not cands:
        return None
    exact = [c for c in cands if f"[T.{focus}]" in c]
    if exact:
        return exact[0]
    hit = [c for c in cands if focus in c]
    if hit:
        return hit[0]
    return cands[0] if len(cands) == 1 else None


def _deseq_contrast(
    counts_df,
    meta_df,
    focus,
    ref,
    *,
    reference_selection: str,
    lfc_threshold: float = DEFAULT_LFC_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
):
    """Run one DESeq2 contrast (focus vs ref). Returns ``(results_df or None, status)``."""
    keep = meta_df["group"].isin([focus, ref])
    c = counts_df.loc[keep.values]
    m = meta_df.loc[keep.values, ["group"]].copy()

    vc = m["group"].value_counts()
    if (
        vc.get(focus, 0) < MIN_SAMPLES_PER_GROUP
        or vc.get(ref, 0) < MIN_SAMPLES_PER_GROUP
    ):
        return (
            None,
            f"insufficient replicates (need >={MIN_SAMPLES_PER_GROUP}/group): {vc.to_dict()}",
        )

    # drop genes with zero total counts across the retained pseudobulk samples
    c = c.loc[:, c.sum(axis=0) > 0]
    if c.shape[1] == 0:
        return None, "no expressed genes"

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except Exception as e:  # pragma: no cover
        logger.debug("%s: falling back after %r", __name__, e)
        return None, f"pydeseq2 unavailable ({e})"

    # The reference level is the CONTROL arm (see contrasts.resolve_reference_group),
    # not the alphabetically-first one, so a positive log2FoldChange always means
    # "higher in the focus/case arm".
    m["group"] = pd.Categorical(m["group"], categories=[ref, focus])
    dds = DeseqDataSet(counts=c, metadata=m, design="~group", quiet=True)
    dds.deseq2()

    # Formal effect-size test: H0 is |LFC| <= lfc_threshold, so surviving genes are
    # significantly LARGER than the threshold rather than merely non-zero and then
    # filtered. lfc_threshold=0 falls back to the classic against-zero Wald test.
    stat_kwargs = dict(contrast=["group", focus, ref], alpha=alpha, quiet=True)
    tested_threshold = float(lfc_threshold or 0.0)
    if tested_threshold > 0:
        stat_kwargs.update(lfc_null=tested_threshold, alt_hypothesis="greaterAbs")
    stat = DeseqStats(dds, **stat_kwargs)
    stat.summary()

    res = stat.results_df.reset_index().rename(columns={"index": "gene"})
    # Keep the unshrunken estimate for anyone who needs the raw MLE.
    if "log2FoldChange" in res.columns:
        res["log2FoldChange_MLE"] = res["log2FoldChange"].to_numpy()

    # apeGLM shrinkage. Leaves p-values untouched (pyDESeq2 documents this); it only
    # makes the reported effect size honest at low counts / few donors.
    shrunk = False
    shrink_note = ""
    coeff = _shrinkage_coeff(stat, focus)
    if coeff is None:
        shrink_note = "no shrinkable coefficient resolved; reporting MLE log2FoldChange"
    else:
        try:
            stat.lfc_shrink(coeff=coeff)
            res_s = stat.results_df.reset_index().rename(columns={"index": "gene"})
            res["log2FoldChange"] = res_s["log2FoldChange"].to_numpy()
            if "lfcSE" in res_s.columns:
                res["lfcSE"] = res_s["lfcSE"].to_numpy()
            shrunk = True
            shrink_note = f"apeGLM shrinkage on coefficient {coeff!r}"
        except Exception as e:  # noqa: BLE001 - shrinkage is an improvement, not a gate
            shrink_note = f"shrinkage failed ({type(e).__name__}: {e}); reporting MLE log2FoldChange"
            logger.warning("[PSEUDOBULK] %s vs %s: %s", focus, ref, shrink_note)

    # Significance/direction. With the threshold test, padj ALREADY encodes the
    # effect-size requirement — no post-hoc |LFC| filter is applied on top of it.
    res["regulation"] = "no_change"
    if {"log2FoldChange", "padj"}.issubset(res.columns):
        sig = res["padj"] < alpha
        res.loc[sig & (res["log2FoldChange"] > 0), "regulation"] = "up"
        res.loc[sig & (res["log2FoldChange"] < 0), "regulation"] = "down"

    # --- audit trail: direction + exactly which test produced these numbers ---
    stamp_contrast(res, focus=focus, ref=ref, reference_selection=reference_selection)
    res["test"] = (
        f"DESeq2 Wald, H0:|log2FC|<={tested_threshold:g} (alt=greaterAbs)"
        if tested_threshold > 0
        else "DESeq2 Wald, H0:log2FC=0"
    )
    res["lfc_threshold"] = tested_threshold
    res["alpha"] = float(alpha)
    res["lfc_shrinkage"] = shrink_note
    res["significance_rule"] = (
        f"padj < {alpha} (effect size tested by H0:|log2FC|<={tested_threshold:g}; "
        f"no post-hoc fold-change filter)"
        if tested_threshold > 0
        else f"padj < {alpha}"
    )
    res["unit_of_replication"] = "sample (donor-level pseudobulk, summed raw counts)"
    return res, ("ok" if shrunk else "ok (unshrunken LFC)")


def _all_pairs_deseq(
    counts_df,
    meta_df,
    label,
    out_file,
    *,
    reference_group=None,
    lfc_threshold: float = DEFAULT_LFC_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
):
    """Run DESeq2 for every group pair; write a combined CSV. Returns ``(n_written, rows)``.

    ``rows`` is the per-contrast audit record (which arm was the reference, why, and
    what happened) that :func:`compute_pseudobulk_de` writes to the design table.
    """
    groups = sorted(meta_df["group"].astype(str).unique())
    if len(groups) < 2:
        logger.info("[PSEUDOBULK] %s: <2 groups present; skip.", label)
        return 0, []

    pairs, ref_group, reason = ordered_contrasts(groups, reference=reference_group)
    if ref_group is None:
        logger.warning("[PSEUDOBULK] %s: %s", label, reason)
    else:
        logger.info(
            "[PSEUDOBULK] %s: reference group = %r — %s", label, ref_group, reason
        )
    frames, audit = [], []
    for focus, ref in pairs:
        res, status = _deseq_contrast(
            counts_df,
            meta_df,
            focus,
            ref,
            reference_selection=reason,
            lfc_threshold=lfc_threshold,
            alpha=alpha,
        )
        record = {
            "scope": label,
            "comparison": f"{focus}_vs_{ref}",
            "focus_group": focus,
            "reference_group": ref,
            "contrast_direction": f"positive log2FoldChange = higher in {focus} than in {ref}",
            "reference_selection": reason,
            "lfc_threshold": float(lfc_threshold or 0.0),
            "alpha": float(alpha),
            "status": status,
            "n_up": 0,
            "n_down": 0,
        }
        if res is None:
            logger.info(
                "[PSEUDOBULK] %s: %s vs %s skipped (%s).", label, focus, ref, status
            )
            audit.append(record)
            continue
        n_up = int((res["regulation"] == "up").sum())
        n_down = int((res["regulation"] == "down").sum())
        record.update(n_up=n_up, n_down=n_down)
        audit.append(record)
        logger.info(
            "[PSEUDOBULK] %s: %s vs %s OK — %s up / %s down in %s (padj<%s, H0:|log2FC|<=%s).",
            label,
            focus,
            ref,
            n_up,
            n_down,
            focus,
            alpha,
            format(float(lfc_threshold or 0.0), "g"),
        )
        frames.append(res)

    if not frames:
        return 0, audit
    out = pd.concat(frames, ignore_index=True)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    write_table(out, out_file, index=False)
    logger.info("[PSEUDOBULK] wrote %s DE -> %s", label, out_file)
    return len(frames), audit


def compute_pseudobulk_de(
    adata: AnnData,
    *,
    group_col: str = "group",
    sample_col: str = "sample",
    celltype_col: Optional[str] = None,
    out_dir: Path,
    reference_group: Optional[str] = None,
    lfc_threshold: float = DEFAULT_LFC_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """Pseudobulk (sample-level) DESeq2 DE across groups.

    Writes, under ``out_dir``:
      * ``pseudobulk_overall_de.csv``   — all cells aggregated per sample
      * ``per_celltype/<ct>_pseudobulk_de.csv`` — cells of one type per sample
      * ``pseudobulk_contrast_design.csv`` — one row per contrast: which arm was the
        reference, why it was chosen, the test used, and how many genes moved.

    ``reference_group`` pins the control arm explicitly; when omitted the baseline is
    detected from the group names and, failing that, the direction falls back to
    alphabetical with that fact recorded in every row.
    """
    out_dir = Path(out_dir)
    if group_col not in adata.obs.columns or sample_col not in adata.obs.columns:
        logger.info(
            "[PSEUDOBULK] need obs['%s'] and obs['%s']; skip.", group_col, sample_col
        )
        return

    n_groups = adata.obs[group_col].astype(str).nunique()
    n_samples = adata.obs[sample_col].astype(str).nunique()
    if n_groups < 2 or n_samples < 2 * MIN_SAMPLES_PER_GROUP:
        logger.info(
            "[PSEUDOBULK] not enough replication (groups=%s, samples=%s); pseudobulk DE needs >=2 groups and >=%s samples/group.",
            n_groups,
            n_samples,
            MIN_SAMPLES_PER_GROUP,
        )
        return

    try:
        import pydeseq2  # noqa: F401
    except Exception as e:
        logger.warning(
            "[PSEUDOBULK] pyDESeq2 not installed (%s); skipping pseudobulk DE (install with: pip install pydeseq2).",
            e,
            exc_info=True,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []

    # --- overall (all cells per sample) ---
    counts_df, meta_df = _build_pseudobulk(adata, sample_col, group_col)
    if counts_df is None:
        logger.info("[PSEUDOBULK] overall: too few valid pseudobulk samples; skip.")
    else:
        logger.info(
            "[PSEUDOBULK] overall: %s pseudobulk samples x %s genes.",
            counts_df.shape[0],
            counts_df.shape[1],
        )
        _, rows = _all_pairs_deseq(
            counts_df,
            meta_df,
            "overall",
            out_dir / "pseudobulk_overall_de.csv",
            reference_group=reference_group,
            lfc_threshold=lfc_threshold,
            alpha=alpha,
        )
        audit_rows.extend(rows)

    # --- per cell type ---
    if celltype_col and celltype_col in adata.obs.columns:
        per_ct_dir = out_dir / "per_celltype"
        cts = sorted(adata.obs[celltype_col].astype(str).unique())
        for ct in cts:
            mask = (adata.obs[celltype_col].astype(str) == ct).values
            c_df, m_df = _build_pseudobulk(adata, sample_col, group_col, cell_mask=mask)
            if c_df is None:
                logger.info(
                    "[PSEUDOBULK] celltype='%s': too few pseudobulk samples; skip.", ct
                )
                continue
            ct_safe = safe_filename(ct)
            _, rows = _all_pairs_deseq(
                c_df,
                m_df,
                f"celltype={ct}",
                per_ct_dir / f"{ct_safe}_pseudobulk_de.csv",
                reference_group=reference_group,
                lfc_threshold=lfc_threshold,
                alpha=alpha,
            )
            for r in rows:
                r["celltype"] = ct
            audit_rows.extend(rows)

    if audit_rows:
        design_path = out_dir / "pseudobulk_contrast_design.csv"
        atomic_to_csv(pd.DataFrame(audit_rows), design_path, index=False)
        logger.info("[PSEUDOBULK] wrote contrast design/audit table -> %s", design_path)
