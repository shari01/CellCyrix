"""
integration.py — what batch integration may influence, and what it must not.

The design fact
---------------
In a cohort study each donor belongs to exactly one arm, so ``sample`` is NESTED
inside ``group``. That is the standard case-control design, not a metadata error —
nothing here aborts because of it.

But it does mean batch correction cannot be selective: BBKNN forces every cell's
neighbourhood to be drawn from several samples, and because samples do not straddle
arms, the between-condition differences get mixed away along with the technical ones.
Over-integration is therefore a real risk in exactly the designs this pipeline is
built for.

The policy
----------
Integration is **annotation-only**. It rewrites the neighbour graph, and the graph
feeds UMAP / Leiden / DPT — i.e. how cells are grouped and named. It never touches
the numbers any statistical test reads:

    integration writes ->  obsp['connectivities'], obsp['distances'], uns['neighbors']
    and thence          ->  obsm['X_umap'], obs['leiden'], obsm['X_diffmap']

    DE reads            ->  layers['counts'] (raw), .raw (log-norm), obs['group'],
                            obs['sample'], obs[celltype_col]

Pseudobulk DE sums raw counts per donor and models them with DESeq2; the cell-level
exploratory DE reads log-normalized ``.raw``. Neither consults a graph, a corrected
embedding, or a corrected expression matrix — BBKNN does not produce one.
:func:`check_de_inputs_uncorrected` verifies that boundary at run time rather than
trusting the comment.

The one indirect path, stated plainly
-------------------------------------
Per-cell-type DE is stratified by ``celltype``, which comes from clustering, which
comes from the corrected graph. So integration cannot bias a *test statistic*, but it
can change *which cells are grouped together* before the test runs. That is a real
dependency and it is recorded verbatim in the manifest instead of being implied by
silence — a reviewer should be able to see it without reading this file.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

INTEGRATION_SCOPE = "annotation_only"

# Keys an integration step writes, directly or through the graph it replaces.
INTEGRATION_DERIVED_OBSP = ("connectivities", "distances")
INTEGRATION_DERIVED_UNS = ("neighbors",)
INTEGRATION_DERIVED_OBSM = ("X_umap", "X_diffmap")
INTEGRATION_DERIVED_OBS = ("leiden",)

# What differential expression is allowed to read.
DE_INPUT_LAYERS = ("counts",)
DE_INPUT_DESIGN_OBS = ("group", "sample")

# Nesting verdicts.
NESTED = "nested"  # every batch sits in exactly one group  -> confounded
CROSSED = "crossed"  # at least one batch spans groups        -> separable
IDENTICAL = "identical"  # batch_key IS the condition             -> fatal upstream
UNDEFINED = "undefined"  # not enough metadata to tell


def describe_batch_design(
    adata: AnnData,
    *,
    batch_key: Optional[str],
    group_col: str = "group",
) -> Dict[str, object]:
    """Is the integration batch nested inside the biological condition, or crossed?

    Returns a record with the verdict, the per-batch group span, and a plain-language
    ``interpretation``. Nesting is reported, never punished: it is what a cohort
    design looks like.
    """
    out: Dict[str, object] = {
        "batch_key": batch_key,
        "group_col": group_col,
        "verdict": UNDEFINED,
        "confounded_with_condition": None,
        "n_batches": None,
        "n_groups": None,
        "batches_spanning_groups": [],
        "interpretation": "",
    }
    if not batch_key:
        out["interpretation"] = "no integration batch configured."
        return out
    if batch_key not in adata.obs.columns or group_col not in adata.obs.columns:
        out["interpretation"] = (
            f"cannot assess: obs['{batch_key}'] and/or obs['{group_col}'] absent."
        )
        return out

    if str(batch_key) == str(group_col):
        out.update(
            verdict=IDENTICAL,
            confounded_with_condition=True,
            interpretation=(
                f"batch_key IS the condition column ('{batch_key}'): integrating on it "
                f"removes the effect under test."
            ),
        )
        return out

    df = pd.DataFrame(
        {
            "_b": adata.obs[batch_key].astype(str).to_numpy(),
            "_g": adata.obs[group_col].astype(str).to_numpy(),
        }
    )
    span = df.groupby("_b")["_g"].nunique()
    spanning = sorted(span[span > 1].index.tolist())
    n_groups = int(df["_g"].nunique())

    out.update(
        n_batches=int(span.size),
        n_groups=n_groups,
        batches_spanning_groups=spanning,
    )
    if n_groups < 2:
        out.update(
            verdict=CROSSED,
            confounded_with_condition=False,
            interpretation=(
                f"only {n_groups} group level present; no condition effect for "
                f"integration to remove."
            ),
        )
        return out

    if not spanning:
        out.update(
            verdict=NESTED,
            confounded_with_condition=True,
            interpretation=(
                f"every one of the {int(span.size)} '{batch_key}' level(s) belongs to "
                f"exactly one '{group_col}' — the standard cohort design. Batch and "
                f"condition are therefore not separable: correcting one also corrects "
                f"the other."
            ),
        )
    else:
        out.update(
            verdict=CROSSED,
            confounded_with_condition=False,
            interpretation=(
                f"{len(spanning)} '{batch_key}' level(s) span more than one "
                f"'{group_col}' ({spanning[:5]}), so the batch effect is estimable "
                f"separately from the condition effect."
            ),
        )
    return out


def resolve_integration_policy(
    adata: AnnData,
    *,
    integration_method: Optional[str],
    batch_key: Optional[str],
    group_col: str = "group",
    analysis_name: str = "analysis",
) -> Dict[str, object]:
    """Warn about a confounded integration and record the policy. Never aborts.

    Nesting is the design, so the correct response is a loud, specific warning plus a
    durable record of the scope — not refusing to run. The returned dict goes into the
    provenance manifest and into ``adata.uns`` (see :func:`stamp_integration_provenance`).
    """
    design = describe_batch_design(adata, batch_key=batch_key, group_col=group_col)
    requested = integration_method or "none"
    will_integrate = (
        bool(integration_method) and bool(batch_key) and batch_key in adata.obs.columns
    )

    record: Dict[str, object] = {
        "scope": INTEGRATION_SCOPE,
        "method_requested": requested,
        "method_used": None,  # filled in by the caller after it runs
        "batch_key": batch_key,
        "batch_design": design,
        "affects": [
            "obsp['connectivities'] / obsp['distances'] / uns['neighbors']",
            "obsm['X_umap']",
            "obs['leiden']",
            "obsm['X_diffmap'] (if DPT ran)",
            "cell-type annotation (voters read per-cluster markers)",
        ],
        "does_not_affect": [
            "layers['counts'] (raw counts — the pseudobulk DE input)",
            ".raw (log-normalized — the cell-level DE input)",
            "obs['group'] / obs['sample'] (the experimental design)",
            "pseudobulk DESeq2 statistics",
            "cell-level Wilcoxon statistics",
        ],
        "indirect_dependency": (
            "per-cell-type DE is stratified by obs['celltype'], which is derived from "
            "clustering on the integrated graph. Integration cannot bias a test "
            "statistic, but it does determine which cells are grouped together before "
            "the test runs."
        ),
        "warnings": [],
    }

    if not will_integrate:
        record["warnings"].append(
            f"no integration applied (method={requested!r}, batch_key={batch_key!r})."
        )
        logger.info("[%s] [INTEGRATION] %s", analysis_name, record["warnings"][-1])
        return record

    if design["verdict"] == NESTED:
        msg = (
            f"CONFOUNDED INTEGRATION: {design['interpretation']} Running "
            f"{requested!r} anyway, because this is the normal case-control design and "
            f"batch structure still has to be handled — but be aware that clustering, "
            f"UMAP and cell-type annotation may under-separate the conditions. "
            f"Differential expression is NOT affected: it reads raw counts per donor, "
            f"never the corrected graph (scope={INTEGRATION_SCOPE}). If the UMAP shows "
            f"conditions implausibly intermixed, re-run with integration_method=null "
            f"and compare."
        )
        record["warnings"].append(msg)
        logger.warning("[%s] [INTEGRATION] %s", analysis_name, msg)
    elif design["verdict"] == IDENTICAL:
        msg = f"batch_key == group_col: {design['interpretation']}"
        record["warnings"].append(msg)
        logger.error("[%s] [INTEGRATION] %s", analysis_name, msg)
    else:
        logger.info(
            "[%s] [INTEGRATION] batch design %s: %s",
            analysis_name,
            design["verdict"],
            design["interpretation"],
        )

    logger.info(
        "[%s] [INTEGRATION] policy scope=%s — the corrected graph drives clustering/annotation only; DE reads layers['counts'] and .raw.",
        analysis_name,
        INTEGRATION_SCOPE,
    )
    return record


# obs columns that are downstream of the corrected graph. Stratifying DE by one of
# these is the DECLARED indirect dependency, not a boundary violation — but it must be
# named in the record rather than passed over.
CLUSTERING_DERIVED_OBS = (
    "leiden",
    "celltype",
    "celltype_consensus",
    "celltype_subtype",
)


def check_de_inputs_uncorrected(
    adata: AnnData,
    *,
    celltype_col: Optional[str] = None,
    analysis_name: str = "analysis",
) -> Dict[str, object]:
    """Verify DE is about to read uncorrected data. Returns an auditable record.

    Enforces the boundary instead of asserting it in prose, and distinguishes two
    different things that are easy to conflate:

    * **Violation** — a *statistic* input (the count matrix, ``.raw``, or a design
      column) is something an integration step writes. That would corrupt the test
      itself, and sets ``ok=False``.
    * **Declared indirect dependency** — DE is *stratified* by a clustering-derived
      label. Legitimate and expected (it is how per-cell-type DE works), but reported
      explicitly in ``stratification`` so a reviewer sees the path from the corrected
      graph to the cell groupings.

    Reports rather than raises: a failed expectation means the policy needs revisiting,
    and that judgement is the analyst's.
    """
    integration_keys = (
        [
            f"obsp['{k}']"
            for k in INTEGRATION_DERIVED_OBSP
            if k in getattr(adata, "obsp", {})
        ]
        + [f"uns['{k}']" for k in INTEGRATION_DERIVED_UNS if k in adata.uns]
        + [f"obsm['{k}']" for k in INTEGRATION_DERIVED_OBSM if k in adata.obsm]
        + [f"obs['{k}']" for k in INTEGRATION_DERIVED_OBS if k in adata.obs.columns]
    )

    statistic_inputs: List[str] = []
    problems: List[str] = []

    for layer in DE_INPUT_LAYERS:
        if layer in adata.layers:
            statistic_inputs.append(f"layers['{layer}']")
        else:
            problems.append(
                f"layers['{layer}'] missing — pseudobulk DE cannot run on raw counts."
            )
    if getattr(adata, "raw", None) is not None:
        statistic_inputs.append(".raw")
    else:
        problems.append(".raw missing — cell-level DE has no log-normalized source.")

    design_inputs = [
        f"obs['{c}']" for c in DE_INPUT_DESIGN_OBS if c in adata.obs.columns
    ]

    # Compared on the SAME key spelling as integration_keys — an annotated label like
    # "obs['leiden'] (stratification only)" would never match, which would make this
    # check incapable of failing.
    violations = sorted(set(statistic_inputs) | set(design_inputs))
    violations = sorted(set(violations) & set(integration_keys))
    if violations:
        problems.append(
            f"DE statistic/design input(s) {violations} are integration-derived; the "
            f"annotation-only boundary is violated."
        )

    stratification = None
    if celltype_col and celltype_col in adata.obs.columns:
        derived = (
            celltype_col in CLUSTERING_DERIVED_OBS
            or f"obs['{celltype_col}']" in integration_keys
        )
        stratification = {
            "column": f"obs['{celltype_col}']",
            "clustering_derived": bool(derived),
            "note": (
                "DE is stratified by a label derived from clustering on the integrated "
                "graph. This does not enter any test statistic, but it determines which "
                "cells are grouped together before testing — the declared indirect "
                "dependency."
            )
            if derived
            else "stratification column is not derived from the integrated graph.",
        }

    ok = not violations and not any("missing" in p for p in problems)
    record = {
        "scope": INTEGRATION_SCOPE,
        "de_statistic_inputs": statistic_inputs,
        "de_design_inputs": design_inputs,
        "stratification": stratification,
        "integration_derived_keys_present": integration_keys,
        "boundary_violations": violations,
        "ok": ok,
        "problems": problems,
        "statement": (
            "differential expression read raw counts / log-normalized .raw and the "
            "design columns; no corrected graph, corrected embedding or corrected "
            "expression matrix entered a test statistic."
        )
        if ok
        else "BOUNDARY NOT VERIFIED — see problems.",
    }
    if violations:
        logger.error(
            "[%s] [INTEGRATION] %s %s", analysis_name, record["statement"], problems
        )
    elif not ok:
        logger.warning(
            "[%s] [INTEGRATION] %s %s", analysis_name, record["statement"], problems
        )
    else:
        logger.info(
            "[%s] [INTEGRATION] DE boundary verified: statistics read %s, design %s; none integration-derived.",
            analysis_name,
            statistic_inputs,
            design_inputs,
        )
    return record


def stamp_integration_provenance(adata: AnnData, record: Dict[str, object]) -> None:
    """Persist the integration policy into ``adata.uns`` so it travels with the h5ad."""
    try:
        adata.uns["integration_policy"] = {
            "scope": record.get("scope"),
            "method_requested": record.get("method_requested"),
            "method_used": record.get("method_used"),
            "batch_key": record.get("batch_key"),
            "batch_design_verdict": (record.get("batch_design") or {}).get("verdict"),
            "confounded_with_condition": (
                (record.get("batch_design") or {}).get("confounded_with_condition")
            ),
            "indirect_dependency": record.get("indirect_dependency"),
        }
    except Exception as e:  # noqa: BLE001 - provenance must never sink a run
        logger.warning(
            "[INTEGRATION] could not stamp policy into uns (%s).", e, exc_info=True
        )
