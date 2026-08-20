"""
agent.py — the ONLY module that talks to the LLM (OpenRouter).

Two agents:
  * llm_annotate_cluster  — Annotator C (Stage 5): names a cluster from markers.
  * llm_adjudicate        — the Stage 8 tie-breaker when methods disagree or the
                            majority is contradicted by the lineage gate.

Disease-agnostic: prompts receive ONLY marker genes + tissue/organ context.
The word "disease" never enters a prompt. No silent failures: transport errors
retry up to cfg.llm_max_retries then raise; JSON parse failure gets one repair
retry, then falls back to a documented "Unassigned (LLM parse fail)" label.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..exceptions import PipelineComputationError
from . import llm_cache
from .config import ConsensusConfig

logger = logging.getLogger(__name__)

PARSE_FAIL_LABEL = "Unassigned (LLM parse fail)"

# Few-shot, disease-agnostic annotation prompt. Rules + examples live in the
# system message (stable/cacheable); the per-cluster query stays short. Examples
# span every major lineage (immune, epithelial, stromal, endothelial) so the
# model never defaults to immune labels the way an immune-only classifier does.
_ANNOTATE_SYSTEM = (
    "You are a single-cell RNA-seq cell-type annotation expert. Name the single "
    "most likely cell type for a cluster from its ranked marker genes and the "
    "tissue/organ context.\n"
    "Rules:\n"
    "- Decide ONLY from the marker genes and tissue. Never infer, use, or mention "
    "any disease, condition, tumor status, or treatment — they do not change cell "
    "identity.\n"
    '- Return a canonical cell-type name (e.g. "T cell", "Epithelial cell", '
    '"Fibroblast"), never a disease, activation state, cluster number, or tissue '
    "name.\n"
    "- Weigh all lineages — immune, epithelial, stromal, endothelial — and match "
    "the markers to the lineage they actually define.\n"
    "- If markers are mixed or non-specific, lower the confidence and give a real "
    "alternative. Do not assert a type you cannot support.\n"
    "- Output ONLY one JSON object with keys cell_type, confidence, reasoning, "
    "alternative_cell_type. confidence is 0.0-1.0; keep reasoning to one short "
    "sentence naming the decisive markers.\n"
    "Examples:\n"
    'markers: EPCAM, KRT8, KRT18, KRT19, CDH1 -> {"cell_type":"Epithelial cell",'
    '"confidence":0.95,"reasoning":"Pan-epithelial keratins with EPCAM/CDH1.",'
    '"alternative_cell_type":"Luminal epithelial cell"}\n'
    'markers: CD3D, CD3E, TRAC, CD8A, IL7R -> {"cell_type":"T cell",'
    '"confidence":0.96,"reasoning":"CD3 complex, TCR chain, CD8/IL7R.",'
    '"alternative_cell_type":"NK cell"}\n'
    'markers: MS4A1, CD79A, CD79B, CD19 -> {"cell_type":"B cell",'
    '"confidence":0.95,"reasoning":"CD20/CD79/CD19 B-lineage markers.",'
    '"alternative_cell_type":"Plasma cell"}\n'
    'markers: LYZ, CD68, CD14, C1QA, AIF1 -> {"cell_type":"Macrophage",'
    '"confidence":0.9,"reasoning":"Myeloid LYZ/CD68/C1QA program.",'
    '"alternative_cell_type":"Monocyte"}\n'
    'markers: COL1A1, COL1A2, DCN, LUM, PDGFRB -> {"cell_type":"Fibroblast",'
    '"confidence":0.94,"reasoning":"Fibrillar collagens with decorin/lumican.",'
    '"alternative_cell_type":"Myofibroblast"}\n'
    'markers: PECAM1, VWF, CLDN5, CDH5 -> {"cell_type":"Endothelial cell",'
    '"confidence":0.95,"reasoning":"PECAM1/VWF/CDH5 endothelial markers.",'
    '"alternative_cell_type":"Lymphatic endothelial cell"}'
)
_ANNOTATE_TEMPLATE = (
    "Tissue/organ context: {tissue}\n"
    "Ranked top marker genes for this cluster: {markers}\n"
    "Return ONLY the JSON object."
)

_ADJUDICATE_SYSTEM = (
    "You are a single-cell RNA-seq annotation adjudicator. Independent methods "
    "disagreed on one cluster. Choose the best-supported cell type using ONLY the "
    "candidate labels, their confidences, the marker genes, and the coarse lineage "
    "flag. Do not invent types outside the candidates unless all are clearly wrong; "
    "in that case return the lineage flag as the label. Never introduce a disease, "
    "condition, or activation state — judge cell identity only. Pick the candidate "
    "whose markers best fit the lineage flag."
)


_ROOT_GROUP_SYSTEM = (
    "You pick the BASELINE / earliest group in a cohort, to use as the origin "
    "(root) for pseudotime ordering of single cells.\n"
    "Rules:\n"
    "- Choose the group that represents the NORMAL / healthy / untreated / "
    "earliest / least-advanced state — the biological starting point of the "
    "progression.\n"
    "- Judge ONLY from the group names. If the groups are NOT ordered stages of "
    "one progression (e.g. unrelated conditions, or a single state), return "
    '"NONE" — pseudotime is not meaningful then.\n'
    '- Return a group name copied EXACTLY from the list, or the literal "NONE".\n'
    '- Output ONLY a JSON object: {"root_group": <exact-name-or-NONE>, '
    '"reasoning": <one short sentence>}.'
)

_TISSUE_INFER_SYSTEM = (
    "You extract the TISSUE / ORGAN of a single-cell RNA-seq study from its "
    "metadata text.\n"
    "Rules:\n"
    '- Return the organ/tissue only (e.g. "cervix", "blood", "lung", '
    '"breast", "colon"). Lowercase, 1-3 words.\n'
    "- Report the tissue/organ ONLY. Never return a disease, cancer type, "
    "condition, or cell-line name — those are not tissues.\n"
    '- If the text does not clearly indicate a tissue/organ, return "unknown".\n'
    '- Output ONLY a JSON object: {"tissue": <organ-or-unknown>, "reasoning": '
    "<one short sentence>}."
)

_MODEL_SELECT_SYSTEM = (
    "You select the single best CellTypist model for annotating a scRNA-seq "
    "dataset, based ONLY on its tissue/organ.\n"
    "Rules:\n"
    "- Match on tissue/organ alone. NEVER use any disease, condition, or "
    "treatment to decide — a model's disease context is irrelevant; only the "
    "organ it profiles matters.\n"
    "- Prefer a model whose tissue is the same organ (or the same organ system) "
    "as the target tissue and that covers epithelial/stromal cells, not just "
    "immune, when the target is a solid tissue.\n"
    "- If NO model's organ is a reasonable match for the target tissue, return "
    '"NONE" so the caller falls back to the general immune model. Do not force '
    "a distant match.\n"
    "- You MUST return a model name copied EXACTLY from the provided list, or the "
    'literal string "NONE".\n'
    '- Output ONLY a JSON object: {"model": <exact-name-or-NONE>, "reasoning": '
    "<one short sentence>}."
)


_REF_SELECT_SYSTEM = (
    "You select the single best SingleR (celldex) reference for annotating a "
    "scRNA-seq dataset, based ONLY on its species and tissue/organ.\n"
    "Rules:\n"
    "- Match on species first (never pick a mouse reference for human data or "
    "vice versa), then on tissue/organ.\n"
    "- NEVER use any disease, condition, or treatment to decide — only species "
    "and organ matter.\n"
    "- For solid tissues with epithelial/stromal cells, prefer a BROAD reference "
    "(covers non-immune lineages), not an immune-only one. Use immune-only "
    "references only for blood/immune datasets.\n"
    '- If no reference is a reasonable match, return "NONE" so the caller uses '
    "its broad general fallback. Do not force a distant match.\n"
    "- You MUST return a reference name copied EXACTLY from the provided list, or "
    'the literal string "NONE".\n'
    '- Output ONLY a JSON object: {"reference": <exact-name-or-NONE>, '
    '"reasoning": <one short sentence>}.'
)


class OpenRouterError(PipelineComputationError):
    """Raised when the LLM transport fails after all retries."""


def _chat(cfg: ConsensusConfig, system: str, user: str) -> str:
    """One chat completion via OpenRouter, served from cache when possible.

    Reproducibility, in order of strength: the response cache (exact replay of a
    previous identical request), then greedy decoding (``temperature`` 0.0, ``top_p``
    1.0) plus a fixed ``seed``. The cache is what actually makes a published run
    regenerable, because sampling parameters cannot stop a hosted endpoint from
    changing the model behind a floating alias — see :mod:`llm_cache`.

    Retries transport errors; raises OpenRouterError when retries are exhausted.
    Returns the message content.
    """
    try:
        import requests
    except ImportError as e:
        raise OpenRouterError("`requests` is required for the LLM layer.") from e

    max_tokens = getattr(cfg, "llm_max_tokens", 800)
    top_p = getattr(cfg, "llm_top_p", 1.0)
    seed = getattr(cfg, "llm_seed", 0)

    # Cache lookup happens BEFORE the credential check: a fully-cached run must work
    # with no API key at all, which is what lets a reviewer reproduce the published
    # numbers offline from the shipped cache directory.
    key = llm_cache.cache_key(
        model=cfg.openrouter_model or "",
        system=system,
        user=user,
        temperature=cfg.llm_temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
    )
    cached = llm_cache.get(key)
    if cached is not None:
        return cached

    if not cfg.openrouter_api_key or not cfg.openrouter_model:
        raise OpenRouterError("OPENROUTER_API_KEY / OPENROUTER_MODEL not configured.")

    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.openrouter_model,
        "temperature": cfg.llm_temperature,
        # Greedy decoding needs top_p pinned too: temperature 0 with a provider-default
        # top_p below 1.0 still truncates the distribution differently across providers.
        "top_p": top_p,
        # Best-effort determinism where the provider honours it. Not sufficient alone,
        # which is why the cache above is the real control.
        "seed": seed,
        "max_tokens": max_tokens,  # bound reply so JSON can't truncate
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, cfg.llm_max_retries + 1):
        try:
            resp = requests.post(
                cfg.endpoint, headers=headers, json=body, timeout=cfg.llm_timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
            content = (data["choices"][0]["message"]["content"] or "").strip()
            # Store the model the endpoint actually served, not just the one requested:
            # with a floating alias those differ, and the served id is what a reader
            # needs to know which weights produced the label.
            llm_cache.put(
                key,
                content,
                meta={
                    "model_requested": cfg.openrouter_model,
                    "model_served": data.get("model"),
                    "temperature": cfg.llm_temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "max_tokens": max_tokens,
                },
            )
            return content
        except Exception as e:  # transport/HTTP/shape error — retry with backoff
            last_err = e
            wait = min(2 ** (attempt - 1), 8)
            logger.warning(
                f"[LLM] attempt {attempt}/{cfg.llm_max_retries} failed ({e}); "
                f"retrying in {wait}s."
                if attempt < cfg.llm_max_retries
                else f"[LLM] attempt {attempt}/{cfg.llm_max_retries} failed ({e}); no retries left.",
                exc_info=True,
            )
            if attempt < cfg.llm_max_retries:
                time.sleep(wait)
    raise OpenRouterError(
        f"OpenRouter call failed after {cfg.llm_max_retries} attempts: {last_err}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating code fences."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*", "", s).strip().rstrip("`").strip()
    # grab the outermost object if there is surrounding prose
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


# ===========================================================================
# Stage 5 — Annotator C
# ===========================================================================
def llm_annotate_cluster(
    cfg: ConsensusConfig,
    cluster_id: str,
    markers: List[str],
    tissue: Optional[str],
) -> Dict[str, object]:
    """Name a cluster from its markers. Returns
    {cell_type, confidence, reasoning, alternative_cell_type}.

    Empty marker set -> Unassigned (no guess). Transport failure -> raises
    (surfaced by the orchestrator). JSON failure -> one repair retry, then the
    documented parse-fail label."""
    if not markers:
        logger.warning(
            "[KB-ANNOTATE] cluster %s: empty marker set; Unassigned.", cluster_id
        )
        return {
            "cell_type": "Unassigned",
            "confidence": 0.0,
            "reasoning": "No marker genes available.",
            "alternative_cell_type": "",
        }

    user = _ANNOTATE_TEMPLATE.format(
        tissue=tissue or "unspecified", markers=", ".join(markers)
    )
    raw = _chat(cfg, _ANNOTATE_SYSTEM, user)
    try:
        obj = _extract_json(raw)
    except Exception as e:
        logger.warning(
            "[KB-ANNOTATE] cluster %s: JSON parse failed (%s); repair retry.",
            cluster_id,
            e,
            exc_info=True,
        )
        repair = (
            user
            + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
        )
        try:
            obj = _extract_json(_chat(cfg, _ANNOTATE_SYSTEM, repair))
        except Exception as e2:
            logger.exception(
                "[KB-ANNOTATE] cluster %s: repair failed (%s); parse-fail label.",
                cluster_id,
                e2,
            )
            return {
                "cell_type": PARSE_FAIL_LABEL,
                "confidence": 0.0,
                "reasoning": f"parse failure: {e2}",
                "alternative_cell_type": "",
            }

    return {
        "cell_type": str(obj.get("cell_type", "Unassigned")).strip() or "Unassigned",
        "confidence": _coerce_conf(obj.get("confidence", 0.0)),
        "reasoning": str(obj.get("reasoning", "")).strip(),
        "alternative_cell_type": str(obj.get("alternative_cell_type", "")).strip(),
    }


# ===========================================================================
# Stage 8 — adjudicator
# ===========================================================================
def llm_adjudicate(
    cfg: ConsensusConfig,
    cluster_id: str,
    candidates: List[Dict[str, object]],
    markers: List[str],
    lineage_flag: str,
    tissue: Optional[str],
    fallback_label: Optional[str] = None,
) -> Dict[str, str]:
    """Break a tie. `candidates` = [{method, label, confidence}]. Returns
    {label, reasoning}. Transport failure raises.

    When the model returns no usable label, fall back to ``fallback_label`` (the
    caller's majority vote) rather than to ``lineage_flag``. The lineage flag is a
    coarse marker-score heuristic with no panel for several real cell types; making
    it the silent default meant a parse failure could overwrite unanimous voters.
    ``lineage_flag`` remains the last resort when no majority exists.
    """
    _fallback = (fallback_label or "").strip() or lineage_flag
    payload = {
        "cluster": cluster_id,
        "tissue": tissue or "unspecified",
        "coarse_lineage_flag": lineage_flag,
        "candidates": candidates,
        "top_markers": markers[:30],
    }
    user = (
        "Adjudicate this cluster. Return ONLY a JSON object with keys: label, reasoning.\n"
        + json.dumps(payload)
    )
    raw = _chat(cfg, _ADJUDICATE_SYSTEM, user)
    try:
        obj = _extract_json(raw)
        label = str(obj.get("label", "")).strip()
        if not label:
            logger.warning(
                "[KB-ADJUDICATE] cluster %s: no label in reply; falling back to '%s'.",
                cluster_id,
                _fallback,
            )
        return {
            "label": label or _fallback,
            "reasoning": str(obj.get("reasoning", "")).strip(),
        }
    except Exception as e:
        logger.warning(
            "[KB-ADJUDICATE] cluster %s: JSON parse failed (%s); falling back to '%s'.",
            cluster_id,
            e,
            _fallback,
            exc_info=True,
        )
        return {"label": _fallback, "reasoning": f"adjudicator parse failure: {e}"}


def _coerce_conf(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


# ===========================================================================
# Biology-context inference (tissue from study metadata; disease-agnostic)
# ===========================================================================
def llm_select_root_group(cfg: ConsensusConfig, groups: List[str]) -> Optional[str]:
    """Pick the baseline/earliest group to root pseudotime at.

    Returns an exact group name, or None when the groups are not a progression /
    the LLM is unavailable. Uses only the group names (no expression, no cells).
    """
    groups = [str(g) for g in (groups or [])]
    if len(groups) < 2:
        return None
    valid = set(groups)
    user = "Groups: " + ", ".join(groups) + "\nReturn ONLY the JSON object."
    try:
        obj = _extract_json(_chat(cfg, _ROOT_GROUP_SYSTEM, user))
    except Exception as e:
        logger.warning(
            "[LLM-ROOT] selection failed (%s); caller will try a heuristic.",
            e,
            exc_info=True,
        )
        return None
    choice = str(obj.get("root_group", "")).strip()
    reasoning = str(obj.get("reasoning", "")).strip()
    if choice in valid:
        logger.info("[LLM-ROOT] baseline group -> %s (%s)", choice, reasoning)
        return choice
    logger.info("[LLM-ROOT] no baseline group (%r); %s", choice, reasoning)
    return None


def llm_infer_tissue(cfg: ConsensusConfig, text: Optional[str]) -> Optional[str]:
    """Infer the tissue/organ of a study from its metadata text.

    Returns a lowercase organ string, or None when it cannot be determined /
    the LLM is unavailable. Disease is never returned (system prompt forbids it).
    """
    if not text or not str(text).strip():
        return None
    user = (
        "Study metadata:\n"
        + str(text).strip()[:4000]
        + "\nReturn ONLY the JSON object."
    )
    try:
        obj = _extract_json(_chat(cfg, _TISSUE_INFER_SYSTEM, user))
    except Exception as e:
        logger.warning(
            "[LLM-TISSUE] inference failed (%s); tissue stays unset.", e, exc_info=True
        )
        return None
    tissue = str(obj.get("tissue", "")).strip()
    reasoning = str(obj.get("reasoning", "")).strip()
    if not tissue or tissue.lower() in {"unknown", "none", "n/a", "na"}:
        logger.info("[LLM-TISSUE] no tissue inferred (%s).", reasoning)
        return None
    logger.info("[LLM-TISSUE] inferred tissue=%r (%s).", tissue.lower(), reasoning)
    return tissue.lower()


# ===========================================================================
# CellTypist model selection (tissue-aware; disease-agnostic)
# ===========================================================================
def llm_select_celltypist_model(
    cfg: ConsensusConfig,
    tissue: Optional[str],
    catalog: List[Dict[str, str]],
) -> Optional[str]:
    """Pick the best CellTypist model for `tissue` from `catalog`.

    `catalog` rows are {model, tissue, description}. Returns an exact model name
    from the catalog, or None when there is no organ match / the LLM is
    unavailable / selection fails — the caller then uses its general fallback.
    Disease is never sent to the model; only tissue/organ is used.
    """
    if not tissue:
        return None  # no tissue context -> caller keeps its configured default

    valid = {row["model"] for row in catalog}
    listing = "\n".join(
        f"{row['model']} :: {row['tissue']} — {row['description']}" for row in catalog
    )
    user = (
        f"Target tissue/organ: {tissue}\n"
        "Available models (name :: tissue — description):\n"
        f"{listing}\n"
        "Return ONLY the JSON object."
    )
    try:
        obj = _extract_json(_chat(cfg, _MODEL_SELECT_SYSTEM, user))
    except Exception as e:
        logger.warning(
            "[LLM-MODEL-SELECT] selection failed (%s); using fallback model.",
            e,
            exc_info=True,
        )
        return None

    choice = str(obj.get("model", "")).strip()
    reasoning = str(obj.get("reasoning", "")).strip()
    if choice in valid:
        logger.info(
            "[LLM-MODEL-SELECT] tissue=%r -> %s (%s)", tissue, choice, reasoning
        )
        return choice

    # "NONE" or a hallucinated name -> fall back.
    logger.info(
        "[LLM-MODEL-SELECT] tissue=%r -> no match (%r); using fallback model. %s",
        tissue,
        choice,
        reasoning,
    )
    return None


def llm_select_singler_reference(
    cfg: ConsensusConfig,
    tissue: Optional[str],
    species: Optional[str],
    references: List[Dict[str, str]],
) -> Optional[str]:
    """Pick the best SingleR/celldex reference for `species` + `tissue`.

    `references` rows are {reference, species, tissue, description}. Returns an
    exact reference name from the list, or None when there is no match / the LLM
    is unavailable / selection fails — the caller then uses its broad fallback.
    Disease is never sent; only species and tissue/organ are used.
    """
    if not tissue and not species:
        return None

    valid = {row["reference"] for row in references}
    listing = "\n".join(
        f"{row['reference']} :: species={row['species']} :: {row['tissue']} — {row['description']}"
        for row in references
    )
    user = (
        f"Target species: {species or 'unspecified'}\n"
        f"Target tissue/organ: {tissue or 'unspecified'}\n"
        "Available references (name :: species :: tissue — description):\n"
        f"{listing}\n"
        "Return ONLY the JSON object."
    )
    try:
        obj = _extract_json(_chat(cfg, _REF_SELECT_SYSTEM, user))
    except Exception as e:
        logger.warning(
            "[LLM-REF-SELECT] selection failed (%s); using fallback reference.",
            e,
            exc_info=True,
        )
        return None

    choice = str(obj.get("reference", "")).strip()
    reasoning = str(obj.get("reasoning", "")).strip()
    if choice in valid:
        logger.info(
            "[LLM-REF-SELECT] species=%r tissue=%r -> %s (%s)",
            species,
            tissue,
            choice,
            reasoning,
        )
        return choice

    logger.info(
        "[LLM-REF-SELECT] species=%r tissue=%r -> no match (%r); using fallback reference. %s",
        species,
        tissue,
        choice,
        reasoning,
    )
    return None
