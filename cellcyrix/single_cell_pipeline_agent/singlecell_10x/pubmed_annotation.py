"""
pubmed_annotation.py — literature-grounded (PubMed RAG) cell-type annotation.

A SELF-CONTAINED, optional annotation method that can act as a 4th voter in the
consensus. Unlike the parametric LLM voter, every call here is grounded in freshly
retrieved PubMed abstracts and returns the **PMIDs** it used, so each annotation is
auditable.

End-to-end flow (per Leiden cluster)
------------------------------------
    top marker genes                      (from rank_genes_groups; low-info genes dropped)
        |
        v
    build a DISEASE- and BIOSAMPLE-aware PubMed query
        |
        v
    esearch -> PMIDs -> efetch -> abstracts        (rate-limited + disk-cached)
        |
        v
    LLM adjudication over ONLY the retrieved abstracts + markers   (few-shot, strict)
        |
        v
    {cell_type, lineage, state, supporting/contradicting markers, PMIDs, confidence}
        |
        v
    confidence score (LLM self-report + objective evidence signals)
        |
        v
    consensus evidence TABLE (.csv) + confidence GRAPH (.png)
        |
        v
    optional: harmonized vote -> tally_votes  (integrate into existing consensus)

Design principles (biologically sound)
---------------------------------------
* Identity is disease-agnostic; DISEASE + BIOSAMPLE are used as SOFT context only.
* Cell IDENTITY is kept separate from cell STATE (activated/exhausted/cycling/...).
* Low-information genes (ribosomal / mito / MALAT1 / hemoglobin) are removed from
  queries so they never drive identity.
* The model may ABSTAIN ("Unknown") — never forced to guess.
* Only PMIDs actually retrieved may be cited; markers may not be invented.
* Retrieval is CACHED and PMIDs are recorded, so a run is reproducible/auditable.

Requires
--------
    requests, pandas, matplotlib   (all in requirements.txt)
    OPENROUTER_API_KEY + OPENROUTER_MODEL in the environment (.env) for the LLM step
    NCBI_API_KEY + NCBI_EMAIL (optional) to raise the PubMed rate limit 3->10 req/s

CLI
---
    python -m cellcyrix.single_cell_pipeline_agent.singlecell_10x.pubmed_annotation \
        --h5ad combined_all_samples_processed_scanpy_output.h5ad \
        --cluster-col leiden --disease "cervical cancer" --biosample "cervix" \
        --species human --out-dir ./pubmed_annotation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .atomic_io import atomic_to_csv
from .figure_style import FIGURE_DPI

if TYPE_CHECKING:  # annotations only; anndata stays a runtime-optional import
    from anndata import AnnData

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Genes that are expressed almost everywhere / are technical — they must NOT drive
# cell identity, so they are stripped from the marker set before querying PubMed.
_LOW_INFO_RX = re.compile(
    r"^(RPL\d|RPS\d|MRPL\d|MRPS\d|MT-|MTRNR|HB[ABDEGQZ]\d?|MALAT1$|NEAT1$|XIST$|"
    r"FOS$|FOSB$|JUN$|JUNB$|EGR1$|HSPA|HSPB|DNAJ|LINC\d|MIR\d|AC\d{6}|AL\d{6}|AP\d{6})",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class PubMedAnnotationConfig:
    """Configuration for a PubMed-RAG annotation run (context, retrieval, credentials)."""

    disease: str = ""  # SOFT context, e.g. "cervical cancer"
    biosample: str = ""  # tissue / biosample, e.g. "cervix"
    species: str = "human"
    top_n_genes: int = 30  # markers per cluster fed to the query/LLM
    retmax: int = 12  # abstracts retrieved per cluster
    # --- credentials (env by default) ---
    ncbi_api_key: str = field(
        default_factory=lambda: os.environ.get("NCBI_API_KEY", "").strip()
    )
    ncbi_email: str = field(
        default_factory=lambda: os.environ.get("NCBI_EMAIL", "").strip()
    )
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_MODEL", "").strip()
    )
    # --- runtime ---
    cache_dir: Path = field(default_factory=lambda: Path(".pubmed_cache"))
    tool_name: str = "sc_pipeline_pubmed_annotation"
    request_timeout: int = 30
    max_retries: int = 3

    @property
    def min_interval(self) -> float:
        """Minimum seconds between NCBI requests (3 req/s without a key, up to 10 with one)."""
        return 0.11 if self.ncbi_api_key else 0.34


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class PubMedAnnotation:
    """One cluster's literature-grounded annotation (call, evidence, PMIDs, confidence)."""

    cluster: str
    cell_type: str = "Unknown"
    broad_lineage: str = "Unknown"
    cell_subtype: Optional[str] = None  # finer identity when the literature supports it
    cell_state: Optional[str] = None
    supporting_markers: List[str] = field(default_factory=list)
    contradicting_markers: List[str] = field(default_factory=list)
    pmids: List[str] = field(default_factory=list)
    confidence: str = "low"  # high | medium | low
    confidence_score: float = 0.0  # 0..1
    review_required: bool = True
    reasoning: str = ""
    query: str = ""
    n_abstracts: int = 0
    retrieval_level: str = ""  # which query level actually returned evidence

    def as_row(self) -> dict[str, Any]:
        """Flatten this annotation into a single dict row for the evidence CSV table."""
        return {
            "cluster": self.cluster,
            "cell_type": self.cell_type,
            "cell_subtype": self.cell_subtype or "",
            "broad_lineage": self.broad_lineage,
            "cell_state": self.cell_state or "",
            "confidence": self.confidence,
            "confidence_score": round(self.confidence_score, 3),
            "review_required": self.review_required,
            "supporting_markers": ";".join(self.supporting_markers),
            "contradicting_markers": ";".join(self.contradicting_markers),
            "n_pmids": len(set(self.pmids)),
            "pmids": ";".join(dict.fromkeys(self.pmids)),  # dedup, keep order
            "n_abstracts_retrieved": self.n_abstracts,
            "retrieval_level": self.retrieval_level,
            "reasoning": self.reasoning,
            "pubmed_query": self.query,
        }


# --------------------------------------------------------------------------- #
#  PubMed E-utilities client (rate-limited + disk-cached for reproducibility)
# --------------------------------------------------------------------------- #
class PubMedClient:
    """Rate-limited, disk-cached NCBI E-utilities client (esearch + efetch) for reproducible retrieval."""

    def __init__(self, cfg: PubMedAnnotationConfig):
        self.cfg = cfg
        self._last = 0.0
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.cfg.min_interval:
            time.sleep(self.cfg.min_interval - dt)
        self._last = time.monotonic()

    def _common_params(self) -> dict[str, str]:
        p = {"db": "pubmed", "tool": self.cfg.tool_name}
        if self.cfg.ncbi_api_key:
            p["api_key"] = self.cfg.ncbi_api_key
        if self.cfg.ncbi_email:
            p["email"] = self.cfg.ncbi_email
        return p

    def _cache_path(self, kind: str, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{kind}_{h}.json"

    def _get(self, endpoint: str, params: dict[str, str], expect_json: bool):
        last_err = None
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                r = requests.get(
                    f"{EUTILS}/{endpoint}",
                    params=params,
                    timeout=self.cfg.request_timeout,
                )
                if r.status_code == 200:
                    return r.json() if expect_json else r.text
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(0.5 * (attempt + 1))
                    last_err = f"HTTP {r.status_code}"
                    continue
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                break
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(0.5 * (attempt + 1))
        logger.warning("[PUBMED] %s failed after retries (%s).", endpoint, last_err)
        return None

    def esearch(self, query: str, retmax: int) -> List[str]:
        """Return up to ``retmax`` PMIDs for ``query`` (relevance-sorted; cached on disk)."""
        cache = self._cache_path("esearch", f"{query}|{retmax}")
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        params = {
            **self._common_params(),
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        }
        data = self._get("esearch.fcgi", params, expect_json=True)
        pmids = []
        if data:
            pmids = list(data.get("esearchresult", {}).get("idlist", []) or [])
        cache.write_text(json.dumps(pmids), encoding="utf-8")
        return pmids

    def efetch_abstracts(self, pmids: List[str]) -> List[dict]:
        """Fetch title/abstract/journal/year records for ``pmids`` (cached on disk)."""
        if not pmids:
            return []
        cache = self._cache_path("efetch", ",".join(pmids))
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        params = {
            **self._common_params(),
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        xml_text = self._get("efetch.fcgi", params, expect_json=False)
        records = self._parse_efetch_xml(xml_text) if xml_text else []
        cache.write_text(json.dumps(records), encoding="utf-8")
        return records

    @staticmethod
    def _parse_efetch_xml(xml_text: str) -> List[dict]:
        out: List[dict] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning("[PUBMED] efetch XML parse error (%s).", e, exc_info=True)
            return out
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//MedlineCitation/PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            title_el = art.find(".//Article/ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abst_parts = []
            for ab in art.findall(".//Article/Abstract/AbstractText"):
                label = ab.get("Label")
                txt = "".join(ab.itertext()).strip()
                abst_parts.append(f"{label}: {txt}" if label else txt)
            abstract = " ".join(p for p in abst_parts if p)
            year_el = art.find(".//Article/Journal/JournalIssue/PubDate/Year")
            year = year_el.text if year_el is not None else ""
            jrn_el = art.find(".//Article/Journal/Title")
            journal = jrn_el.text if jrn_el is not None else ""
            if pmid and (title or abstract):
                out.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "journal": journal,
                    }
                )
        return out


# --------------------------------------------------------------------------- #
#  Marker cleaning + query building
# --------------------------------------------------------------------------- #
def clean_markers(
    genes: List[str], species: str = "human", top_n: int = 30
) -> List[str]:
    """Uppercase (human), drop low-information genes, dedup, keep order, cap at top_n."""
    out: List[str] = []
    seen = set()
    for g in genes:
        s = str(g).strip()
        if not s:
            continue
        if species.lower() in ("human", "hs", "homo sapiens"):
            s = s.upper()
        if _LOW_INFO_RX.match(s):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= top_n:
            break
    return out


def build_query(markers: List[str], disease: str, biosample: str, species: str) -> str:
    """Disease- and biosample-aware PubMed query targeting cell-type marker literature."""
    gene_clause = " OR ".join(f'"{g}"[tiab]' for g in markers[:12])  # top genes only
    ctx_terms = []
    if biosample:
        ctx_terms.append(f'"{biosample}"[tiab]')
    if disease:
        ctx_terms.append(f'"{disease}"[tiab]')
    ctx_clause = f" AND ({' OR '.join(ctx_terms)})" if ctx_terms else ""
    concept = (
        '("cell type"[tiab] OR "cell population"[tiab] OR marker[tiab] OR '
        '"single-cell"[tiab] OR "scRNA-seq"[tiab] OR immunohistochemistry[tiab])'
    )
    sp = ""
    if species.lower() in ("human", "hs", "homo sapiens"):
        sp = ' AND ("Humans"[Mesh] OR human[tiab])'
    return f"({gene_clause}){ctx_clause} AND {concept}{sp}"


def build_query_ladder(
    markers: List[str], disease: str, biosample: str, species: str
) -> List[Tuple[str, str]]:
    """Progressively broader queries so a cluster is never left with zero evidence
    just because the disease/tissue term was too narrow. Most specific first:

      1. tissue+concept  genes AND (biosample OR disease) AND cell-type-concept AND human
      2. concept         genes AND cell-type-concept AND human   (pan-tissue marker evidence)
      3. gene-only       genes AND human                         (last resort)

    Returns [(level_label, query), ...]. The label is recorded on the annotation so it
    is transparent whether a call used tissue-specific or only pan-tissue evidence.
    """
    gene_clause = " OR ".join(f'"{g}"[tiab]' for g in markers[:12])
    concept = (
        '("cell type"[tiab] OR "cell population"[tiab] OR marker[tiab] OR '
        '"single-cell"[tiab] OR "scRNA-seq"[tiab] OR immunohistochemistry[tiab])'
    )
    sp = ""
    if species.lower() in ("human", "hs", "homo sapiens"):
        sp = ' AND ("Humans"[Mesh] OR human[tiab])'
    ctx_terms = []
    if biosample:
        ctx_terms.append(f'"{biosample}"[tiab]')
    if disease:
        ctx_terms.append(f'"{disease}"[tiab]')

    ladder: List[Tuple[str, str]] = []
    if ctx_terms:
        ctx = f" AND ({' OR '.join(ctx_terms)})"
        ladder.append(("tissue+concept", f"({gene_clause}){ctx} AND {concept}{sp}"))
    ladder.append(("concept", f"({gene_clause}) AND {concept}{sp}"))
    ladder.append(("gene-only", f"({gene_clause}){sp}"))
    return ladder


# --------------------------------------------------------------------------- #
#  LLM adjudication (few-shot, strict, biologically grounded)
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are an expert single-cell biologist and pathologist annotating \
a cell cluster. You are given: the cluster's top marker genes, the tissue/biosample, an \
optional disease context, and a set of REAL PubMed abstracts retrieved for those markers.

Annotate the cluster's CELL IDENTITY using ONLY the evidence provided. Rules:
1. Base the call on the retrieved abstracts + the marker genes. Do NOT use outside facts \
that contradict the abstracts, and do NOT invent markers or PMIDs.
2. Cite ONLY PMIDs that appear in the provided abstracts and that actually support the call.
3. Tissue and disease are SOFT context, not identity. Do NOT let disease/stress/interferon/\
proliferation genes decide identity.
4. Separate IDENTITY (e.g. "CD8-positive T cell") from STATE (e.g. "exhausted", "cycling", \
"interferon-responsive"). Put state in cell_state, never in cell_type.
5. If the evidence is weak, mixed across unrelated lineages, or absent, return \
cell_type "Unknown" and set review_required true. Never force a guess.
6. Report contradicting markers (lineage markers present that argue against your call).

Return ONLY a JSON object with EXACTLY these keys:
{"broad_lineage": str, "cell_type": str, "cell_subtype": str|null, "cell_state": str|null,
 "supporting_markers": [str], "contradicting_markers": [str], "pmids": [str],
 "confidence": "high"|"medium"|"low", "review_required": bool, "reasoning": str}"""

# Few-shot examples teach the format AND the biology (identity vs state, abstain, aliases).
_FEWSHOT: List[Tuple[str, str]] = [
    (
        "Tissue: lung | Disease: lung adenocarcinoma | Cluster markers: EPCAM, KRT8, "
        "KRT18, KRT19, MSLN, SFTPC\n"
        "Abstracts:\n[PMID 31978346] EPCAM and cytokeratins KRT8/KRT18/KRT19 are "
        "canonical epithelial markers; MSLN marks malignant lung epithelium.\n"
        "[PMID 33915094] SFTPC identifies alveolar type II epithelial cells in lung.",
        '{"broad_lineage": "epithelial", "cell_type": "epithelial cell", '
        '"cell_subtype": "alveolar type II cell", "cell_state": null, '
        '"supporting_markers": ["EPCAM","KRT8","KRT18","SFTPC"], '
        '"contradicting_markers": [], "pmids": ["31978346","33915094"], '
        '"confidence": "high", "review_required": false, '
        '"reasoning": "Epithelial cytokeratins + EPCAM define epithelial identity; SFTPC narrows to AT2. MSLN is a tumor-associated state, not identity."}',
    ),
    (
        "Tissue: peripheral blood | Disease: none | Cluster markers: NKG7, GNLY, KLRD1, "
        "FCGR3A, NCAM1, PRF1\n"
        "Abstracts:\n[PMID 29942094] NKG7, GNLY, KLRD1 (CD94) and NCAM1 (CD56) mark "
        "natural killer cells; FCGR3A encodes CD16 on CD16+ NK cells.\n"
        "[PMID 30726743] Cytotoxic effector PRF1 is shared by NK and CD8 T cells; CD3 "
        "absence distinguishes NK from T cells.",
        '{"broad_lineage": "immune", "cell_type": "natural killer cell", '
        '"cell_subtype": "CD16-positive NK cell", "cell_state": null, '
        '"supporting_markers": ["NKG7","GNLY","KLRD1","FCGR3A","NCAM1"], '
        '"contradicting_markers": [], "pmids": ["29942094","30726743"], '
        '"confidence": "high", "review_required": false, '
        '"reasoning": "NK-specific markers with FCGR3A/CD16; PRF1 is cytotoxic effector shared with CD8 T but no CD3, so NK not T."}',
    ),
    (
        "Tissue: cervix | Disease: cervical cancer | Cluster markers: MALAT1, RPL13, "
        "ACTB, MT-CO1, XIST\n"
        "Abstracts:\n[PMID 00000000] (no specific cell-type marker evidence retrieved).",
        '{"broad_lineage": "Unknown", "cell_type": "Unknown", "cell_subtype": null, '
        '"cell_state": "low-quality/ambient", "supporting_markers": [], '
        '"contradicting_markers": [], "pmids": [], "confidence": "low", '
        '"review_required": true, '
        '"reasoning": "Only housekeeping/technical genes (MALAT1, ribosomal, mito, XIST); no identity evidence. Abstain."}',
    ),
]


def _llm_chat(cfg: PubMedAnnotationConfig, user_prompt: str) -> str:
    if not cfg.openrouter_api_key or not cfg.openrouter_model:
        raise RuntimeError(
            "OPENROUTER_API_KEY / OPENROUTER_MODEL not set; cannot run LLM adjudication."
        )
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for u, a in _FEWSHOT:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg.openrouter_model,
            "temperature": 0,
            "max_tokens": 700,
            "messages": messages,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict[str, Any]:
    """Robustly pull the JSON object out of an LLM reply."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?", "", t).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)


def _build_evidence_prompt(
    cluster: str,
    markers: List[str],
    abstracts: List[dict],
    cfg: PubMedAnnotationConfig,
    other_predictions: Optional[Dict[str, str]] = None,
) -> str:
    lines = [
        f"Tissue/biosample: {cfg.biosample or 'unspecified'}",
        f"Disease (soft context): {cfg.disease or 'none'}",
        f"Species: {cfg.species}",
        f"Cluster: {cluster}",
        f"Top marker genes: {', '.join(markers)}",
    ]
    if other_predictions:
        preds = "; ".join(f"{k}={v}" for k, v in other_predictions.items() if v)
        if preds:
            lines.append(f"Other methods' predictions (context only): {preds}")
    lines.append("\nRetrieved PubMed abstracts:")
    if abstracts:
        for a in abstracts:
            snippet = (a.get("abstract") or a.get("title") or "")[:700]
            lines.append(f"[PMID {a['pmid']}] {a.get('title', '')} — {snippet}")
    else:
        lines.append("(no abstracts retrieved)")
    lines.append("\nReturn the JSON object now.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Confidence
# --------------------------------------------------------------------------- #
def compute_confidence(
    result: dict[str, Any], n_abstracts: int
) -> Tuple[str, float, bool]:
    """Blend the LLM's self-report with objective evidence signals -> (band, score, review)."""
    label = str(result.get("cell_type") or "").strip()
    n_support = len({str(g).upper() for g in (result.get("supporting_markers") or [])})
    n_pmids = len({str(p) for p in (result.get("pmids") or []) if str(p).strip("0")})
    n_contra = len(result.get("contradicting_markers") or [])
    base = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(
        str(result.get("confidence", "")).lower(), 0.4
    )

    score = base
    score += 0.05 if n_support >= 3 else 0.0
    score += 0.05 if n_pmids >= 2 else 0.0
    score -= 0.20 if n_pmids == 0 else 0.0  # uncited -> weak
    score -= 0.10 * min(n_contra, 2)  # contradictions hurt
    if n_abstracts == 0:
        score = min(score, 0.25)  # nothing retrieved -> cap low
    if n_pmids == 0:
        # A literature-grounded voter that cites NO PubMed IDs is not literature-grounded:
        # it can never be "high" confidence and must always be flagged for human review.
        score = min(score, 0.6)

    unknown = (not label) or label.lower() in {"unknown", "unclear", "none", "na"}
    if unknown:
        score = min(score, 0.2)
    score = float(max(0.0, min(1.0, score)))
    band = "high" if score >= 0.75 else ("medium" if score >= 0.5 else "low")
    review = (
        bool(result.get("review_required", True))
        or unknown
        or band == "low"
        or n_pmids == 0
    )
    return band, score, review


# --------------------------------------------------------------------------- #
#  Per-cluster + full run
# --------------------------------------------------------------------------- #
def annotate_cluster(
    cfg: PubMedAnnotationConfig,
    client: PubMedClient,
    cluster: str,
    raw_markers: List[str],
    other_predictions: Optional[Dict[str, str]] = None,
) -> PubMedAnnotation:
    """Annotate one cluster: clean markers -> graded PubMed retrieval -> LLM adjudication -> confidence."""
    markers = clean_markers(raw_markers, cfg.species, cfg.top_n_genes)
    ann = PubMedAnnotation(cluster=str(cluster), supporting_markers=[])
    if not markers:
        ann.reasoning = "No informative markers after low-information filtering."
        return ann

    # Graded retrieval: tissue/disease-specific first, then relax so a cluster is
    # never left with zero evidence just because the disease term was too narrow.
    abstracts: List[dict] = []
    used_query = ""
    for level, q in build_query_ladder(
        markers, cfg.disease, cfg.biosample, cfg.species
    ):
        used_query = q
        ids = client.esearch(q, cfg.retmax)
        if not ids:
            continue
        fetched = client.efetch_abstracts(ids)
        if fetched:
            abstracts = fetched
            ann.retrieval_level = level
            if level != "tissue+concept":
                logger.info(
                    "[PUBMED] cluster %s: no tissue/disease-specific hits; relaxed retrieval to '%s'.",
                    cluster,
                    level,
                )
            break
    ann.query = used_query
    ann.n_abstracts = len(abstracts)

    prompt = _build_evidence_prompt(
        str(cluster), markers, abstracts, cfg, other_predictions
    )
    try:
        result = _parse_json(_llm_chat(cfg, prompt))
    except Exception as e:
        logger.warning(
            "[PUBMED] cluster %s: LLM adjudication failed (%s); abstaining.",
            cluster,
            e,
            exc_info=True,
        )
        ann.reasoning = f"LLM adjudication failed: {e}"
        return ann

    # only cite PMIDs that were actually retrieved
    retrieved = {a["pmid"] for a in abstracts}
    cited = [str(p) for p in (result.get("pmids") or []) if str(p) in retrieved]

    ann.cell_type = str(result.get("cell_type") or "Unknown")
    ann.broad_lineage = str(result.get("broad_lineage") or "Unknown")
    ann.cell_subtype = result.get("cell_subtype") or None
    ann.cell_state = result.get("cell_state") or None
    ann.supporting_markers = [str(g) for g in (result.get("supporting_markers") or [])]
    ann.contradicting_markers = [
        str(g) for g in (result.get("contradicting_markers") or [])
    ]
    ann.pmids = cited
    ann.reasoning = str(result.get("reasoning") or "")
    result["pmids"] = cited  # confidence should reflect verified citations only
    ann.confidence, ann.confidence_score, ann.review_required = compute_confidence(
        result, len(abstracts)
    )
    logger.info(
        "[PUBMED] cluster %s -> %s (%s, %s, %s PMIDs).",
        cluster,
        ann.cell_type,
        ann.confidence,
        format(ann.confidence_score, ".2f"),
        len(cited),
    )
    return ann


def annotate_with_pubmed(
    cluster_markers: Dict[str, List[str]],
    cfg: PubMedAnnotationConfig,
    other_predictions: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, PubMedAnnotation]:
    """Run the PubMed voter over {cluster -> [marker genes]}. Returns {cluster -> annotation}."""
    client = PubMedClient(cfg)
    results: Dict[str, PubMedAnnotation] = {}
    for cl, genes in cluster_markers.items():
        op = (other_predictions or {}).get(str(cl))
        results[str(cl)] = annotate_cluster(cfg, client, str(cl), genes, op)
    return results


# --------------------------------------------------------------------------- #
#  Marker extraction from an AnnData
# --------------------------------------------------------------------------- #
def extract_cluster_markers(
    adata: AnnData,
    cluster_col: str = "leiden",
    top_n: int = 60,
    key: str = "rank_genes_groups",
) -> Dict[str, List[str]]:
    """Top positive markers per cluster from rank_genes_groups (computed if absent)."""
    import scanpy as sc

    if key not in adata.uns:
        logger.info(
            "[PUBMED] rank_genes_groups not found; computing on '%s'.", cluster_col
        )
        sc.tl.rank_genes_groups(
            adata, groupby=cluster_col, method="wilcoxon", key_added=key
        )
    df = sc.get.rank_genes_groups_df(adata, group=None, key=key)
    out: Dict[str, List[str]] = {}
    lfc_col = "logfoldchanges" if "logfoldchanges" in df.columns else None
    sort_col = "scores" if "scores" in df.columns else df.columns[-1]
    for grp, sub in df.groupby("group"):
        if lfc_col:
            sub = sub[sub[lfc_col] > 0]
        sub = sub.sort_values(sort_col, ascending=False)
        out[str(grp)] = sub["names"].astype(str).tolist()[:top_n]
    return out


# --------------------------------------------------------------------------- #
#  Consensus table + graph
# --------------------------------------------------------------------------- #
def build_evidence_table(results: Dict[str, PubMedAnnotation]) -> pd.DataFrame:
    """Assemble per-cluster annotations into one tidy evidence DataFrame (cluster-sorted)."""
    rows = [results[k].as_row() for k in sorted(results, key=lambda c: (len(c), c))]
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write the evidence table to ``out_dir/pubmed_annotation_table.csv`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "pubmed_annotation_table.csv"
    atomic_to_csv(df, p, index=False)
    logger.info("[PUBMED] wrote evidence table -> %s", p)
    return p


def plot_confidence(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """Per-cluster confidence graph: bar length = confidence, color = band, label = cell type + #PMIDs."""
    if df.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.copy()
    d["clabel"] = d["cluster"].astype(str)
    colors = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
    bar_colors = [colors.get(str(b), "#7f7f7f") for b in d["confidence"]]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["confidence_score"], color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"cluster {c}" for c in d["clabel"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("PubMed-evidence confidence (0–1)")
    ax.set_title("Literature (PubMed) cell-type annotation — per-cluster confidence")
    for yi, row in zip(y, d.to_dict("records"), strict=True):
        ax.text(
            min(row["confidence_score"] + 0.01, 0.99),
            yi,
            f" {row['cell_type']}  ({int(row['n_pmids'])} PMIDs)",
            va="center",
            ha="left",
            fontsize=8,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[b]) for b in ("high", "medium", "low")
    ]
    ax.legend(
        handles,
        ["high", "medium", "low"],
        title="confidence",
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "pubmed_annotation_confidence.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("[PUBMED] wrote confidence graph -> %s", p)
    return p


# --------------------------------------------------------------------------- #
#  Consensus adapter (optional: feed the existing tally_votes)
# --------------------------------------------------------------------------- #
def as_consensus_votes(
    results: Dict[str, PubMedAnnotation],
) -> Dict[str, Tuple[str, float]]:
    """{cluster -> (harmonized_label, confidence_score)} for the existing consensus vote.

    Harmonizes to the shared controlled vocabulary if the consensus module is importable,
    so the vote aligns with CellTypist/SingleR/LLM; otherwise returns the raw label.
    """
    try:
        from .celltype_consensus.tools import harmonize_label  # type: ignore
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        logger.debug("%s: falling back after %r", __name__, exc)
        harmonize_label = lambda x: x  # noqa: E731
    votes: Dict[str, Tuple[str, float]] = {}
    for cl, ann in results.items():
        label = (
            "Unassigned"
            if ann.cell_type.lower() == "unknown"
            else harmonize_label(ann.cell_type)
        )
        votes[str(cl)] = (label, ann.confidence_score)
    return votes


def run_pubmed_annotation(
    adata: AnnData,
    *,
    cluster_col: str = "leiden",
    disease: str = "",
    biosample: str = "",
    species: str = "human",
    out_dir: Path,
    top_n_genes: int = 30,
    other_predictions: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """End-to-end: extract markers -> PubMed RAG annotate -> write table + graph. Returns the table."""
    cfg = PubMedAnnotationConfig(
        disease=disease,
        biosample=biosample,
        species=species,
        top_n_genes=top_n_genes,
        cache_dir=Path(out_dir) / "_pubmed_cache",
    )
    markers = extract_cluster_markers(
        adata, cluster_col=cluster_col, top_n=max(top_n_genes * 2, 60)
    )
    results = annotate_with_pubmed(markers, cfg, other_predictions)
    df = build_evidence_table(results)
    save_table(df, out_dir)
    plot_confidence(df, out_dir)
    return df


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _cli(argv=None):
    ap = argparse.ArgumentParser(
        description="PubMed literature-grounded cell-type annotation voter."
    )
    ap.add_argument("--h5ad", required=True, help="Processed .h5ad (clustered).")
    ap.add_argument("--cluster-col", default="leiden")
    ap.add_argument(
        "--disease",
        default="",
        help="Disease context (soft prior), e.g. 'cervical cancer'.",
    )
    ap.add_argument("--biosample", default="", help="Tissue/biosample, e.g. 'cervix'.")
    ap.add_argument("--species", default="human")
    ap.add_argument("--top-n-genes", type=int, default=30)
    ap.add_argument("--out-dir", default="./pubmed_annotation")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    import scanpy as sc

    adata = sc.read_h5ad(args.h5ad)
    df = run_pubmed_annotation(
        adata,
        cluster_col=args.cluster_col,
        disease=args.disease,
        biosample=args.biosample,
        species=args.species,
        out_dir=Path(args.out_dir),
        top_n_genes=args.top_n_genes,
    )
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    _cli()
