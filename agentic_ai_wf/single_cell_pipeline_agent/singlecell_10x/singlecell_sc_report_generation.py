"""
singlecell_sc_report_generation.py — build the HTML/PDF analysis report.

Collects every artifact a run produced (QC tables, embeddings, cluster/cell-type
figures, marker and DE tables, pathway results) and renders a single branded,
self-contained report via a Jinja2 template, embedding images as base64 so the
HTML is portable. An optional LLM pass (OpenRouter) writes short narrative
summaries for each section; all LLM calls are deterministic (temperature 0) and
degrade gracefully to templated text when no API key is configured. Images use the
headless matplotlib ``Agg`` backend so the report builds on a server.
"""

import argparse
import ast
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

# Matplotlib headless backend
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import io
import math

from dotenv import load_dotenv
from jinja2 import Template

from . import env_names
from .column_names import GENE_COLUMNS, to_canonical_columns

# ======================
# === Constants & logos
# ======================

logger = logging.getLogger(__name__)

DEFAULT_LOGOS_DIR = Path(__file__).parent / "logos"
LOGO_FILENAMES_PREFERRED = ["left-side-logo.png", "right-side-main.png"]
LOGO_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PDF_PAGE_SIZE = "A4"
PDF_MARGIN_MM = 12


# ======================
# === LLM (OpenRouter, OpenAI-compatible)
# ======================


def _require_openai():
    """Return ``(client, model)`` for the report's LLM sections, or raise.

    Everything goes through OpenRouter, so the model is an OpenRouter slug and the
    provider is chosen by its prefix (``anthropic/...`` vs ``openai/...``).
    """
    load_dotenv()
    try:
        from agentic_ai_wf.llm import settings as llm_settings
        from agentic_ai_wf.llm.clients import get_sync_openrouter
    except Exception as e:
        # Surface the real cause. This used to report only "agentic_ai_wf.llm is
        # required", which hid failures like the wrong `decouple` distribution being
        # installed (the PyPI package is `python-decouple`; a bare `decouple` package
        # also exists and has no `config`).
        raise RuntimeError(
            "agentic_ai_wf.llm is required for LLM report sections, but importing it "
            f"failed with {type(e).__name__}: {e}. If this mentions `decouple`, the "
            "environment has the wrong distribution — install `python-decouple`. "
            "This package also requires Python >= 3.11."
        ) from e

    try:
        client = get_sync_openrouter()
    except RuntimeError as e:
        raise RuntimeError(
            "OPENROUTER_API_KEY missing in .env (LLM-only interpretations required)."
        ) from e

    # Model resolution, first match wins. OPENROUTER_MODEL comes first because that
    # is the variable the rest of the pipeline already reads (the consensus voter
    # uses it), so one .env entry drives every LLM step instead of the report
    # silently falling back to a different vendor's default.
    #   SCPIPE_REPORT_LLM_MODEL  report-only override (legacy: REPORT_LLM_MODEL)
    #   OPENROUTER_MODEL  the pipeline-wide slug (e.g. anthropic/claude-sonnet-4.6)
    #   LLM_MODEL_FAST    the shared "fast tier" slug
    #   OPENAI_MODEL      legacy hook, kept so existing setups do not change
    #   LLM_MODEL_CHAT    last-resort default from llm/settings.py
    model = (
        (env_names.get_env(env_names.REPORT_LLM_MODEL) or "").strip()
        or os.getenv("OPENROUTER_MODEL", "").strip()
        or os.getenv("LLM_MODEL_FAST", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or llm_settings.LLM_MODEL_CHAT
    )
    logger.info("[REPORT] LLM sections via OpenRouter, model=%s", model)
    return client.with_options(timeout=30.0), model


_client, _model = None, None


def _client_model():
    global _client, _model
    if _client is None:
        _client, _model = _require_openai()
    return _client, _model


def _json_maybe_repair(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip(":").strip()
    return s


def _repair_truncated_json(s: str) -> str:
    """Best-effort completion of a truncated JSON object/array: close a dangling
    string, drop a partial trailing key/value, and balance open brackets/braces.

    Handles the common failure where the LLM response is cut off mid-value
    (e.g. hit max_tokens), which leaves an unterminated string and unclosed {}.
    """
    stack: List[str] = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    out = s
    if in_str:
        out += '"'  # close the dangling string
    out = re.sub(r"[,:]\s*$", "", out.rstrip())  # drop a partial trailing key/value
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


def _coerce_json(text: str) -> Optional[dict[str, Any]]:
    """Parse LLM output into a dict, tolerating code fences, surrounding prose,
    and truncation. Returns None only if nothing dict-like can be recovered."""
    text = _json_maybe_repair(text)
    start = text.find("{")
    core = text[start:] if start >= 0 else text
    candidates = [text, core, _repair_truncated_json(core)]
    # Progressive trim: cut back to each comma from the end and re-balance, so all
    # complete fields survive even when the last one was cut off mid-write.
    for m in reversed([mm.start() for mm in re.finditer(r",", core)]):
        candidates.append(_repair_truncated_json(core[:m]))
    for cand in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                val = parser(cand)
            except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
                logger.debug("%s: falling back after %r", __name__, exc)
                continue
            if isinstance(val, dict):
                return val
    return None


def llm_json(system: str, prompt: str, max_tokens: int = 700) -> dict[str, Any]:
    """Call the LLM and parse its reply as a JSON object (``{}`` on failure)."""
    client, model = _client_model()
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt + "\n\nReturn ONLY valid JSON."},
    ]
    out = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=msg,
    )
    text = (out.choices[0].message.content or "").strip()
    parsed = _coerce_json(text)
    if parsed is not None:
        return parsed
    # Non-fatal: never let one unparseable/truncated LLM reply kill the whole
    # report. Callers read fields with .get(), so an empty dict degrades cleanly.
    logger.warning(
        "llm_json: could not parse LLM output; returning empty dict. "
        "First 200 chars:\n%s",
        text[:200],
    )
    return {}


def llm_text(system: str, prompt: str, max_tokens: int = 420) -> str:
    """Call the LLM and return its reply as plain text (deterministic, temperature 0)."""
    client, model = _client_model()
    out = client.chat.completions.create(
        model=model,
        temperature=0.0,  # deterministic: same inputs -> same text across runs
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return (out.choices[0].message.content or "").strip()


# ==============
# === IO utils
# ==============


def read_table(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV/TSV table into a DataFrame, or ``None`` if missing/unreadable.

    Headers are normalised through `column_names.to_canonical_columns`, so this
    reader sees one vocabulary whether the file was written before or after the
    Rule 5.4 rename — an older table's ``Adjusted P-value`` arrives as
    ``p_value_adj``, exactly like a current one.

    Args:
        path: CSV, TSV or JSON table to read.

    Returns:
        The table with canonical headers, or None when absent or unreadable.
    """
    if not path.exists():
        return None
    try:
        suf = path.suffix.lower()
        if suf in {".tsv", ".txt"}:
            return to_canonical_columns(pd.read_csv(path, sep="\t"))
        if suf == ".csv":
            return to_canonical_columns(pd.read_csv(path))
        if suf == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return pd.json_normalize(obj) if not isinstance(obj, pd.DataFrame) else obj
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e, exc_info=True)
    return None


def read_json(path: Path) -> Optional[dict[str, Any]]:
    """Read a JSON file into a dict, or return ``None`` if it is missing/unreadable."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read JSON %s: %s", path, e, exc_info=True)
        return None


# Size limits for an image once embedded in the report. The figures on disk are
# full print resolution (see figure_style.FIGURE_DPI) and stay that way — but
# base64-inlining a folder of them at full size makes a single self-contained HTML
# file tens of megabytes. The report lays images out inside a 1400 px container, so
# the embedded copy stays above that (sharp at 1x, near-2x on a wide figure) and
# the PNG on disk remains the one to put in a slide or a manuscript. Two limits,
# because a panorama and a square UMAP need different handling: total pixels keeps
# the file small, the side cap keeps a wide figure from being over-shrunk.
REPORT_IMG_MAX_PX: int = 2_000_000
REPORT_IMG_MAX_SIDE: int = 2400


def _dedup_pathway_csvs(pathways_combined_dir: Path) -> list[Path]:
    """Deduplicated pathway tables in a directory, newest naming first.

    The current filename is ``*_combined_pathways_dedup.csv``; runs produced before
    the Rule 5.1 lowercasing wrote ``*_combined_pathways_DEDUP.csv``. Both are matched
    so a report can be regenerated over an existing output directory. On a
    case-insensitive filesystem the two globs return the same files, hence the set.

    Args:
        pathways_combined_dir: Directory holding the combined pathway tables.

    Returns:
        Sorted list of matching CSV paths; empty when none are present.
    """
    return sorted(
        set(pathways_combined_dir.glob("*combined_pathways_dedup.csv"))
        | set(pathways_combined_dir.glob("*combined_pathways_DEDUP.csv"))
    )


def b64_img(
    path: Optional[Path],
    *,
    max_px: int = REPORT_IMG_MAX_PX,
    max_side: int = REPORT_IMG_MAX_SIDE,
) -> Optional[str]:
    """Return a base64 ``data:`` URI for an image so it can be embedded inline in the HTML report."""
    if not path or not path.exists():
        return None
    try:
        data, mime = _embed_bytes(path, max_px, max_side)
        return "data:image/%s;base64,%s" % (
            mime,
            base64.b64encode(data).decode("utf-8"),
        )
    except Exception as e:
        logger.warning("Could not b64-encode %s: %s", path, e, exc_info=True)
        return None


def _embed_bytes(path: Path, max_px: int, max_side: int) -> tuple[str, str]:
    """``(bytes, mime)`` for ``path``, shrunk to fit the embedding limits.

    Falls back to the original bytes when Pillow is missing or anything goes wrong —
    an oversized report beats a report with holes in it.
    """
    raw = path.read_bytes()
    raw_mime = "png" if path.suffix.lower() == ".png" else "jpeg"
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            scale = min(
                max_side / float(max(width, height)),
                math.sqrt(max_px / float(max(width * height, 1))),
            )
            if scale >= 1.0:
                return raw, raw_mime
            resized = img.convert("RGB").resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.LANCZOS,
            )
            buf = io.BytesIO()
            resized.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            # Resampling can cost more than it saves on a flat-colour figure (a
            # heatmap re-encoded larger than the original), and bytes are what the
            # report pays for — keep whichever is smaller.
            return (data, "png") if len(data) < len(raw) else (raw, raw_mime)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not downscale %s for embedding: %s", path.name, e, exc_info=True
        )
        return raw, raw_mime


def ensure_dir(p: Path) -> None:
    """Create directory ``p`` (and parents) if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x: float) -> "float | None":
    """Coerce ``x`` to float, returning ``None`` if it cannot be parsed."""
    try:
        return float(x)
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        logger.debug("%s: falling back after %r", __name__, exc)
        return None


# ================
# === Text clean
# ================

_MD_PAT = re.compile(r"[*_`#>]+")


def strip_md(s: str) -> str:
    """Remove markdown markup (``*``, ``#``, ...) from ``s`` for clean plain-text rendering."""
    if not s:
        return ""
    return _MD_PAT.sub("", s).replace("  ", " ").strip()


def clean_line(s: str) -> str:
    """Strip markdown and leading bullet/heading punctuation from a single line."""
    s = strip_md(s or "")
    return re.sub(r"^[\s\-\*\#]+", "", s).strip()


def clean_list(lines: List[str]) -> List[str]:
    """Apply :func:`clean_line` to each entry, dropping any that become empty."""
    return [clean_line(x) for x in lines if clean_line(x)]


# =========================
# === Cell-type coloring
# =========================


def palette(i: int) -> str:
    """Return a distinct, deterministic pastel HSL color for palette index ``i``."""
    hue = (i * 53) % 360
    return f"hsl({hue}, 70%, 80%)"


def build_ct_colors_from_counts(ct_counts: Dict[str, int]) -> Dict[str, str]:
    """Assign each cell type a stable color, ordered by descending cell count."""
    names = [str(k).strip() for k in ct_counts if str(k).strip()]
    uniq = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return {name: palette(i) for i, name in enumerate(uniq)}


def colorize_text(text: str, ct_colors: Dict[str, str]) -> str:
    """Wrap each cell-type name found in ``text`` in a colored HTML ``<span>`` tag."""
    if not text:
        return ""
    s = strip_md(text)
    for name in sorted(ct_colors.keys(), key=len, reverse=True):
        pat = r"\b" + re.escape(name).replace(r"\.", r"[._ ]") + r"\b"
        s = re.sub(
            pat,
            f'<span class="ct-tag" style="background:{ct_colors[name]};">'
            + name.replace(".", " ").replace("_", " ")
            + "</span>",
            s,
            flags=re.IGNORECASE,
        )
    return s


def colorize_list(lines: List[str], ct_colors: Dict[str, str]) -> List[str]:
    """Clean and cell-type-colorize each line (list form of :func:`colorize_text`)."""
    return [colorize_text(clean_line(line), ct_colors) for line in lines]


# ==========================
# === GEO metadata helpers
# ==========================


def _find_sample_entry(
    geo_meta: dict[str, Any], case_id: str
) -> Optional[dict[str, Any]]:
    """
    Try to locate a GSM/sample entry matching case_id in the GEO metadata.
    Supports several possible list keys.
    """
    if not case_id:
        return None
    keys = ["samples", "sample_list", "gsm_list", "sample_metadata"]
    for k in keys:
        arr = geo_meta.get(k)
        if isinstance(arr, list):
            for s in arr:
                acc = str(s.get("accession", "")).strip()
                if acc == str(case_id).strip():
                    return s
    return None


def infer_dataset_context_from_geo(
    meta: dict[str, Any],
    sample_accession: str = "",
    sample_title: str = "",
) -> Tuple[str, str, str]:
    """
    Use LLM to derive (disease_label, biosample_label, context_text) from GEO JSON,
    explicitly taking into account the chosen sample (case_id) when available.
    """
    es = meta.get("esummary_raw", {})
    title = meta.get("title") or es.get("title", "")
    summary = meta.get("summary") or es.get("summary", "")
    taxon = meta.get("taxon") or es.get("taxon", "")
    gdstype = meta.get("gdstype") or es.get("gdstype", "")

    j = llm_json(
        "You are a biomedical metadata curator. Return ONLY JSON.",
        (
            "Given GEO metadata (title, summary, taxon, gdstype) and a specific sample, "
            "extract a concise high-level context.\n"
            f"Series title: {title}\n"
            f"Series summary: {summary}\n"
            f"Taxon: {taxon}\n"
            f"gdstype: {gdstype}\n"
            f"Sample accession: {sample_accession or 'NA'}\n"
            f"Sample title: {sample_title or 'NA'}\n\n"
            'Return JSON: {"disease":"...","biosample":"...","context":"2-3 sentences summarizing the disease, '
            'the biological role of this specific sample, and why transcriptomic profiling is informative (no invented numbers)."}'
        ),
        max_tokens=260,
    )
    disease = clean_line(j.get("disease", "") or "NA")
    biosample = clean_line(j.get("biosample", "") or "NA")
    context = strip_md(j.get("context", "") or "")
    return disease, biosample, context


# ===============================
# === Single-cell summary parse
# ===============================


def load_analysis_summary(
    summary_dir: Path,
) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    """
    Returns:
      analysis_name (str from header),
      kv (dict of key metrics),
      celltype_counts (dict)
    """
    txt_files = sorted(summary_dir.glob("*_analysis_summary.txt"))
    if not txt_files:
        return "singlecell", {}, {}

    path = txt_files[0]
    text = path.read_text(encoding="utf-8", errors="ignore")

    kv = {}
    ct_counts: Dict[str, int] = {}
    analysis_name = "singlecell"

    # Simple key-value parsing
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("===") and line.endswith("==="):
            inside = line.strip("=").strip()
            analysis_name = inside or analysis_name
        m = re.match(r"([^:]+):\s*(.+)", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            kv[key] = val

    # Celltype counts block
    if "Celltype counts:" in text:
        block = text.split("Celltype counts:")[1]
        for line in block.splitlines()[1:]:
            if not line.strip():
                break
            m2 = re.match(r"^\s*(.+?):\s*([0-9]+)\s+cells", line.strip())
            if m2:
                ct = m2.group(1).strip()
                n = int(m2.group(2))
                ct_counts[ct] = n

    return analysis_name, kv, ct_counts


def top_celltypes_line(ct_counts: Dict[str, int]) -> str:
    """Return a one-line summary of the top (up to 8) cell types with their % of cells."""
    if not ct_counts:
        return ""
    total = float(sum(ct_counts.values()))
    rows = sorted(ct_counts.items(), key=lambda x: x[1], reverse=True)
    bits = []
    for name, n in rows[:8]:
        pct = 100.0 * n / total if total > 0 else 0.0
        bits.append(f"{name} ({pct:.1f}% of cells)")
    return ", ".join(bits)


def build_preproc_bullets_from_kv(
    kv: Dict[str, str], disease: str, biosample: str
) -> List[str]:
    """
    Build a professional, stepwise preprocessing summary using the metrics in kv.
    Mirrors the 1–7 bullets you described.
    """
    bullets: List[str] = []

    init_cells = kv.get("Initial cells")
    init_genes = kv.get("Initial genes")
    cells_after_qc = kv.get("Cells after QC filters")
    genes_after_min = kv.get("Genes after min_cells filter")
    hvgs = kv.get("HVGs used")
    n_clusters = kv.get("Leiden clusters")

    if init_cells and init_genes:
        bullets.append(
            f"Initial Data: The dataset starts with {init_cells} cells and {init_genes} genes "
            f"derived from transcriptomic profiling of {biosample} in the context of {disease}."
        )
    else:
        bullets.append(
            f"Initial Data: Transcriptomic profiling was performed on {biosample} in {disease}, "
            "capturing thousands of cells and genes for downstream analysis."
        )

    if cells_after_qc:
        bullets.append(
            f"Quality Control (QC): Low-quality cells are filtered out based on total counts, number of detected genes, "
            f"and mitochondrial gene percentage, resulting in {cells_after_qc} high-quality cells retained for analysis."
        )
    else:
        bullets.append(
            "Quality Control (QC): Low-quality cells are removed using thresholds on library size, detected genes, and "
            "mitochondrial RNA content, retaining a high-confidence set of cells."
        )

    if genes_after_min:
        bullets.append(
            f"Gene Filtering: Genes expressed in very few cells are removed using a minimum-cells filter, reducing the "
            f"feature space to {genes_after_min} genes while preserving informative transcripts."
        )
    else:
        bullets.append(
            "Gene Filtering: Lowly expressed genes are filtered out to focus the analysis on robustly detected transcripts."
        )

    bullets.append(
        "Normalization: Raw counts are normalized across cells (library-size scaling followed by log-transformation) "
        "to make gene expression levels comparable across the dataset."
    )

    if hvgs:
        bullets.append(
            f"Highly Variable Genes (HVGs): {hvgs} highly variable genes are selected to capture the dominant biological "
            "signals and reduce technical noise before downstream modeling."
        )
    else:
        bullets.append(
            "Highly Variable Genes (HVGs): The most variable genes are selected to emphasise key biological variation "
            "while reducing technical noise."
        )

    bullets.append(
        "Dimensionality Reduction: Latent embeddings are constructed from the HVG space, followed by non-linear methods "
        "such as t-SNE and UMAP to visualise cell states and gradients in two dimensions."
    )

    if n_clusters:
        bullets.append(
            f"Clustering: A graph-based Leiden clustering algorithm is run on the embedding space, yielding {n_clusters} "
            "transcriptionally distinct cell populations for downstream interpretation."
        )
    else:
        bullets.append(
            "Clustering: A graph-based Leiden clustering algorithm identifies transcriptionally distinct cell populations, "
            "which are then annotated using canonical marker genes and, where relevant, supervised reference models."
        )

    return bullets


# =======================
# === Markers & pathways
# =======================

# =========================================================================== #
#  Annotation confidence / consensus reporting
# =========================================================================== #
CONSENSUS_TIER_ORDER = ("High", "Medium", "Low/Review")

# (display header, preferred column, legacy fallback column)
_CONSENSUS_DISPLAY_COLUMNS = [
    ("Cluster", "leiden", "cluster"),
    ("Cells", "n_cells", None),
    ("Final cell type", "final_celltype", "consensus"),
    ("Tier", "consensus_tier", "tier"),
    ("Lineage gate", "lineage_coarse", "lineage_gate"),
    ("CellTypist", "celltypist_label", "celltypist"),
    ("SingleR", "singler_label", "singler"),
    ("Knowledge-based", "knowledge_based_label", "knowledge_based"),
    ("PubMed", "pubmed_label", "pubmed"),
    ("Voters disagree", "voters_disagree", None),
    ("Mixed cluster", "mixed_cluster_flag", None),
    ("Subtype", "celltype_subtype", None),
    ("In downstream", "included_in_downstream_analysis", None),
    ("Decision", "decision_reason", None),
]


def _as_bool_series(s) -> "pd.Series":
    """Coerce a CSV column that may hold bools or 'True'/'False' strings."""
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_consensus_annotation(ct_anno_dir: Path) -> "tuple[pd.DataFrame | None, dict]":
    """Read the per-cluster consensus table -> ``(display_df, summary_dict)``.

    Returns ``(None, {})`` when the table is absent (e.g. annotation did not run),
    so the report simply omits the section instead of failing.
    """
    if not ct_anno_dir or not Path(ct_anno_dir).exists():
        return None, {}
    hits = sorted(Path(ct_anno_dir).glob("*_consensus_annotation.csv"))
    if not hits:
        return None, {}
    df = read_table(hits[0])
    if df is None or df.empty:
        return None, {}

    tier_col = "consensus_tier" if "consensus_tier" in df.columns else "tier"
    final_col = "final_celltype" if "final_celltype" in df.columns else "consensus"

    tiers = (
        df[tier_col].astype(str) if tier_col in df.columns else pd.Series([], dtype=str)
    )
    mixed = (
        _as_bool_series(df["mixed_cluster_flag"])
        if "mixed_cluster_flag" in df.columns
        else pd.Series(False, index=df.index)
    )
    included = (
        _as_bool_series(df["included_in_downstream_analysis"])
        if "included_in_downstream_analysis" in df.columns
        else pd.Series(True, index=df.index)
    )
    finals = (
        df[final_col].astype(str)
        if final_col in df.columns
        else pd.Series([], dtype=str)
    )
    disagree = (
        _as_bool_series(df["voters_disagree"])
        if "voters_disagree" in df.columns
        else pd.Series(False, index=df.index)
    )

    summary = {
        "n_clusters": int(len(df)),
        "n_cells": int(pd.to_numeric(df["n_cells"], errors="coerce").fillna(0).sum())
        if "n_cells" in df.columns
        else None,
        "high": int((tiers == "High").sum()),
        "medium": int((tiers == "Medium").sum()),
        "low_review": int((tiers == "Low/Review").sum()),
        "mixed": int(mixed.sum()),
        "excluded": int((~included).sum()),
        "unassigned": int(
            finals.str.strip().str.lower().str.startswith("unassigned").sum()
        ),
        "disagreeing": int(disagree.sum()),
    }
    if "n_cells" in df.columns:
        summary["n_cells_excluded"] = int(
            pd.to_numeric(df.loc[~included, "n_cells"], errors="coerce").fillna(0).sum()
        )
        summary["n_cells_low_review"] = int(
            pd.to_numeric(df.loc[tiers == "Low/Review", "n_cells"], errors="coerce")
            .fillna(0)
            .sum()
        )

    # Build the display frame from whichever column names this CSV actually has.
    out = pd.DataFrame()
    for header, preferred, legacy in _CONSENSUS_DISPLAY_COLUMNS:
        col = (
            preferred
            if preferred in df.columns
            else (legacy if legacy and legacy in df.columns else None)
        )
        if col is None:
            continue
        vals = df[col]
        if header in ("Voters disagree", "Mixed cluster"):
            vals = _as_bool_series(vals).map({True: "yes", False: "no"})
        elif header == "In downstream":
            vals = _as_bool_series(vals).map({True: "yes", False: "EXCLUDED"})
        elif header == "Decision":
            vals = vals.astype(str).str.slice(0, 90)
        out[header] = vals.values
    return out, summary


def read_run_manifest(sc_root: Path) -> dict[str, Any]:
    """Load ``provenance/manifest.json`` (empty dict when unavailable)."""
    path = Path(sc_root) / "provenance" / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Failed to read manifest %s: %s", path, e, exc_info=True)
        return {}


def annotation_resource_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """``[(label, value)]`` describing the annotation resources ACTUALLY used."""
    params = (manifest or {}).get("params") or {}
    anno = params.get("annotation") or {}
    clus = params.get("clustering") or {}
    qc = params.get("qc") or {}
    conf = params.get("confidence_filtering") or {}

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    rows = [
        (
            "Leiden resolution (used)",
            _fmt(clus.get("leiden_resolution") or params.get("leiden_resolution")),
        ),
        ("Leiden clusters", _fmt(clus.get("n_clusters"))),
        ("Integration requested", _fmt(params.get("integration_method"))),
        ("Integration used", _fmt(params.get("integration_method_used"))),
        ("CellTypist model requested", _fmt(anno.get("requested_celltypist_model"))),
        ("CellTypist model used", _fmt(anno.get("resolved_celltypist_model"))),
        ("SingleR reference requested", _fmt(anno.get("requested_singler_reference"))),
        ("SingleR reference used", _fmt(anno.get("resolved_singler_reference"))),
        (
            "Annotation tissue / species",
            f"{_fmt(anno.get('annotation_tissue'))} / "
            f"{_fmt(anno.get('annotation_species'))}",
        ),
        (
            "Voters enabled",
            ", ".join(
                n
                for n, on in (
                    ("CellTypist", anno.get("celltypist_enabled")),
                    ("SingleR", anno.get("singler_enabled")),
                    ("Knowledge-based", anno.get("knowledge_based_enabled")),
                    ("PubMed", anno.get("pubmed_enabled")),
                )
                if on
            )
            or "—",
        ),
        ("Knowledge-based LLM", _fmt(anno.get("llm_model"))),
        ("Marker ranking", _fmt(anno.get("marker_ranking_method"))),
        ("QC thresholds applied", _fmt(qc.get("applied"))),
        ("Low-confidence exclusion", _fmt(conf.get("exclude_low_confidence_de"))),
        (
            "Excluded tiers",
            _fmt(", ".join(conf.get("excluded_consensus_tiers") or []) or None),
        ),
        ("Annotation source", _fmt(anno.get("celltype_source"))),
        (
            "Reused existing annotation",
            _fmt(anno.get("reuse_existing_final_annotation")),
        ),
    ]
    return [(k, v) for k, v in rows if v != "—" or k.endswith("used")]


def prepare_sc_markers_table(
    markers_path: Path, disease: str, biosample: str, ct_colors: Dict[str, str]
) -> Tuple[Optional[pd.DataFrame], str, List[str], str, Dict[str, List[str]]]:
    """
    Use celltype_marker_genes_celltype_all.csv (group=celltype, gene=gene).
    Returns:
      display_df,
      topline,
      bullets,
      note,
      top_markers_per_ct (dict celltype -> list[genes])
    """
    if not markers_path.exists():
        return None, "Marker table not found.", [], "", {}

    df = pd.read_csv(markers_path)
    gene_col = next((c for c in GENE_COLUMNS if c in df.columns), None)
    if "group" not in df.columns or gene_col is None:
        return None, "Marker table not found (missing columns).", [], "", {}

    # group markers per celltype
    grouped = (
        df[["group", gene_col]]
        .dropna()
        .groupby("group")[gene_col]
        .apply(lambda s: list(dict.fromkeys([str(x) for x in s])))
        .reset_index(name="markers")
    )

    top_markers_per_ct: Dict[str, List[str]] = {}
    # limit each cell type to top 10 markers for display and LLM
    grouped["markers"] = grouped["markers"].apply(lambda lst: [g for g in lst][:10])
    for row in grouped.to_dict("records"):
        top_markers_per_ct[str(row["group"])] = list(row["markers"])

    display = grouped.copy()
    display["markers"] = display["markers"].apply(lambda lst: ", ".join(lst))
    display = display.rename(columns={"group": "cell_type"}).head(18)

    topline = "Key marker genes are summarised per cell type."

    j = llm_json(
        "You are a single-cell RNA-seq data analyst. Return ONLY JSON.",
        (
            f"For biosample '{biosample}', interpret the following cell-type marker summary.\n"
            f"Each row has a cell_type and its top markers.\n"
            f"Markers:\n{display.to_dict(orient='records')}\n\n"
            "Ground every statement in the specific marker genes listed; phrase interpretations "
            "as hypotheses. Do NOT discuss clinical outcomes, treatment, immunotherapy, or prognosis.\n"
            'Return JSON: {"bullets":["...","...","...","..."],'
            '"note":"one concise, evidence-grounded note on cell-type composition, naming the markers behind it."}\n'
            "Keep each bullet to one sentence and the note to 2-3 sentences. "
            "Avoid invented percentages or patient numbers."
        ),
        max_tokens=900,
    )

    bullets = colorize_list(j.get("bullets", [])[:6], ct_colors)
    note = colorize_text(clean_line(j.get("note", "")), ct_colors)

    return display, topline, bullets, note, top_markers_per_ct


def summarize_pathways(
    pathways_combined_dir: Path, disease: str, biosample: str, ct_colors: Dict[str, str]
) -> Tuple[Optional[pd.DataFrame], List[str], str]:
    """
    Summarize top pathways across all clusters (global view).
    Show only top 10 pathways, with max 2 per biological_database.
    Do NOT display p-values in the final table (only use them internally for ranking).
    """
    if not pathways_combined_dir.exists():
        return None, ["Pathway enrichment tables not found."], ""

    csvs = _dedup_pathway_csvs(pathways_combined_dir)
    if not csvs:
        csvs = sorted(pathways_combined_dir.glob("*combined_pathways*.csv"))
    if not csvs:
        return None, ["Pathway enrichment tables not found."], ""

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df["source_file"] = f.name
                dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read pathways %s: %s", f, e, exc_info=True)
    if not dfs:
        return None, ["Pathway enrichment tables not found."], ""

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Rank by adj p or combined score
    if "p_value_adj" in all_df.columns:
        all_df["__score"] = all_df["p_value_adj"].fillna(1.0)
        all_df = all_df.sort_values("__score", ascending=True)
    elif "combined_score" in all_df.columns:
        all_df["__score"] = -all_df["combined_score"].fillna(0.0)
        all_df = all_df.sort_values("__score", ascending=True)

    # Keep only descriptive columns for display (no p-values)
    keep_cols = []
    for c in ["biological_database", "pathway", "genes", "source_file", "__score"]:
        if c in all_df.columns:
            keep_cols.append(c)
    display = all_df[keep_cols].copy()

    # limit to top 2 per database, then overall top 10
    if "biological_database" in display.columns:
        top_rows = []
        for _db, sub in display.groupby("biological_database", sort=False):
            sub_sorted = sub.sort_values("__score", ascending=True)
            top_rows.append(sub_sorted.head(2))
        display = pd.concat(top_rows, ignore_index=True)
        display = display.sort_values("__score", ascending=True).head(10)
    else:
        display = display.sort_values("__score", ascending=True).head(10)

    # Build compact list representation to feed into LLM (we can still send p-values if present)
    mini_list = []
    for r in display.to_dict("records"):
        mini_list.append(
            {
                "database": r.get("biological_database", ""),
                "pathway": r.get("pathway", ""),
            }
        )

    j = llm_json(
        "You are a single-cell RNA-seq data analyst. Return ONLY JSON.",
        (
            f"Summarize what the top enriched pathways across clusters suggest about the biology "
            f"of the sampled tissue ('{biosample}').\n"
            f"Top pathways list (each with database and pathway name):\n{mini_list}\n\n"
            "Rules: base every statement ONLY on the pathway names listed above and name the "
            "specific pathway(s) supporting each point; phrase interpretations as hypotheses; "
            "do NOT discuss clinical outcomes, treatment, immunotherapy response, or prognosis; "
            "do NOT fabricate effect sizes, p-values or counts.\n"
            'Return JSON: {"bullets":["...","...","..."],'
            '"note":"one concise, evidence-grounded note on what these pathways suggest and what would confirm it."}'
        ),
        max_tokens=320,
    )

    bullets = colorize_list(j.get("bullets", [])[:5], ct_colors)
    note = colorize_text(clean_line(j.get("note", "")), ct_colors)

    display = display.drop(
        columns=[c for c in display.columns if c == "__score"], errors="ignore"
    )
    return display, bullets, note


def _subset_genes_string(genes_str: str, max_genes: int = 8) -> str:
    """
    Take a long 'genes' string and keep only first N genes as a subset.
    """
    if not isinstance(genes_str, str):
        return ""
    # split on commas, semicolons, or whitespace slashes
    tokens = re.split(r"[,\s;/]+", genes_str.strip())
    tokens = [t for t in tokens if t]
    return ", ".join(tokens[:max_genes])


def summarize_celltype_pathways(
    pathways_combined_dir: Path,
    top_markers_per_ct: Dict[str, List[str]],
    disease: str,
    biosample: str,
    ct_colors: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Per-cluster/cell-type pathway summary.
    For each *combined_pathways_dedup.csv, extract top 3 pathways
    and link them to that cell type and its markers.
    Shows a subset of genes per pathway. p-values are used internally
    but not displayed in the final report.
    """
    items: List[Dict[str, Any]] = []
    if not pathways_combined_dir.exists():
        return items

    csvs = _dedup_pathway_csvs(pathways_combined_dir)
    if not csvs:
        csvs = sorted(pathways_combined_dir.glob("*combined_pathways*.csv"))
    if not csvs:
        return items

    for f in csvs:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e, exc_info=True)
            continue
        if df.empty or "pathway" not in df.columns:
            continue

        stem = f.stem  # e.g. single_dataset_cluster_0_Tcell_combined_pathways_DEDUP
        label = (
            stem.replace("single_dataset_cluster_", "")
            .replace("_combined_pathways_DEDUP", "")
            .replace("_combined_pathways_RAW", "")
        )
        pretty_label = label.replace("_", " ")

        # choose best score
        if "p_value_adj" in df.columns:
            df = df.sort_values("p_value_adj", ascending=True)
        elif "combined_score" in df.columns:
            df = df.sort_values("combined_score", ascending=False)

        top_df = df.head(3).copy()
        top_list = []
        for r in top_df.to_dict("records"):
            genes_sub = _subset_genes_string(str(r.get("genes", "")), max_genes=8)
            if "p_value_adj" in r and pd.notnull(r["p_value_adj"]):
                try:
                    adj_val = float(r["p_value_adj"])
                    adj_str = f"{adj_val:.2e}"  # still used for LLM context only
                except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
                    logger.debug("%s: falling back after %r", __name__, exc)
                    adj_str = ""
            else:
                adj_str = ""
            top_list.append(
                {
                    "db": str(r.get("biological_database", "")),
                    "name": str(r.get("pathway", "")),
                    "adj_p": adj_str,
                    "genes": genes_sub,
                }
            )

        # match markers
        ct_guess = label.split("_", 1)[-1]
        markers = (
            top_markers_per_ct.get(ct_guess)
            or top_markers_per_ct.get(pretty_label)
            or []
        )

        j = llm_json(
            "You are a single-cell RNA-seq data analyst. Return ONLY JSON.",
            textwrap.dedent(
                f"""
                You are interpreting cluster/cell-type–specific pathway enrichment results
                from a single-cell RNA-seq study.

                Biosample / tissue context: {biosample}
                Cell type / cluster label: "{pretty_label}"

                Top enriched pathways (database, name, adj_p, subset of genes):
                {top_list}

                Known marker genes for this cell type:
                {markers}

                Task:
                  • Provide 3–5 bullets on what these pathways and markers suggest about the
                    functional state of this cell population.
                  • Ground every bullet in a SPECIFIC pathway name or marker gene listed above;
                    do not introduce programs that are not represented in the data.
                  • Phrase interpretations as hypotheses ("may", "is consistent with").
                  • Do NOT discuss clinical outcomes, treatment, immunotherapy response or
                    resistance, prognosis, or patient-specific implications.
                  • Do NOT invent numeric percentages, p-values or hazard ratios.

                Return JSON:
                {{
                  "bullets": ["bullet1","bullet2","bullet3","bullet4","bullet5"],
                  "note": "single concise, evidence-grounded note naming the pathways/markers behind the interpretation."
                }}
                """
            ),
            max_tokens=420,
        )

        bullets = colorize_list(j.get("bullets", [])[:5], ct_colors)
        note = colorize_text(clean_line(j.get("note", "")), ct_colors)

        items.append(
            {
                "cell_label": label,
                "pretty_label": pretty_label,
                "top_pathways": top_list,
                "bullets": bullets,
                "note": note,
            }
        )

    return items


# ==========================
# === HTML Template
# ==========================

HTML = Template(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Single-Cell Transcriptomic Report — {{ case_id }}</title>
<style>
:root{
  --bg:#f7f8ff;--card:#ffffff;--ink:#0b1020;--muted:#4b5a8a;--edge:#e3e7ff;
  --accent:#6c5ce7;--glow1:#6c8cff;--glow2:#b06afc;--radius:16px;--maxw:1400px
}
html,body{
  margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  line-height:1.55;
}
.container{max-width:var(--maxw);margin:0 auto;padding:22px 16px}
header{padding:18px 16px 12px;background:#fff;border-bottom:1px solid var(--edge)}
.header-inner{max-width:var(--maxw);margin:0 auto}
.brand{font-weight:800;letter-spacing:.3px;margin-bottom:2px}
h1{margin:4px 0 6px;font-size:24px;letter-spacing:.2px}
.muted{color:var(--muted);font-size:13px}
.pill{
  display:inline-block;padding:4px 8px;border-radius:999px;background:#eef1ff;
  border:1px solid var(--edge);color:#37417a;margin:0 4px;font-size:12px
}
.card{
  background:var(--card);border:1px solid var(--edge);border-radius:var(--radius);
  padding:18px;margin:0 0 16px;box-shadow:0 8px 22px rgba(16,23,51,.10);
  page-break-inside:avoid;break-inside:avoid
}
.page-break-before{
  page-break-before: always;
  break-before: page;
}
h2{
  color:var(--accent);margin:0 0 10px;font-size:18px;letter-spacing:.2px;
  position:relative;page-break-after:avoid;break-after:avoid
}
h2::after{
  content:"";position:absolute;left:0;bottom:-6px;width:48px;height:2px;
  background:linear-gradient(90deg,var(--glow1),var(--glow2));border-radius:2px;opacity:.65
}
.table-wrap{
  overflow:auto;border:1px solid var(--edge);border-radius:12px;
  page-break-inside:avoid;break-inside:avoid
}
table{width:100%;border-collapse:collapse;background:#fff}
th,td{
  border-bottom:1px dashed #dfe5ff;padding:8px 10px;font-size:13px;vertical-align:top
}
th{text-align:left;color:#3a4688;background:#f8f9ff;position:sticky;top:0}
.img-frame{
  border:1px solid var(--edge);border-radius:12px;overflow:hidden;background:#fff;
  page-break-inside:avoid;break-inside:avoid
}
.img-frame img{
  width:100%;height:auto;display:block;image-rendering:auto;
  page-break-inside:avoid;break-inside:avoid
}
.caption{color:#5b69a8;font-size:12px;margin-top:6px}
.panel{
  background:#f3f5ff;border:1px solid #dbe2ff;border-radius:12px;
  padding:12px;margin-top:10px;page-break-inside:avoid;break-inside:avoid
}
.panel h3{margin:0 0 6px;font-size:15px;color:#4957a8}
ul{margin:6px 0 0 18px}
.ct-tag{
  display:inline-block;padding:2px 8px;border-radius:999px;color:#0b1020;
  border:1px solid rgba(0,0,0,.06);margin:0 2px
}

/* annotation-confidence summary tiles + warning panel */
.tier-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0
}
.tier-tile{
  border:1px solid var(--edge);border-radius:10px;padding:8px 10px;background:#fff;
  page-break-inside:avoid;break-inside:avoid
}
.tier-tile .n{font-size:20px;font-weight:700;line-height:1.1}
.tier-tile .k{font-size:11px;color:#5b69a8;text-transform:uppercase;letter-spacing:.3px}
.tier-high{border-left:4px solid #2e9e6b}
.tier-medium{border-left:4px solid #d79a1e}
.tier-low{border-left:4px solid #c8483c}
.tier-mixed{border-left:4px solid #8a5cd1}
.tier-excluded{border-left:4px solid #7b8398}
.warn-panel{
  background:#fff6f5;border:1px solid #f0c8c3;border-radius:12px;padding:12px;margin-top:10px;
  page-break-inside:avoid;break-inside:avoid
}
.warn-panel h3{margin:0 0 6px;font-size:15px;color:#a83b30}

/* two col */
.two-col{display:grid;grid-template-columns:1.05fr 1.35fr;gap:14px}
@media (max-width:900px){.two-col{grid-template-columns:1fr}}

/* Logo header */
.header-flex{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  page-break-inside:avoid;break-inside:avoid
}
.header-center{flex:1 1 auto;text-align:center;min-width:0}
.header-logo{
  height:50px;max-width:220px;object-fit:contain;flex:0 0 auto;
  page-break-inside:avoid;break-inside:avoid
}
.header-center .muted{
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}

/* Print tweaks */
@media print{
  html,body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .card{box-shadow:none;border:1px solid #dfe5ff}
  a[href]:after{content:""!important}
  .two-col{display:block;grid-template-columns:none}
  .two-col>div{page-break-inside:avoid;break-inside:avoid}
  .img-frame,.panel,.table-wrap,.figure-block,.gallery-card{
    page-break-inside:avoid;break-inside:avoid
  }
  h2,.panel h3,.caption{page-break-after:avoid;break-after:avoid}
  img{page-break-inside:avoid;break-inside:avoid}
}

/* Page size for headless browser */
@page{
  size: {{ page_size }};
  margin: {{ pdf_margin }}mm;
}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="header-flex">
      {% if left_logo %}
        <img class="header-logo" src="{{ left_logo }}" alt="Left Logo"/>
      {% else %}
        <div style="width:50px"></div>
      {% endif %}
      <div class="header-center">
        <div class="brand">Ayass Bioscience</div>
        <h1>Single-Cell Transcriptomic Profiling Report</h1>
        <div class="muted">
          Case: <strong>{{ case_id }}</strong>
          · GEO: <strong>{{ accession }}</strong>
          {% if sample_accession %}
            · Sample: <strong>{{ sample_accession }}</strong>{% if sample_title %} — {{ sample_title }}{% endif %}
          {% endif %}
        </div>
        <div class="muted" style="margin-top:4px;">
          Disease: <strong>{{ disease }}</strong>
          {% if n_cells %} · Cells (post-QC): <strong>{{ n_cells }}</strong>{% endif %}
          {% if n_clusters %} · Clusters: <strong>{{ n_clusters }}</strong>{% endif %}
          {% if n_celltypes %} · Cell types: <strong>{{ n_celltypes }}</strong>{% endif %}
        </div>
      </div>
      {% if right_logo %}
        <img class="header-logo" src="{{ right_logo }}" alt="Right Logo"/>
      {% else %}
        <div style="width:50px"></div>
      {% endif %}
    </div>
  </div>
</header>

<div class="container">

  <div class="card">
    <h2>Dataset & Study Context</h2>
    <p><strong>Series title:</strong> {{ title }}</p>
    {% if sample_accession %}
      <p><strong>Sample:</strong> {{ sample_accession }}{% if sample_title %} — {{ sample_title }}{% endif %}</p>
    {% endif %}
    <p><strong>GEO Accession:</strong> {{ accession }} &nbsp; · &nbsp;
       <strong>Taxon:</strong> {{ taxon }} &nbsp; · &nbsp;
       <strong>Type:</strong> {{ gdstype }}</p>
    <p>{{ dataset_context|safe }}</p>
  </div>

  <div class="card page-break-before">
    <h2>Preprocessing & Quality Control Pipeline</h2>
    <ul>
      {% for b in preproc_bullets %}
        <li>{{ b }}</li>
      {% endfor %}
    </ul>
    {% if qc_cells_line %}
      <p class="caption"><strong>Key QC metrics:</strong> {{ qc_cells_line }}</p>
    {% endif %}
    {% if qc_gallery %}
      <div class="two-col" style="margin-top:10px;">
        {% for fig in qc_gallery %}
          <div class="img-frame figure-block" style="margin-bottom:10px;">
            <img src="{{ fig.src }}" alt="{{ fig.caption }}"/>
            <div class="caption"><strong>{{ fig.caption }}</strong></div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>

  <div class="card page-break-before">
    <h2>Embeddings (t-SNE, UMAP) & Clustering</h2>
    {% if embedding_bullets and embedding_bullets|length > 0 %}
      <ul>
        {% for b in embedding_bullets %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
    {% endif %}
    {% if embed_gallery %}
      <div class="two-col">
        {% for fig in embed_gallery %}
          <div class="img-frame figure-block" style="margin-bottom:10px;">
            <img src="{{ fig.src }}" alt="{{ fig.caption }}"/>
            <div class="caption"><strong>{{ fig.caption }}</strong></div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>

  <div class="card page-break-before">
    <h2>Cell-Type Landscape</h2>
    <p><strong>Dominant cell types:</strong> {{ top_celltypes_html|safe }}</p>
    {% if celltype_barplot %}
      <div class="img-frame figure-block">
        <img src="{{ celltype_barplot }}" alt="Cell-type composition barplot"/>
      </div>
    {% endif %}
    {% if celltype_umap %}
      <div class="img-frame figure-block" style="margin-top:10px;">
        <img src="{{ celltype_umap }}" alt="UMAP colored by cell type"/>
      </div>
    {% endif %}
    {% if celltype_bullets and celltype_bullets|length > 0 %}
      <div class="panel">
        <h3>Cell-type interpretation</h3>
        <ul>
          {% for b in celltype_bullets %}
            <li>{{ b|safe }}</li>
          {% endfor %}
        </ul>
        <p class="caption"><strong>Interpretation (hypothesis):</strong> {{ celltype_note|safe }}</p>
      </div>
    {% endif %}
  </div>

  {% if consensus_summary %}
  <div class="card page-break-before">
    <h2>Annotation Confidence &amp; Consensus</h2>
    <p class="caption">
      Cell identity is assigned per Leiden cluster by majority vote across the enabled
      annotators, gated against canonical lineage markers. Tiers are NOT equally
      certain: <strong>High</strong> = all voters agreed and the lineage gate concurred;
      <strong>Medium</strong> = a majority agreed; <strong>Low/Review</strong> = the
      voters split or the lineage gate contradicted them, so that label is a hypothesis
      requiring manual review, not a result.
    </p>

    <div class="tier-grid">
      <div class="tier-tile tier-high"><div class="n">{{ consensus_summary.high }}</div>
        <div class="k">High-confidence clusters</div></div>
      <div class="tier-tile tier-medium"><div class="n">{{ consensus_summary.medium }}</div>
        <div class="k">Medium-confidence clusters</div></div>
      <div class="tier-tile tier-low"><div class="n">{{ consensus_summary.low_review }}</div>
        <div class="k">Low/Review clusters</div></div>
      <div class="tier-tile tier-mixed"><div class="n">{{ consensus_summary.mixed }}</div>
        <div class="k">Mixed clusters</div></div>
      <div class="tier-tile tier-excluded"><div class="n">{{ consensus_summary.excluded }}</div>
        <div class="k">Excluded clusters</div></div>
      <div class="tier-tile"><div class="n">{{ consensus_summary.unassigned }}</div>
        <div class="k">Unassigned clusters</div></div>
    </div>

    {% if consensus_warnings and consensus_warnings|length > 0 %}
      <div class="warn-panel">
        <h3>Annotation caveats</h3>
        <ul>
          {% for w in consensus_warnings %}<li>{{ w }}</li>{% endfor %}
        </ul>
      </div>
    {% endif %}

    {% if consensus_table %}
      <h3 style="margin-top:14px;font-size:15px;color:#4957a8">Per-cluster annotation detail</h3>
      <p class="caption">
        One row per Leiden cluster: the final consensus label, its tier, every
        individual voter's call, whether the voters disagreed, whether CellTypist found
        the cluster heterogeneous, and whether the cluster entered the inferential
        (DE / composition) analyses.
      </p>
      <div class="table-wrap">{{ consensus_table|safe }}</div>
    {% endif %}

    {% if annotation_resources and annotation_resources|length > 0 %}
      <div class="panel">
        <h3>Annotation resources actually used</h3>
        <div class="table-wrap">
          <table>
            <tbody>
            {% for k, v in annotation_resources %}
              <tr><th style="width:34%">{{ k }}</th><td>{{ v }}</td></tr>
            {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    {% endif %}
  </div>
  {% endif %}

  <div class="card page-break-before">
    <h2>Marker Genes by Cell Type</h2>
    {% if markers_topline_items_html and markers_topline_items_html|length > 0 %}
      <ul>
        {% for line in markers_topline_items_html %}
          <li>{{ line|safe }}</li>
        {% endfor %}
      </ul>
    {% endif %}
    {% if markers_table %}
      <div class="table-wrap">{{ markers_table|safe }}</div>
    {% endif %}
    <div class="panel">
      <h3>Marker-based biological interpretation</h3>
      <ul>
        {% for b in markers_bullets_html %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
      <p class="caption"><strong>Interpretation (hypothesis):</strong> {{ markers_note_html|safe }}</p>
    </div>
  </div>

  {% if pathways_table %}
  <div class="card page-break-before">
    <h2>Global Pathway Programs & Cell-State Shifts</h2>
    <p class="caption">
      Top enriched pathways across all clusters (maximum 2 pathways per database), highlighting dominant signalling
      programs across the profiled cell populations. p-values are used for ranking but not shown here.
    </p>
    <div class="table-wrap">{{ pathways_table|safe }}</div>
    <div class="panel">
      <h3>Pathway interpretation (global view)</h3>
      <ul>
        {% for b in pathways_bullets_html %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
      <p class="caption"><strong>Interpretation (hypothesis):</strong> {{ pathways_note_html|safe }}</p>
    </div>
  </div>
  {% endif %}

  {% if celltype_pathways and celltype_pathways|length > 0 %}
  <div class="card page-break-before">
    <h2>Cell-Type–Specific Pathway Programs</h2>
    <p>
      For each major cell population, pathway enrichment of its defining marker genes highlights functional programs
      such as cytotoxicity, metabolic activity, and stromal activation across the profiled cell populations.
      Only the top pathways per population are shown.
    </p>
  </div>

  {% for ct in celltype_pathways %}
  <div class="card">
    <h2>{{ ct.pretty_label }}</h2>
    <div class="table-wrap">
      <table>
        <tr>
          <th>Biological Database</th>
          <th>Pathway</th>
          <th>Genes (subset)</th>
        </tr>
        {% for p in ct.top_pathways %}
        <tr>
          <td>{{ p.db }}</td>
          <td>{{ p.name }}</td>
          <td>{{ p.genes }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
    <div class="panel">
      <h3>Functional interpretation</h3>
      <ul>
        {% for b in ct.bullets %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
      <p class="caption"><strong>Interpretation (hypothesis):</strong> {{ ct.note|safe }}</p>
    </div>
  </div>
  {% endfor %}
  {% endif %}

  <div class="card page-break-before">
    <h2>Key Takeaways & Summary</h2>
    {% if key_takeaways_html and key_takeaways_html|length > 0 %}
    <ul>
      {% for b in key_takeaways_html %}
        <li>{{ b|safe }}</li>
      {% endfor %}
    </ul>
    {% endif %}
    <p style="margin-top:10px;">
      {{ clinical_conclusion_html|safe }}
    </p>
  </div>

</div>

</body>
</html>
""")


# ======================================
# === HTML → PDF via headless browser
# ======================================


def _file_url(path: Path) -> str:
    return urljoin("file:", quote(str(path.resolve()).replace("\\", "/")))


def _find_browser_exe() -> Optional[str]:
    candidates = []
    if sys.platform.startswith("win"):
        candidates += [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
    elif sys.platform == "darwin":
        candidates += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            shutil.which("chrome"),
            shutil.which("google-chrome"),
            shutil.which("msedge"),
            shutil.which("chromium"),
        ]
    else:
        candidates += [
            shutil.which("google-chrome"),
            shutil.which("chrome"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
            shutil.which("msedge"),
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _timestamped_pdf_path(outdir: Path, case_id: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pdf_name = f"Single-Cell Scanpy — Ayass Bioscience Report — {case_id} — {ts}.pdf"
    return outdir / pdf_name


def write_pdf_via_browser(html_path: Path, pdf_path: Path) -> bool:
    """
    Generate PDF using Chrome/Edge headless browser.
    Tries Playwright first (supports disabling headers/footers), then falls back to Chrome CDP.
    """
    file_url = _file_url(html_path)

    # Try Playwright first (most reliable way to disable headers/footers)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser_instance = p.chromium.launch(headless=True)
            page = browser_instance.new_page()
            page.goto(file_url, wait_until="networkidle", timeout=30000)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                margin={
                    "top": "12mm",
                    "bottom": "12mm",
                    "left": "12mm",
                    "right": "12mm",
                },
                print_background=True,
                display_header_footer=False,
            )
            browser_instance.close()
            if pdf_path.exists():
                return True
    except ImportError:
        logger.info(
            "Playwright not installed. Install with: pip install playwright && "
            "playwright install chromium"
        )
        logger.info("Falling back to Chrome print-to-pdf (may include headers/footers)")
    except Exception as e:
        logger.warning("Playwright PDF export failed: %s", e, exc_info=True)
        logger.info("Falling back to Chrome print-to-pdf (may include headers/footers)")

    # Fallback: Use Chrome's basic print-to-pdf (will include headers/footers)
    browser = _find_browser_exe()
    if not browser:
        logger.warning("No Chrome/Edge found. Install one for PDF export.")
        return False

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--disable-print-preview",
        f"--print-to-pdf={str(pdf_path)}",
        file_url,
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )
        if pdf_path.exists():
            logger.info("PDF generated. Note: Browser headers/footers may appear.")
            logger.info(
                "To remove headers/footers, install Playwright: pip install "
                "playwright && playwright install chromium"
            )
            return True
    except Exception as e:
        logger.warning("Browser PDF export failed: %s", e, exc_info=True)

    return False


# =====================
# === Logo helpers
# =====================


def _find_logo_files(dirpath: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not dirpath or not dirpath.exists():
        return None, None
    left = dirpath / LOGO_FILENAMES_PREFERRED[0]
    right = dirpath / LOGO_FILENAMES_PREFERRED[1]
    if left.exists() and right.exists():
        return left, right

    imgs = [
        p for p in dirpath.iterdir() if p.suffix.lower() in LOGO_EXTS and p.is_file()
    ]

    left_guess = None
    right_guess = None
    for p in imgs:
        n = p.name.lower()
        if ("left" in n or "ayass" in n) and left_guess is None:
            left_guess = p
        if ("right" in n or "logo" in n) and right_guess is None:
            right_guess = p
    if not left_guess and imgs:
        imgs_sorted = sorted(imgs, key=lambda x: x.name.lower())
        left_guess = imgs_sorted[0]
        if len(imgs_sorted) > 1:
            right_guess = imgs_sorted[1]
    return left_guess, right_guess


def load_logos(logos_dir: Optional[Path]) -> Tuple[Optional[str], Optional[str]]:
    """Return (left, right) header logos as base64 data URIs, falling back to the default logos dir."""
    if not logos_dir or not logos_dir.exists():
        logos_dir = DEFAULT_LOGOS_DIR
    l_path, r_path = _find_logo_files(logos_dir)
    left = b64_img(l_path) if l_path else None
    right = b64_img(r_path) if r_path else None
    return left, right


# ============================
# === Figure discovery helper
# ============================


def first_match(dirpath: Path, patterns: List[str]) -> Optional[Path]:
    """Return the first file in ``dirpath`` matching any glob in ``patterns`` (else ``None``)."""
    if not dirpath.exists():
        return None
    for pat in patterns:
        hits = sorted(dirpath.glob(pat))
        if hits:
            return hits[0]
    return None


# ==================
# === Builder main
# ==================


def build_singlecell_report(
    sc_root: Path,
    geo_json_path: Optional[Path] = None,
    case_id: str = "",
    logos_dir: Optional[Path] = None,
) -> None:
    """
    sc_root: SC_RESULTS directory from your single-cell pipeline.
    geo_json_path: path to GSE*_metadata.json (or similar). Optional - if None, report will be generated without GEO metadata.
    case_id: the sample/case identifier used in the report (e.g. GSM6360688).
    """
    outdir = sc_root / "singlecell_report"
    ensure_dir(outdir)

    # ---- GEO metadata ----
    geo_meta = {}
    if geo_json_path is not None:
        geo_meta = read_json(geo_json_path) or {}
    es = geo_meta.get("esummary_raw", {})
    accession = geo_meta.get("accession") or es.get("accession", "")
    title = geo_meta.get("title") or es.get("title", "")
    taxon = geo_meta.get("taxon") or es.get("taxon", "")
    gdstype = geo_meta.get("gdstype") or es.get("gdstype", "")

    # Sample-level info from GEO JSON based on case_id
    sample_entry = _find_sample_entry(geo_meta, case_id)
    sample_accession = ""
    sample_title = ""
    if sample_entry:
        sample_accession = str(sample_entry.get("accession", "")).strip()
        sample_title = str(sample_entry.get("title", "")).strip()

    disease, biosample, dataset_context = infer_dataset_context_from_geo(
        geo_meta,
        sample_accession=sample_accession or case_id,
        sample_title=sample_title,
    )

    # ---- Scanpy summary ----
    summary_dir = sc_root / "00_analysis_summary"
    qc_dir = sc_root / "01_qc_and_filtering"
    dim_dir = sc_root / "03_dimensionality_reduction_and_embeddings"
    clust_dir = sc_root / "04_clustering_and_cell_states"
    ct_anno_dir = sc_root / "05_celltype_analysis" / "celltype_annotation"
    ct_markers_dir = sc_root / "05_celltype_analysis" / "celltype_specific_markers"
    pathways_combined_dir = (
        sc_root
        / "07_pathway_enrichment"
        / "cluster_marker_enrichment"
        / "pathways"
        / "combined"
    )

    analysis_name, kv, ct_counts = load_analysis_summary(summary_dir)

    # Parse key metrics
    n_cells = 0
    n_clusters = 0
    if "Cells after QC filters" in kv:
        try:
            n_cells = int(kv["Cells after QC filters"])
        except (TypeError, ValueError) as exc:
            logger.debug("unparseable 'Cells after QC filters': %s", exc)
    if "Leiden clusters" in kv:
        try:
            n_clusters = int(kv["Leiden clusters"])
        except (TypeError, ValueError) as exc:
            logger.debug("unparseable 'Leiden clusters': %s", exc)
    n_celltypes = len(ct_counts) if ct_counts else 0

    # Preprocessing bullets (1–7 style)
    preproc_bullets = build_preproc_bullets_from_kv(
        kv, disease=disease, biosample=biosample
    )

    # Short QC metrics line
    qc_bits = []
    for key_label in [
        ("Initial cells", "initial cells"),
        ("Initial genes", "initial genes"),
        ("Genes after min_cells filter", "genes after min_cells filter"),
        ("Cells after QC filters", "cells after QC filters"),
        ("HVGs used", "highly variable genes"),
    ]:
        k, label = key_label
        if k in kv:
            qc_bits.append(f"{label}: {kv[k]}")
    qc_cells_line = " · ".join(qc_bits)

    # ---- Cell-type colors & top line ----
    ct_colors = build_ct_colors_from_counts(ct_counts)
    top_ct_line = top_celltypes_line(ct_counts)
    top_celltypes_html = colorize_text(top_ct_line, ct_colors)

    # ---- Annotation confidence / consensus section ----
    # Surfaces the per-cluster voter calls, tiers, mixed-cluster flags and downstream
    # inclusion that the pipeline already computes. Without this the report presents a
    # Low/Review label with the same authority as a unanimous one.
    consensus_df, consensus_summary = read_consensus_annotation(ct_anno_dir)
    run_manifest = read_run_manifest(sc_root)
    annotation_resources = annotation_resource_rows(run_manifest)
    consensus_table = None
    consensus_warnings: List[str] = []
    if consensus_df is not None and not consensus_df.empty:
        consensus_table = consensus_df.to_html(index=False, border=0)
        s = consensus_summary
        if s.get("low_review"):
            _cells = s.get("n_cells_low_review")
            consensus_warnings.append(
                f"{s['low_review']} of {s['n_clusters']} clusters are Low/Review"
                + (f" ({_cells} cells)" if _cells else "")
                + " — the annotators disagreed or the lineage gate contradicted them. "
                "Treat those labels as hypotheses pending manual review."
            )
        if s.get("mixed"):
            consensus_warnings.append(
                f"{s['mixed']} cluster(s) are heterogeneous: CellTypist's per-cell "
                "predictions were split, so the single cluster label may be averaging "
                "over more than one cell type. Consider a finer clustering resolution "
                "for those clusters."
            )
        if s.get("excluded"):
            consensus_warnings.append(
                f"{s['excluded']} cluster(s)"
                + (
                    f" ({s.get('n_cells_excluded')} cells)"
                    if s.get("n_cells_excluded")
                    else ""
                )
                + " were excluded from differential expression and composition "
                "comparisons by the confidence filter. They remain in the exported "
                "h5ad and in every audit table."
            )
        if s.get("unassigned"):
            consensus_warnings.append(
                f"{s['unassigned']} cluster(s) could not be assigned a cell type."
            )
        if s.get("disagreeing") and not s.get("low_review"):
            consensus_warnings.append(
                f"{s['disagreeing']} cluster(s) had at least one dissenting voter."
            )

    # ---- QC figures (violin + hist) ----
    qc_violin = first_match(
        qc_dir, ["*qc_violin*.png", "violin*qc*.png", "*qc_violin*.pdf"]
    )
    qc_hist = first_match(qc_dir, ["*qc_metric_histograms*.png", "*histograms*.png"])
    qc_gallery = []
    if qc_violin:
        qc_gallery.append(
            {
                "src": b64_img(qc_violin),
                "caption": "QC violin — genes, counts, mitochondrial %",
            }
        )
    if qc_hist:
        qc_gallery.append({"src": b64_img(qc_hist), "caption": "QC metric histograms"})

    # ---- Embeddings & clustering figures ----
    # Lowercase first: the figure is written as `*_pca_variance_explained.png` since
    # the Rule 5.1 rename. `Path.glob` is case-sensitive on Linux, so an uppercase-only
    # pattern list would find nothing in deployment while still matching on Windows.
    pca_plot = first_match(
        dim_dir,
        [
            "*pca*variance*.png",
            "*pca*.png",
            "*PCA*variance*.png",  # pre-rename runs
            "*PCA*.png",
            "*pc_variance*.png",
        ],
    )
    umap_leiden = first_match(clust_dir, ["umap*leiden*.png", "*UMAP_leiden*.png"])
    tsne_leiden = first_match(clust_dir, ["tsne*leiden*.png", "*TSNE_leiden*.png"])
    umap_samples = first_match(
        dim_dir,
        ["umap*UMAP_samples_groups*.png", "umap*group*.png", "*UMAP_samples*.png"],
    )

    embed_gallery = []
    if pca_plot:
        embed_gallery.append(
            {
                "src": b64_img(pca_plot),
                "caption": "PCA embedding of major cell populations",
            }
        )
    if umap_samples:
        embed_gallery.append(
            {"src": b64_img(umap_samples), "caption": "UMAP coloured by sample/group"}
        )
    if umap_leiden:
        embed_gallery.append(
            {"src": b64_img(umap_leiden), "caption": "UMAP coloured by Leiden clusters"}
        )
    if tsne_leiden:
        embed_gallery.append(
            {
                "src": b64_img(tsne_leiden),
                "caption": "t-SNE coloured by Leiden clusters",
            }
        )

    # LLM: embeddings & clustering summary as bullet points
    embedding_j = llm_json(
        "You are a single-cell RNA-seq data analyst. Return ONLY JSON.",
        (
            f"Describe, in 3–5 bullet points, how t-SNE and UMAP embeddings together with Leiden clustering "
            f"resolve discrete and transitional cell states in this '{biosample}' dataset. "
            "Describe separation of the annotated cell populations and any transitional gradients. "
            "Do NOT discuss clinical outcomes, treatment, or prognosis; "
            "do NOT discuss PCA variance structure or exact numerical percentages.\n\n"
            'Return JSON: {"bullets":["...","...","..."]}'
        ),
        max_tokens=320,
    )
    embedding_bullets_raw = clean_list(embedding_j.get("bullets", [])[:5])
    embedding_bullets = colorize_list(embedding_bullets_raw, ct_colors)

    # ---- Cell-type figures ----
    celltype_barplot = first_match(
        ct_anno_dir, ["*celltype_composition_barplot*.png", "*celltype*barplot*.png"]
    )
    celltype_umap = first_match(
        ct_anno_dir, ["umap*celltype*.png", "*UMAP_celltypes*.png"]
    )

    # LLM: cell-type landscape summary
    celltype_summary_j = llm_json(
        "You are a single-cell RNA-seq data analyst. Return ONLY JSON.",
        (
            f"Summarize the cell-type landscape for biosample '{biosample}'. "
            f"Dominant cell types string: {top_ct_line or 'NA'}.\n"
            "Phrase interpretations as hypotheses; do NOT discuss clinical outcomes, treatment, or prognosis.\n"
            'Return JSON: {"bullets":["...","...","..."],'
            '"note":"one sentence on the overall cell-type composition observed in this dataset."}\n'
            "Avoid fabricated fractions; you may refer qualitatively to dominant cell types."
        ),
        max_tokens=260,
    )
    celltype_bullets = colorize_list(
        celltype_summary_j.get("bullets", [])[:4], ct_colors
    )
    celltype_note = colorize_text(
        clean_line(celltype_summary_j.get("note", "")), ct_colors
    )

    # ---- Markers ----
    markers_all_path = ct_markers_dir / "celltype_marker_genes_celltype_all.csv"
    if not markers_all_path.is_file():  # pre-rename runs
        markers_all_path = ct_markers_dir / "celltype_marker_genes_celltype_ALL.csv"
    (
        markers_display,
        markers_topline,
        markers_bullets,
        markers_note,
        top_markers_per_ct,
    ) = prepare_sc_markers_table(
        markers_all_path,
        disease=disease,
        biosample=biosample,
        ct_colors=ct_colors,
    )
    markers_table_html = (
        markers_display.to_html(index=False, border=0)
        if markers_display is not None
        else None
    )

    # Build bullet-style per-cell-type marker lines for the top section
    markers_topline_items_html: List[str] = []
    if markers_display is not None and not markers_display.empty:
        marker_lines = [
            f"{row['cell_type']}: {row['markers']}"
            for row in markers_display.to_dict("records")
        ]
        markers_topline_items_html = colorize_list(marker_lines, ct_colors)

    markers_bullets_html = colorize_list(markers_bullets, ct_colors)
    markers_note_html = colorize_text(markers_note, ct_colors)

    # ---- Pathways: global summary ----
    pathways_table = None
    pathways_bullets_html: List[str] = []
    pathways_note_html = ""
    pw_display, pw_bullets, pw_note = summarize_pathways(
        pathways_combined_dir,
        disease=disease,
        biosample=biosample,
        ct_colors=ct_colors,
    )

    if pw_display is not None:
        pathways_table = pw_display.to_html(index=False, border=0)
        pathways_bullets_html = colorize_list(pw_bullets, ct_colors)
        pathways_note_html = colorize_text(pw_note, ct_colors)

    # ---- Pathways: per cell-type/cluster ----
    celltype_pathways = summarize_celltype_pathways(
        pathways_combined_dir=pathways_combined_dir,
        top_markers_per_ct=top_markers_per_ct,
        disease=disease,
        biosample=biosample,
        ct_colors=ct_colors,
    )

    # ---- Global key takeaways & clinical conclusion ----
    ct_names = sorted(top_markers_per_ct.keys())
    ct_prog_text = "; ".join(
        f"{ct['pretty_label']}: " + ", ".join(p["name"] for p in ct["top_pathways"])
        for ct in celltype_pathways
    )[:2500]

    summary_j = llm_json(
        "You are a single-cell RNA-seq data analyst. Return ONLY JSON.",
        textwrap.dedent(
            f"""
            You are summarising a Scanpy-based single-cell RNA-seq analysis.

            Biosample: {biosample}
            GEO title: {title}
            Sample accession: {sample_accession or case_id}
            Sample title: {sample_title or "NA"}

            Annotated cell types (from markers): {ct_names}
            Cell-type–specific pathway programs (name → pathways):
            {ct_prog_text}

            Write:
              1) 4–7 high-level bullet points capturing:
                 • which cell types are most prominent in this dataset,
                 • which pathway programs are enriched in which cell populations —
                   naming the SPECIFIC pathways listed above as support.
              2) One concise paragraph summarising the cell types, pathways and markers observed.

            Rules:
              • Base every statement ONLY on the cell types and pathways provided above.
              • Phrase interpretations as hypotheses, not established findings.
              • Do NOT discuss clinical outcomes, treatment, immunotherapy response/resistance,
                combination regimens, prognosis, or patient-specific implications.
              • Do NOT invent numeric effect sizes, p-values, or patient counts.

            Return JSON:
            {{
              "bullets": ["...","...","..."],
              "conclusion": "final evidence-grounded summary paragraph (biology only, no clinical claims)."
            }}
            """
        ),
        max_tokens=520,
    )

    key_takeaways = clean_list(summary_j.get("bullets", [])[:8])
    key_takeaways_html = colorize_list(key_takeaways, ct_colors)
    clinical_conclusion_raw = strip_md(summary_j.get("conclusion", "") or "")
    clinical_conclusion_html = colorize_text(clinical_conclusion_raw, ct_colors)

    # ---- Logos ----
    left_logo, right_logo = load_logos(logos_dir)

    # ---- Render HTML ----
    html = HTML.render(
        case_id=case_id,
        accession=accession,
        title=title,
        taxon=taxon,
        gdstype=gdstype,
        disease=disease,
        biosample=biosample,
        dataset_context=dataset_context,
        n_cells=n_cells,
        n_clusters=n_clusters,
        n_celltypes=n_celltypes,
        preproc_bullets=preproc_bullets,
        qc_cells_line=qc_cells_line,
        qc_gallery=qc_gallery,
        embed_gallery=embed_gallery,
        embedding_bullets=embedding_bullets,
        celltype_barplot=b64_img(celltype_barplot) if celltype_barplot else None,
        celltype_umap=b64_img(celltype_umap) if celltype_umap else None,
        top_celltypes_html=top_celltypes_html,
        celltype_bullets=celltype_bullets,
        celltype_note=celltype_note,
        consensus_summary=(consensus_summary or None),
        consensus_table=consensus_table,
        consensus_warnings=consensus_warnings,
        annotation_resources=annotation_resources,
        markers_topline_items_html=markers_topline_items_html,
        markers_table=markers_table_html,
        markers_bullets_html=markers_bullets_html,
        markers_note_html=markers_note_html,
        pathways_table=pathways_table,
        pathways_bullets_html=pathways_bullets_html,
        pathways_note_html=pathways_note_html,
        celltype_pathways=celltype_pathways,
        key_takeaways_html=key_takeaways_html,
        clinical_conclusion_html=clinical_conclusion_html,
        left_logo=left_logo,
        right_logo=right_logo,
        page_size=PDF_PAGE_SIZE,
        pdf_margin=PDF_MARGIN_MM,
        sample_accession=sample_accession or case_id,
        sample_title=sample_title,
    )

    out_html = outdir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    logger.info("Single-cell report HTML written to: %s", out_html)

    pdf_path = _timestamped_pdf_path(outdir, case_id)
    if write_pdf_via_browser(out_html, pdf_path):
        logger.info("PDF written to: %s", pdf_path)
    else:
        logger.warning(
            "Could not create PDF. Ensure Chrome/Edge is installed and accessible."
        )


# ========= CLI =========


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments for the standalone report-generation CLI.

    Args:
        argv: Argument list; defaults to `sys.argv[1:]`.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Generate Ayass Bioscience LLM-interpreted single-cell Scanpy report (HTML + PDF)."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Path to SC_RESULTS folder (output of single-cell Scanpy pipeline).",
    )
    p.add_argument(
        "--geo-json",
        required=True,
        help="Path to GEO metadata JSON (e.g. GSE233203_metadata.json).",
    )
    p.add_argument(
        "--case-id",
        default=None,
        help="Case/sample label (e.g. GSM ID); also used to select sample context from GEO JSON when available.",
    )
    p.add_argument(
        "--logos-dir",
        default=None,
        help="Optional directory with logos (defaults to Ayass logo folder).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the report from command-line arguments.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success.
    """
    args = parse_args(argv)
    build_singlecell_report(
        sc_root=Path(args.root),
        geo_json_path=Path(args.geo_json),
        case_id=args.case_id,
        logos_dir=Path(args.logos_dir) if args.logos_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
