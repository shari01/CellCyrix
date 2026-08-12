"""
consensus.py — orchestrator for multi-method consensus annotation.

Runs, per Leiden cluster: markers (S2) -> CellTypist (S3) -> SingleR (S4, optional)
-> LLM annotator (S5) -> harmonize (S6) -> lineage gate (S7) -> reconcile (S8,
counting=tool, tie-break=agent) -> broadcast/validate/export (S9).

Invariants enforced here:
  * disease-agnostic  — only `tissue` is ever passed to annotators/prompts.
  * total-in == total-out — asserted in tools.broadcast_and_validate.
  * no silent failures — LLM transport errors are caught, logged, and degrade to
    a documented fallback; SingleR failure (when enabled) propagates loudly.
  * LLM only in the agent layer — this file calls agent.py for LLM work; all
    counting/gating/harmonization is in tools.py.

The single function the host pipeline calls is `run_consensus_annotation`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import tools
from .agent import (
    OpenRouterError,
    llm_adjudicate,
    llm_annotate_cluster,
    llm_infer_tissue,
    llm_select_celltypist_model,
    llm_select_singler_reference,
)
from .celltypist_catalog import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_SINGLER_REFERENCE,
    HUMAN_CELLTYPIST_MODELS,
    SINGLER_REFERENCES,
    VALID_MODEL_NAMES,
    VALID_SINGLER_REFERENCES,
)
from .config import ConsensusConfig, load_config

if TYPE_CHECKING:  # annotation only — anndata is not imported at module load
    from anndata import AnnData

logger = logging.getLogger(__name__)

CELLTYPE_CONSENSUS_COL = "celltype_consensus"
TIER_HIGH, TIER_MEDIUM, TIER_LOW = "High", "Medium", "Low/Review"
BRIDGE_SCRIPT = Path(__file__).parent / "singler_bridge.R"

# adata.uns key holding what was ACTUALLY resolved at run time (model/reference/
# tissue/species/marker ranking), so the provenance manifest and the report can
# record resolved values instead of the literal "auto" that was requested.
CONSENSUS_UNS_KEY = "celltype_consensus_run"

# CellTypist models that only know immune cells. These are the general fallback
# used when no tissue-specific model matches the study's organ. CellTypist cannot
# abstain, so such a model forces an immune label onto EVERY cluster — including
# epithelial/stromal ones. On a cluster the lineage gate calls non-immune, that
# vote is out-of-domain and is dropped (see the reconciliation loop below).
_IMMUNE_ONLY_CELLTYPIST = {"Immune_All_Low.pkl", "Immune_All_High.pkl"}


def _metadata_text(meta: dict[str, Any]) -> str:
    """Assemble title + summary + sample titles from a GEO metadata JSON for
    tissue inference (disease-agnostic downstream — the prompt forbids disease)."""
    es = (
        meta.get("esummary_raw", {})
        if isinstance(meta.get("esummary_raw"), dict)
        else {}
    )
    parts: List[str] = []
    for k in ("title", "summary"):
        v = meta.get(k) or es.get(k)
        if v:
            parts.append(f"{k}: {v}")
    samples = es.get("samples") or []
    titles = [str(s.get("title", "")) for s in samples if isinstance(s, dict)]
    titles = [t for t in titles if t]
    if titles:
        parts.append("sample_titles: " + "; ".join(titles)[:1500])
    return "\n".join(parts)


def _species_from_taxon(meta: dict[str, Any]) -> Optional[str]:
    """Deterministic species from the GEO 'taxon' field ('Homo sapiens'->human)."""
    es = (
        meta.get("esummary_raw", {})
        if isinstance(meta.get("esummary_raw"), dict)
        else {}
    )
    taxon = str(meta.get("taxon") or es.get("taxon") or "").lower()
    if "homo" in taxon or "sapiens" in taxon:
        return "human"
    if "mus" in taxon or "musculus" in taxon:
        return "mouse"
    return None


def _resolve_biocontext(cfg: ConsensusConfig, geo_json_path, _log) -> None:
    """Fill cfg.tissue / cfg.species from a GEO metadata JSON when they are unset
    or set to 'auto'. Species is deterministic (taxon); tissue is inferred by the
    LLM from title/summary/sample titles. Disease is never read. Any 'auto' that
    cannot be resolved is normalized to None so downstream fallbacks apply."""
    need_tissue = str(cfg.tissue or "").strip().lower() in {"", "auto"}
    need_species = str(cfg.species or "").strip().lower() in {"", "auto"}
    if not (need_tissue or need_species):
        return

    meta = None
    if geo_json_path:
        try:
            meta = json.loads(Path(geo_json_path).read_text(encoding="utf-8"))
        except Exception as e:
            _log(
                f"[CONSENSUS] could not read metadata JSON ({e}); biocontext stays auto->none.",
                level=logging.WARNING,
            )

    if need_species:
        cfg.species = _species_from_taxon(meta) if meta else None
        _log("[CONSENSUS] species (auto) -> %r", cfg.species)
    if need_tissue:
        inferred = None
        if meta is not None and cfg.enable_llm:
            inferred = llm_infer_tissue(cfg, _metadata_text(meta))
        cfg.tissue = inferred
        _log("[CONSENSUS] tissue (auto) -> %r", cfg.tissue)


def run_consensus_annotation(
    adata: AnnData,
    out_dir: str | Path,
    analysis_name: str,
    config: Optional[ConsensusConfig] = None,
    *,
    tissue: Optional[str] = None,
    enable_llm: bool = True,
    enable_singler: bool = False,
    cluster_col: str = "leiden",
    geo_json_path: str | Path | None = None,
) -> AnnData:
    """Annotate `adata` in place with consensus labels; returns `adata`.

    Adds obs columns: lineage_coarse, celltype_celltypist, [celltype_singler],
    celltype_knowledge_based, celltype_consensus, consensus_tier, annotation_provenance.
    Writes a per-cluster provenance CSV and a run log to `out_dir`.
    """
    cfg = config or load_config(
        enable_llm=enable_llm,
        enable_singler=enable_singler,
        tissue=tissue,
        cluster_col=cluster_col,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log: List[str] = []

    def _log(msg: str, *args: object, level: int = logging.INFO) -> None:
        """Log lazily and keep the rendered line for the run log.

        ``level`` is keyword-only: with lazy ``%s`` formatting a positional second
        argument is a format argument, and letting it bind to ``level`` would send
        a string where logging expects an int.
        """
        logger.log(level, msg, *args)
        run_log.append(msg % args if args else msg)

    # Auto-fill tissue/species from the study metadata JSON when unset/"auto".
    # These drive model/reference selection only; disease is never read.
    _resolve_biocontext(cfg, geo_json_path, _log)

    n_in = adata.n_obs
    _log(
        "[CONSENSUS] start | cells=%s | tissue=%r | species=%r | llm=%s | singler=%s | pubmed=%s",
        n_in,
        cfg.tissue,
        cfg.species,
        cfg.enable_llm,
        cfg.enable_singler,
        getattr(cfg, "enable_pubmed", False),
    )

    if cfg.cluster_col not in adata.obs.columns:
        raise ValueError(
            f"cluster column '{cfg.cluster_col}' not in adata.obs — cluster first."
        )
    clusters = sorted(
        adata.obs[cfg.cluster_col].astype(str).unique(), key=tools._natural_key
    )

    # Build a log-normalized working copy for ALL logic voters. This is robust to
    # scale-corrupted adata.raw and guarantees CellTypist gets valid input.
    adata_ln, ln_src = tools.get_lognorm(adata)
    _log("[CONSENSUS] annotation expression source: %s", ln_src)

    # Everything actually RESOLVED at run time (vs. what was requested as "auto").
    # Published to adata.uns[CONSENSUS_UNS_KEY] at the end so the provenance
    # manifest and the report can record real values, not the literal "auto".
    resolved: Dict[str, object] = {
        "requested_celltypist_model": cfg.celltypist_model,
        "resolved_celltypist_model": None,
        "requested_singler_reference": cfg.singler_reference,
        "resolved_singler_reference": None,
        "celltypist_enabled": bool(getattr(cfg, "enable_celltypist", True)),
        "singler_enabled": bool(cfg.enable_singler),
        "knowledge_based_enabled": bool(cfg.enable_llm),
        "pubmed_enabled": bool(getattr(cfg, "enable_pubmed", False)),
        "annotation_tissue": cfg.tissue,
        "annotation_species": cfg.species,
        "cluster_col": cfg.cluster_col,
        "expression_source": ln_src,
        "marker_ranking_method": tools.MARKER_RANKING_METHOD,
        "marker_min_detection_fraction": float(
            getattr(cfg, "min_detection_fraction", 0.10)
        ),
        "top_n_markers": int(cfg.top_n_markers),
        "mixed_cluster_min_dominant_fraction": float(
            getattr(cfg, "mixed_cluster_min_dominant_fraction", 0.70)
        ),
        "mixed_cluster_second_label_fraction": float(
            getattr(cfg, "mixed_cluster_second_label_fraction", 0.20)
        ),
        "use_subtypes_for_downstream": bool(
            getattr(cfg, "use_subtypes_for_downstream", False)
        ),
        "openrouter_model": cfg.openrouter_model if cfg.enable_llm else None,
        # Which curated reference actually backed this run's annotation. Both fall
        # back silently to the built-in tables if the reference data is absent, so
        # without this the run looks identical either way.
        "lineage_panel_source": tools.LINEAGE_MARKERS_PROVENANCE.get("source"),
        "lineage_panel_table": tools.LINEAGE_MARKERS_PROVENANCE.get("table"),
        "lineage_panel_fallback_reason": tools.LINEAGE_MARKERS_PROVENANCE.get("reason")
        or None,
        "lineage_panel_sizes": tools.LINEAGE_MARKERS_PROVENANCE.get("panel_sizes"),
        "lineage_min_score": tools.MIN_LINEAGE_SCORE,
        "label_resolver": (
            "cell_hierarchy (subtype-ref)"
            if tools._hierarchy() is not None
            else "keyword table only"
        ),
        "label_resolver_min_confidence": tools.MIN_RESOLVE_CONFIDENCE,
    }
    _log(
        "[CONSENSUS] lineage panels: %s %s | label resolver: %s",
        resolved["lineage_panel_source"],
        resolved["lineage_panel_sizes"],
        resolved["label_resolver"],
    )

    # -- Stage 2: markers --------------------------------------------------
    markers_by_cluster, empty_clusters = tools.compute_cluster_markers(
        adata_ln,
        cluster_col=cfg.cluster_col,
        top_n=cfg.top_n_markers,
        min_detection_fraction=float(getattr(cfg, "min_detection_fraction", 0.10)),
    )
    if empty_clusters:
        _log(
            f"[CONSENSUS] clusters with empty marker sets (flagged): {empty_clusters}",
            level=logging.WARNING,
        )

    # -- Stage 2b: state programmes (per cluster) --------------------------
    # Which clusters are defined by a STATE (cell cycle, dissociation stress,
    # interferon, ambient haemoglobin) rather than by an identity. Advisory for the
    # label, but it CAPS the confidence tier and suppresses the subtype: a cluster
    # whose evidence is 10/15 heat-shock genes has no identity evidence to be
    # confident about. See tools.state_programme_profile.
    state_profiles: Dict[str, dict] = {
        cl: tools.state_programme_profile(markers_by_cluster.get(cl, []))
        for cl in clusters
    }
    _state_flagged = {cl: p for cl, p in state_profiles.items() if p["state_dominated"]}
    for cl, p in _state_flagged.items():
        _log(
            f"[CONSENSUS] cluster {cl}: {p['state_fraction']:.0%} of its top "
            f"{p['n_markers_scored']} markers are cell-state genes "
            f"(dominant programme: {p['dominant_programme']}) — this is a state, not "
            f"an identity. Tier capped at {TIER_LOW} and the subtype suppressed.",
            level=logging.WARNING,
        )
    resolved["state_dominated_clusters"] = sorted(_state_flagged)
    resolved["state_dominance_threshold"] = tools.STATE_DOMINANCE_THRESHOLD

    # -- Stage 2c: observed lineage composition ----------------------------
    # What the DATA contains, by absolute lineage-panel detection. Used to check
    # that an auto-selected organ model actually covers the compartment present
    # (see tools.observed_lineage_profile for why detection fraction and not
    # score_genes / marker overlap).
    try:
        obs_lineage = tools.observed_lineage_profile(
            adata_ln, cluster_col=cfg.cluster_col
        )
    except Exception as e:  # never sink annotation over a descriptive statistic
        _log(
            f"[CONSENSUS] observed-lineage profiling failed ({e}); "
            f"model refinement unavailable.",
            level=logging.WARNING,
        )
        obs_lineage = {
            "dominant_lineage": "Other",
            "dominant_fraction": 0.0,
            "cluster_counts": {},
            "per_cluster": {},
        }
    resolved["observed_lineage_counts"] = obs_lineage.get("cluster_counts")
    resolved["observed_dominant_lineage"] = obs_lineage.get("dominant_lineage")
    resolved["observed_dominant_lineage_fraction"] = obs_lineage.get(
        "dominant_fraction"
    )

    # -- Stage 3: CellTypist (optional voter) ------------------------------
    # Model choice: "auto" => LLM picks the tissue-appropriate model from the
    # human catalog (disease-agnostic, tissue-only), falling back to the general
    # immune model when no organ matches / the LLM is off. A concrete .pkl name
    # in config is used as-is (no selection).
    ct_raw: Dict[str, tuple] = {}
    ct_metrics: Dict[
        str, dict
    ] = {}  # per-cluster heterogeneity / mixed-cluster metrics
    ct_immune_only = False  # True when the immune-only fallback model is in use
    if getattr(cfg, "enable_celltypist", True):
        ct_model = str(cfg.celltypist_model or "").strip()
        if ct_model.lower() == "auto":
            chosen = None
            if cfg.enable_llm:
                chosen = llm_select_celltypist_model(
                    cfg, cfg.tissue, HUMAN_CELLTYPIST_MODELS
                )
            ct_model = chosen or DEFAULT_FALLBACK_MODEL
            _log(
                f"[CONSENSUS] CellTypist model (auto, tissue={cfg.tissue!r}) -> {ct_model}"
                + ("" if chosen else " [general fallback]")
            )
            # The organ was inferred from study metadata text; the COMPARTMENT is
            # read from the data. A sorted-immune dataset from skin must not get a
            # keratinocyte/stromal model. Only applied to an auto choice — an
            # explicit model in config is the caller's decision and is respected.
            _refined, _why = tools.refine_celltypist_model_for_observed_lineage(
                ct_model, obs_lineage, valid_models=frozenset(VALID_MODEL_NAMES)
            )
            if _why:
                _log(
                    f"[CONSENSUS] CellTypist model refined: {_why}.",
                    level=logging.WARNING,
                )
                resolved["celltypist_model_before_lineage_refinement"] = ct_model
                resolved["celltypist_model_refinement_reason"] = _why
                ct_model = _refined
        else:
            _log("[CONSENSUS] CellTypist model (fixed) -> %s", ct_model)

        resolved["resolved_celltypist_model"] = ct_model
        ct_immune_only = ct_model in _IMMUNE_ONLY_CELLTYPIST
        resolved["celltypist_immune_only_model"] = bool(ct_immune_only)
        if ct_immune_only:
            _log(
                "[CONSENSUS] CellTypist is an immune-only model; its vote is dropped "
                "on clusters the lineage gate calls non-immune (out-of-domain)."
            )

        # Detailed variant: same dominant-label vote as before, PLUS the per-cluster
        # label heterogeneity that used to be discarded (see tools.summarize_celltypist_by_cluster).
        ct_raw, ct_metrics = tools.annotate_celltypist_detailed(
            adata_ln,
            cluster_col=cfg.cluster_col,
            model_name=ct_model,
            min_dominant_fraction=float(
                getattr(cfg, "mixed_cluster_min_dominant_fraction", 0.70)
            ),
            second_label_fraction=float(
                getattr(cfg, "mixed_cluster_second_label_fraction", 0.20)
            ),
        )
        if not ct_raw:
            _log(
                "[CONSENSUS] CellTypist voter abstained (see log above).",
                level=logging.WARNING,
            )
        else:
            _n_mixed = sum(
                1 for m in ct_metrics.values() if m.get("mixed_cluster_flag")
            )
            _log(
                "[CONSENSUS] CellTypist flagged %s/%s cluster(s) as mixed/heterogeneous (advisory only; the consensus label is unchanged).",
                _n_mixed,
                len(ct_metrics),
            )
    else:
        _log("[CONSENSUS] CellTypist voter disabled (enable_celltypist=False).")

    # -- Stage 4: SingleR (optional; loud on failure) ----------------------
    sr_raw: Dict[str, tuple] = {}
    if cfg.enable_singler:
        # Reference choice: "auto" => LLM picks by species+tissue (disease-agnostic),
        # falling back to the broad general reference; a concrete name is used as-is.
        sr_ref = str(cfg.singler_reference or "").strip()
        if sr_ref.lower() == "auto":
            chosen_ref = None
            if cfg.enable_llm:
                chosen_ref = llm_select_singler_reference(
                    cfg, cfg.tissue, cfg.species, SINGLER_REFERENCES
                )
            sr_ref = chosen_ref or DEFAULT_SINGLER_REFERENCE
            _log(
                f"[CONSENSUS] SingleR reference (auto, species={cfg.species!r}, "
                f"tissue={cfg.tissue!r}) -> {sr_ref}"
                + ("" if chosen_ref else " [general fallback]")
            )
        else:
            # An explicit name bypasses the selector, so validate it here: the R
            # bridge would happily load a mouse reference for human data, and the
            # resulting labels look plausible. Fail before SingleR runs, not after.
            if sr_ref not in VALID_SINGLER_REFERENCES:
                raise ValueError(
                    f"singler_reference={sr_ref!r} is not an accepted reference. "
                    f"This pipeline is human-only; choose one of: "
                    f"{', '.join(sorted(VALID_SINGLER_REFERENCES))}. "
                    f"(celldex::MouseRNAseqData is deliberately not offered — see "
                    f"celltypist_catalog.py for why and what mouse support needs.)"
                )
            _log("[CONSENSUS] SingleR reference (fixed) -> %s", sr_ref)
        resolved["resolved_singler_reference"] = sr_ref
        _log("[CONSENSUS] running SingleR (%s) via R bridge.", sr_ref)
        sr_raw = tools.run_singler(  # raises RuntimeError on non-zero exit (no silent fallback)
            adata_ln,
            cluster_col=cfg.cluster_col,
            reference=sr_ref,
            bridge_script=BRIDGE_SCRIPT,
            rscript_exe=cfg.rscript_exe,
        )

    # -- Stage 5: LLM annotator (agent layer) ------------------------------
    llm_raw: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}
    llm_available = cfg.enable_llm
    if cfg.enable_llm:
        for cl in clusters:
            if not llm_available:
                llm_raw[cl] = "Unassigned"
                continue
            try:
                res = llm_annotate_cluster(
                    cfg, cl, markers_by_cluster.get(cl, []), cfg.tissue
                )
                llm_raw[cl] = str(res.get("cell_type", "Unassigned"))
                llm_conf[cl] = float(res.get("confidence", 0.0))
            except OpenRouterError as e:
                # documented fallback: log once, stop hammering, remaining clusters abstain
                _log(
                    f"[CONSENSUS] LLM transport failed on cluster {cl} ({e}); "
                    f"remaining clusters abstain from the LLM voter.",
                    level=logging.ERROR,
                )
                llm_available = False
                llm_raw[cl] = "Unassigned"

    # -- Stage 5b: PubMed literature voter (optional; opt-in, needs network) --
    # Disease-agnostic: uses tissue (biosample) ONLY, never a disease term, to
    # honor the ConsensusConfig invariant. Any failure -> abstains (never fatal).
    pm_raw: Dict[str, tuple] = {}
    pm_detail: Dict[
        str, object
    ] = {}  # cluster -> PubMedAnnotation (markers, PMIDs, subtype)
    if getattr(cfg, "enable_pubmed", False):
        try:
            from ..pubmed_annotation import (
                PubMedAnnotationConfig,
                annotate_with_pubmed,
                build_evidence_table,
                plot_confidence,
                save_table,
            )

            pm_cfg = PubMedAnnotationConfig(
                disease="",  # invariant: no disease drives the vote
                biosample=(cfg.tissue or ""),
                species=(cfg.species or "human"),
                cache_dir=out_dir / "_pubmed_cache",
            )
            if not pm_cfg.openrouter_api_key:
                # Enabled but unusable: skip fast rather than slow-failing every cluster.
                _log(
                    "[CONSENSUS] PubMed voter enabled but OPENROUTER_API_KEY is missing; "
                    "skipping (voter abstains).",
                    level=logging.WARNING,
                )
            else:
                _log(
                    "[CONSENSUS] running PubMed literature voter (biosample=%r).",
                    cfg.tissue,
                )
                pm_results = annotate_with_pubmed(
                    {cl: markers_by_cluster.get(cl, []) for cl in clusters}, pm_cfg
                )
                for cl, ann in pm_results.items():
                    lbl = (
                        None
                        if str(ann.cell_type).strip().lower() in ("", "unknown")
                        else ann.cell_type
                    )
                    pm_raw[str(cl)] = (lbl, float(ann.confidence_score))
                    pm_detail[str(cl)] = ann
                # Surface the PubMed EVIDENCE (PMIDs + supporting markers + subtype)
                # as its own auditable table + confidence graph — otherwise the
                # literature markers that justify each call are never written out.
                try:
                    _pm_df = build_evidence_table(pm_results)
                    save_table(_pm_df, out_dir)  # -> pubmed_annotation_table.csv
                    plot_confidence(
                        _pm_df, out_dir
                    )  # -> pubmed_annotation_confidence.png
                except Exception as e:
                    _log(
                        f"[CONSENSUS] PubMed evidence table/graph failed ({e}).",
                        level=logging.WARNING,
                    )
        except Exception as e:
            _log(
                f"[CONSENSUS] PubMed voter failed ({e}); abstaining.",
                level=logging.WARNING,
            )
            pm_raw = {}
            pm_detail = {}

    # -- Stage 7: lineage sanity gate --------------------------------------
    gate = tools.lineage_gate_per_cluster(adata_ln, cluster_col=cfg.cluster_col)

    # -- Stages 6 + 8: harmonize & reconcile per cluster -------------------
    consensus_label: Dict[str, str] = {}
    consensus_tier: Dict[str, str] = {}
    # Full adjudicator reasoning, kept untruncated. `decision_reason` is capped at
    # 120 chars for readability, which hides exactly why a contested cluster was
    # called the way it was — the audit trail you need when a label looks wrong.
    adj_reasoning: Dict[str, str] = {}
    provenance: Dict[str, str] = {}
    ct_label_out: Dict[str, str] = {}
    sr_label_out: Dict[str, str] = {}
    llm_label_out: Dict[str, str] = {}
    pm_label_out: Dict[str, str] = {}
    subtype_map: Dict[str, str] = {}  # per-cluster finer label (consensus-consistent)
    subtype_source_map: Dict[str, str] = {}  # which annotator produced the subtype
    subtype_conf_map: Dict[str, str] = {}  # that annotator's native confidence
    subtype_rejected_map: Dict[str, str] = {}  # subtypes turned away, and why
    state_flag_map: Dict[str, str] = {}  # state-dominated flag (as str for obs)
    state_prog_map: Dict[str, str] = {}  # dominant state programme
    state_frac_map: Dict[str, str] = {}  # state fraction of top markers
    mixed_flag_map: Dict[
        str, str
    ] = {}  # CellTypist heterogeneity flag (as str for obs)
    ct_metric_maps: Dict[str, Dict[str, str]] = {
        k: {} for k in tools.CELLTYPIST_METRIC_KEYS if k != "celltypist_dominant_label"
    }
    cluster_sizes = adata.obs[cfg.cluster_col].astype(str).value_counts()
    table_rows: List[dict] = []

    # Rank-normalize each voter's confidences WITHIN this run so their
    # incompatible native scales (CellTypist cell-fraction, SingleR Spearman rho,
    # LLM self-report, PubMed evidence score) become comparable before they are
    # used to break ties / feed the adjudicator. See tools.normalize_confidences.
    norm_conf_by_method: Dict[str, Dict[str, float]] = {
        "celltypist": tools.normalize_confidences({c: v[1] for c, v in ct_raw.items()}),
        "singler": tools.normalize_confidences({c: v[1] for c, v in sr_raw.items()}),
        "knowledge_based": tools.normalize_confidences(dict(llm_conf)),
        "pubmed": tools.normalize_confidences({c: v[1] for c, v in pm_raw.items()}),
    }

    # A voter that returns one label for nearly every cluster is saturated, not
    # voting. Its permanent dissent otherwise sets voters_disagree everywhere and
    # collapses the tier column (measured: SingleR 'Astrocyte' on 18/20 clusters of
    # GSE157827 pushed 83.5% of cells to Low/Review with 0 clusters High). Detected
    # once over the whole run, since a single cluster cannot reveal saturation. Raw
    # calls are still reported per cluster — only the vote is withheld.
    _harm = tools.harmonize_label
    _labels_by_voter: Dict[str, Dict[str, str]] = {
        "celltypist": {c: _harm(v[0]) for c, v in ct_raw.items() if v[0]},
        "singler": {c: _harm(v[0]) for c, v in sr_raw.items() if v[0]},
        "knowledge_based": {c: _harm(v) for c, v in llm_raw.items() if v},
        "pubmed": {c: _harm(v[0]) for c, v in pm_raw.items() if v[0]},
    }
    saturated = tools.degenerate_voters(_labels_by_voter)
    for _v, _info in sorted(saturated.items()):
        _log(
            f"[CONSENSUS] voter {_v!r} returned {_info['modal_label']!r} for "
            f"{_info['modal_fraction']:.0%} of {_info['n_clusters']} clusters — "
            f"saturated, so it carries no discriminative signal. Its vote is "
            f"withheld from the tally (raw calls still reported).",
            level=logging.WARNING,
        )
    resolved["saturated_voters"] = {k: v for k, v in sorted(saturated.items())}

    for cl in clusters:
        gate_lin = gate.get(cl, "Other")
        ct_l = ct_raw.get(cl, (None, None))[0]
        sr_l = sr_raw.get(cl, (None, None))[0]
        llm_l = llm_raw.get(cl)
        pm_l = pm_raw.get(cl, (None, None))[0]
        ct_label_out[cl] = ct_l or tools.UNASSIGNED
        sr_label_out[cl] = sr_l or tools.UNASSIGNED
        llm_label_out[cl] = llm_l or tools.UNASSIGNED
        pm_label_out[cl] = pm_l or tools.UNASSIGNED

        # Stage 6 — harmonize each method's label to the controlled vocabulary
        method_h: Dict[str, str] = {}
        ct_suppressed = False
        if ct_l is not None:
            # Immune-only CellTypist cannot abstain; on a non-immune cluster its
            # forced immune label is out-of-domain, so drop the vote (the raw call
            # is still kept in the celltype_celltypist column for transparency).
            if ct_immune_only and gate_lin not in ("Immune", "Other"):
                ct_suppressed = True
            else:
                method_h["celltypist"] = tools.harmonize_label(ct_l)
        if cfg.enable_singler and sr_l is not None:
            method_h["singler"] = tools.harmonize_label(sr_l)
        if cfg.enable_llm and llm_l is not None:
            method_h["knowledge_based"] = tools.harmonize_label(llm_l)
        if getattr(cfg, "enable_pubmed", False) and pm_l is not None:
            method_h["pubmed"] = tools.harmonize_label(pm_l)
        # Withhold saturated voters' votes (see degenerate_voters above).
        withheld = sorted(set(method_h) & set(saturated))
        for _v in withheld:
            method_h.pop(_v, None)

        # per-method confidences (rank-normalized so scales are comparable) —
        # used ONLY to break ties deterministically.
        method_conf: Dict[str, float] = {}
        for _m in method_h:
            _c = norm_conf_by_method.get(_m, {}).get(cl)
            if _c is not None:
                method_conf[_m] = _c

        # Stage 8a — count (tool)
        tally = tools.tally_votes(method_h, method_conf)
        majority = str(tally["majority_label"])
        maj_coarse = tools.coarse_lineage_of(majority)
        contradicted = (
            gate_lin != "Other"
            and maj_coarse != "Other"
            and maj_coarse != gate_lin
            # A gate built from a handful of pan-lineage markers must not overturn
            # every independent annotator at once. When the voters are UNANIMOUS the
            # gate is the more likely thing to be wrong (it has no panel for e.g.
            # mast cells or dendritic cells), so the vote stands. Split votes still
            # route to the adjudicator below.
            and not tally["unanimous"]
        )
        if (
            gate_lin != "Other"
            and maj_coarse != "Other"
            and maj_coarse != gate_lin
            and tally["unanimous"]
        ):
            _log(
                "[CONSENSUS] cluster %s: lineage gate says %r but all %s voters agree on %r; keeping the unanimous vote.",
                cl,
                gate_lin,
                tally["n_methods"],
                majority,
            )

        # Stage 8a2 — reference-scope check. A closed-vocabulary voter handed a
        # reference that does not contain the population in front of it cannot
        # abstain; it emits a forced, near-chance label. Two such forced guesses
        # must not outvote one confident marker-driven call. See
        # tools.out_of_domain_deference for the measured case this corrects.
        ood = tools.out_of_domain_deference(
            method_h,
            tally,
            open_confidences={
                "knowledge_based": llm_conf.get(cl),
                "pubmed": pm_raw.get(cl, (None, None))[1],
            },
            celltypist_unreliable=bool(
                ct_metrics.get(cl, {}).get("mixed_cluster_flag", False)
            ),
        )
        if ood is not None:
            ood_label, ood_voter, ood_reason = ood
            _log(f"[CONSENSUS] cluster {cl}: {ood_reason}.", level=logging.WARNING)

        # Stage 8b — decide tier / route to adjudicator (agent)
        if ood is not None:
            # Never promoted above review: this is the better guess, not a
            # corroborated identity.
            final, tier, reason = ood_label, TIER_LOW, ood_reason
        elif tally["unanimous"] and not contradicted:
            final, tier = majority, TIER_HIGH
            reason = "unanimous, lineage-consistent"
        elif tally["has_majority"] and not contradicted:
            final, tier = majority, TIER_MEDIUM
            reason = "majority, lineage-consistent"
        else:
            tier = TIER_LOW
            if cfg.enable_llm and llm_available and tally["n_methods"] >= 1:
                candidates = _candidates(method_h, norm_conf_by_method, cl)
                try:
                    adj = llm_adjudicate(
                        cfg,
                        cl,
                        candidates,
                        markers_by_cluster.get(cl, []),
                        gate_lin,
                        cfg.tissue,
                        fallback_label=majority,
                    )
                    final = tools.harmonize_label(adj["label"])
                    adj_reasoning[cl] = str(adj.get("reasoning", ""))
                    reason = f"adjudicated ({adj_reasoning[cl][:120]})"
                except OpenRouterError as e:
                    final = majority if tally["n_methods"] else gate_lin
                    reason = f"adjudicator transport failed ({e}); fell back to {final}"
                    _log(f"[CONSENSUS] cluster {cl}: {reason}", level=logging.ERROR)
            elif contradicted and gate_lin != "Other":
                # No adjudicator, and the vote conflicts with pan-lineage biology:
                # trust the lineage gate. This is what corrects non-immune clusters
                # (e.g. EPCAM+ voted "T cell") even when the LLM voter is disabled.
                final = gate_lin
                reason = (
                    "lineage-contradicted, no adjudicator; deferred to lineage gate"
                )
            else:
                # no adjudicator: prefer a usable majority, else the lineage flag.
                final = majority if tally["n_methods"] else gate_lin
                reason = "no adjudicator; used majority/lineage fallback"
                # principled tie-break: when the top vote is tied, prefer the tied
                # label whose coarse lineage matches the gate (biology over an
                # arbitrary pick). Confidence already broke the tally-level tie.
                if tally.get("tied") and gate_lin != "Other":
                    gate_ok = [
                        label
                        for label in tally.get("top_labels", [])
                        if tools.coarse_lineage_of(label) == gate_lin
                    ]
                    if gate_ok:
                        final = sorted(gate_ok)[0]
                        reason = f"tie broken by lineage gate -> {gate_lin}"
        if not final or final == tools.UNASSIGNED:
            final = gate_lin if gate_lin != "Other" else tools.UNASSIGNED

        # Stage 8c — a cluster defined by a STATE cannot be a High-confidence
        # identity. The voters may well have named the right lineage, but they did
        # it from stress/cycle/interferon genes, so the call is not corroborated by
        # identity evidence. The label is kept; only the confidence claim is
        # withdrawn. See tools.state_programme_profile for the measured case.
        st = state_profiles.get(cl, {})
        if st.get("state_dominated") and tier != TIER_LOW:
            reason = (
                f"{reason} | tier capped: {st['state_fraction']:.0%} of top "
                f"{st['n_markers_scored']} markers are "
                f"{st['dominant_programme']} state genes, not identity markers"
            )
            tier = TIER_LOW

        consensus_label[cl] = final
        consensus_tier[cl] = tier
        prov = (
            f"votes[{tally['pattern']}] | gate={gate_lin} | "
            f"pattern={'unanimous' if tally['unanimous'] else ('majority' if tally['has_majority'] else 'split')} | "
            f"{reason}"
            + (
                " | celltypist vote dropped (immune-only model on non-immune cluster)"
                if ct_suppressed
                else ""
            )
        )
        provenance[cl] = prov

        # --- subtype layer: the finest voter label whose harmonized coarse
        # identity EQUALS the consensus node (so a subtype can never contradict
        # the coarse call). Priority: PubMed literature subtype -> LLM label ->
        # CellTypist fine label -> PubMed cell_type; else the coarse consensus.
        # The producing annotator and its native confidence are recorded, so a
        # subtype is never mistaken for a consensus-validated identity.
        pm_ann = pm_detail.get(cl)
        pm_sub = getattr(pm_ann, "cell_subtype", None) if pm_ann is not None else None
        # A subtype must be backed by MARKERS PRESENT IN THIS CLUSTER, not merely by
        # coarse-identity agreement. A state-dominated cluster gets no subtype at
        # all: there is no identity evidence for a finer call to rest on.
        subtype, subtype_source, subtype_conf, subtype_rejected = (
            pick_subtype_with_source(
                final,
                [
                    (pm_sub, "pubmed", pm_raw.get(cl, (None, None))[1]),
                    (llm_l, "knowledge_based", llm_conf.get(cl)),
                    (ct_l, "celltypist", ct_raw.get(cl, (None, None))[1]),
                    (pm_l, "pubmed", pm_raw.get(cl, (None, None))[1]),
                ],
                markers=markers_by_cluster.get(cl, []),
                suppress=bool(state_profiles.get(cl, {}).get("state_dominated")),
            )
        )
        for _rej_label, _rej_why in subtype_rejected:
            _log(
                f"[CONSENSUS] cluster {cl}: rejected subtype {_rej_label!r} — {_rej_why}.",
                level=logging.WARNING,
            )
        subtype_map[cl] = subtype
        subtype_source_map[cl] = subtype_source
        subtype_conf_map[cl] = (
            "" if subtype_conf is None else str(round(float(subtype_conf), 3))
        )
        subtype_rejected_map[cl] = "; ".join(
            f"{label} ({w})" for label, w in subtype_rejected
        )

        # --- state programme (caps the tier; see Stage 8c)
        state_flag_map[cl] = str(bool(st.get("state_dominated", False)))
        state_prog_map[cl] = str(st.get("dominant_programme", "") or "")
        state_frac_map[cl] = str(st.get("state_fraction", 0.0))

        # --- CellTypist heterogeneity metrics (advisory; do not alter the vote)
        cl_metrics = ct_metrics.get(cl, {})
        mixed_flag_map[cl] = str(bool(cl_metrics.get("mixed_cluster_flag", False)))
        for _mk in ct_metric_maps:
            _mv = cl_metrics.get(_mk, "")
            ct_metric_maps[_mk][cl] = "" if _mv == "" or _mv is None else str(_mv)

        # PubMed evidence (markers actually cited from the retrieved abstracts).
        pm_support = (
            ";".join(getattr(pm_ann, "supporting_markers", []) or [])
            if pm_ann is not None
            else ""
        )
        pm_pmids = (
            ";".join(dict.fromkeys(getattr(pm_ann, "pmids", []) or []))
            if pm_ann is not None
            else ""
        )
        pm_state = (
            (getattr(pm_ann, "cell_state", None) or "") if pm_ann is not None else ""
        )

        _en_ct = getattr(cfg, "enable_celltypist", True)
        _en_pm = getattr(cfg, "enable_pubmed", False)
        # One master row per cluster: every method's call + its native confidence,
        # the consensus + subtype + tier, the marker evidence (source), and PubMed
        # citations. This is the single "all models + consensus + markers" table.
        _ct_conf = (
            _fmt_conf(ct_raw.get(cl, (None, None))[1]) if _en_ct else "(disabled)"
        )
        _sr_conf = (
            _fmt_conf(sr_raw.get(cl, (None, None))[1])
            if cfg.enable_singler
            else "(disabled)"
        )
        _kb_conf = _fmt_conf(llm_conf.get(cl)) if cfg.enable_llm else "(disabled)"
        _pm_conf = (
            _fmt_conf(pm_raw.get(cl, (None, None))[1]) if _en_pm else "(disabled)"
        )
        _mixed = bool(cl_metrics.get("mixed_cluster_flag", False))
        table_rows.append(
            {
                # --- historical column names (unchanged; downstream readers depend on them)
                "cluster": cl,
                "consensus": final,
                "celltype_subtype": subtype,
                "tier": tier,
                "lineage_gate": gate_lin,
                "celltypist": ct_label_out[cl] if _en_ct else "(disabled)",
                "celltypist_conf": _ct_conf,
                "singler": sr_label_out[cl] if cfg.enable_singler else "(disabled)",
                "singler_conf": _sr_conf,
                "knowledge_based": llm_label_out[cl]
                if cfg.enable_llm
                else "(disabled)",
                "knowledge_based_conf": _kb_conf,
                "pubmed": pm_label_out[cl] if _en_pm else "(disabled)",
                "pubmed_conf": _pm_conf,
                "pubmed_supporting_markers": pm_support if _en_pm else "(disabled)",
                "pubmed_pmids": pm_pmids if _en_pm else "(disabled)",
                "pubmed_cell_state": pm_state if _en_pm else "(disabled)",
                "harmonized_agreement": tally["pattern"],
                "markers_empty": cl in empty_clusters,
                "top_markers": ", ".join(markers_by_cluster.get(cl, [])[:15]),
                "provenance": prov,
                # --- explicit aliases + new fields (added; nothing above was renamed)
                "leiden": cl,
                "final_celltype": final,
                "consensus_tier": tier,
                "lineage_coarse": gate_lin,
                "celltypist_label": ct_label_out[cl] if _en_ct else "(disabled)",
                "celltypist_confidence": _ct_conf,
                "singler_label": sr_label_out[cl]
                if cfg.enable_singler
                else "(disabled)",
                "singler_confidence": _sr_conf,
                "knowledge_based_label": llm_label_out[cl]
                if cfg.enable_llm
                else "(disabled)",
                "knowledge_based_confidence": _kb_conf,
                "pubmed_label": pm_label_out[cl] if _en_pm else "(disabled)",
                "pubmed_confidence": _pm_conf,
                "n_cells": int(cluster_sizes.get(cl, 0)),
                # Honest: did ANY voter that produced a usable label — including one
                # whose vote was withheld for saturation — end up disagreeing with the
                # label that shipped? The old definition read only the tally, so it
                # reported False whenever the tally happened to be clean even though the
                # adjudicator or a deference rule had overridden it, hiding exactly the
                # cases worth auditing. `harmonized_agreement` still carries the tally
                # pattern, so nothing is lost.
                "voters_disagree": bool(
                    any(
                        _h != final
                        for _h in (
                            tools.harmonize_label(_r)
                            for _r in (
                                ct_label_out[cl],
                                sr_label_out[cl],
                                llm_label_out[cl],
                                pm_label_out[cl],
                            )
                            if _r and _r != tools.UNASSIGNED
                        )
                    )
                ),
                # Voters whose saturated output was excluded from this cluster's tally.
                "voters_withheld": ", ".join(withheld),
                "n_voters": int(tally["n_methods"]),
                "mixed_cluster_flag": _mixed,
                "celltypist_dominant_fraction": cl_metrics.get(
                    "celltypist_dominant_fraction", ""
                ),
                "celltypist_second_label": cl_metrics.get(
                    "celltypist_second_label", ""
                ),
                "celltypist_second_fraction": cl_metrics.get(
                    "celltypist_second_fraction", ""
                ),
                "celltypist_label_entropy": cl_metrics.get(
                    "celltypist_label_entropy", ""
                ),
                "celltypist_unique_label_count": cl_metrics.get(
                    "celltypist_unique_label_count", ""
                ),
                "celltype_subtype_source": subtype_source,
                "celltype_subtype_confidence": subtype_conf_map[cl],
                # Subtypes that were offered and turned away for lack of marker support,
                # with the reason. A blank cell means nothing was rejected.
                "celltype_subtype_rejected": subtype_rejected_map[cl],
                # Cell-state evidence: is this cluster defined by a programme rather than
                # an identity? `state_dominated` caps the tier at Low/Review.
                "state_dominated": bool(st.get("state_dominated", False)),
                "state_programme": state_prog_map[cl],
                "state_marker_fraction": st.get("state_fraction", 0.0),
                "observed_lineage": (obs_lineage.get("per_cluster") or {}).get(cl, ""),
                "decision_reason": reason,
                "adjudicator_reasoning_full": adj_reasoning.get(cl, ""),
                # Filled in by the host pipeline once tier gating is resolved; the
                # placeholder keeps the column present even when the CSV is read on its own.
                "included_in_downstream_analysis": "",
            }
        )

    # -- Stage 9: broadcast to cells, validate conservation, export --------
    # `celltype_subtype` IS broadcast to adata.obs, together with the annotator that
    # produced it and that annotator's native confidence. It is purely ADDITIVE:
    # `celltype` / `celltype_consensus` remain the coarse, consensus-validated
    # identity and are what every downstream step reads. Carrying the subtype in obs
    # (not only in the CSV) is what lets the exported h5ad and the report show the
    # finer call without ever promoting a single-annotator label to the main label.
    per_cluster_columns = {
        "lineage_coarse": gate,
        "celltype_celltypist": ct_label_out,
        "celltype_knowledge_based": llm_label_out,
        CELLTYPE_CONSENSUS_COL: consensus_label,
        "consensus_tier": consensus_tier,
        "annotation_provenance": provenance,
        "celltype_subtype": subtype_map,
        "celltype_subtype_source": subtype_source_map,
        "celltype_subtype_confidence": subtype_conf_map,
        # State evidence travels with the h5ad so a downstream reader can exclude
        # or inspect state-driven clusters without re-reading the CSV.
        "state_dominated": state_flag_map,
        "state_programme": state_prog_map,
        "state_marker_fraction": state_frac_map,
    }
    if cfg.enable_singler:
        per_cluster_columns["celltype_singler"] = sr_label_out
    if getattr(cfg, "enable_pubmed", False):
        per_cluster_columns["celltype_pubmed"] = pm_label_out
    # CellTypist heterogeneity: written only when the voter actually produced
    # metrics, so a disabled/abstaining CellTypist cannot create a misleading
    # all-False mixed_cluster_flag column.
    if ct_metrics:
        per_cluster_columns["mixed_cluster_flag"] = mixed_flag_map
        for _mk, _mmap in ct_metric_maps.items():
            per_cluster_columns[_mk] = _mmap

    tools.broadcast_and_validate(adata, cfg.cluster_col, per_cluster_columns)
    _cast_numeric_and_bool_obs(adata)
    table_path = tools.write_consensus_table(table_rows, out_dir, analysis_name)

    # Publish what was actually resolved so the provenance manifest and the report
    # can record real values instead of the requested "auto". Lives in .uns, so it
    # travels with the exported h5ad.
    resolved["n_clusters"] = len(clusters)
    resolved["consensus_table"] = str(table_path)
    resolved["clusters_with_empty_markers"] = list(empty_clusters)
    resolved["n_mixed_clusters"] = int(
        sum(1 for m in ct_metrics.values() if m.get("mixed_cluster_flag"))
    )
    resolved["tier_counts"] = {
        t: int(sum(1 for v in consensus_tier.values() if v == t))
        for t in (TIER_HIGH, TIER_MEDIUM, TIER_LOW)
    }
    adata.uns[CONSENSUS_UNS_KEY] = dict(resolved)

    # Visual summaries (best-effort; never sink the run): per-sample cell-type
    # composition + cross-method agreement/confidence + one graph per method.
    from .consensus_plots import (
        plot_method_agreement,
        plot_per_method_calls,
        plot_sample_celltype_composition,
    )

    plot_sample_celltype_composition(
        adata, out_dir, analysis_name, celltype_col=CELLTYPE_CONSENSUS_COL
    )
    plot_method_agreement(
        table_rows,
        out_dir,
        analysis_name,
        enable_singler=cfg.enable_singler,
        enable_llm=cfg.enable_llm,
        enable_pubmed=getattr(cfg, "enable_pubmed", False),
    )
    plot_per_method_calls(
        table_rows,
        out_dir,
        analysis_name,
        enable_celltypist=getattr(cfg, "enable_celltypist", True),
        enable_singler=cfg.enable_singler,
        enable_llm=cfg.enable_llm,
        enable_pubmed=getattr(cfg, "enable_pubmed", False),
    )

    tier_counts = {
        t: sum(1 for v in consensus_tier.values() if v == t)
        for t in (TIER_HIGH, TIER_MEDIUM, TIER_LOW)
    }
    _log(
        "[CONSENSUS] done | clusters=%s | tiers=%s | n_out=%s (n_in=%s)",
        len(clusters),
        tier_counts,
        adata.n_obs,
        n_in,
    )
    if adata.n_obs != n_in:
        raise tools.CellConservationError(
            f"annotation must not change the cell count: n_in={n_in} n_out={adata.n_obs}"
        )

    log_path = out_dir / f"{analysis_name}_consensus_run.log"
    log_path.write_text("\n".join(run_log), encoding="utf-8")
    logger.info("[CONSENSUS] run log: %s", log_path)
    return adata


#: obs columns that must end up numeric rather than categorical strings.
_NUMERIC_OBS_COLS = (
    "celltypist_dominant_fraction",
    "celltypist_second_fraction",
    "celltypist_label_entropy",
    "celltypist_unique_label_count",
    "celltype_subtype_confidence",
    "state_marker_fraction",
)
#: obs columns that must end up real booleans.
_BOOL_OBS_COLS = ("mixed_cluster_flag", "state_dominated")


def _cast_numeric_and_bool_obs(adata) -> None:
    """Re-type the metric columns after ``broadcast_and_validate``.

    ``broadcast_and_validate`` writes every per-cluster mapping as a categorical of
    strings (correct for labels). The heterogeneity metrics are numbers and a flag,
    so they are cast back here — otherwise ``obs['mixed_cluster_flag'] == True``
    would silently be False for every cell.
    """
    import pandas as _pd

    for col in _BOOL_OBS_COLS:
        if col in adata.obs.columns:
            adata.obs[col] = (
                adata.obs[col].astype(str).str.strip().str.lower().eq("true")
            )
    for col in _NUMERIC_OBS_COLS:
        if col in adata.obs.columns:
            adata.obs[col] = _pd.to_numeric(
                adata.obs[col].astype(str).replace({"": None}), errors="coerce"
            )


def _fmt_conf(x):
    """Round a native confidence to 3 dp for the master table; '' if missing/NaN."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if f != f else round(f, 3)  # f != f catches NaN


#: Source recorded when no annotator offered a consensus-consistent finer label.
SUBTYPE_SOURCE_COARSE = "consensus_coarse"


#: Source recorded when a state-dominated cluster is denied a subtype entirely.
SUBTYPE_SOURCE_SUPPRESSED = "suppressed_state_dominated"


def pick_subtype_with_source(
    final: bool,
    candidates: Iterable[float] | None,
    *,
    markers: Mapping[str, Sequence[str]] = None,
    suppress: bool = False,
) -> tuple[str, str, float | None, list]:
    """Finest subtype that is BOTH consensus-consistent and marker-supported.

    ``candidates`` is an ordered list of ``(label, source, confidence)``. A
    candidate wins when it clears three gates, in order:

      1. its harmonized coarse identity EQUALS ``final`` (and it is not just
         ``final`` restated) — so a subtype can never contradict the vote:
         'CD16+ NK cell' is kept under an 'NK cell' consensus, 'DC1' is rejected
         under a 'B cell' consensus;
      2. every marker-defined claim in its NAME is present in ``markers`` — see
         tools.subtype_marker_support. This is the gate that stops an annotator
         asserting 'CD8-positive T cell' on a cluster with no CD8A/CD8B;
      3. ``suppress`` is False. A state-dominated cluster gets no subtype: there is
         no identity evidence for a finer call to rest on.

    Returns ``(subtype, source, confidence, rejected)`` where ``rejected`` is a list
    of ``(label, reason)`` for every candidate turned away by gate 2 or 3 — the
    audit trail for a subtype that did NOT ship. When nothing survives, the coarse
    ``final`` is returned with source ``consensus_coarse``/``suppressed_state_dominated``
    and confidence ``None``, so a subtype is never presented as though a voter had
    asserted it.
    """
    fin = str(final).strip()
    # Harmonize BOTH sides. Comparing a harmonized candidate against a raw `final`
    # only worked while `final` happened to already be canonical; once labels
    # resolve through the cell_hierarchy ("NK cell" -> "Natural killer cell") a raw
    # `final` would never match and every subtype would silently fall back to the
    # coarse label. harmonize_label is idempotent, so this is safe either way.
    fin_h = tools.harmonize_label(fin)
    rejected: List[tuple] = []
    for cand, source, conf in candidates:
        if not cand:
            continue
        s = str(cand).strip()
        if not s or s.lower() in ("unknown", "unassigned", "none", "nan"):
            continue
        if tools.harmonize_label(s) != fin_h or s.lower() == fin.lower():
            continue  # gate 1 (silent: not a subtype)
        if suppress:  # gate 3
            rejected.append(
                (s, "cluster is state-dominated; no identity evidence for a finer call")
            )
            continue
        ok, why = tools.subtype_marker_support(s, markers)
        if not ok:  # gate 2
            rejected.append((s, why))
            continue
        return s, str(source), conf, rejected
    src = (
        SUBTYPE_SOURCE_SUPPRESSED if (suppress and rejected) else SUBTYPE_SOURCE_COARSE
    )
    return final, src, None, rejected


def _pick_subtype(final, *candidates):
    """Backward-compatible positional wrapper around ``pick_subtype_with_source``.

    Returns only the subtype label. Kept because existing callers/tests pass the
    candidates positionally in priority order. Marker gating is not applied here —
    the wrapper has no markers to check against — so callers that need it must use
    ``pick_subtype_with_source`` directly.
    """
    subtype, _source, _conf, _rejected = pick_subtype_with_source(
        final, [(c, "unknown", None) for c in candidates]
    )
    return subtype


def _candidates(method_h, norm_conf_by_method, cl):
    """Build the adjudicator candidate list with rank-normalized per-method
    confidences (comparable across voters; see tools.normalize_confidences)."""
    out = []
    for method, harmonized in method_h.items():
        if harmonized == tools.UNASSIGNED:
            continue
        conf = norm_conf_by_method.get(method, {}).get(cl)
        out.append(
            {
                "method": method,
                "label": harmonized,
                "confidence": None if conf is None else round(float(conf), 3),
            }
        )
    return out


# ===========================================================================
# Standalone CLI — run consensus on an existing (clustered) .h5ad
# ===========================================================================
def _cli():  # pragma: no cover - thin wrapper
    import scanpy as sc

    p = argparse.ArgumentParser(
        description="Multi-method consensus cell-type annotation."
    )
    p.add_argument(
        "--h5ad", required=True, help="Input .h5ad (ideally already clustered)."
    )
    p.add_argument(
        "--out-dir", default=None, help="Output dir (default: alongside input)."
    )
    p.add_argument("--analysis-name", default="consensus")
    p.add_argument("--tissue", default=None, help="Organ/tissue context (NOT disease).")
    p.add_argument("--cluster-col", default="leiden")
    p.add_argument("--no-llm", action="store_true", help="Disable the LLM voter.")
    p.add_argument(
        "--enable-singler", action="store_true", help="Enable SingleR (needs R)."
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    adata = sc.read_h5ad(args.h5ad)

    if args.cluster_col not in adata.obs.columns:
        logger.info(
            "[CLI] '%s' absent; computing a minimal Leiden clustering.",
            args.cluster_col,
        )
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1), random_state=0)
        sc.pp.neighbors(adata, n_neighbors=15, random_state=0)
        sc.tl.leiden(adata, resolution=0.5, key_added=args.cluster_col, random_state=0)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.h5ad).parent
    cfg = load_config(
        enable_llm=not args.no_llm,
        enable_singler=args.enable_singler,
        tissue=args.tissue,
        cluster_col=args.cluster_col,
    )
    run_consensus_annotation(adata, out_dir, args.analysis_name, config=cfg)

    out_h5ad = out_dir / f"{args.analysis_name}_consensus_annotated.h5ad"
    adata.write_h5ad(out_h5ad)
    logger.info("[CLI] wrote annotated h5ad: %s", out_h5ad)


if __name__ == "__main__":
    _cli()
