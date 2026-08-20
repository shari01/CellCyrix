"""
config.py — .env-driven configuration for consensus annotation.

Reads only from the environment / .env (never hardcodes a model). Enforces the
"fail fast" rule: if the LLM layer is enabled but OPENROUTER_API_KEY is absent,
construction raises immediately with a clear message — we never silently skip a
voter.

Env vars
--------
OPENROUTER_API_KEY   required when enable_llm=True
OPENROUTER_MODEL     model slug (no default hardcoded model name; must be set to use LLM)
OPENROUTER_ENDPOINT  optional, default https://openrouter.ai/api/v1/chat/completions
LLM_MAX_RETRIES      optional, default 3
LLM_TIMEOUT_S        optional, default 60
LLM_TEMPERATURE      optional, default 0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from .. import env_names
from ..exceptions import PipelineInputError

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class ConsensusConfigError(PipelineInputError):
    """Raised when configuration is invalid (e.g. LLM enabled but no API key)."""


def _load_dotenv_best_effort() -> None:
    """Load .env if python-dotenv is importable. Missing dotenv is logged, not fatal:
    env vars can still come from the shell/process environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.info(
            "[CONFIG] python-dotenv not installed; reading OS environment only."
        )
        return
    load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning(
            "[CONFIG] %s=%r is not an int; using default %s.", name, raw, default
        )
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning(
            "[CONFIG] %s=%r is not a float; using default %s.", name, raw, default
        )
        return default


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Accepts 1/0, true/false, yes/no, on/off (any case)."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "[CONFIG] %s=%r is not a boolean; using default %s.", name, raw, default
    )
    return default


@dataclass
class ConsensusConfig:
    """Immutable-ish configuration bag passed to tools and the agent layer.

    NOTE: `tissue` (organ context) is allowed and is cell biology, NOT disease.
    There is deliberately NO `disease` field anywhere in this object — the
    disease-agnostic invariant is enforced by construction.
    """

    # LLM (agent layer)
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    endpoint: str = DEFAULT_ENDPOINT
    llm_max_retries: int = 3
    llm_timeout_s: int = 60
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800  # pin the reply length so JSON can't silently truncate
    # Greedy decoding needs top_p pinned as well as temperature: temperature 0 with a
    # provider-default top_p below 1.0 still truncates the distribution, and different
    # providers behind the same alias default differently.
    llm_top_p: float = 1.0
    # Sent as `seed`. Best-effort only — providers are not required to honour it — so
    # the response cache in llm_cache.py is what actually makes a run reproducible.
    llm_seed: int = 0

    # Voter toggles (any voter can be turned off; the consensus uses whoever is on)
    enable_celltypist: bool = True  # CellTypist ML classifier voter
    enable_llm: bool = True  # LLM marker-reasoning voter
    enable_singler: bool = False  # requires R + SingleR + celldex; explicit opt-in
    enable_pubmed: bool = (
        False  # literature (PubMed RAG) voter; needs network + OpenRouter
    )

    # --- mixed-cluster refinement ---
    # The consensus emits ONE label per cluster, so a cluster holding several cell types
    # can only be given one of them. CellTypist's heterogeneity metrics already detect
    # that case (mixed_cluster_flag); with this on, those clusters are split and the
    # voters run on the sub-clusters instead of on the merged group.
    #
    # Measured motivation: on a lung cohort 8 of 12 clusters were flagged mixed, and one
    # of them held 2,852 smooth-muscle cells, 650 myofibroblasts and 532 neurons under a
    # single "Airway smooth muscle cell" label.
    #
    # Costs extra per-cluster voter calls (one per new sub-cluster), which is what
    # refine_max_new_clusters bounds.
    refine_mixed_clusters: bool = True
    refine_resolution: float = 0.30
    refine_min_subcluster_cells: int = 30
    refine_max_new_clusters: int = 24

    # Biology context (disease-agnostic)
    tissue: Optional[str] = None
    species: Optional[str] = (
        None  # e.g. "human"/"mouse"; used to pick the SingleR reference
    )
    # "auto" => LLM picks a tissue-appropriate model from the human catalog,
    # falling back to the general immune model. A concrete .pkl name is used
    # as-is. Selection is tissue-driven, never disease-driven.
    celltypist_model: str = "auto"
    # "auto" => LLM picks a species+tissue-appropriate SingleR/celldex reference,
    # falling back to the broad general reference below. A concrete name is used
    # as-is. Selection is species/tissue-driven, never disease-driven.
    singler_reference: str = "auto"

    # Marker computation
    top_n_markers: int = 50
    cluster_col: str = "leiden"
    # Fraction of a cluster's cells that must express a gene for it to count as a
    # marker. See tools.MARKER_RANKING_METHOD — markers are ranked by effect size,
    # not by adjusted p-value.
    min_detection_fraction: float = 0.10

    # Mixed / heterogeneous cluster detection from the per-cell CellTypist labels.
    # A cluster whose dominant label covers < min_dominant_fraction of cells, or
    # whose runner-up covers >= second_label_fraction, is flagged mixed. The flag is
    # advisory: it never changes the consensus label, only reports uncertainty.
    mixed_cluster_min_dominant_fraction: float = 0.70
    mixed_cluster_second_label_fraction: float = 0.20

    # Whether downstream (DE/composition) analyses may key off celltype_subtype
    # instead of the coarse consensus. Default False — subtypes are single-annotator
    # calls and are not consensus-validated.
    use_subtypes_for_downstream: bool = False

    # SingleR bridge
    rscript_exe: str = "Rscript"

    def validate(self) -> "ConsensusConfig":
        """Validate voter/credential combinations and return ``self`` for chaining.

        Raises ``ConsensusConfigError`` when a voter is enabled without the
        credentials/inputs it needs (e.g. ``enable_llm=True`` with no
        ``OPENROUTER_API_KEY``), so misconfiguration fails fast rather than at run time.
        """
        if self.enable_llm:
            if not self.openrouter_api_key:
                raise ConsensusConfigError(
                    "enable_llm=True but OPENROUTER_API_KEY is not set. "
                    "Add it to .env, or disable the LLM voter explicitly (enable_llm=False)."
                )
            if not self.openrouter_model:
                raise ConsensusConfigError(
                    "enable_llm=True but OPENROUTER_MODEL is not set. "
                    "Set OPENROUTER_MODEL in .env (no default model is hardcoded)."
                )
        if self.top_n_markers < 1:
            raise ConsensusConfigError(
                f"top_n_markers must be >= 1, got {self.top_n_markers}."
            )
        for _name in (
            "min_detection_fraction",
            "mixed_cluster_min_dominant_fraction",
            "mixed_cluster_second_label_fraction",
        ):
            _v = getattr(self, _name)
            if (
                not isinstance(_v, (int, float))
                or isinstance(_v, bool)
                or not (0.0 <= float(_v) <= 1.0)
            ):
                raise ConsensusConfigError(
                    f"{_name} must be a fraction in [0, 1], got {_v!r}."
                )
        return self


def load_config(
    *,
    enable_celltypist: bool = True,
    enable_llm: bool = True,
    enable_singler: bool = False,
    enable_pubmed: bool = False,
    tissue: Optional[str] = None,
    species: Optional[str] = None,
    celltypist_model: Optional[str] = None,
    singler_reference: Optional[str] = None,
    top_n_markers: int = 50,
    cluster_col: str = "leiden",
    rscript_exe: Optional[str] = None,
    min_detection_fraction: Optional[float] = None,
    mixed_cluster_min_dominant_fraction: Optional[float] = None,
    mixed_cluster_second_label_fraction: Optional[float] = None,
    use_subtypes_for_downstream: bool = False,
) -> ConsensusConfig:
    """Build a validated ConsensusConfig from .env + explicit overrides.

    Explicit kwargs win over the environment for the biology/toggle settings so
    the host pipeline can drive behaviour; LLM credentials always come from .env.

    DELIBERATE EXCEPTION to the 5-parameter soft limit (Rule 7.2). This function IS the
    validation boundary for ``ConsensusConfig``: each parameter is named here so a
    caller's value can be checked and defaulted against the environment before the
    frozen config is built. Replacing the list with an options object would restate the
    same 15 names in a second place and move the validation away from the boundary. All
    parameters are keyword-only (Rule 6.2), so no call site can pass them positionally.
    """
    _load_dotenv_best_effort()

    cfg = ConsensusConfig(
        openrouter_api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip() or None,
        openrouter_model=(os.getenv("OPENROUTER_MODEL") or "").strip() or None,
        endpoint=(os.getenv("OPENROUTER_ENDPOINT") or DEFAULT_ENDPOINT).strip(),
        llm_max_retries=_get_int("LLM_MAX_RETRIES", 3),
        llm_timeout_s=_get_int("LLM_TIMEOUT_S", 60),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.0),
        llm_max_tokens=_get_int("LLM_MAX_TOKENS", 800),
        llm_top_p=_get_float("LLM_TOP_P", 1.0),
        llm_seed=_get_int("LLM_SEED", 0),
        refine_mixed_clusters=_get_bool("REFINE_MIXED_CLUSTERS", True),
        refine_resolution=_get_float("REFINE_RESOLUTION", 0.30),
        refine_min_subcluster_cells=_get_int("REFINE_MIN_SUBCLUSTER_CELLS", 30),
        refine_max_new_clusters=_get_int("REFINE_MAX_NEW_CLUSTERS", 24),
        enable_celltypist=enable_celltypist,
        enable_llm=enable_llm,
        enable_singler=enable_singler,
        enable_pubmed=enable_pubmed,
        tissue=tissue,
        species=species,
        celltypist_model=celltypist_model or "auto",
        singler_reference=singler_reference or "auto",
        top_n_markers=top_n_markers,
        cluster_col=cluster_col,
        rscript_exe=(
            rscript_exe or env_names.get_env(env_names.RSCRIPT_EXE) or "Rscript"
        ),
        # None => keep the dataclass default (0.10 / 0.70 / 0.20). Values are passed
        # through unconverted so validate() reports a bad one with a clear message.
        **{
            k: v
            for k, v in (
                ("min_detection_fraction", min_detection_fraction),
                (
                    "mixed_cluster_min_dominant_fraction",
                    mixed_cluster_min_dominant_fraction,
                ),
                (
                    "mixed_cluster_second_label_fraction",
                    mixed_cluster_second_label_fraction,
                ),
            )
            if v is not None
        },
        use_subtypes_for_downstream=bool(use_subtypes_for_downstream),
    )
    return cfg.validate()
