"""Multi-method consensus cell-type annotation (CellTypist + SingleR + LLM).

Tools/agent separation:
  * tools.py   — logic only (markers, CellTypist, SingleR bridge, harmonize,
                 lineage gate, vote counting). ZERO LLM calls.
  * agent.py   — the only module that calls the LLM (OpenRouter).
  * consensus.py — orchestrator; `run_consensus_annotation` is the entry point.
  * config.py  — .env-driven configuration.
"""

from .config import ConsensusConfig, ConsensusConfigError, load_config
from .consensus import (
    CELLTYPE_CONSENSUS_COL,
    CONSENSUS_UNS_KEY,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    run_consensus_annotation,
)
from .model_provenance import (
    ModelNumericsError,
    collect_model_provenance,
    model_provenance_record,
    pickled_sklearn_versions,
    record_validated_model,
    reset_run_provenance,
    run_provenance,
    validate_model_numerics,
)

__all__ = [
    "run_consensus_annotation",
    "CELLTYPE_CONSENSUS_COL",
    "CONSENSUS_UNS_KEY",
    "TIER_HIGH",
    "TIER_MEDIUM",
    "TIER_LOW",
    "ConsensusConfig",
    "ConsensusConfigError",
    "load_config",
    # model provenance / scikit-learn version-skew validation
    "ModelNumericsError",
    "pickled_sklearn_versions",
    "validate_model_numerics",
    "model_provenance_record",
    "collect_model_provenance",
    "record_validated_model",
    "run_provenance",
    "reset_run_provenance",
]
