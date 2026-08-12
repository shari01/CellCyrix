"""
On-disk folder segments and workflow node identifiers.

**Authoritative ids** are :class:`AgentName` values (``*_node``). They are the only
accepted strings for disk paths and node resolution.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Final, Union

try:
    from analysisapp.constants import GUIDED_PIPELINE_RUN_PLAN_SOURCE
except Exception:  # pragma: no cover
    GUIDED_PIPELINE_RUN_PLAN_SOURCE = "guided_pipeline_run"


class AgentName(str, Enum):
    """Workflow node id and disk segment (same string).

    For logs: each visit to a plan node must emit at least two persisted rows via
    :class:`~agentic_ai_wf.helpers.logging_utils.AsyncWorkflowLogger` — a
    ``node_started`` row and a terminal row (``node_completed`` | ``node_failed`` |
    ``node_skipped``). See :mod:`agentic_ai_wf.pipeline_node_log` phase constants.
    """

    COHORT_RETRIEVAL_NODE = "cohort_retrieval_node"
    DEG_NODE = "deg_node"
    GENE_PRIORITIZATION_NODE = "gene_prioritization_node"
    PATHWAY_ENRICHMENT_NODE = "pathway_enrichment_node"
    GSEA_PATHWAYS_ENRICHMENT_NODE = "gsea_pathways_enrichment_node"
    DECONVOLUTION_NODE = "deconvolution_node"
    TEMPORAL_NODE = "temporal_node"
    TEMPORAL_GRANGER_NODE = "temporal_granger_node"
    HARMONIZATION_NODE = "harmonization_node"
    MDP_NODE = "mdp_node"
    PERTURBATION_NODE = "perturbation_node"
    MULTIOMICS_NODE = "multiomics_node"
    FASTQ_PROCESSING_NODE = "fastq_processing_node"
    MOLECULAR_REPORT_NODE = "molecular_report_node"
    CRISPR_PERTURB_SEQ_NODE = "crispr_perturb_seq_node"
    CRISPR_SCREENING_NODE = "crispr_screening_node"
    CRISPR_TARGETED_NODE = "crispr_targeted_node"
    GWAS_MR_NODE = "gwas_mr_node"
    SIGNOR_CAUSALITY_NODE = "signor_causality_node"
    CAUSAL_DAG_BUILDER_NODE = "causal_dag_builder_node"
    CAUSALITY_NODE = "causality_node"
    IPAA_NODE = "ipaa_node"
    SINGLE_CELL_NODE = "single_cell_node"
    DRUG_DISCOVERY_NODE = "drug_discovery_node"
    CLINICAL_REPORT_NODE = "clinical_report_node"
    FINALIZATION_NODE = "finalization_node"


AGENT_IDENTIFIER_INDEX: Final[dict[str, AgentName]] = {m.value: m for m in AgentName}

AGENT_ID_TO_AGENT_NAME: Final[dict[str, AgentName]] = AGENT_IDENTIFIER_INDEX
NODE_KEY_TO_AGENT_NAME: Final[dict[str, AgentName]] = AGENT_IDENTIFIER_INDEX


def resolve_node_id_for_adapter(module_id: str) -> str:
    """Return ``module_id`` if it is a valid :class:`AgentName` value, else raise."""
    if not module_id:
        return module_id
    s = module_id.strip()
    mapped = AGENT_IDENTIFIER_INDEX.get(s)
    if mapped is not None:
        return mapped.value
    raise ValueError(
        f"Unknown module id {module_id!r}. Use an AgentName.value (e.g. {AgentName.DEG_NODE.value!r})."
    )


def disk_segment_for_agent(agent_id: Union[str, AgentName]) -> str:
    """Return the on-disk path segment for this agent (``*_node``)."""
    if isinstance(agent_id, AgentName):
        return agent_id.value + "_results"
    aid = (agent_id or "").strip()
    if not aid:
        raise ValueError("empty agent_id")
    canonical = resolve_node_id_for_adapter(aid)
    mapped = AGENT_IDENTIFIER_INDEX.get(canonical)
    if mapped is not None:
        return mapped.value + "_results"
    raise ValueError(
        f"Unknown agent id {agent_id!r}. Use an AgentName.value string (e.g. {AgentName.DEG_NODE.value!r})."
    )


def agent_folder_for_config_node_key(node_key: str) -> str:
    """On-disk folder segment for a pipeline-config node key.

    Args:
        node_key: The config node key naming an agent.

    Returns:
        The filesystem-safe folder segment for that agent's outputs.
    """
    return disk_segment_for_agent(node_key)


def guided_fs_segment() -> str:
    """
    Safe directory segment for guided (pipeline designer) outputs: ``shared/<segment>/``.

    Override with env ``GUIDED_PIPELINE_FS_SEGMENT``. Default: slug from
    ``GUIDED_PIPELINE_RUN_PLAN_SOURCE`` (``:`` and ``/`` → ``_``).
    """
    env = (os.environ.get("GUIDED_PIPELINE_FS_SEGMENT") or "").strip().strip("/")
    if env:
        return env
    return GUIDED_PIPELINE_RUN_PLAN_SOURCE.replace(":", "_").replace("/", "_")


def multi_node_fs_segment() -> str:
    """
    Backward-compatible alias for the non-guided supervisor shared directory segment.

    Multi-step non-guided outputs use the same layout as single-step (see
    :func:`agentic_ai_wf.supervisor_agent.output_paths.supervisor_output_fs_segment`).
    """
    try:
        from agentic_ai_wf.supervisor_agent.output_paths import (
            supervisor_output_fs_segment,
        )

        return supervisor_output_fs_segment()
    except Exception:  # pragma: no cover
        return "supervisor_agent_results"
