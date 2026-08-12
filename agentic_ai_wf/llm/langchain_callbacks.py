"""LangChain callback: log token usage with tenant context (no message bodies)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agentic_ai_wf.llm.invocation_context import get_llm_invocation_context
from agentic_ai_wf.llm.llm_usage_logger import log_llm_invocation
from agentic_ai_wf.llm.tenant_llm_config import TenantLLMConfig

logger = logging.getLogger(__name__)


class TenantLLMUsageCallbackHandler(BaseCallbackHandler):
    """Attach to ``ChatOpenAI`` / chat models; reads ``LLMResult.llm_output`` token_usage."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Emit token usage for a completion-model call. Never raises."""
        try:
            self._emit(response, kwargs)
        except Exception as exc:
            logger.debug("TenantLLMUsageCallbackHandler: %s", exc)

    def on_chat_model_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Emit token usage for a chat-model call. Never raises."""
        try:
            self._emit(response, kwargs)
        except Exception as exc:
            logger.debug("TenantLLMUsageCallbackHandler (chat): %s", exc)

    def _emit(self, response: LLMResult, _kwargs: Any) -> None:
        inv = get_llm_invocation_context()
        cfg = TenantLLMConfig.for_organization((inv.tenant_id if inv else None))
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        if not isinstance(usage, dict) and usage:
            usage = {}
        model = (
            out.get("model_name")
            or out.get("model")
            or (getattr(response, "generations", None) and out.get("id"))
            or "unknown"
        )
        in_t = usage.get("prompt_tokens")
        out_t = usage.get("completion_tokens")
        tot = usage.get("total_tokens")
        tid = (inv.tenant_id if inv else None) or cfg.organization_id
        log_llm_invocation(
            provider="openrouter",
            model=str(model),
            input_tokens=in_t if isinstance(in_t, int) else None,
            output_tokens=out_t if isinstance(out_t, int) else None,
            total_tokens=tot if isinstance(tot, int) else None,
            tenant_id=tid,
            user_id=str(inv.user_id) if inv and inv.user_id is not None else None,
            phi_policy=inv.phi_policy if inv else cfg.phi_allowed,
        )
