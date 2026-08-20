"""LangChain ChatOpenAI pointed at OpenRouter."""

from __future__ import annotations

from typing import Any, List, Union

from langchain_openai import ChatOpenAI

from cellcyrix.llm import settings as ls
from cellcyrix.llm.clients import _default_headers
from cellcyrix.llm.env_names import OPENROUTER_API_KEY_ENV
from cellcyrix.llm.invocation_context import get_llm_invocation_context
from cellcyrix.llm.langchain_callbacks import TenantLLMUsageCallbackHandler
from cellcyrix.llm.tenant_llm_config import TenantLLMConfig, coalesce_tenant_key


def chat_openrouter(*, model: str | None = None, **kwargs: Any) -> ChatOpenAI:
    """
    LangChain chat model: tenant API key (if configured), usage logging, PHI policy on context.
    Set :func:`use_llm_invocation_context` for tenant/user in meta-agent / jobs.
    """
    ctx = get_llm_invocation_context()
    cfg = TenantLLMConfig.for_organization(ctx.tenant_id if ctx else None)
    api_key = coalesce_tenant_key(cfg, OPENROUTER_API_KEY_ENV)
    cbs: List[Union[TenantLLMUsageCallbackHandler, Any]] = []
    raw = kwargs.pop("callbacks", None)
    if raw is not None:
        if isinstance(raw, (list, tuple)):
            cbs.extend(raw)
        else:
            cbs.append(raw)
    cbs.append(TenantLLMUsageCallbackHandler())
    return ChatOpenAI(
        model=model or ls.LLM_MODEL_CHAT,
        api_key=api_key,
        base_url=ls.OPENROUTER_BASE_URL,
        default_headers=_default_headers(),
        callbacks=cbs,
        **kwargs,
    )
