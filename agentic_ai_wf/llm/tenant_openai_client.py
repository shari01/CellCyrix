"""OpenAI SDK (OpenRouter) client with tenant model policy + usage logging (sync)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from agentic_ai_wf.llm.invocation_context import (
    LLMInvocationContext,
    get_llm_invocation_context,
)
from agentic_ai_wf.llm.llm_usage_logger import log_llm_invocation
from agentic_ai_wf.llm.tenant_llm_config import TenantLLMConfig

logger = logging.getLogger(__name__)

_PROVIDER = "openrouter"


def _log_completion(
    *,
    config: TenantLLMConfig,
    inv: Optional[LLMInvocationContext],
    model: str,
    resp: ChatCompletion,
) -> None:
    usage = resp.usage
    in_t = getattr(usage, "prompt_tokens", None) if usage else None
    out_t = getattr(usage, "completion_tokens", None) if usage else None
    tot = getattr(usage, "total_tokens", None) if usage else None
    tid = (inv.tenant_id if inv else None) or config.organization_id
    uid = inv.user_id if inv else None
    phi = inv.phi_policy if inv else config.phi_allowed
    log_llm_invocation(
        provider=_PROVIDER,
        model=model or (resp.model or ""),
        input_tokens=in_t,
        output_tokens=out_t,
        total_tokens=tot,
        tenant_id=tid,
        user_id=str(uid) if uid is not None else None,
        phi_policy=bool(phi),
    )


@dataclass
class _TenantCompletions:
    _inner: Any
    _config: TenantLLMConfig
    _inv: Optional[LLMInvocationContext]

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion after checking the model against the policy.

        Raises:
            ValueError: If the requested model is not allowed for this tenant.
        """
        model = (kwargs.get("model") or (args[0] if args else None) or "").strip()
        if model and not self._config.is_model_allowed(model):
            raise ValueError(f"Model {model!r} is not allowed for this tenant policy")
        resp: ChatCompletion = self._inner.create(*args, **kwargs)
        try:
            _log_completion(
                config=self._config,
                inv=self._inv,
                model=str(model or resp.model or ""),
                resp=resp,
            )
        except Exception as exc:
            logger.debug("llm usage log failed: %s", exc)
        return resp


@dataclass
class _TenantChat:
    _client: "TenantOpenAIClient"

    @property
    def completions(self) -> _TenantCompletions:
        """Policy-enforcing view of `chat.completions`."""
        return _TenantCompletions(
            self._client._raw.chat.completions,
            self._client._config,
            self._client._inv,
        )


class TenantOpenAIClient:
    """Narrow surface: ``.chat.completions.create`` used by bio self-coder and others."""

    __slots__ = ("_raw", "_config", "_inv")

    def __init__(
        self,
        raw: OpenAI,
        config: TenantLLMConfig,
        inv: Optional[LLMInvocationContext],
    ) -> None:
        self._raw = raw
        self._config = config
        self._inv = inv

    @property
    def chat(self) -> _TenantChat:
        """Policy-enforcing view of the `chat` namespace."""
        return _TenantChat(self)

    def with_options(self, *args: Any, **kwargs: Any) -> "TenantOpenAIClient":
        """Return a re-wrapped client carrying the same tenant policy."""
        r2 = self._raw.with_options(*args, **kwargs)
        if not isinstance(r2, OpenAI):
            return self
        return TenantOpenAIClient(
            r2, self._config, self._inv or get_llm_invocation_context()
        )


def build_tenant_openai(
    base: OpenAI,
    cfg: Optional[TenantLLMConfig] = None,
    inv: Optional[LLMInvocationContext] = None,
) -> TenantOpenAIClient:
    """Wrap an OpenAI client so every call is checked against a tenant policy.

    Args:
        base: The underlying OpenAI client.
        cfg: Tenant policy. Resolved from `inv`'s tenant when omitted.
        inv: Invocation context supplying tenant/user attribution.

    Returns:
        A policy-enforcing wrapper around `base`.
    """
    if cfg is None:
        cfg = TenantLLMConfig.for_organization((inv.tenant_id if inv else None))
    if inv is None:
        inv = get_llm_invocation_context()
    return TenantOpenAIClient(base, cfg, inv)
