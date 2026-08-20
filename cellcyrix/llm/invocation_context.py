"""Per-request LLM invocation context (tenant, user, PHI policy) via ``ContextVar``."""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator, Optional

_llm_ctx: contextvars.ContextVar[Optional["LLMInvocationContext"]] = (
    contextvars.ContextVar("llm_invocation", default=None)
)


@dataclass(frozen=True, slots=True)
class LLMInvocationContext:
    """Carried across nested LLM calls in one workflow (Celery task / async run)."""

    tenant_id: Optional[str]
    user_id: Optional[str]
    phi_policy: bool
    """True when prompts may include PHI; used for compliance logging (not content logging)."""


def get_llm_invocation_context() -> Optional[LLMInvocationContext]:
    """The invocation context bound to the current task, or None."""
    return _llm_ctx.get()


@contextlib.contextmanager
def use_llm_invocation_context(ctx: LLMInvocationContext) -> Iterator[None]:
    """Bind `ctx` for the `with` block, restoring the previous context after."""
    token = _llm_ctx.set(ctx)
    try:
        yield
    finally:
        _llm_ctx.reset(token)


def bind_llm_invocation_context(ctx: LLMInvocationContext) -> contextvars.Token:
    """Bind `ctx` and return the token that undoes it.

    The paired bind/unbind form exists so a large callable does not have to be
    indented inside a `with` block.

    Args:
        ctx: The invocation context to bind.

    Returns:
        A token to pass to :func:`unbind_llm_invocation_context`.
    """
    return _llm_ctx.set(ctx)


def unbind_llm_invocation_context(token: contextvars.Token) -> None:
    """Restore the context that `bind_llm_invocation_context` replaced.

    Args:
        token: The token returned by the matching bind call.
    """
    _llm_ctx.reset(token)
