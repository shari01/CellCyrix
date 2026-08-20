"""Structured logging for LLM invocations (no prompt/body content)."""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from cellcyrix.llm.telemetry_logging import llm_usage_console_enabled

LOG = logging.getLogger("agenticaib.llm_usage")

# `django` is a platform-deployment dependency, absent when this package is used
# standalone. Resolved HERE rather than inside the try block below, because
# `except OperationalError:` referencing a name imported by that same try raises
# NameError while the handler is being evaluated — and a NameError in the except
# clause propagates past the `except Exception` that follows it. Telemetry would then
# break the caller's request, which is precisely what it must never do.
try:
    from django.db.utils import OperationalError
except ImportError:  # pragma: no cover - exercised only outside the platform

    class OperationalError(Exception):  # type: ignore[no-redef]
        """Stand-in so the handler below is always a valid exception type."""


def log_llm_invocation(
    *,
    provider: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    tenant_id: Optional[str],
    user_id: Optional[str],
    phi_policy: bool,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Record one LLM call for cost and usage accounting.

    Args:
        provider: Provider id (e.g. "openrouter").
        model: Model id actually invoked.
        input_tokens: Prompt tokens, if the provider reported them.
        output_tokens: Completion tokens, if reported.
        total_tokens: Total tokens, if reported.
        tenant_id: Owning tenant, for per-tenant attribution.
        user_id: Requesting user, for per-user attribution.
        phi_policy: True when the call ran under the PHI-restricted policy.
        extra: Additional non-sensitive fields to attach to the record.
    """
    payload = {
        "event": "llm_invocation",
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "phi_policy": phi_policy,
    }
    if extra:
        payload["extra"] = {k: v for k, v in extra.items() if v is not None}
    if llm_usage_console_enabled():
        LOG.info("%s", payload)
    try:
        # `core` is the platform's Django app: present in the deployment, absent when
        # this package runs standalone. Imported lazily so the absence is a skipped
        # audit record, not an import-time failure for every caller.
        from core.models import TenantAuditEventType
        from core.services.tenant_audit import record_tenant_audit_event

        tid = None
        if tenant_id:
            try:
                tid = UUID(str(tenant_id))
            except (ValueError, TypeError):
                tid = None
        uid = None
        if user_id:
            try:
                uid = UUID(str(user_id))
            except (ValueError, TypeError):
                uid = None
        # DPA / subprocessor list: map provider string to legal vendor list in ops (not in code)
        record_tenant_audit_event(
            event_type=TenantAuditEventType.LLM_CALL,
            organization_id=tid,
            actor_id=uid,
            resource_type="llm",
            resource_id=(model or "")[:256],
            metadata={
                "provider": provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "phi_policy": phi_policy,
            },
        )
    except OperationalError as exc:
        LOG.warning("llm_usage: tenant_audit skipped (db): %s", exc)
    except Exception:  # noqa: BLE001 - telemetry must never break the caller's request
        LOG.debug("llm_usage: tenant_audit skipped (unexpected)", exc_info=True)
