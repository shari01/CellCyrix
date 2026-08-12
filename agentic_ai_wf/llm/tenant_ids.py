"""Tenant-scoped identifiers for checkpointers / LangGraph ``thread_id`` (Phase 7, strict mode Phase 9.2)."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# tnt:{uuid}:{logical} — logical may contain colons; split after 5th segment for UUID
_TNT_PREFIX = re.compile(r"^tnt:([0-9a-fA-F-]{36}):(.+)$")

_WARNED_UNSCOPED = False


class MissingTenantContextError(ValueError):
    """
    *organization_id* is required to compute a tenant-scoped ``thread_id``.

    Raised when :data:`TENANCY_STRICT_THREAD_IDS` is active and ``scoped_thread_id``
    is called without a tenant.
    """


class ThreadTenantMismatchError(PermissionError):
    """A checkpoint ``thread_id`` is not owned by the expected organization."""


def _strict_thread_ids() -> bool:
    """
    When True, :func:`scoped_thread_id` requires a non-empty *organization_id*.

    If Django is not configured (e.g. ad-hoc scripts), returns False so local tools keep working.
    """
    try:
        from django.conf import settings

        if not getattr(settings, "configured", False):
            return False
        return bool(getattr(settings, "TENANCY_STRICT_THREAD_IDS", True))
    except Exception as exc:  # noqa: BLE001 - best-effort boundary; falls back below
        logger.debug("%s: falling back after %r", __name__, exc)
        return False


def _audit_thread_context_event(
    *,
    event_type: str,
    logical_id: str = "",
    expected_organization_id: str = "",
    thread_id: str = "",
    reason: str = "",
) -> None:
    try:
        from core.services.tenant_audit import record_tenant_audit_event

        oid: Optional[uuid.UUID] = None
        s = (expected_organization_id or "").strip()
        if s:
            try:
                oid = uuid.UUID(s)
            except (ValueError, TypeError, AttributeError):
                oid = None
        meta: dict[str, Any] = {"reason": reason}
        if logical_id:
            meta["logical_id"] = (logical_id or "")[:128]
        if thread_id:
            meta["thread_id"] = (thread_id or "")[:200]
        record_tenant_audit_event(
            event_type=event_type,
            organization_id=oid,
            resource_type="langgraph.thread",
            resource_id=(logical_id or thread_id or "")[:256],
            metadata=meta,
        )
    except Exception:
        logger.debug("tenancy audit skipped", exc_info=True)


def tenant_scoped_thread_id(organization_id: str | None, logical_id: str) -> str:
    """
    Namespace checkpoint keys per tenant to avoid cross-tenant collision in shared storage.

    Format: ``tnt:{tenant_uuid}:{logical_id}``; no org: ``u:{logical_id}`` (namespaced, not raw).
    """
    lid = (logical_id or "").strip() or "unknown"
    oid = (organization_id or "").strip()
    if oid:
        return f"tnt:{oid}:{lid}"
    return f"u:{lid}"


def scoped_thread_id(organization_id: str | None, logical_id: str) -> str:
    """
    LangGraph checkpointer / session ``thread_id``.

    When *organization_id* is set: ``tnt:{org}:{logical_id}``.

    When *organization_id* is empty and :data:`TENANCY_STRICT_THREAD_IDS` is True, raises
    :class:`MissingTenantContextError` and records an audit row (Django available).

    When strict mode is off, returns *logical_id* only (legacy) and logs a warning (once per process
    for the unscoped case).
    """
    global _WARNED_UNSCOPED
    lid = (logical_id or "").strip() or "unknown"
    oid = (organization_id or "").strip()
    if oid:
        return f"tnt:{oid}:{lid}"
    if _strict_thread_ids():
        _audit_thread_context_event(
            event_type="access.tenancy_thread_context_required",
            logical_id=lid,
            reason="scoped_thread_id_missing_organization",
        )
        raise MissingTenantContextError(
            "organization_id is required to construct a tenant-scoped thread_id when "
            "TENANCY_STRICT_THREAD_IDS is enabled (see settings)."
        )
    if not _WARNED_UNSCOPED:
        logger.warning(
            "thread_id has no organization scope (legacy key); set organization_id to "
            "enforce isolation: logical_id_len=%s",
            len(lid),
        )
        _WARNED_UNSCOPED = True
    return lid


def parse_scoped_thread_id(thread_id: str) -> Tuple[Optional[uuid.UUID], str]:
    """
    Split ``tnt:{org}:{rest}`` into (organization_id, logical_id).
    For legacy raw *thread_id*, returns (None, thread_id).
    """
    raw = (thread_id or "").strip()
    m = _TNT_PREFIX.match(raw)
    if not m:
        return (None, raw)
    org_s, logical = m.group(1), m.group(2)
    try:
        return (uuid.UUID(org_s), logical)
    except (ValueError, TypeError):
        return (None, raw)


def require_thread_tenant_match(
    thread_id: str,
    expected_organization_id: str | uuid.UUID | None,
) -> bool:
    """True if *thread_id* is tenant-scoped and its org matches (or thread is legacy and expected is None)."""
    if not expected_organization_id:
        return True
    p_org, _ = parse_scoped_thread_id(thread_id)
    if p_org is None:
        return False
    exp = (
        expected_organization_id
        if isinstance(expected_organization_id, uuid.UUID)
        else uuid.UUID(str(expected_organization_id))
    )
    return p_org == exp


def ensure_thread_tenant_match(
    thread_id: str,
    expected_organization_id: str | uuid.UUID,
    *,
    audit: bool = True,
) -> None:
    """
    Enforce that *thread_id* is scoped to *expected_organization_id*.

    Raises :class:`ThreadTenantMismatchError` for legacy *thread_id* when a tenant is required,
    or when the embedded org does not match.
    """
    if require_thread_tenant_match(thread_id, expected_organization_id):
        return
    if audit:
        exp_s = str(expected_organization_id)
        _audit_thread_context_event(
            event_type="access.tenancy_thread_mismatch",
            thread_id=thread_id,
            expected_organization_id=exp_s,
            reason="thread_tenant_mismatch",
        )
    raise ThreadTenantMismatchError(
        "thread_id is not scoped to the active organization; refusing to load checkpoint state"
    )


def reset_scoped_thread_warning_for_tests() -> None:
    """Re-arm the once-per-process unscoped-thread warning. Test helper only."""
    global _WARNED_UNSCOPED
    _WARNED_UNSCOPED = False


# Alias for call sites that refer to "context" at checkpoint boundaries
require_thread_tenant_context = ensure_thread_tenant_match
