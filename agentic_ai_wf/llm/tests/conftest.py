"""Test setup for the LLM layer.

Why this exists
---------------
``llm_usage_logger.log_llm_invocation`` writes a tenant audit record through the
platform's Django app (``core.models`` / ``core.services.tenant_audit``). That app is
a deployment dependency, not a dependency of this package, so it is not installed
here — and ``test_telemetry_logging.py`` patches ``core.services.tenant_audit`` by
name, which ``unittest.mock.patch`` can only do if the module is importable.

Without this file those two tests failed with ``ModuleNotFoundError: No module named
'core'``. They were invisible for a while because ``testpaths`` did not include this
directory; once it did, they failed. A unit test must not depend on a service being
installed (standards §14.8), so the service is stubbed instead: minimal in-memory
modules registered in ``sys.modules`` for the duration of the session, which makes
the audit branch actually execute and be assertable.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest


def _build_core_stub() -> dict[str, types.ModuleType]:
    """Build the minimal ``core`` package the audit branch imports.

    Returns:
        Mapping of module name to stub module, ready for `sys.modules`.
    """
    core = types.ModuleType("core")
    core.__path__ = []  # mark as a package so submodule imports resolve

    models = types.ModuleType("core.models")

    class TenantAuditEventType:
        """Stub of the platform's audit event enum; only LLM_CALL is used here."""

        LLM_CALL = "llm_call"

    models.TenantAuditEventType = TenantAuditEventType

    services = types.ModuleType("core.services")
    services.__path__ = []

    tenant_audit = types.ModuleType("core.services.tenant_audit")

    def record_tenant_audit_event(**kwargs: object) -> None:
        """No-op stand-in. Tests patch this name to assert it was called."""

    tenant_audit.record_tenant_audit_event = record_tenant_audit_event

    core.models = models
    core.services = services
    services.tenant_audit = tenant_audit

    return {
        "core": core,
        "core.models": models,
        "core.services": services,
        "core.services.tenant_audit": tenant_audit,
    }


@pytest.fixture(autouse=True, scope="session")
def stub_platform_core() -> Iterator[None]:
    """Register the ``core`` stub for the session, restoring `sys.modules` after.

    Autouse and session-scoped: the import happens inside `log_llm_invocation`, so
    the stub has to be in place for any test that calls it, and registering it once
    keeps the patch target stable across tests.

    Yields:
        None, for the duration of the test session.
    """
    stubs = _build_core_stub()
    # A real `core` (running inside the platform) must win over the stub.
    stubs = {name: mod for name, mod in stubs.items() if name not in sys.modules}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
