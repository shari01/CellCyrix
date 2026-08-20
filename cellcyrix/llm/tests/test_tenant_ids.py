import uuid

import pytest

from cellcyrix.llm.tenant_ids import (
    MissingTenantContextError,
    ThreadTenantMismatchError,
    ensure_thread_tenant_match,
    parse_scoped_thread_id,
    require_thread_tenant_match,
    reset_scoped_thread_warning_for_tests,
    scoped_thread_id,
    tenant_scoped_thread_id,
)


def test_scoped_thread_id_with_org():
    assert (
        scoped_thread_id("00000000-0000-0000-0000-000000000001", "abc")
        == "tnt:00000000-0000-0000-0000-000000000001:abc"
    )


def test_scoped_thread_id_no_org_legacy(monkeypatch, caplog):
    monkeypatch.setattr("cellcyrix.llm.tenant_ids._strict_thread_ids", lambda: False)
    reset_scoped_thread_warning_for_tests()
    with caplog.at_level("WARNING"):
        assert scoped_thread_id(None, "run-1") == "run-1"
        assert scoped_thread_id("", "run-1") == "run-1"
    assert "thread_id has no organization scope" in caplog.text


def test_scoped_thread_id_strict_missing_org(monkeypatch):
    monkeypatch.setattr("cellcyrix.llm.tenant_ids._strict_thread_ids", lambda: True)
    with pytest.raises(MissingTenantContextError):
        scoped_thread_id(None, "run-x")


def test_parse_scoped_thread_id_roundtrip():
    oid = uuid.UUID("00000000-0000-0000-0000-0000000000ab")
    tid = scoped_thread_id(str(oid), "same-analysis")
    p_org, logical = parse_scoped_thread_id(tid)
    assert p_org == oid
    assert logical == "same-analysis"
    assert parse_scoped_thread_id("legacy-only")[0] is None


def test_same_logical_id_different_tenants_distinct_threads():
    a = uuid.uuid4()
    b = uuid.uuid4()
    assert scoped_thread_id(str(a), "run-x") != scoped_thread_id(str(b), "run-x")


def test_require_thread_tenant_match():
    oid = uuid.uuid4()
    tid = scoped_thread_id(str(oid), "x")
    assert require_thread_tenant_match(tid, oid)
    assert not require_thread_tenant_match(tid, uuid.uuid4())
    assert require_thread_tenant_match("legacy", None)
    assert not require_thread_tenant_match("legacy", oid)


def test_ensure_thread_tenant_mismatch(monkeypatch):
    monkeypatch.setattr("cellcyrix.llm.tenant_ids._strict_thread_ids", lambda: True)
    oid = uuid.uuid4()
    other = uuid.uuid4()
    tid = scoped_thread_id(str(oid), "s")
    with pytest.raises(ThreadTenantMismatchError):
        ensure_thread_tenant_match(tid, other, audit=False)


def test_ensure_rejects_legacy_thread_when_org_required():
    with pytest.raises(ThreadTenantMismatchError):
        ensure_thread_tenant_match("legacy-raw", uuid.uuid4(), audit=False)


def test_tenant_scoped_thread_id_no_org_u_prefix():
    assert tenant_scoped_thread_id(None, "x") == "u:x"
