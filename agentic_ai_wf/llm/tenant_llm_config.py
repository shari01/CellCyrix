"""Tenant-scoped LLM policy: API keys, model allowlists (placeholders), PHI defaults."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional

from decouple import config

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", re.I
)

# JSON: { "<tenant_uuid>": { "api_key": "sk-...", "allowed_models": ["a/b", "c/d"], "phi_allowed": true } }
_TENANT_LLM_POLICIES_ENV = "TENANT_LLM_POLICIES_JSON"
# Optional global allowlist: comma-separated OpenRouter slugs (empty = no restriction)
_GLOBAL_MODEL_ALLOWLIST_ENV = "TENANT_LLM_GLOBAL_MODEL_ALLOWLIST"


@dataclass(frozen=True, slots=True)
class TenantLLMConfig:
    """
    Resolved policy for one organization.

    **Placeholders** (until DB-backed config):
    - Per-tenant OpenRouter key via :envvar:`TENANT_LLM_POLICIES_JSON`
    - Per-tenant allowed model slugs
    - ``phi_allowed`` default per tenant (when omitted, true)
    """

    organization_id: Optional[str]
    openrouter_api_key: Optional[str]
    allowed_models: Optional[FrozenSet[str]]
    phi_allowed: bool

    def is_model_allowed(self, model: str) -> bool:
        """True when `model` is permitted by this tenant's allow-list.

        An empty allow-list means "no restriction", not "nothing allowed".
        """
        if not self.allowed_models:
            return True
        m = (model or "").strip()
        return m in self.allowed_models

    @classmethod
    def for_organization(cls, organization_id: Optional[str]) -> "TenantLLMConfig":
        """Build the config for an organization, else the global default policy.

        Args:
            organization_id: The tenant/org id, or None for the default policy.

        Returns:
            The resolved per-tenant LLM config.
        """
        oid = (organization_id or "").strip() or None
        key: Optional[str] = None
        allow: Optional[FrozenSet[str]] = None
        phi = True
        if oid and _UUID_RE.match(oid):
            raw = (config(_TENANT_LLM_POLICIES_ENV, default="") or "").strip()
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Invalid %s: %s", _TENANT_LLM_POLICIES_ENV, exc, exc_info=True
                    )
                else:
                    if isinstance(data, dict) and isinstance(data.get(oid), dict):
                        ent = data[oid]
                        k = ent.get("api_key") or ent.get("openrouter_api_key")
                        if isinstance(k, str) and k.strip():
                            key = k.strip()
                        am = ent.get("allowed_models")
                        if isinstance(am, list):
                            allow = frozenset(
                                str(x).strip()
                                for x in am
                                if isinstance(x, (str, int, float)) and str(x).strip()
                            )
                        if "phi_allowed" in ent:
                            phi = bool(ent.get("phi_allowed"))
        g_raw = (config(_GLOBAL_MODEL_ALLOWLIST_ENV, default="") or "").strip()
        if g_raw and allow is None:
            allow = frozenset(p.strip() for p in g_raw.split(",") if p.strip())
        return cls(
            organization_id=oid,
            openrouter_api_key=key,
            allowed_models=allow,
            phi_allowed=phi,
        )


def coalesce_tenant_key(config_obj: TenantLLMConfig, primary_env_key: str) -> str:
    """Return tenant key if set, else fall back to global OpenRouter key resolver."""
    if config_obj.openrouter_api_key:
        return config_obj.openrouter_api_key
    from agentic_ai_wf.llm.settings import primary_llm_api_key

    return (primary_llm_api_key() or config(primary_env_key, default="") or "").strip()


class TenantLLMConfigService:
    """Application entry to resolve :class:`TenantLLMConfig` (Phase 7)."""

    @staticmethod
    def for_organization(organization_id: str | None) -> TenantLLMConfig:
        """Alias for `TenantLLMConfig.for_organization`, for holders of this type."""
        return TenantLLMConfig.for_organization(organization_id)
