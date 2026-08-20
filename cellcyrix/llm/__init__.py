"""Central OpenRouter-backed LLM access."""

from . import env_names, fallbacks, models, settings
from .agents_openrouter import (
    openrouter_run_config,
    reset_agents_openrouter_client_cache,
)
from .clients import (
    OpenRouterSyncClient,
    create_async_openrouter,
    ensure_openrouter_env_for_openai_sdk,
    get_async_openrouter,
    get_sync_openrouter,
    openrouter_client_for_key,
    sync_openrouter_with_timeout,
    try_get_sync_openrouter,
)
from .fallbacks import (
    async_chat_completions_with_fallbacks,
    chat_completions_with_fallbacks,
    should_failover,
)
from .models import (
    ModelTier,
    models_for_tier,
    pick_model_for_call,
    resolve_responses_model,
    routing_model,
)
from .settings import LLM_RESPONSES_MODEL

__all__ = [
    "ModelTier",
    "OpenRouterSyncClient",
    "async_chat_completions_with_fallbacks",
    "chat_completions_with_fallbacks",
    "create_async_openrouter",
    "ensure_openrouter_env_for_openai_sdk",
    "get_async_openrouter",
    "get_sync_openrouter",
    "models_for_tier",
    "openrouter_client_for_key",
    "openrouter_run_config",
    "reset_agents_openrouter_client_cache",
    "pick_model_for_call",
    "LLM_RESPONSES_MODEL",
    "resolve_responses_model",
    "routing_model",
    "should_failover",
    "sync_openrouter_with_timeout",
    "try_get_sync_openrouter",
    "env_names",
    "fallbacks",
    "models",
    "settings",
]
