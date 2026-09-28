from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any

from app_core.state import (
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_APP_TITLE,
    DEFAULT_OPENROUTER_HTTP_REFERER,
    AppPreferences,
    ProviderConnection,
    normalize_openrouter_app_title,
    normalize_openrouter_http_referer,
)
from app_core.runtime_providers import normalize_provider, runtime_base_url
from app_core.secret_store import get_secret

logger = logging.getLogger(__name__)
_PROVIDER_API_KEY_ENV_NAMES = {
    "openrouter": ("OPENROUTER_API_KEY", "LLM_API_KEY"),
    "ollama": ("OLLAMA_API_KEY", "LLM_API_KEY"),
    "lmstudio": ("LLM_API_KEY",),
    "openai_compatible": ("LLM_API_KEY",),
}


@dataclass
class RuntimeConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    connection_id: str = ""
    label: str = ""
    is_local: bool = False
    provider_metadata: dict[str, Any] = field(default_factory=dict)


def _provider_env_api_key(provider: str) -> str:
    for name in _PROVIDER_API_KEY_ENV_NAMES.get(provider, ("LLM_API_KEY",)):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _connection_api_key(connection: ProviderConnection, provider: str) -> str:
    if connection.secret_ref:
        secret = get_secret(connection.secret_ref)
        if secret:
            return secret
        logger.warning("Saved secret %s is unavailable; checking provider environment variables.", connection.secret_ref)
    return _provider_env_api_key(provider)


def active_connection(prefs: AppPreferences) -> ProviderConnection | None:
    connections = list(getattr(prefs, "connections", []) or [])
    if not connections:
        return None
    if getattr(prefs, "active_connection_id", ""):
        match = next((item for item in connections if item.connection_id == prefs.active_connection_id), None)
        if match is not None:
            return match
    match = next((item for item in connections if item.is_default), None)
    if match is not None:
        return match
    return connections[0]


def resolve_connection_runtime(connection: ProviderConnection) -> RuntimeConfig:
    provider = normalize_provider(connection.provider_kind)
    metadata = dict(connection.provider_metadata or {})
    runtime = RuntimeConfig(
        provider=provider,
        model=str(connection.default_model or DEFAULT_MODEL),
        base_url=runtime_base_url(provider, connection.base_url),
        api_key=_connection_api_key(connection, provider),
        connection_id=str(connection.connection_id or ""),
        label=str(connection.label or ""),
        is_local=bool(connection.is_local),
        provider_metadata=metadata,
    )
    if provider == "openrouter":
        runtime.extra_headers["HTTP-Referer"] = normalize_openrouter_http_referer(metadata.get("http_referer"))
        title = normalize_openrouter_app_title(metadata.get("app_title"))
        runtime.extra_headers["X-OpenRouter-Title"] = title
        runtime.extra_headers["X-Title"] = title
    return runtime


def resolve_runtime_config(prefs: AppPreferences) -> RuntimeConfig:
    connection = active_connection(prefs)
    if connection is None:
        raise ValueError("No active runtime connection is configured.")
    return resolve_connection_runtime(connection)


def sync_runtime_fields(prefs: AppPreferences) -> RuntimeConfig:
    runtime = resolve_runtime_config(prefs)
    prefs.provider = runtime.provider
    prefs.model = runtime.model
    prefs.llm_base_url = runtime.base_url
    prefs.llm_api_key = runtime.api_key
    if runtime.provider == "openrouter":
        prefs.openrouter_http_referer = runtime.extra_headers.get("HTTP-Referer", DEFAULT_OPENROUTER_HTTP_REFERER)
        prefs.openrouter_app_title = runtime.extra_headers.get("X-Title", DEFAULT_OPENROUTER_APP_TITLE)
    return runtime
