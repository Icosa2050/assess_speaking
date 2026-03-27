from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app_shell.runtime_providers import normalize_provider, runtime_base_url
from app_shell.secret_store import get_secret
from app_shell.state import DEFAULT_MODEL, DEFAULT_OPENROUTER_APP_TITLE, DEFAULT_OPENROUTER_HTTP_REFERER, AppPreferences, ProviderConnection


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


def _provider_env_vars(provider: str) -> tuple[str, ...]:
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return ("OPENROUTER_API_KEY", "LLM_API_KEY")
    if normalized == "ollama":
        return ("OLLAMA_API_KEY", "LLM_API_KEY")
    return ("LLM_API_KEY",)


def _env_secret(provider: str) -> str:
    for env_name in _provider_env_vars(provider):
        value = str(os.getenv(env_name) or "").strip()
        if value:
            return value
    return ""


def _connection_api_key(connection: ProviderConnection) -> str:
    if connection.secret_ref:
        secret = get_secret(connection.secret_ref, env_var_names=_provider_env_vars(connection.provider_kind))
        if secret:
            return secret
    return _env_secret(connection.provider_kind)


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
        api_key=_connection_api_key(connection),
        connection_id=str(connection.connection_id or ""),
        label=str(connection.label or ""),
        is_local=bool(connection.is_local),
        provider_metadata=metadata,
    )
    if provider == "openrouter":
        runtime.extra_headers["HTTP-Referer"] = str(
            metadata.get("http_referer") or DEFAULT_OPENROUTER_HTTP_REFERER
        ).strip()
        title = str(metadata.get("app_title") or DEFAULT_OPENROUTER_APP_TITLE).strip()
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
