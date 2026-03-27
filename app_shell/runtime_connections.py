from __future__ import annotations

from typing import Any
from uuid import uuid4

from app_shell.runtime_providers import connection_secret_ref, default_base_url, default_connection_label, normalize_provider
from app_shell.state import DEFAULT_MODEL, ProviderConnection


def _is_local_url(url: str) -> bool:
    candidate = str(url or "").strip().lower()
    return candidate.startswith("http://localhost") or candidate.startswith("http://127.0.0.1")


def deserialize_connections(raw_connections: object) -> list[ProviderConnection]:
    if not isinstance(raw_connections, list):
        return []
    connections: list[ProviderConnection] = []
    for raw in raw_connections:
        if not isinstance(raw, dict):
            continue
        connection_id = str(raw.get("connection_id") or "").strip() or uuid4().hex
        provider_kind = normalize_provider(raw.get("provider_kind") or raw.get("provider"))
        provider_metadata = raw.get("provider_metadata") if isinstance(raw.get("provider_metadata"), dict) else {}
        connection = ProviderConnection(
            connection_id=connection_id,
            provider_kind=provider_kind,
            label=str(raw.get("label") or default_connection_label(provider_kind)).strip() or default_connection_label(provider_kind),
            base_url=str(raw.get("base_url") or default_base_url(provider_kind)).strip(),
            default_model=str(raw.get("default_model") or raw.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            auth_mode="bearer" if str(raw.get("auth_mode") or "").strip().lower() == "bearer" else "none",
            secret_ref=str(raw.get("secret_ref") or connection_secret_ref(connection_id)).strip() or connection_secret_ref(connection_id),
            is_default=bool(raw.get("is_default")),
            is_local=bool(raw.get("is_local")) or _is_local_url(str(raw.get("base_url") or "")),
            last_test_status=str(raw.get("last_test_status") or "").strip(),
            last_tested_at=str(raw.get("last_tested_at") or "").strip(),
            provider_metadata=dict(provider_metadata),
        )
        connections.append(connection)
    return connections


def serialize_connections(connections: list[ProviderConnection]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for connection in connections:
        payload.append(
            {
                "connection_id": connection.connection_id,
                "provider_kind": normalize_provider(connection.provider_kind),
                "label": connection.label,
                "base_url": connection.base_url,
                "default_model": connection.default_model,
                "auth_mode": connection.auth_mode,
                "secret_ref": connection.secret_ref,
                "is_default": connection.is_default,
                "is_local": connection.is_local,
                "last_test_status": connection.last_test_status,
                "last_tested_at": connection.last_tested_at,
                "provider_metadata": dict(connection.provider_metadata or {}),
            }
        )
    return payload


def ensure_single_default_connection(
    connections: list[ProviderConnection],
    active_connection_id: str = "",
) -> tuple[list[ProviderConnection], str]:
    normalized = list(connections or [])
    if not normalized:
        return [], ""
    chosen_active = str(active_connection_id or "").strip()
    if not chosen_active or not any(item.connection_id == chosen_active for item in normalized):
        chosen_default = next((item.connection_id for item in normalized if item.is_default), "")
        chosen_active = chosen_default or normalized[0].connection_id
    saw_default = False
    for connection in normalized:
        connection.is_default = connection.connection_id == chosen_active
        if connection.is_default:
            saw_default = True
    if not saw_default:
        normalized[0].is_default = True
        chosen_active = normalized[0].connection_id
    return normalized, chosen_active
