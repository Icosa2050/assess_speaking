from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

SERVICE_NAME = "Speaking Studio"
_SESSION_SECRETS: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class SecretStoreStatus:
    persistent: bool
    backend_name: str
    detail: str = ""


class SecretStore(Protocol):
    def get_secret(self, service: str, account: str) -> str: ...

    def set_secret(self, service: str, account: str, value: str) -> None: ...

    def delete_secret(self, service: str, account: str) -> None: ...

    def is_persistent_supported(self) -> bool: ...


def _load_keyring_module() -> tuple[Any | None, SecretStoreStatus]:
    try:
        import keyring  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency boundary
        return None, SecretStoreStatus(persistent=False, backend_name="unavailable", detail=str(exc))

    try:
        backend = keyring.get_keyring()
        backend_name = backend.__class__.__name__
        if "fail" in backend_name.lower():
            return keyring, SecretStoreStatus(
                persistent=False,
                backend_name=backend_name,
                detail="No secure keyring backend is active.",
            )
        return keyring, SecretStoreStatus(persistent=True, backend_name=backend_name)
    except Exception as exc:  # pragma: no cover - backend boundary
        return keyring, SecretStoreStatus(persistent=False, backend_name="unknown", detail=str(exc))


class KeyringSecretStore:
    def __init__(self) -> None:
        self._keyring, self.status = _load_keyring_module()

    def get_secret(self, service: str, account: str) -> str:
        if self._keyring is None or not self.status.persistent:
            return ""
        try:
            return str(self._keyring.get_password(service, account) or "")
        except Exception:  # pragma: no cover - backend boundary
            return ""

    def set_secret(self, service: str, account: str, value: str) -> None:
        if self._keyring is None or not self.status.persistent:
            return
        self._keyring.set_password(service, account, value)

    def delete_secret(self, service: str, account: str) -> None:
        if self._keyring is None or not self.status.persistent:
            return
        try:
            self._keyring.delete_password(service, account)
        except Exception:
            pass

    def is_persistent_supported(self) -> bool:
        return self.status.persistent


class EnvFallbackSecretStore:
    def __init__(self, env_var_names: tuple[str, ...]) -> None:
        self.env_var_names = tuple(name for name in env_var_names if name)

    def get_secret(self, service: str, account: str) -> str:
        del service, account
        for env_name in self.env_var_names:
            value = str(os.getenv(env_name) or "").strip()
            if value:
                return value
        return ""

    def set_secret(self, service: str, account: str, value: str) -> None:
        del service, account, value
        # Environment fallback is read-only from the app's perspective.
        return None

    def delete_secret(self, service: str, account: str) -> None:
        del service, account
        return None

    def is_persistent_supported(self) -> bool:
        return False


class SessionSecretStore:
    def get_secret(self, service: str, account: str) -> str:
        return _SESSION_SECRETS.get((service, account), "")

    def set_secret(self, service: str, account: str, value: str) -> None:
        if value:
            _SESSION_SECRETS[(service, account)] = value
        else:
            _SESSION_SECRETS.pop((service, account), None)

    def delete_secret(self, service: str, account: str) -> None:
        _SESSION_SECRETS.pop((service, account), None)

    def is_persistent_supported(self) -> bool:
        return False


def _active_secret_store(*, env_var_names: tuple[str, ...] = ()) -> tuple[SecretStore, SecretStoreStatus]:
    keyring_store = KeyringSecretStore()
    if keyring_store.is_persistent_supported():
        return keyring_store, keyring_store.status
    env_store = EnvFallbackSecretStore(env_var_names)
    if env_store.get_secret(SERVICE_NAME, "__probe__"):
        detail = ", ".join(env_var_names)
        return env_store, SecretStoreStatus(persistent=False, backend_name="environment", detail=f"Using environment fallback: {detail}")
    return SessionSecretStore(), SecretStoreStatus(
        persistent=False,
        backend_name="session",
        detail=keyring_store.status.detail or "Secure storage unavailable; using session-only secrets.",
    )


def secret_store_status(*, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    _store, status = _active_secret_store(env_var_names=env_var_names)
    return status


def get_secret(account: str, *, service: str = SERVICE_NAME, env_var_names: tuple[str, ...] = ()) -> str:
    keyring_store = KeyringSecretStore()
    secret = keyring_store.get_secret(service, account)
    if secret:
        return secret
    secret = SessionSecretStore().get_secret(service, account)
    if secret:
        return secret
    env_store = EnvFallbackSecretStore(env_var_names)
    secret = env_store.get_secret(service, account)
    if secret:
        return secret
    return ""


def set_secret(account: str, value: str, *, service: str = SERVICE_NAME, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    if not value:
        return delete_secret(account, service=service, env_var_names=env_var_names)
    store, status = _active_secret_store(env_var_names=env_var_names)
    try:
        if isinstance(store, EnvFallbackSecretStore):
            SessionSecretStore().set_secret(service, account, value)
            return SecretStoreStatus(
                persistent=False,
                backend_name="session",
                detail="Environment values are read-only here; secret is stored for this session only.",
            )
        store.set_secret(service, account, value)
        return status
    except Exception as exc:  # pragma: no cover - backend boundary
        SessionSecretStore().set_secret(service, account, value)
        return SecretStoreStatus(persistent=False, backend_name=status.backend_name, detail=str(exc))


def delete_secret(account: str, *, service: str = SERVICE_NAME, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    keyring_store = KeyringSecretStore()
    status = keyring_store.status
    try:
        keyring_store.delete_secret(service, account)
    except Exception:
        pass
    SessionSecretStore().delete_secret(service, account)
    if not status.persistent:
        store, fallback_status = _active_secret_store(env_var_names=env_var_names)
        if isinstance(store, EnvFallbackSecretStore):
            return fallback_status
    return status
