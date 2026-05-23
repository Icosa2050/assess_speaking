from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

SERVICE_NAME = "Vostavo"
LEGACY_SERVICE_NAME = "Speaking Studio"
_SESSION_SECRETS: dict[tuple[str, str], str] = {}

logger = logging.getLogger(__name__)


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
            logger.warning("Could not read secret %s from service %s.", account, service, exc_info=True)
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
            logger.warning("Could not delete secret %s from service %s.", account, service, exc_info=True)

    def is_persistent_supported(self) -> bool:
        return self.status.persistent


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


def _active_secret_store() -> tuple[SecretStore, SecretStoreStatus]:
    keyring_store = KeyringSecretStore()
    return keyring_store, keyring_store.status


def _configured_env_keys(env_var_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in env_var_names if str(os.getenv(name) or "").strip())


def _status_with_env_hint(status: SecretStoreStatus, *, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    configured = _configured_env_keys(env_var_names)
    if not configured:
        return status
    detail_prefix = f"{status.detail} " if status.detail else ""
    return SecretStoreStatus(
        persistent=status.persistent,
        backend_name=status.backend_name,
        detail=f"{detail_prefix}Environment variables are present but ignored in the desktop app: {', '.join(configured)}",
    )


def secret_store_status(*, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    _store, status = _active_secret_store()
    return _status_with_env_hint(status, env_var_names=env_var_names)


def _should_use_legacy_fallback(service: str) -> bool:
    return service == SERVICE_NAME


def _legacy_secret(account: str) -> str:
    legacy_secret = KeyringSecretStore().get_secret(LEGACY_SERVICE_NAME, account)
    return legacy_secret if legacy_secret else ""


def _has_legacy_stored_secret(account: str) -> bool:
    return bool(KeyringSecretStore().get_secret(LEGACY_SERVICE_NAME, account))


def _copy_secret_to_primary(account: str, value: str) -> None:
    if not value:
        return
    store, status = _active_secret_store()
    if not status.persistent:
        return
    try:
        store.set_secret(SERVICE_NAME, account, value)
    except Exception:
        logger.warning("Could not copy legacy secret %s to primary service.", account, exc_info=True)
        return


def get_secret(account: str, *, service: str = SERVICE_NAME, env_var_names: tuple[str, ...] = ()) -> str:
    del env_var_names
    keyring_store = KeyringSecretStore()
    secret = keyring_store.get_secret(service, account)
    if secret:
        return secret
    if _should_use_legacy_fallback(service):
        secret = _legacy_secret(account)
        if secret:
            _copy_secret_to_primary(account, secret)
            return secret
    return ""


def set_secret(account: str, value: str, *, service: str = SERVICE_NAME, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    if not value:
        return delete_secret(account, service=service, env_var_names=env_var_names)
    legacy_present = _should_use_legacy_fallback(service) and _has_legacy_stored_secret(account)
    store, status = _active_secret_store()
    if not status.persistent:
        return _status_with_env_hint(status, env_var_names=env_var_names)
    try:
        store.set_secret(service, account, value)
        if legacy_present:
            store.set_secret(LEGACY_SERVICE_NAME, account, value)
        return status
    except Exception as exc:  # pragma: no cover - backend boundary
        return SecretStoreStatus(persistent=False, backend_name=status.backend_name, detail=str(exc))


def delete_secret(account: str, *, service: str = SERVICE_NAME, env_var_names: tuple[str, ...] = ()) -> SecretStoreStatus:
    keyring_store = KeyringSecretStore()
    status = keyring_store.status
    try:
        keyring_store.delete_secret(service, account)
    except Exception:
        logger.warning("Could not clear stored secret %s from service %s.", account, service, exc_info=True)
    if _should_use_legacy_fallback(service):
        try:
            keyring_store.delete_secret(LEGACY_SERVICE_NAME, account)
        except Exception:
            logger.warning(
                "Could not clear stored secret %s from legacy service %s.",
                account,
                LEGACY_SERVICE_NAME,
                exc_info=True,
            )
    SessionSecretStore().delete_secret(service, account)
    if _should_use_legacy_fallback(service):
        SessionSecretStore().delete_secret(LEGACY_SERVICE_NAME, account)
    return _status_with_env_hint(status, env_var_names=env_var_names)
