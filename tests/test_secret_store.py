import os
import unittest
from unittest import mock

from app_shell import secret_store


class FakeKeyringModule:
    def __init__(self, initial: dict[tuple[str, str], str] | None = None) -> None:
        self.secrets = dict(initial or {})

    def get_password(self, service: str, account: str) -> str | None:
        return self.secrets.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.secrets[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.secrets.pop((service, account), None)


class SecretStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        secret_store._SESSION_SECRETS.clear()

    def tearDown(self) -> None:
        secret_store._SESSION_SECRETS.clear()

    def test_session_store_round_trip(self):
        store = secret_store.SessionSecretStore()
        store.set_secret(secret_store.SERVICE_NAME, "account-1", "secret-123")
        self.assertEqual(store.get_secret(secret_store.SERVICE_NAME, "account-1"), "secret-123")
        store.delete_secret(secret_store.SERVICE_NAME, "account-1")
        self.assertEqual(store.get_secret(secret_store.SERVICE_NAME, "account-1"), "")

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_secret_store_status_reports_ignored_environment_variables(self, mock_load_keyring):
        mock_load_keyring.return_value = (
            None,
            secret_store.SecretStoreStatus(persistent=False, backend_name="unavailable", detail="no keyring"),
        )
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "env-key"}, clear=False):
            status = secret_store.secret_store_status(env_var_names=("LLM_API_KEY",))
        self.assertFalse(status.persistent)
        self.assertEqual(status.backend_name, "unavailable")
        self.assertIn("ignored in the desktop app", status.detail)

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_set_secret_requires_persistent_secure_storage(self, mock_load_keyring):
        mock_load_keyring.return_value = (
            None,
            secret_store.SecretStoreStatus(persistent=False, backend_name="unavailable", detail="no keyring"),
        )
        status = secret_store.set_secret("account-2", "saved-key", env_var_names=("OPENROUTER_API_KEY",))
        self.assertFalse(status.persistent)
        self.assertEqual(secret_store.get_secret("account-2"), "")

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_get_secret_ignores_environment_fallback(self, mock_load_keyring):
        mock_load_keyring.return_value = (
            None,
            secret_store.SecretStoreStatus(persistent=False, backend_name="unavailable", detail="no keyring"),
        )
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            self.assertEqual(secret_store.get_secret("account-3", env_var_names=("OPENROUTER_API_KEY",)), "")

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_get_secret_falls_back_to_legacy_service_name_in_secure_storage(self, mock_load_keyring):
        keyring = FakeKeyringModule({(secret_store.LEGACY_SERVICE_NAME, "account-3"): "legacy-key"})
        mock_load_keyring.return_value = (
            keyring,
            secret_store.SecretStoreStatus(persistent=True, backend_name="mock-keyring"),
        )

        self.assertEqual(secret_store.get_secret("account-3"), "legacy-key")
        self.assertEqual(keyring.get_password(secret_store.SERVICE_NAME, "account-3"), "legacy-key")

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_set_secret_mirrors_legacy_entry_when_present(self, mock_load_keyring):
        keyring = FakeKeyringModule({(secret_store.LEGACY_SERVICE_NAME, "account-4"): "old-key"})
        mock_load_keyring.return_value = (
            keyring,
            secret_store.SecretStoreStatus(persistent=True, backend_name="mock-keyring"),
        )

        secret_store.set_secret("account-4", "new-key")

        self.assertEqual(keyring.get_password(secret_store.SERVICE_NAME, "account-4"), "new-key")
        self.assertEqual(keyring.get_password(secret_store.LEGACY_SERVICE_NAME, "account-4"), "new-key")

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_set_secret_does_not_treat_environment_variables_as_legacy_entry(self, mock_load_keyring):
        mock_load_keyring.return_value = (
            None,
            secret_store.SecretStoreStatus(persistent=False, backend_name="unavailable", detail="no keyring"),
        )
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            status = secret_store.set_secret("account-5", "new-key", env_var_names=("OPENROUTER_API_KEY",))

        self.assertFalse(status.persistent)
        self.assertEqual(secret_store.SessionSecretStore().get_secret(secret_store.SERVICE_NAME, "account-5"), "")
        self.assertEqual(secret_store.SessionSecretStore().get_secret(secret_store.LEGACY_SERVICE_NAME, "account-5"), "")


if __name__ == "__main__":
    unittest.main()
