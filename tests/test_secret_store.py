import os
import unittest
from unittest import mock

from app_shell import secret_store


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
    def test_secret_store_status_reports_environment_fallback(self, mock_load_keyring):
        mock_load_keyring.return_value = (
            None,
            secret_store.SecretStoreStatus(persistent=False, backend_name="unavailable", detail="no keyring"),
        )
        with mock.patch.dict(os.environ, {"LLM_API_KEY": "env-key"}, clear=False):
            status = secret_store.secret_store_status(env_var_names=("LLM_API_KEY",))
        self.assertFalse(status.persistent)
        self.assertEqual(status.backend_name, "environment")

    @mock.patch("app_shell.secret_store._load_keyring_module")
    def test_set_secret_falls_back_to_session_when_secure_storage_is_unavailable(self, mock_load_keyring):
        mock_load_keyring.return_value = (
            None,
            secret_store.SecretStoreStatus(persistent=False, backend_name="unavailable", detail="no keyring"),
        )
        status = secret_store.set_secret("account-2", "saved-key", env_var_names=("OPENROUTER_API_KEY",))
        self.assertFalse(status.persistent)
        self.assertEqual(secret_store.get_secret("account-2"), "saved-key")


if __name__ == "__main__":
    unittest.main()
