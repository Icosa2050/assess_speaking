import unittest
from unittest import mock

from app_shell.runtime_resolver import resolve_runtime_config, sync_runtime_fields
from app_shell.state import AppPreferences, ProviderConnection


class RuntimeResolverTests(unittest.TestCase):
    def test_resolve_runtime_config_requires_active_connection(self):
        with self.assertRaisesRegex(ValueError, "No active runtime connection is configured."):
            resolve_runtime_config(AppPreferences())

    @mock.patch("app_shell.runtime_resolver.get_secret", return_value="local-token")
    def test_resolve_runtime_config_adds_v1_for_local_ollama_connection(self, _mock_get_secret):
        prefs = AppPreferences(
            connections=[
                ProviderConnection(
                    connection_id="conn-1",
                    provider_kind="ollama",
                    label="Ollama Local",
                    base_url="http://localhost:11434",
                    default_model="llama3",
                    secret_ref="connection:conn-1",
                    is_default=True,
                    is_local=True,
                    provider_metadata={"deployment": "local"},
                )
            ],
            active_connection_id="conn-1",
        )
        runtime = resolve_runtime_config(prefs)
        self.assertEqual(runtime.provider, "ollama")
        self.assertEqual(runtime.base_url, "http://localhost:11434/v1")
        self.assertEqual(runtime.model, "llama3")

    @mock.patch("app_shell.runtime_resolver.get_secret", return_value="saved-key")
    def test_sync_runtime_fields_uses_openrouter_connection_metadata(self, _mock_get_secret):
        prefs = AppPreferences(
            connections=[
                ProviderConnection(
                    connection_id="conn-2",
                    provider_kind="openrouter",
                    label="OpenRouter",
                    base_url="https://openrouter.ai/api/v1",
                    default_model="google/gemini-3.1-pro-preview",
                    secret_ref="connection:conn-2",
                    is_default=True,
                    provider_metadata={
                        "http_referer": "http://localhost:8503",
                        "app_title": "Speaking Studio",
                    },
                )
            ],
            active_connection_id="conn-2",
        )
        sync_runtime_fields(prefs)
        self.assertEqual(prefs.provider, "openrouter")
        self.assertEqual(prefs.llm_api_key, "saved-key")
        self.assertEqual(prefs.openrouter_http_referer, "http://localhost:8503")
        self.assertEqual(prefs.openrouter_app_title, "Speaking Studio")


if __name__ == "__main__":
    unittest.main()
