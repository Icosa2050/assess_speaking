import unittest

from app_shell.migrations import ensure_single_default_connection, legacy_connection_from_prefs
from app_shell.state import ProviderConnection


class RuntimeMigrationTests(unittest.TestCase):
    def test_ensure_single_default_connection_promotes_active_connection(self):
        connections = [
            ProviderConnection(connection_id="one", provider_kind="openrouter", label="One"),
            ProviderConnection(connection_id="two", provider_kind="ollama", label="Two"),
        ]
        normalized, active_id = ensure_single_default_connection(connections, active_connection_id="two")
        self.assertEqual(active_id, "two")
        self.assertFalse(normalized[0].is_default)
        self.assertTrue(normalized[1].is_default)

    def test_legacy_connection_from_prefs_normalizes_legacy_openrouter_title(self):
        connection = legacy_connection_from_prefs(
            {
                "provider": "openrouter",
                "model": "google/gemini-3.1-pro-preview",
                "openrouter_http_referer": "https://example.test/app",
                "openrouter_app_title": "Assess Speaking",
            }
        )
        self.assertIsNotNone(connection)
        self.assertEqual(connection.provider_kind, "openrouter")
        self.assertEqual(connection.provider_metadata["http_referer"], "https://example.test/app")
        self.assertEqual(connection.provider_metadata["app_title"], "Speaking Studio")


if __name__ == "__main__":
    unittest.main()
