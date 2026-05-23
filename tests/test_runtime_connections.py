import unittest
from unittest import mock

from app_core.runtime_connections import deserialize_connections, ensure_single_default_connection, serialize_connections
from app_core.state import DEFAULT_MODEL, ProviderConnection


class RuntimeConnectionTests(unittest.TestCase):
    def test_deserialize_connections_returns_empty_for_non_lists(self):
        self.assertEqual(deserialize_connections(None), [])

    def test_deserialize_connections_normalizes_values_and_skips_invalid_rows(self):
        raw_connections = [
            "skip-me",
            {
                "connection_id": "primary",
                "provider": "OLLAMA",
                "base_url": " http://localhost:11434/v1 ",
                "model": "llama3",
                "auth_mode": "Bearer",
                "is_default": 1,
                "last_test_status": " ok ",
                "last_tested_at": " now ",
                "provider_metadata": {"deployment": "local"},
            },
            {
                "connection_id": " ",
                "provider_kind": "unsupported-provider",
                "label": " ",
                "base_url": None,
                "default_model": " ",
                "secret_ref": " ",
                "provider_metadata": "invalid",
            },
        ]

        with mock.patch("app_core.runtime_connections.uuid4", return_value=mock.Mock(hex="generated-id")):
            connections = deserialize_connections(raw_connections)

        self.assertEqual(len(connections), 2)

        primary, fallback = connections
        self.assertEqual(primary.connection_id, "primary")
        self.assertEqual(primary.provider_kind, "ollama")
        self.assertEqual(primary.label, "Ollama")
        self.assertEqual(primary.base_url, "http://localhost:11434/v1")
        self.assertEqual(primary.default_model, "llama3")
        self.assertEqual(primary.auth_mode, "bearer")
        self.assertEqual(primary.secret_ref, "connection:primary")
        self.assertTrue(primary.is_default)
        self.assertTrue(primary.is_local)
        self.assertEqual(primary.last_test_status, "ok")
        self.assertEqual(primary.last_tested_at, "now")
        self.assertEqual(primary.provider_metadata, {"deployment": "local"})

        self.assertEqual(fallback.connection_id, "generated-id")
        self.assertEqual(fallback.provider_kind, "openrouter")
        self.assertEqual(fallback.label, "OpenRouter")
        self.assertEqual(fallback.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(fallback.default_model, DEFAULT_MODEL)
        self.assertEqual(fallback.auth_mode, "none")
        self.assertEqual(fallback.secret_ref, "connection:generated-id")
        self.assertFalse(fallback.is_default)
        self.assertFalse(fallback.is_local)
        self.assertEqual(fallback.provider_metadata, {})

    def test_serialize_connections_preserves_payload_shape(self):
        payload = serialize_connections(
            [
                ProviderConnection(
                    connection_id="one",
                    provider_kind="ollama",
                    label="One",
                    base_url="http://localhost:11434/v1",
                    default_model="llama3",
                    auth_mode="bearer",
                    secret_ref="connection:one",
                    is_default=True,
                    is_local=True,
                    last_test_status="ok",
                    last_tested_at="now",
                    provider_metadata={"deployment": "local"},
                ),
                ProviderConnection(
                    connection_id="two",
                    provider_kind="unexpected",
                    label="Two",
                    provider_metadata=None,
                ),
            ]
        )

        self.assertEqual(payload[0]["provider_kind"], "ollama")
        self.assertEqual(payload[0]["provider_metadata"], {"deployment": "local"})
        self.assertTrue(payload[0]["is_local"])
        self.assertEqual(payload[1]["provider_kind"], "openrouter")
        self.assertEqual(payload[1]["provider_metadata"], {})

    def test_ensure_single_default_connection_promotes_active_connection(self):
        connections = [
            ProviderConnection(connection_id="one", provider_kind="openrouter", label="One"),
            ProviderConnection(connection_id="two", provider_kind="ollama", label="Two"),
        ]
        normalized, active_id = ensure_single_default_connection(connections, active_connection_id="two")
        self.assertEqual(active_id, "two")
        self.assertFalse(normalized[0].is_default)
        self.assertTrue(normalized[1].is_default)

    def test_ensure_single_default_connection_returns_empty_for_no_connections(self):
        normalized, active_id = ensure_single_default_connection([])

        self.assertEqual(normalized, [])
        self.assertEqual(active_id, "")

    def test_ensure_single_default_connection_falls_back_to_existing_default(self):
        connections = [
            ProviderConnection(connection_id="one", provider_kind="openrouter", label="One", is_default=True),
            ProviderConnection(connection_id="two", provider_kind="ollama", label="Two"),
        ]

        normalized, active_id = ensure_single_default_connection(connections, active_connection_id="missing")

        self.assertEqual(active_id, "one")
        self.assertTrue(normalized[0].is_default)
        self.assertFalse(normalized[1].is_default)

    def test_ensure_single_default_connection_uses_first_entry_without_any_default(self):
        connections = [
            ProviderConnection(connection_id="one", provider_kind="openrouter", label="One"),
            ProviderConnection(connection_id="two", provider_kind="ollama", label="Two"),
        ]

        normalized, active_id = ensure_single_default_connection(connections, active_connection_id="")

        self.assertEqual(active_id, "one")
        self.assertTrue(normalized[0].is_default)
        self.assertFalse(normalized[1].is_default)


if __name__ == "__main__":
    unittest.main()
