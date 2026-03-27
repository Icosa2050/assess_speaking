import unittest

from app_shell.runtime_connections import ensure_single_default_connection
from app_shell.state import ProviderConnection


class RuntimeConnectionTests(unittest.TestCase):
    def test_ensure_single_default_connection_promotes_active_connection(self):
        connections = [
            ProviderConnection(connection_id="one", provider_kind="openrouter", label="One"),
            ProviderConnection(connection_id="two", provider_kind="ollama", label="Two"),
        ]
        normalized, active_id = ensure_single_default_connection(connections, active_connection_id="two")
        self.assertEqual(active_id, "two")
        self.assertFalse(normalized[0].is_default)
        self.assertTrue(normalized[1].is_default)


if __name__ == "__main__":
    unittest.main()
