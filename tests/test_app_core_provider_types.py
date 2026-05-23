import unittest
from typing import get_type_hints

from app_core.provider_types import LMStudioMetadata, OllamaMetadata, OpenAICompatibleMetadata, OpenRouterMetadata


class ProviderTypesTests(unittest.TestCase):
    def test_provider_metadata_typed_dicts_behave_like_optional_mappings(self):
        openrouter = OpenRouterMetadata(http_referer="https://example.com", app_title="Vostavo")
        ollama = OllamaMetadata(deployment="local")
        lmstudio = LMStudioMetadata(deployment="local", token_optional=True)
        compatible = OpenAICompatibleMetadata(deployment="custom")

        self.assertIsInstance(openrouter, dict)
        self.assertEqual(openrouter["app_title"], "Vostavo")
        self.assertEqual(ollama["deployment"], "local")
        self.assertTrue(lmstudio["token_optional"])
        self.assertEqual(compatible["deployment"], "custom")

    def test_provider_metadata_annotations_expose_expected_keys(self):
        self.assertIn("http_referer", get_type_hints(OpenRouterMetadata))
        self.assertIn("deployment", get_type_hints(OllamaMetadata))
        self.assertIn("token_optional", get_type_hints(LMStudioMetadata))
        self.assertIn("deployment", get_type_hints(OpenAICompatibleMetadata))


if __name__ == "__main__":
    unittest.main()
