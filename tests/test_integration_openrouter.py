import os
import unittest

from llm_client import generate_rubric


@unittest.skipUnless(
    os.getenv("RUN_OPENROUTER_INTEGRATION") == "1" and os.getenv("OPENROUTER_API_KEY"),
    "Set RUN_OPENROUTER_INTEGRATION=1 and OPENROUTER_API_KEY to run integration test",
)
class OpenRouterIntegrationTests(unittest.TestCase):
    def test_generate_rubric_round_trip(self):
        prompt = """
Rispondi SOLO con JSON valido:
{
  "fluency": 4,
  "cohesion": 4,
  "accuracy": 4,
  "range": 4,
  "overall": 4,
  "comments_fluency": "ok",
  "comments_cohesion": "ok",
  "comments_accuracy": "ok",
  "comments_range": "ok",
  "overall_comment": "ok",
  "on_topic": true
}
"""
        rubric, raw = generate_rubric(
            provider="openrouter",
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-pro-preview"),
            prompt=prompt,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.assertGreaterEqual(rubric.overall, 1)
        self.assertIn("overall", raw)


if __name__ == "__main__":
    unittest.main()
