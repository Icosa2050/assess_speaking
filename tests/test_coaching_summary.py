import unittest
from unittest import mock

from assessment_runtime import llm_client
from assessment_runtime.feedback import build_fallback_coaching


VALID_COACHING_JSON = """
{
  "strengths": ["Resti sul tema."],
  "top_3_priorities": ["Più connettivi", "Meno filler", "Passato più stabile"],
  "next_focus": "Ordina meglio gli eventi",
  "next_exercise": "Racconta di nuovo il viaggio usando prima, poi e alla fine.",
  "coach_summary": "Buona base, ma serve una struttura narrativa più chiara."
}
"""


class CoachingSummaryTests(unittest.TestCase):
    def test_build_fallback_coaching_returns_three_priorities(self):
        coaching = build_fallback_coaching(
            metrics={
                "word_count": 50,
                "fillers": 6,
                "cohesion_markers": 0,
                "wpm": 70,
                "complexity_index": 0,
            },
            checks={"language_pass": True, "topic_pass": True, "duration_pass": False},
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180,
        )
        self.assertEqual(len(coaching["top_3_priorities"]), 3)
        self.assertIn("Il mio ultimo viaggio all'estero", coaching["next_exercise"])

    def test_build_fallback_coaching_uses_ui_locale(self):
        coaching = build_fallback_coaching(
            metrics={
                "word_count": 50,
                "fillers": 6,
                "cohesion_markers": 0,
                "wpm": 70,
                "complexity_index": 0,
            },
            checks={"language_pass": False, "topic_pass": True, "duration_pass": False},
            theme="Remote work",
            target_duration_sec=180,
            ui_locale="en",
            learning_language="it",
        )
        self.assertIn("Complete the full task in Italian", coaching["top_3_priorities"][0])
        self.assertTrue(coaching["coach_summary"].startswith("This is a useful starting point."))
        self.assertIn("Complete the full task in Italian", coaching["next_exercise"])

    def test_build_fallback_coaching_calls_out_short_language_mismatch_transcript(self):
        coaching = build_fallback_coaching(
            metrics={
                "word_count": 3,
                "fillers": 0,
                "cohesion_markers": 0,
                "wpm": 15,
                "complexity_index": 0,
            },
            checks={"language_pass": False, "topic_pass": None, "duration_pass": False},
            theme="cambiamento climatico",
            target_duration_sec=180,
            ui_locale="de",
            learning_language="it",
            transcript="Eins, zwei, drei.",
            detected_language="de",
        )
        self.assertIn("Eins, zwei, drei.", coaching["coach_summary"])
        self.assertIn("Deutsch", coaching["coach_summary"])
        self.assertNotIn("Du bist beim vorgegebenen Thema geblieben.", coaching["strengths"])
        self.assertIn("laengere Antwort in Italienisch", coaching["top_3_priorities"][0])

    @mock.patch("assessment_runtime.llm_client._chat_completion", return_value=VALID_COACHING_JSON)
    def test_generate_coaching_summary_returns_valid_payload(self, _mock_chat):
        coaching, raw = llm_client.generate_coaching_summary(
            provider="ollama",
            model="x",
            prompt="p",
        )
        self.assertEqual(len(coaching.top_3_priorities), 3)
        self.assertIn("coach_summary", raw)

    def test_generate_coaching_summary_retries_then_fails(self):
        with mock.patch("assessment_runtime.llm_client._chat_completion", return_value='{"coach_summary":"x"}'):
            with self.assertRaises(llm_client.LLMClientError):
                llm_client.generate_coaching_summary(
                    provider="ollama",
                    model="x",
                    prompt="p",
                    max_validation_retries=1,
                )


if __name__ == "__main__":
    unittest.main()
