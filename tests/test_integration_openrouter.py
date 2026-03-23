import os
import unittest

from assessment_runtime.assessment_prompts import coaching_prompt_it, rubric_prompt_it
from assessment_runtime.llm_client import generate_coaching_summary, generate_rubric


@unittest.skipUnless(
    os.getenv("RUN_OPENROUTER_INTEGRATION") == "1" and os.getenv("OPENROUTER_API_KEY"),
    "Set RUN_OPENROUTER_INTEGRATION=1 and OPENROUTER_API_KEY to run integration test",
)
class OpenRouterIntegrationTests(unittest.TestCase):
    def test_generate_rubric_round_trip(self):
        metrics = {
            "duration_sec": 180.0,
            "speaking_time_sec": 162.0,
            "pause_total_sec": 18.0,
            "pause_count": 12,
            "word_count": 230,
            "wpm": 85.2,
            "fillers": 7,
            "cohesion_markers": 3,
            "complexity_index": 2,
        }
        transcript = (
            "L'anno scorso ho fatto il mio ultimo viaggio all'estero in Portogallo. "
            "Prima sono arrivato a Lisbona, poi ho visitato Porto e alla fine sono tornato a casa. "
            "Il viaggio era molto bello ma ho avuto qualche problema con i trasporti."
        )
        prompt = rubric_prompt_it(transcript, metrics, "Il mio ultimo viaggio all'estero")
        rubric, raw = generate_rubric(
            provider="openrouter",
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-pro-preview"),
            prompt=prompt,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.assertGreaterEqual(rubric.overall, 1)
        self.assertGreaterEqual(rubric.topic_relevance_score, 1)
        self.assertTrue(isinstance(rubric.evidence_quotes, list))
        self.assertIn("overall", raw)

    def test_generate_coaching_summary_round_trip(self):
        metrics = {
            "duration_sec": 180.0,
            "speaking_time_sec": 150.0,
            "pause_total_sec": 30.0,
            "pause_count": 18,
            "word_count": 210,
            "wpm": 84.0,
            "fillers": 9,
            "cohesion_markers": 1,
            "complexity_index": 1,
        }
        rubric = {
            "fluency": 3,
            "cohesion": 2,
            "accuracy": 3,
            "range": 3,
            "overall": 3,
            "comments_fluency": "Il ritmo è ancora spezzato.",
            "comments_cohesion": "La sequenza narrativa è debole.",
            "comments_accuracy": "Ci sono errori ricorrenti di preposizione.",
            "comments_range": "Il lessico è sufficiente ma ripetitivo.",
            "overall_comment": "Prestazione comprensibile ma da rendere più fluida.",
            "on_topic": True,
            "topic_relevance_score": 4,
            "language_ok": True,
            "recurring_grammar_errors": [
                {
                    "category": "preposition_choice",
                    "explanation": "Scelta incerta delle preposizioni.",
                    "examples": ["sono andato a Spagna"],
                }
            ],
            "coherence_issues": [
                {
                    "category": "missing_sequence_markers",
                    "explanation": "Mancano connettivi temporali.",
                    "examples": ["poi", "alla fine"],
                }
            ],
            "lexical_gaps": [],
            "evidence_quotes": ["sono andato a Spagna"],
            "confidence": "medium",
        }
        prompt = coaching_prompt_it(
            metrics=metrics,
            rubric=rubric,
            theme="Il mio ultimo viaggio all'estero",
            target_duration_sec=180.0,
        )
        coaching, raw = generate_coaching_summary(
            provider="openrouter",
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-pro-preview"),
            prompt=prompt,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.assertEqual(len(coaching.top_3_priorities), 3)
        self.assertTrue(coaching.next_exercise)
        self.assertIn("coach_summary", raw)


if __name__ == "__main__":
    unittest.main()
