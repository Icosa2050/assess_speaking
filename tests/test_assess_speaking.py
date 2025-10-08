import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import assess_speaking


class MetricsTests(unittest.TestCase):
    def test_metrics_from_basic_sample(self):
        words = [
            {"text": "ciao"},
            {"text": "eh"},
            {"text": "parlo"},
            {"text": "per"},
            {"text": "quanto"},
            {"text": "riguarda"},
        ]
        audio_feats = {"duration_sec": 10.0, "pauses": [(1.0, 1.6, 0.6), (4.0, 4.4, 0.4)]}

        metrics = assess_speaking.metrics_from(words, audio_feats)

        self.assertEqual(metrics["duration_sec"], 10.0)
        self.assertEqual(metrics["pause_total_sec"], 1.0)
        self.assertEqual(metrics["speaking_time_sec"], 9.0)
        self.assertEqual(metrics["pause_count"], 2)
        self.assertEqual(metrics["word_count"], 6)
        self.assertEqual(metrics["wpm"], 40.0)
        self.assertEqual(metrics["fillers"], 1)
        self.assertEqual(metrics["cohesion_markers"], 1)
        self.assertEqual(metrics["complexity_index"], 0)


class PromptTests(unittest.TestCase):
    def test_rubric_prompt_includes_metrics_and_transcript(self):
        transcript = "Questo e un test."
        metrics = {
            "duration_sec": 12.3,
            "speaking_time_sec": 10.0,
            "pause_total_sec": 2.3,
            "pause_count": 2,
            "word_count": 20,
            "wpm": 120.0,
            "fillers": 2,
            "cohesion_markers": 1,
            "complexity_index": 3,
        }

        prompt = assess_speaking.rubric_prompt_it(transcript, metrics)

        self.assertIn("Durata: 12.3 s", prompt)
        self.assertIn("WPM: 120.0", prompt)
        self.assertIn("TRASCRITTO:", prompt)
        self.assertIn(transcript.strip(), prompt)
        self.assertIn("RISPONDI IN JSON", prompt)


class DependencyTests(unittest.TestCase):
    def test_extract_rubric_json_from_code_block(self):
        payload = """
        Risposta:
        ```json
        {"overall": 3.5, "fluency": 4}
        ```
        """
        parsed = assess_speaking.extract_rubric_json(payload)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["overall"], 3.5)

    def test_extract_rubric_json_invalid_returns_none(self):
        self.assertIsNone(assess_speaking.extract_rubric_json("not json"))

    def test_evaluate_baseline_returns_expected_flags(self):
        metrics = {"wpm": 120, "fillers": 3, "cohesion_markers": 1, "complexity_index": 1}
        result = assess_speaking.evaluate_baseline("B2", metrics)
        self.assertIsNotNone(result)
        self.assertTrue(result["passed"])
        self.assertTrue(result["targets"]["wpm"]["ok"])
        self.assertTrue(result["targets"]["fillers"]["ok"])

    def test_evaluate_baseline_handles_missing_level(self):
        metrics = {"wpm": 50, "fillers": 10, "cohesion_markers": 0, "complexity_index": 0}
        result = assess_speaking.evaluate_baseline("B2", metrics)
        self.assertFalse(result["passed"])

    def test_load_audio_features_requires_parselmouth(self):
        with mock.patch.object(assess_speaking, "parselmouth", None), \
             mock.patch.object(assess_speaking, "call", None):
            with self.assertRaises(RuntimeError) as ctx:
                assess_speaking.load_audio_features(Path("dummy.wav"))
        self.assertIn("praat-parselmouth", str(ctx.exception))

    def test_call_ollama_returns_error_payload_when_curl_fails(self):
        err = subprocess.CalledProcessError(returncode=1, cmd="curl", stderr="boom")
        with mock.patch("subprocess.run", side_effect=err):
            resp = assess_speaking.call_ollama("llama3", "prompt")
        payload = json.loads(resp)
        self.assertEqual(payload["error"], "ollama_not_running_or_model_missing")
        self.assertIn("boom", payload["detail"])


if __name__ == "__main__":
    unittest.main()
