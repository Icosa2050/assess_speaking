import os
import unittest
from pathlib import Path

import assess_speaking

SAMPLE_PRESETS = {
    ("it", "B1"): Path("samples/cefr/it/B1/travel_story.wav"),
    ("it", "B2"): Path("samples/cefr/it/B2/remote_work.wav"),
    ("it", "C1"): Path("samples/cefr/it/C1/public_debate.wav"),
    ("en", "B1"): Path("samples/cefr/en/B1/travel_story.wav"),
    ("en", "B2"): Path("samples/cefr/en/B2/remote_work.wav"),
    ("en", "C1"): Path("samples/cefr/en/C1/public_debate.wav"),
}


class SampleLibraryTests(unittest.TestCase):
    def test_repo_ships_expected_en_it_cefr_samples(self):
        for (language, level), sample_path in SAMPLE_PRESETS.items():
            with self.subTest(language=language, level=level):
                self.assertTrue(sample_path.exists(), f"Missing sample audio: {sample_path}")
                self.assertGreater(sample_path.stat().st_size, 1024, f"Sample audio is unexpectedly small: {sample_path}")

    def test_each_language_has_b1_b2_c1_samples(self):
        levels_by_language: dict[str, set[str]] = {}
        for language, level in SAMPLE_PRESETS:
            levels_by_language.setdefault(language, set()).add(level)

        self.assertEqual(levels_by_language["it"], {"B1", "B2", "C1"})
        self.assertEqual(levels_by_language["en"], {"B1", "B2", "C1"})


@unittest.skipUnless(
    os.getenv("RUN_AUDIO_INTEGRATION") == "1",
    "Set RUN_AUDIO_INTEGRATION=1 to run real audio integration tests.",
)
class SampleAudioIntegrationTests(unittest.TestCase):
    def test_sample_wav_transcription_and_metrics_for_all_cefr_presets(self):
        model_size = os.getenv("WHISPER_MODEL", "tiny")
        for (language, level), sample_path in SAMPLE_PRESETS.items():
            with self.subTest(language=language, level=level, sample=sample_path.name):
                audio_features = assess_speaking.load_audio_features(sample_path)
                try:
                    asr_result = assess_speaking.transcribe(sample_path, model_size=model_size)
                except (ImportError, RuntimeError) as exc:
                    self.skipTest(f"ASR prerequisites unavailable: {exc}")
                metrics = assess_speaking.metrics_from(asr_result["words"], audio_features)

                self.assertGreater(metrics["duration_sec"], 1.0)
                self.assertGreater(metrics["word_count"], 10)
                self.assertGreater(metrics["wpm"], 10.0)
                self.assertTrue(asr_result["text"].strip(), f"Expected non-empty transcript for {sample_path}")


if __name__ == "__main__":
    unittest.main()
