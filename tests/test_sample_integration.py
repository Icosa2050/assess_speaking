import os
import unittest
from pathlib import Path

import assess_speaking


@unittest.skipUnless(
    os.getenv("RUN_AUDIO_INTEGRATION") == "1",
    "Set RUN_AUDIO_INTEGRATION=1 to run real audio integration tests.",
)
class SampleAudioIntegrationTests(unittest.TestCase):
    def test_sample_wav_transcription_and_metrics(self):
        sample_path = Path("samples/italian_demo.wav")
        self.assertTrue(sample_path.exists(), f"Missing sample audio: {sample_path}")

        model_size = os.getenv("WHISPER_MODEL", "tiny")
        audio_features = assess_speaking.load_audio_features(sample_path)
        try:
            asr_result = assess_speaking.transcribe(sample_path, model_size=model_size)
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"ASR prerequisites unavailable: {exc}")
        metrics = assess_speaking.metrics_from(asr_result["words"], audio_features)

        self.assertGreater(metrics["duration_sec"], 1.0)
        self.assertGreater(metrics["word_count"], 3)
        self.assertGreater(metrics["wpm"], 10.0)
        self.assertTrue(asr_result["text"].strip(), "Expected non-empty transcript")


if __name__ == "__main__":
    unittest.main()
