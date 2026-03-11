import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import assess_speaking


DEFAULT_REAL_AUDIO_PATH = Path(__file__).resolve().parent / "audio" / "test1.m4a"


def _require_real_audio_env() -> Path:
    if os.getenv("RUN_REAL_AUDIO_ASSESSMENT") != "1":
        raise unittest.SkipTest("Set RUN_REAL_AUDIO_ASSESSMENT=1 to run the real audio assessment test.")
    audio_path = os.getenv("ASSESS_SPEAKING_REAL_AUDIO_PATH")
    path = Path(audio_path).expanduser() if audio_path else DEFAULT_REAL_AUDIO_PATH
    if not path.exists():
        raise unittest.SkipTest(f"Real audio file not found: {path}")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise unittest.SkipTest("OPENROUTER_API_KEY is required for the real audio assessment test.")
    return path


def _faster_whisper_cache_exists(model_name: str) -> bool:
    hub_cache = Path(
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    return (hub_cache / f"models--Systran--faster-whisper-{model_name}").exists()


class RealAudioAssessmentTests(unittest.TestCase):
    def test_real_audio_file_returns_report_and_feedback(self):
        audio_path = _require_real_audio_env()
        whisper_model = os.getenv("ASSESS_SPEAKING_REAL_WHISPER_MODEL", "tiny")
        llm_model = os.getenv("ASSESS_SPEAKING_REAL_LLM_MODEL")
        provider = os.getenv("ASSESS_SPEAKING_REAL_PROVIDER", "openrouter")
        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as stack:
                if _faster_whisper_cache_exists(whisper_model):
                    stack.enter_context(mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False))
                result = assess_speaking.run_assessment(
                    audio_path,
                    provider=provider,
                    llm_model=llm_model,
                    whisper_model=whisper_model,
                    theme="tema libero",
                    task_family="real_audio_smoke",
                    speaker_id="real-audio-test",
                    target_duration_sec=60.0,
                    train_dir=Path(tmpdir),
                )
        report = result["report"]
        self.assertIn("metrics", report)
        self.assertIn("checks", report)
        self.assertIn("scores", report)
        self.assertIn("coaching", report)
        self.assertTrue(result["transcript_preview"])
        self.assertEqual(report["input"]["speaker_id"], "real-audio-test")
        self.assertEqual(report["input"]["task_family"], "real_audio_smoke")
        self.assertEqual(report["input"]["theme"], "tema libero")
        self.assertEqual(len(report["coaching"]["top_3_priorities"]), 3)
        self.assertIn("next_exercise", report["coaching"])
        self.assertIn(report["scores"]["mode"], {"hybrid", "deterministic_only"})


if __name__ == "__main__":
    unittest.main()
