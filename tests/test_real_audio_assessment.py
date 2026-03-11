import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import assess_speaking


DEFAULT_REAL_AUDIO_PATH = Path(__file__).resolve().parent / "audio" / "test1.m4a"
BETTER_REAL_AUDIO_PATH = Path(__file__).resolve().parent / "audio" / "test2.m4a"


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


def _require_fixture(path: Path) -> Path:
    if not path.exists():
        raise unittest.SkipTest(f"Real audio file not found: {path}")
    return path


def _faster_whisper_cache_exists(model_name: str) -> bool:
    hub_cache = Path(
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    return (hub_cache / f"models--Systran--faster-whisper-{model_name}").exists()


class RealAudioAssessmentTests(unittest.TestCase):
    def _run_real_assessment(self, audio_path: Path, *, task_family: str, speaker_id: str) -> dict:
        whisper_model = os.getenv("ASSESS_SPEAKING_REAL_WHISPER_MODEL", "tiny")
        llm_model = os.getenv("ASSESS_SPEAKING_REAL_LLM_MODEL")
        provider = os.getenv("ASSESS_SPEAKING_REAL_PROVIDER", "openrouter")
        with tempfile.TemporaryDirectory() as tmpdir:
            with ExitStack() as stack:
                if _faster_whisper_cache_exists(whisper_model):
                    stack.enter_context(mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False))
                return assess_speaking.run_assessment(
                    audio_path,
                    provider=provider,
                    llm_model=llm_model,
                    whisper_model=whisper_model,
                    theme="tema libero",
                    task_family=task_family,
                    speaker_id=speaker_id,
                    target_duration_sec=30.0,
                    train_dir=Path(tmpdir),
                )

    def test_real_audio_file_returns_report_and_feedback(self):
        audio_path = _require_real_audio_env()
        result = self._run_real_assessment(
            audio_path,
            task_family="real_audio_smoke",
            speaker_id="real-audio-test",
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

    def test_real_audio_grading_ranks_second_sample_higher(self):
        _require_real_audio_env()
        first_audio = _require_fixture(DEFAULT_REAL_AUDIO_PATH)
        second_audio = _require_fixture(BETTER_REAL_AUDIO_PATH)

        first_result = self._run_real_assessment(
            first_audio,
            task_family="real_audio_grading_compare",
            speaker_id="real-audio-compare",
        )
        second_result = self._run_real_assessment(
            second_audio,
            task_family="real_audio_grading_compare",
            speaker_id="real-audio-compare",
        )

        first_report = first_result["report"]
        second_report = second_result["report"]
        first_scores = first_report["scores"]
        second_scores = second_report["scores"]
        first_rubric = first_report.get("rubric") or {}
        second_rubric = second_report.get("rubric") or {}

        for report in (first_report, second_report):
            self.assertTrue(report["checks"]["language_pass"])
            self.assertTrue(report["checks"]["topic_pass"])
            self.assertTrue(report["checks"]["duration_pass"])

        self.assertGreater(
            second_scores["final"],
            first_scores["final"],
            msg=f"Expected test2.m4a to outrank test1.m4a, got {first_scores['final']} vs {second_scores['final']}",
        )
        self.assertGreater(
            second_scores["deterministic"],
            first_scores["deterministic"],
            msg="Expected the stronger sample to have better deterministic speaking metrics.",
        )
        self.assertGreaterEqual(
            second_rubric.get("overall", 0),
            first_rubric.get("overall", 0) + 1,
            msg=f"Expected better overall rubric for test2.m4a, got {first_rubric.get('overall')} vs {second_rubric.get('overall')}",
        )
        self.assertGreaterEqual(
            second_rubric.get("fluency", 0),
            first_rubric.get("fluency", 0) + 1,
            msg=f"Expected better fluency rubric for test2.m4a, got {first_rubric.get('fluency')} vs {second_rubric.get('fluency')}",
        )
        self.assertGreaterEqual(
            second_rubric.get("range", 0),
            first_rubric.get("range", 0) + 1,
            msg=f"Expected better range rubric for test2.m4a, got {first_rubric.get('range')} vs {second_rubric.get('range')}",
        )


if __name__ == "__main__":
    unittest.main()
