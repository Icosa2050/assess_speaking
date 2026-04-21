import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run


def _assessment_payload() -> dict:
    return {
        "metrics": {
            "duration_sec": 30.0,
            "wpm": 100.0,
            "word_count": 50,
        },
        "transcript_full": "hello world",
        "transcript_preview": "hello world",
        "llm_rubric": '{"ok": true}',
        "report": {
            "session_id": "sess-1",
            "schema_version": 2,
            "input": {
                "provider": "openrouter",
                "llm_model": "google/gemini-3.1-pro-preview",
                "learning_language": "en",
                "feedback_language": "en",
                "task_family": "free_monologue",
                "speaker_id": "bern",
                "theme": "travel",
                "target_duration_sec": 90,
            },
            "checks": {
                "duration_pass": True,
                "topic_pass": True,
                "language_pass": True,
            },
            "scores": {
                "final": 78.0,
                "band": "B2",
            },
            "rubric": {
                "fluency": 4,
                "cohesion": 4,
                "accuracy": 4,
                "range": 4,
                "overall": 4,
                "recurring_grammar_errors": [],
                "coherence_issues": [],
            },
            "coaching": {
                "top_3_priorities": ["More connectors", "", ""],
                "next_focus": "Use transitions",
            },
            "warnings": [],
            "errors": [],
            "requires_human_review": False,
        },
    }


class AssessmentRunnerTests(unittest.TestCase):
    @mock.patch("assess_speaking.build_progress_delta", return_value=None)
    @mock.patch("assess_speaking.run_assessment")
    def test_execute_assessment_run_no_log_returns_stdout_payload(self, mock_run_assessment, _mock_progress):
        mock_run_assessment.return_value = _assessment_payload()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_assessment_run(
                AssessmentRunRequest(
                    audio=Path(tmpdir) / "sample.wav",
                    whisper_model="small",
                    asr_provider="faster_whisper_chunked",
                    llm_model="google/gemini-3.1-pro-preview",
                    provider="openrouter",
                    expected_language="en",
                    feedback_language="en",
                    task_family="free_monologue",
                    speaker_id="bern",
                    theme="travel",
                    target_duration_sec=90,
                    log_dir=Path(tmpdir),
                    no_log=True,
                )
            )

        payload = json.loads(result.stdout_json)
        self.assertEqual(result.report_path, None)
        self.assertIsNone(result.saved_payload)
        self.assertEqual(payload["report"]["session_id"], "sess-1")
        self.assertEqual(result.meta["provider"], "openrouter")
        self.assertEqual(result.meta["asr_provider"], "faster_whisper_chunked")

    @mock.patch("assess_speaking.build_progress_delta", return_value=None)
    @mock.patch("assess_speaking.run_assessment")
    def test_execute_assessment_run_persists_report_and_history(self, mock_run_assessment, _mock_progress):
        mock_run_assessment.return_value = _assessment_payload()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.wav"
            audio_path.write_bytes(b"fake-audio")
            result = execute_assessment_run(
                AssessmentRunRequest(
                    audio=audio_path,
                    whisper_model="small",
                    asr_provider="faster_whisper_chunked",
                    llm_model="google/gemini-3.1-pro-preview",
                    provider="openrouter",
                    expected_language="en",
                    feedback_language="en",
                    task_family="free_monologue",
                    speaker_id="bern",
                    theme="travel",
                    target_duration_sec=90,
                    log_dir=Path(tmpdir),
                    notes="Remember examples",
                )
            )

            self.assertIsNotNone(result.report_path)
            self.assertTrue(result.report_path.exists())
            self.assertTrue((Path(tmpdir) / "history.csv").exists())
            saved = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["transcript_full"], "hello world")
            self.assertEqual(saved["notes"], "Remember examples")

    @mock.patch("assess_speaking.build_progress_delta", return_value=None)
    @mock.patch("assess_speaking.run_assessment")
    def test_execute_assessment_run_forwards_status_callback(self, mock_run_assessment, _mock_progress):
        events: list[str] = []

        def _fake_run_assessment(*_args, **kwargs):
            callback = kwargs.get("status_callback")
            self.assertIsNotNone(callback)
            callback("transcribing")
            callback("finalizing_report")
            return _assessment_payload()

        mock_run_assessment.side_effect = _fake_run_assessment

        with tempfile.TemporaryDirectory() as tmpdir:
            execute_assessment_run(
                AssessmentRunRequest(
                    audio=Path(tmpdir) / "sample.wav",
                    whisper_model="small",
                    asr_provider="faster_whisper_chunked",
                    llm_model="google/gemini-3.1-pro-preview",
                    provider="openrouter",
                    expected_language="en",
                    feedback_language="en",
                    task_family="free_monologue",
                    speaker_id="bern",
                    theme="travel",
                    target_duration_sec=90,
                    log_dir=Path(tmpdir),
                    no_log=True,
                ),
                status_callback=events.append,
            )

        self.assertEqual(events, ["transcribing", "finalizing_report"])

    @mock.patch("assess_speaking.build_progress_delta", return_value=None)
    @mock.patch("assess_speaking.run_assessment")
    def test_execute_assessment_run_forwards_asr_provider(self, mock_run_assessment, _mock_progress):
        mock_run_assessment.return_value = _assessment_payload()

        with tempfile.TemporaryDirectory() as tmpdir:
            execute_assessment_run(
                AssessmentRunRequest(
                    audio=Path(tmpdir) / "sample.wav",
                    whisper_model="small",
                    asr_provider="faster_whisper_chunked",
                    llm_model="google/gemini-3.1-pro-preview",
                    provider="openrouter",
                    log_dir=Path(tmpdir),
                    no_log=True,
                )
            )

        self.assertEqual(mock_run_assessment.call_args.kwargs["asr_provider"], "faster_whisper_chunked")


if __name__ == "__main__":
    unittest.main()
