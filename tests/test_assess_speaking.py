import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import assess_speaking
import asr
import audio_features
from llm_client import LLMClientError
from schemas import RubricResult


def _sample_report(*, overall: int = 4) -> dict:
    rubric = {
        "fluency": overall,
        "cohesion": overall,
        "accuracy": overall,
        "range": overall,
        "overall": overall,
        "comments_fluency": "ok",
        "comments_cohesion": "ok",
        "comments_accuracy": "ok",
        "comments_range": "ok",
        "overall_comment": "ok",
        "on_topic": True,
    }
    return {
        "timestamp_utc": "2026-03-07T10:00:00+00:00",
        "input": {
            "provider": "openrouter",
            "llm_model": "google/gemini-3.1-pro-preview",
            "whisper_model": "large-v3",
            "expected_language": "it",
            "detected_language": "it",
            "detected_language_probability": 0.99,
            "theme": "la mia città",
            "target_duration_sec": 60.0,
            "prompt_version": "rubric_it_v1",
            "asr_compute_type": "default",
            "asr_fallback_compute_type": "int8",
            "asr_compute_type_used": "default",
            "asr_compute_fallback_used": False,
            "pause_threshold_offset_db": -10.0,
        },
        "metrics": {
            "duration_sec": 30.0,
            "pause_count": 1,
            "pause_total_sec": 1.0,
            "speaking_time_sec": 29.0,
            "word_count": 40,
            "wpm": 82.8,
            "fillers": 1,
            "cohesion_markers": 1,
            "complexity_index": 1,
        },
        "checks": {
            "duration_pass": False,
            "topic_pass": True,
            "min_words_pass": True,
            "language_pass": True,
            "asr_speaking_time_sec": 28.5,
            "speaking_time_delta_sec": 0.5,
            "asr_pause_consistent": True,
        },
        "scores": {
            "deterministic": 3.2,
            "llm": float(overall),
            "final": 3.68,
            "band": 4,
            "mode": "hybrid",
        },
        "requires_human_review": False,
        "transcript_preview": "ciao mondo",
        "warnings": [],
        "errors": [],
        "rubric": rubric,
        "timings_ms": {"audio_features": 10.0, "asr": 20.0, "llm": 30.0},
    }


class MetricsAndPromptTests(unittest.TestCase):
    def test_metrics_from_handles_fillers_and_markers(self):
        words = [
            {"text": "buongiorno"},
            {"text": "eh"},
            {"text": "oggi"},
            {"text": "parlo"},
            {"text": "di"},
            {"text": "efficienza"},
            {"text": "energetica"},
            {"text": "inoltre"},
            {"text": "che"},
            {"text": "qualora"},
        ]
        audio = {"duration_sec": 12.0, "pauses": [(1.0, 1.5, 0.5), (5.0, 5.7, 0.7)]}
        result = assess_speaking.metrics_from(words, audio)
        self.assertEqual(result["duration_sec"], 12.0)
        self.assertEqual(result["pause_count"], 2)
        self.assertAlmostEqual(result["pause_total_sec"], 1.2)
        self.assertAlmostEqual(result["speaking_time_sec"], 10.8)
        self.assertEqual(result["word_count"], 10)
        self.assertAlmostEqual(result["wpm"], 55.6)
        self.assertEqual(result["fillers"], 1)
        self.assertEqual(result["cohesion_markers"], 1)
        self.assertEqual(result["complexity_index"], 2)

    def test_rubric_prompt_it_includes_theme_and_schema(self):
        metrics = {
            "duration_sec": 12,
            "speaking_time_sec": 10,
            "pause_total_sec": 2,
            "pause_count": 3,
            "word_count": 120,
            "wpm": 72.5,
            "fillers": 1,
            "cohesion_markers": 2,
            "complexity_index": 4,
        }
        prompt = assess_speaking.rubric_prompt_it("Trascritto", metrics, "la mia città")
        self.assertIn("la mia città", prompt)
        self.assertIn("on_topic", prompt)
        self.assertIn("Rispondi SOLO con JSON valido", prompt)

    def test_rubric_prompt_sanitizes_triple_quotes(self):
        metrics = {
            "duration_sec": 12,
            "speaking_time_sec": 10,
            "pause_total_sec": 2,
            "pause_count": 3,
            "word_count": 120,
            "wpm": 72.5,
            "fillers": 1,
            "cohesion_markers": 2,
            "complexity_index": 4,
        }
        prompt = assess_speaking.rubric_prompt_it('Ho detto """ciao"""', metrics, "la mia città")
        self.assertNotIn('"""ciao"""', prompt)
        self.assertIn("'''ciao'''", prompt)


class ParsingAndBaselineTests(unittest.TestCase):
    def test_extract_rubric_json_from_code_block(self):
        payload = """
        Risposta:
        ```json
        {"overall": 4, "fluency": 4}
        ```
        """
        parsed = assess_speaking.extract_rubric_json(payload)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["overall"], 4)

    def test_extract_rubric_json_handles_trailing_braces(self):
        payload = 'prefix {"overall":4,"comments":"{ok}"} suffix }'
        parsed = assess_speaking.extract_rubric_json(payload)
        self.assertEqual(parsed["overall"], 4)
        self.assertEqual(parsed["comments"], "{ok}")

    def test_extract_rubric_json_invalid_returns_none(self):
        self.assertIsNone(assess_speaking.extract_rubric_json("not json"))

    def test_evaluate_baseline_returns_expected_flags(self):
        metrics = {"wpm": 120, "fillers": 3, "cohesion_markers": 1, "complexity_index": 1}
        result = assess_speaking.evaluate_baseline("B2", metrics)
        self.assertTrue(result["passed"])
        self.assertTrue(result["targets"]["wpm"]["ok"])

    def test_load_audio_features_requires_parselmouth(self):
        with mock.patch.object(audio_features, "parselmouth", None), mock.patch.object(audio_features, "call", None):
            with self.assertRaises(RuntimeError) as ctx:
                audio_features.load_audio_features(Path("dummy.wav"))
        self.assertIn("praat-parselmouth", str(ctx.exception))


class OllamaHelpersTests(unittest.TestCase):
    @mock.patch("assess_speaking.subprocess.run")
    def test_call_ollama_parses_json_response(self, mock_run):
        mock_run.return_value = mock.Mock(stdout=json.dumps({"response": "ok"}))
        result = assess_speaking.call_ollama("llama", "prompt")
        self.assertEqual(result, "ok")

    @mock.patch("assess_speaking.subprocess.run")
    def test_call_ollama_returns_raw_on_invalid_json(self, mock_run):
        mock_run.return_value = mock.Mock(stdout="not-json")
        result = assess_speaking.call_ollama("llama", "prompt")
        self.assertEqual(result, "not-json")

    @mock.patch(
        "assess_speaking.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "curl", stderr="boom"),
    )
    def test_call_ollama_handles_subprocess_errors(self, _mock_run):
        result = assess_speaking.call_ollama("llama", "prompt")
        payload = json.loads(result)
        self.assertEqual(payload["error"], "ollama_not_running_or_model_missing")
        self.assertIn("boom", payload["detail"])

    @mock.patch("assess_speaking.subprocess.run")
    def test_list_ollama_models_success(self, mock_run):
        mock_run.return_value = mock.Mock(stdout="models")
        self.assertEqual(assess_speaking.list_ollama_models(), "models")


class SelftestAndTranscribeTests(unittest.TestCase):
    @mock.patch("assess_speaking.call_ollama", return_value="ok")
    def test_selftest_uses_legacy_ollama_when_model_looks_local(self, mock_call):
        result = assess_speaking.selftest("llama3.1")
        self.assertEqual(result, "ok")
        self.assertIn("la mia città", mock_call.call_args[0][1])

    @mock.patch("assess_speaking.generate_rubric")
    def test_selftest_uses_openrouter_when_provider_selected(self, mock_generate):
        mock_generate.return_value = (
            RubricResult(
                fluency=3,
                cohesion=3,
                accuracy=3,
                range=3,
                overall=3,
                comments_fluency="ok",
                comments_cohesion="ok",
                comments_accuracy="ok",
                comments_range="ok",
                overall_comment="ok",
                on_topic=True,
            ),
            "{}",
        )
        payload = json.loads(assess_speaking.selftest(model="google/gemini-3.1-pro-preview", provider="openrouter"))
        self.assertEqual(payload["overall"], 3)

    def test_transcribe_requires_whisper_model(self):
        with mock.patch.object(asr, "WhisperModel", None):
            with self.assertRaises(RuntimeError):
                assess_speaking.transcribe(Path("sample.wav"))

    def test_transcribe_rewords_missing_socksio(self):
        class BrokenModel:
            def __init__(self, *args, **kwargs):
                raise ImportError("Using SOCKS proxy, but the 'socksio' package is not installed.")

        with mock.patch.object(asr, "WhisperModel", BrokenModel):
            with self.assertRaises(RuntimeError) as ctx:
                assess_speaking.transcribe(Path("sample.wav"))
        self.assertIn("SOCKS proxy detected", str(ctx.exception))

    def test_transcribe_rewords_proxy_download_failure(self):
        class ProxyError(Exception):
            __module__ = "httpx"

        class BrokenModel:
            def __init__(self, *args, **kwargs):
                raise ProxyError("403 Forbidden")

        with mock.patch.object(asr, "WhisperModel", BrokenModel):
            with self.assertRaises(RuntimeError) as ctx:
                assess_speaking.transcribe(Path("sample.wav"))
        self.assertIn("Whisper model download failed", str(ctx.exception))

    def test_transcribe_collects_segments(self):
        class DummyWord:
            def __init__(self, word, start, end):
                self.word = word
                self.start = start
                self.end = end

        class DummySegment:
            def __init__(self, text, words):
                self.text = text
                self.words = words

        class DummyInfo:
            language = "it"
            language_probability = 0.99

        class DummyModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, path, **kwargs):
                segments = [
                    DummySegment(
                        " Ciao mondo ",
                        [DummyWord("Ciao", 0.0, 0.4), DummyWord("Mondo", 0.4, 0.8)],
                    )
                ]
                return segments, DummyInfo()

        with mock.patch.object(asr, "WhisperModel", DummyModel):
            result = assess_speaking.transcribe(Path("sample.wav"), model_size="tiny")
        self.assertEqual(result["text"], "Ciao mondo")
        self.assertEqual(result["detected_language"], "it")
        self.assertEqual(
            result["words"],
            [
                {"t0": 0.0, "t1": 0.4, "text": "ciao"},
                {"t0": 0.4, "t1": 0.8, "text": "mondo"},
            ],
        )


class LmsConfigTests(unittest.TestCase):
    def test_resolve_lms_token_prefers_cli_token(self):
        with mock.patch.dict("os.environ", {"CANVAS_TOKEN": "env-token"}, clear=False):
            token, source = assess_speaking.resolve_lms_token("canvas", "cli-token")
        self.assertEqual(token, "cli-token")
        self.assertEqual(source, "cli")


class RunAssessmentTests(unittest.TestCase):
    @mock.patch.object(assess_speaking, "generate_feedback", return_value=[{"id": "res"}])
    @mock.patch.object(assess_speaking, "evaluate_baseline", return_value={"level": "B1", "passed": True})
    @mock.patch.object(
        assess_speaking,
        "generate_rubric",
        return_value=(
            RubricResult(
                fluency=4,
                cohesion=4,
                accuracy=3,
                range=4,
                overall=4,
                comments_fluency="ok",
                comments_cohesion="ok",
                comments_accuracy="ok",
                comments_range="ok",
                overall_comment="ok",
                on_topic=True,
            ),
            '{"overall":4,"on_topic":true}',
        ),
    )
    @mock.patch.object(assess_speaking, "load_audio_features", return_value={"duration_sec": 70.0, "pauses": [(5.0, 6.0, 1.0)]})
    @mock.patch.object(
        assess_speaking,
        "transcribe",
        return_value={
            "text": "parlo della mia città e dei trasporti pubblici",
            "detected_language": "it",
            "language_probability": 0.99,
            "compute_type_used": "default",
            "compute_fallback_used": False,
            "words": [
                {"t0": 0.0, "t1": 10.0, "text": "parlo"},
                {"t0": 10.0, "t1": 20.0, "text": "della"},
                {"t0": 20.0, "t1": 30.0, "text": "mia"},
                {"t0": 30.0, "t1": 40.0, "text": "città"},
                {"t0": 40.0, "t1": 69.0, "text": "trasporti"},
            ],
        },
    )
    def test_run_assessment_returns_legacy_payload_plus_report(
        self,
        _mock_transcribe,
        _mock_audio,
        _mock_generate_rubric,
        _mock_baseline,
        _mock_feedback,
    ):
        result = assess_speaking.run_assessment(
            Path("sample.wav"),
            llm_model="google/gemini-3.1-pro-preview",
            provider="openrouter",
            feedback_enabled=True,
            target_cefr="B1",
            theme="la mia città",
            target_duration_sec=60.0,
        )
        self.assertIn("metrics", result)
        self.assertEqual(result["transcript_preview"], "parlo della mia città e dei trasporti pubblici")
        self.assertIn("baseline_comparison", result)
        self.assertIn("suggested_training", result)
        self.assertIn("report", result)
        self.assertEqual(result["report"]["input"]["provider"], "openrouter")
        self.assertEqual(result["report"]["scores"]["mode"], "hybrid")
        self.assertFalse(result["report"]["requires_human_review"])

    @mock.patch.object(assess_speaking, "generate_rubric")
    @mock.patch.object(assess_speaking, "load_audio_features", return_value={"duration_sec": 20.0, "pauses": []})
    @mock.patch.object(
        assess_speaking,
        "transcribe",
        return_value={
            "text": "I am talking about my city.",
            "detected_language": "en",
            "language_probability": 0.95,
            "compute_type_used": "default",
            "compute_fallback_used": False,
            "words": [
                {"t0": 0.0, "t1": 3.0, "text": "i"},
                {"t0": 3.0, "t1": 6.0, "text": "am"},
                {"t0": 6.0, "t1": 9.0, "text": "talking"},
                {"t0": 9.0, "t1": 12.0, "text": "about"},
                {"t0": 12.0, "t1": 15.0, "text": "city"},
            ],
        },
    )
    def test_run_assessment_skips_llm_on_language_mismatch(self, _mock_transcribe, _mock_audio, mock_generate):
        result = assess_speaking.run_assessment(
            Path("sample.wav"),
            llm_model="google/gemini-3.1-pro-preview",
            provider="openrouter",
            expected_language="it",
        )
        self.assertTrue(result["report"]["requires_human_review"])
        self.assertIn("language_mismatch", result["report"]["warnings"])
        self.assertIsNone(result["report"]["rubric"])
        mock_generate.assert_not_called()

    @mock.patch.object(assess_speaking, "load_audio_features", return_value={"duration_sec": 50.0, "pauses": []})
    @mock.patch.object(
        assess_speaking,
        "transcribe",
        return_value={
            "text": "parlo della mia città e dei trasporti pubblici",
            "detected_language": "it",
            "language_probability": 0.99,
            "compute_type_used": "default",
            "compute_fallback_used": False,
            "words": [
                {"t0": 0.0, "t1": 10.0, "text": "parlo"},
                {"t0": 10.0, "t1": 20.0, "text": "della"},
                {"t0": 20.0, "t1": 30.0, "text": "mia"},
                {"t0": 30.0, "t1": 40.0, "text": "città"},
                {"t0": 40.0, "t1": 49.0, "text": "trasporti"},
            ],
        },
    )
    @mock.patch.object(assess_speaking, "generate_rubric", side_effect=LLMClientError("Request timed out after 3.0s"))
    def test_run_assessment_marks_timeout_as_degraded(self, _mock_generate, _mock_transcribe, _mock_audio):
        result = assess_speaking.run_assessment(
            Path("sample.wav"),
            llm_model="google/gemini-3.1-pro-preview",
            provider="openrouter",
            llm_timeout_sec=3.0,
        )
        self.assertIn("llm_unavailable", result["report"]["warnings"])
        self.assertIn("timed out", " ".join(result["report"]["errors"]))
        self.assertTrue(result["report"]["requires_human_review"])
        self.assertEqual(result["report"]["scores"]["mode"], "deterministic_only")

    @mock.patch.object(assess_speaking, "call_ollama", return_value=json.dumps(_sample_report()["rubric"]))
    @mock.patch.object(assess_speaking, "load_audio_features", return_value={"duration_sec": 40.0, "pauses": [(1.0, 2.0, 1.0)]})
    @mock.patch.object(
        assess_speaking,
        "transcribe",
        return_value={
            "text": "ciao mondo adesso parlo della mia città",
            "detected_language": "it",
            "language_probability": 0.99,
            "compute_type_used": "default",
            "compute_fallback_used": False,
            "words": [
                {"t0": 0.0, "t1": 1.0, "text": "ciao"},
                {"t0": 1.0, "t1": 2.0, "text": "mondo"},
                {"t0": 2.0, "t1": 3.0, "text": "adesso"},
                {"t0": 3.0, "t1": 4.0, "text": "parlo"},
                {"t0": 4.0, "t1": 5.0, "text": "della"},
                {"t0": 5.0, "t1": 6.0, "text": "mia"},
                {"t0": 6.0, "t1": 7.0, "text": "città"},
            ],
        },
    )
    def test_run_assessment_legacy_ollama_path_still_builds_report(self, _mock_transcribe, _mock_audio, _mock_call):
        result = assess_speaking.run_assessment(Path("sample.wav"), llm_model="llama3.1")
        self.assertEqual(result["report"]["input"]["provider"], "ollama")
        self.assertEqual(result["report"]["scores"]["mode"], "hybrid")
        self.assertIn("llm_rubric", result)


class MainCliTests(unittest.TestCase):
    def test_main_without_arguments_exits(self):
        buf = io.StringIO()
        with mock.patch("sys.argv", ["assess_speaking.py"]), contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit) as exc:
                assess_speaking.main()
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("Bitte Audio-Datei", buf.getvalue())

    def test_main_list_ollama(self):
        buf = io.StringIO()
        with (
            mock.patch("sys.argv", ["assess_speaking.py", "--list-ollama"]),
            mock.patch.object(assess_speaking, "list_ollama_models", return_value="models"),
            contextlib.redirect_stdout(buf),
        ):
            assess_speaking.main()
        self.assertEqual(buf.getvalue().strip(), "models")

    def test_main_selftest(self):
        buf = io.StringIO()
        with (
            mock.patch("sys.argv", ["assess_speaking.py", "--selftest", "--llm", "llama3.1"]),
            mock.patch.object(assess_speaking, "selftest", return_value='{"ok": true}'),
            contextlib.redirect_stdout(buf),
        ):
            assess_speaking.main()
        self.assertEqual(buf.getvalue().strip(), '{"ok": true}')

    @mock.patch("assess_speaking.run_assessment")
    def test_main_normalizes_none_as_disabled_asr_fallback(self, mock_run_assessment):
        mock_run_assessment.return_value = {
            "metrics": _sample_report()["metrics"],
            "transcript_full": "ciao mondo",
            "transcript_preview": "ciao mondo",
            "llm_rubric": json.dumps(_sample_report()["rubric"]),
            "report": _sample_report(),
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        args = [
            "assess_speaking.py",
            "sample.wav",
            "--no-log",
            "--asr-fallback-compute-type",
            "none",
        ]
        with (
            mock.patch("sys.argv", args),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            assess_speaking.main()
        self.assertIsNone(mock_run_assessment.call_args.kwargs["asr_fallback_compute_type"])

    @mock.patch("assess_speaking.run_assessment")
    def test_main_lms_dry_run_uses_env_token_and_skips_upload(self, mock_run_assessment):
        stdout = io.StringIO()
        stderr = io.StringIO()
        mock_run_assessment.return_value = {
            "metrics": {"word_count": 4, "duration_sec": 5.0, "wpm": 48.0},
            "transcript_full": "ciao",
            "transcript_preview": "ciao",
            "llm_rubric": '{"overall": 4}',
            "report": _sample_report(),
        }
        args = [
            "assess_speaking.py",
            "sample.wav",
            "--no-log",
            "--lms-type",
            "canvas",
            "--lms-url",
            "https://canvas.example.edu",
            "--lms-course-id",
            "7",
            "--lms-assign-id",
            "42",
            "--lms-dry-run",
        ]
        with (
            mock.patch("sys.argv", args),
            mock.patch.dict("os.environ", {"CANVAS_TOKEN": "env-token"}, clear=False),
            mock.patch.object(assess_speaking, "upload_to_canvas") as mock_upload,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            assess_speaking.main()
        mock_upload.assert_not_called()
        self.assertIn('"metrics"', stdout.getvalue())
        self.assertIn("[lms] Dry run:", stderr.getvalue())
        self.assertIn('"token_source": "env:CANVAS_TOKEN"', stderr.getvalue())

    @mock.patch("assess_speaking.run_assessment")
    def test_main_logs_history_with_report_payload(self, mock_run_assessment):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_run_assessment.return_value = {
                "metrics": _sample_report()["metrics"],
                "transcript_full": "ciao mondo",
                "transcript_preview": "ciao mondo",
                "llm_rubric": json.dumps(_sample_report()["rubric"]),
                "report": _sample_report(),
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            args = [
                "assess_speaking.py",
                "sample.wav",
                "--log-dir",
                tmpdir,
                "--theme",
                "la mia città",
            ]
            with (
                mock.patch("sys.argv", args),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                assess_speaking.main()
            payload = json.loads(stdout.getvalue())
            self.assertIn("report", payload)
            history = Path(tmpdir) / "history.csv"
            self.assertTrue(history.exists())
            self.assertIn("Ergebnis gespeichert", stderr.getvalue())


class ConvertToWavTests(unittest.TestCase):
    class _NamedTempFileStub:
        def __init__(self, path: Path):
            self.name = str(path)
            self._path = path

        def __enter__(self):
            self._path.touch()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    @mock.patch("assess_speaking.subprocess.run", side_effect=FileNotFoundError("ffmpeg"))
    def test_convert_to_wav_requires_ffmpeg(self, _mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "in-conversion.wav"
            with mock.patch(
                "assess_speaking.tempfile.NamedTemporaryFile",
                return_value=self._NamedTempFileStub(tmp_wav),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    assess_speaking._convert_to_wav(Path("in.mp3"))
        self.assertIn("ffmpeg is required", str(ctx.exception))
        self.assertFalse(tmp_wav.exists())

    @mock.patch(
        "assess_speaking.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"bad input"),
    )
    def test_convert_to_wav_handles_ffmpeg_failure(self, _mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "broken-conversion.wav"
            with mock.patch(
                "assess_speaking.tempfile.NamedTemporaryFile",
                return_value=self._NamedTempFileStub(tmp_wav),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    assess_speaking._convert_to_wav(Path("broken.mp3"))
        self.assertIn("Audio conversion failed", str(ctx.exception))
        self.assertFalse(tmp_wav.exists())


if __name__ == "__main__":
    unittest.main()
