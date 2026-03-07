import contextlib
import io
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


class MetricsFromTests(unittest.TestCase):
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


class RubricPromptTests(unittest.TestCase):
    def test_rubric_prompt_it_includes_metrics(self):
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

        prompt = assess_speaking.rubric_prompt_it("Trascritto", metrics)

        for fragment in (
            "METRICHE OGGETTIVE",
            "Durata: 12",
            "WPM: 72.5",
            "Trascritto",
            "RISPONDI IN JSON",
        ):
            self.assertIn(fragment, prompt)


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
        with mock.patch.object(assess_speaking, "parselmouth", None), mock.patch.object(
            assess_speaking, "call", None
        ):
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


class CallOllamaTests(unittest.TestCase):
    @mock.patch("assess_speaking.subprocess.run")
    def test_call_ollama_parses_json_response(self, mock_run):
        mock_run.return_value = mock.Mock(stdout=json.dumps({"response": "ok"}))
        result = assess_speaking.call_ollama("llama", "prompt")
        self.assertEqual(result, "ok")
        self.assertTrue(mock_run.called)

    @mock.patch("assess_speaking.subprocess.run")
    def test_call_ollama_returns_raw_on_invalid_json(self, mock_run):
        mock_run.return_value = mock.Mock(stdout="not-json")
        result = assess_speaking.call_ollama("llama", "prompt")
        self.assertEqual(result, "not-json")

    @mock.patch(
        "assess_speaking.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "curl", stderr="boom"),
    )
    def test_call_ollama_handles_subprocess_errors(self, mock_run):
        result = assess_speaking.call_ollama("llama", "prompt")
        payload = json.loads(result)
        self.assertEqual(payload["error"], "ollama_not_running_or_model_missing")
        self.assertIn("boom", payload["detail"])


class ListOllamaTests(unittest.TestCase):
    @mock.patch("assess_speaking.subprocess.run")
    def test_list_ollama_models_success(self, mock_run):
        mock_run.return_value = mock.Mock(stdout="models")
        self.assertEqual(assess_speaking.list_ollama_models(), "models")

    @mock.patch(
        "assess_speaking.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "curl", stderr="fail"),
    )
    def test_list_ollama_models_error(self, mock_run):
        payload = json.loads(assess_speaking.list_ollama_models())
        self.assertEqual(payload["error"], "ollama_tags_failed")
        self.assertIn("fail", payload["detail"])


class SelfTestTests(unittest.TestCase):
    @mock.patch("assess_speaking.call_ollama", return_value="ok")
    def test_selftest_uses_default_prompt(self, mock_call):
        result = assess_speaking.selftest("llama")
        self.assertEqual(result, "ok")
        self.assertIn("Valuta brevemente", mock_call.call_args[0][1])


class TranscribeTests(unittest.TestCase):
    def test_transcribe_requires_whisper_model(self):
        with mock.patch.object(assess_speaking, "WhisperModel", None):
            with self.assertRaises(RuntimeError):
                assess_speaking.transcribe(Path("sample.wav"))

    def test_transcribe_rewords_missing_socksio(self):
        class BrokenModel:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[socks]`."
                )

        with mock.patch.object(assess_speaking, "WhisperModel", BrokenModel):
            with self.assertRaises(RuntimeError) as ctx:
                assess_speaking.transcribe(Path("sample.wav"))

        self.assertIn("SOCKS proxy detected", str(ctx.exception))
        self.assertIn("socksio", str(ctx.exception))

    def test_transcribe_rewords_proxy_download_failure(self):
        class ProxyError(Exception):
            __module__ = "httpx"

        class BrokenModel:
            def __init__(self, *args, **kwargs):
                raise ProxyError("403 Forbidden")

        with mock.patch.object(assess_speaking, "WhisperModel", BrokenModel):
            with self.assertRaises(RuntimeError) as ctx:
                assess_speaking.transcribe(Path("sample.wav"))

        self.assertIn("Whisper model download failed", str(ctx.exception))
        self.assertIn("Hugging Face", str(ctx.exception))

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
                return segments, {"dummy": True}

        with mock.patch.object(assess_speaking, "WhisperModel", DummyModel):
            result = assess_speaking.transcribe(Path("sample.wav"), model_size="tiny")
        self.assertEqual(result["text"], "Ciao mondo")
        self.assertEqual(
            result["words"],
            [
                {"t0": 0.0, "t1": 0.4, "text": "ciao"},
                {"t0": 0.4, "t1": 0.8, "text": "mondo"},
            ],
        )


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
        with mock.patch("sys.argv", ["assess_speaking.py", "--list-ollama"]), mock.patch.object(
            assess_speaking,
            "list_ollama_models",
            return_value="models",
        ), contextlib.redirect_stdout(buf):
            assess_speaking.main()
        self.assertEqual(buf.getvalue().strip(), "models")

    def test_main_selftest(self):
        buf = io.StringIO()
        with mock.patch("sys.argv", ["assess_speaking.py", "--selftest", "--llm", "llama"]), mock.patch.object(
            assess_speaking,
            "selftest",
            return_value="ok",
        ), contextlib.redirect_stdout(buf):
            assess_speaking.main()
        self.assertEqual(buf.getvalue().strip(), "ok")

    def test_main_canvas_upload_requires_course_id(self):
        args = [
            "assess_speaking.py",
            "sample.wav",
            "--no-log",
            "--lms-type",
            "canvas",
            "--lms-url",
            "https://canvas.example.edu",
            "--lms-token",
            "token",
            "--lms-assign-id",
            "42",
        ]

        with mock.patch("sys.argv", args), mock.patch.object(assess_speaking, "upload_to_canvas") as mock_upload:
            with self.assertRaises(RuntimeError) as ctx:
                assess_speaking.main()

        mock_upload.assert_not_called()
        self.assertIn("--lms-course-id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
