import unittest
from pathlib import Path
from unittest import mock

from assessment_runtime import asr


class AsrTests(unittest.TestCase):
    @staticmethod
    def _dummy_transcription_result():
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
            language_probability = 0.95

        return [DummySegment(" Ciao ", [DummyWord("Ciao", 0.0, 0.5)])], DummyInfo()

    def test_transcribe_uses_fallback_compute_type(self):
        class DummyModel:
            calls = []

            def __init__(self, model_size, compute_type="default"):
                self.calls.append(compute_type)
                if compute_type == "default":
                    raise RuntimeError("unsupported")

            def transcribe(self, path, **kwargs):
                return AsrTests._dummy_transcription_result()

        with mock.patch.object(asr, "WhisperModel", DummyModel):
            result = asr.transcribe(Path("sample.wav"), compute_type="default", fallback_compute_type="int8")
        self.assertEqual(result["compute_type_used"], "int8")
        self.assertTrue(result["compute_fallback_used"])

    def test_transcribe_prefers_cached_snapshot_path(self):
        class DummyModel:
            calls = []

            def __init__(self, model_size, compute_type="default"):
                self.calls.append((model_size, compute_type))

            def transcribe(self, path, **kwargs):
                return AsrTests._dummy_transcription_result()

        cached_snapshot = "/tmp/faster-whisper-tiny-snapshot"
        with (
            mock.patch.object(asr, "WhisperModel", DummyModel),
            mock.patch.object(asr, "_resolve_cached_model_path", return_value=cached_snapshot),
        ):
            result = asr.transcribe(Path("sample.wav"), model_size="tiny", compute_type="default")
        self.assertEqual(DummyModel.calls[0], (cached_snapshot, "default"))
        self.assertEqual(result["compute_type_used"], "default")
        self.assertFalse(result["compute_fallback_used"])

    def test_resolve_cached_model_path_prefers_main_ref(self):
        with mock.patch.dict(
            asr.os.environ,
            {"HF_HUB_CACHE": "/cache"},
            clear=False,
        ), mock.patch.object(asr.Path, "home", return_value=Path("/home/test")):
            cache_root = Path("/cache")
            preferred_snapshot = cache_root / "models--Systran--faster-whisper-tiny" / "snapshots" / "preferred"
            alternate_snapshot = cache_root / "models--Systran--faster-whisper-tiny" / "snapshots" / "older"
            main_ref = cache_root / "models--Systran--faster-whisper-tiny" / "refs" / "main"

            def fake_exists(path_self):
                return path_self in {
                    cache_root / "models--Systran--faster-whisper-tiny" / "snapshots",
                    preferred_snapshot,
                    alternate_snapshot,
                    main_ref,
                }

            def fake_is_dir(path_self):
                return path_self in {preferred_snapshot, alternate_snapshot}

            def fake_iterdir(path_self):
                if path_self == cache_root / "models--Systran--faster-whisper-tiny" / "snapshots":
                    return iter([alternate_snapshot, preferred_snapshot])
                raise AssertionError(f"Unexpected iterdir call for {path_self}")

            def fake_read_text(path_self, encoding="utf-8"):
                if path_self == main_ref:
                    return "preferred\n"
                raise AssertionError(f"Unexpected read_text call for {path_self}")

            with (
                mock.patch.object(Path, "exists", fake_exists),
                mock.patch.object(Path, "is_dir", fake_is_dir),
                mock.patch.object(Path, "iterdir", fake_iterdir),
                mock.patch.object(Path, "read_text", fake_read_text),
            ):
                resolved = asr._resolve_cached_model_path("tiny")

        self.assertEqual(str(preferred_snapshot), resolved)

    def test_recommend_model_choice_prefers_cached_highest_quality(self):
        with mock.patch.object(asr, "describe_model_availability") as mock_describe:
            mock_describe.side_effect = lambda model: {
                "large-v3": {"cached": False},
                "medium": {"cached": True},
                "small": {"cached": True},
                "tiny": {"cached": True},
            }[model]
            recommendation = asr.recommend_model_choice()
        self.assertEqual(recommendation["model"], "medium")

    def test_describe_model_availability_treats_dry_run_as_cached(self):
        with (
            mock.patch.dict(asr.os.environ, {"ASSESS_SPEAKING_DRY_RUN": "1"}, clear=False),
            mock.patch.object(asr, "_resolve_cached_model_path", return_value=None),
        ):
            availability = asr.describe_model_availability("tiny")
        self.assertTrue(availability["cached"])
        self.assertIsNone(availability["cached_path"])

    def test_ensure_model_downloaded_returns_cached_model_without_init(self):
        with (
            mock.patch.object(asr, "WhisperModel", object()),
            mock.patch.object(asr, "describe_model_availability", return_value={"cached": True, "cached_path": "/tmp/model"}),
            mock.patch.object(asr, "_initialize_whisper_model") as mock_init,
        ):
            availability = asr.ensure_model_downloaded("tiny")
        self.assertEqual(availability["cached_path"], "/tmp/model")
        mock_init.assert_not_called()

    def test_ensure_model_downloaded_initializes_when_missing(self):
        availability_side_effect = [
            {"cached": False, "cached_path": None},
            {"cached": True, "cached_path": "/tmp/model"},
        ]
        with (
            mock.patch.object(asr, "WhisperModel", object()),
            mock.patch.object(asr, "describe_model_availability", side_effect=availability_side_effect),
            mock.patch.object(asr, "_initialize_whisper_model", return_value=(object(), "default", False)) as mock_init,
        ):
            availability = asr.ensure_model_downloaded("tiny")
        self.assertEqual(availability["cached_path"], "/tmp/model")
        mock_init.assert_called_once()


if __name__ == "__main__":
    unittest.main()
