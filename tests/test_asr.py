import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
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

    def test_describe_model_availability_does_not_force_cache_in_dry_run(self):
        with (
            mock.patch.dict(asr.os.environ, {"ASSESS_SPEAKING_DRY_RUN": "1"}, clear=False),
            mock.patch.object(asr, "_resolve_cached_model_path", return_value=None),
        ):
            availability = asr.describe_model_availability("tiny")
        self.assertFalse(availability["cached"])
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

    def test_ensure_model_downloaded_reports_progress_for_hub_downloads(self):
        availability_side_effect = [
            {"cached": False, "cached_path": None},
            {"cached": True, "cached_path": "/tmp/model"},
        ]
        planned_files = [
            SimpleNamespace(
                filename="config.json",
                file_size=200,
                commit_hash="abc123",
                local_path="/tmp/model/config.json",
                will_download=False,
            ),
            SimpleNamespace(
                filename="model.bin",
                file_size=800,
                commit_hash="abc123",
                local_path="/tmp/model/model.bin",
                will_download=True,
            ),
        ]
        events: list[dict] = []

        def fake_hf_hub_download(_repo_id, filename, revision, tqdm_class=None):
            self.assertEqual(revision, "abc123")
            if tqdm_class is not None:
                progress = tqdm_class(total=800, initial=0)
                progress.update(800)
            return f"/tmp/model/{filename}"

        with (
            mock.patch.object(asr, "WhisperModel", object()),
            mock.patch.object(asr, "describe_model_availability", side_effect=availability_side_effect),
            mock.patch.object(asr, "_plan_snapshot_download", return_value=planned_files),
            mock.patch.object(asr, "hf_hub_download", side_effect=fake_hf_hub_download),
            mock.patch.object(asr, "_initialize_whisper_model", return_value=(object(), "default", False)),
        ):
            availability = asr.ensure_model_downloaded("tiny", progress_callback=events.append)

        self.assertEqual(availability["cached_path"], "/tmp/model")
        self.assertEqual(events[0]["stage"], "checking_cache")
        self.assertIn("starting_download", [event["stage"] for event in events])
        self.assertIn("finalizing", [event["stage"] for event in events])
        self.assertEqual(events[-1]["stage"], "ready")
        downloading_events = [event for event in events if event["stage"] == "downloading"]
        self.assertTrue(any(event.get("current_file") == "model.bin" for event in downloading_events))

    def test_transcribe_rejects_unknown_asr_provider(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported ASR provider"):
            asr.transcribe(Path("sample.wav"), asr_provider="unknown-provider")

    def test_available_asr_providers_lists_chunked_variant(self):
        providers = asr.available_asr_providers()
        self.assertIn("faster_whisper", providers)
        self.assertIn("faster_whisper_chunked", providers)

    def test_known_asr_providers_matches_registered_providers(self):
        self.assertEqual(asr.KNOWN_ASR_PROVIDERS, asr.available_asr_providers())

    def test_normalize_asr_provider_accepts_chunked_alias(self):
        self.assertEqual(asr._normalize_asr_provider_key("whisper_chunked"), "faster_whisper_chunked")

    def test_transcribe_with_native_strategy_uses_provider_native_path(self):
        class DummyProvider:
            provider_id = "dummy"
            capabilities = asr.ASRCapabilities(
                supports_native_file_transcription=True,
                prefers_native_file_transcription=True,
                supports_chunked_fallback=True,
            )

            def load_model(self, *, model_size, compute_type, fallback_compute_type):
                return asr.LoadedASRModel(
                    model=object(),
                    compute_type_used=compute_type,
                    compute_fallback_used=False,
                )

            def transcribe_file_native(self, loaded_model, path, *, language, time_offset_sec=0.0):
                return {
                    "text": f"native:{path.name}",
                    "words": [{"t0": time_offset_sec, "t1": time_offset_sec + 0.5, "text": "native"}],
                    "compute_type_used": loaded_model.compute_type_used,
                    "compute_fallback_used": loaded_model.compute_fallback_used,
                    "detected_language": language,
                    "language_probability": 1.0,
                }

        with mock.patch.object(asr, "_resolve_asr_provider", return_value=DummyProvider()):
            result = asr.transcribe(
                Path("sample.wav"),
                asr_provider="dummy",
                file_strategy="native",
                language="it",
            )

        self.assertEqual(result["text"], "native:sample.wav")
        self.assertEqual(result["detected_language"], "it")

    def test_transcribe_with_chunked_strategy_merges_chunk_offsets(self):
        class DummyProvider:
            provider_id = "dummy"
            capabilities = asr.ASRCapabilities(
                supports_native_file_transcription=True,
                prefers_native_file_transcription=False,
                supports_chunked_fallback=True,
            )

            def load_model(self, *, model_size, compute_type, fallback_compute_type):
                return asr.LoadedASRModel(
                    model=object(),
                    compute_type_used="int8",
                    compute_fallback_used=True,
                )

            def transcribe_file_native(self, loaded_model, path, *, language, time_offset_sec=0.0):
                token = path.stem.replace("chunk-", "")
                return {
                    "text": f"text-{token}",
                    "words": [{"t0": time_offset_sec, "t1": time_offset_sec + 0.25, "text": token}],
                    "compute_type_used": loaded_model.compute_type_used,
                    "compute_fallback_used": loaded_model.compute_fallback_used,
                    "detected_language": "it",
                    "language_probability": 0.91,
                }

        @contextmanager
        def fake_chunk_context(_path, *, chunk_duration_sec):
            self.assertEqual(chunk_duration_sec, 30)
            yield [Path("chunk-000.wav"), Path("chunk-001.wav")]

        with (
            mock.patch.object(asr, "_resolve_asr_provider", return_value=DummyProvider()),
            mock.patch.object(asr, "_chunk_audio_for_asr", fake_chunk_context),
            mock.patch.object(asr, "_wav_duration_sec", side_effect=[2.0, 3.0]),
        ):
            result = asr.transcribe(
                Path("meeting.mp3"),
                asr_provider="dummy",
                file_strategy="chunked",
                chunk_duration_sec=30,
            )

        self.assertEqual(result["text"], "text-000 text-001")
        self.assertEqual(
            result["words"],
            [
                {"t0": 0.0, "t1": 0.25, "text": "000"},
                {"t0": 2.0, "t1": 2.25, "text": "001"},
            ],
        )
        self.assertEqual(result["compute_type_used"], "int8")
        self.assertTrue(result["compute_fallback_used"])
        self.assertEqual(result["detected_language"], "it")

    def test_chunked_provider_uses_chunked_path_in_auto_mode(self):
        class DummyProvider:
            provider_id = "faster_whisper_chunked"
            capabilities = asr.ASRCapabilities(
                supports_native_file_transcription=True,
                prefers_native_file_transcription=False,
                supports_chunked_fallback=True,
            )

            def load_model(self, *, model_size, compute_type, fallback_compute_type):
                return asr.LoadedASRModel(
                    model=object(),
                    compute_type_used=compute_type,
                    compute_fallback_used=False,
                )

            def transcribe_file_native(self, loaded_model, path, *, language, time_offset_sec=0.0):
                return {
                    "text": "native",
                    "words": [],
                    "compute_type_used": loaded_model.compute_type_used,
                    "compute_fallback_used": loaded_model.compute_fallback_used,
                    "detected_language": language,
                    "language_probability": 1.0,
                }

        with (
            mock.patch.object(asr, "_resolve_asr_provider", return_value=DummyProvider()),
            mock.patch.object(asr, "_transcribe_file_chunked", return_value={"text": "chunked", "words": []}) as mock_chunked,
        ):
            result = asr.transcribe(Path("sample.wav"), asr_provider="faster_whisper_chunked")

        self.assertEqual(result["text"], "chunked")
        mock_chunked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
