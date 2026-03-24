from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmarking.synthetic_benchmark_generation import (
    estimate_render_duration,
    render_seed_manifest,
    resolve_render_config,
)
from benchmarking.synthetic_seed_manifests import load_seed_manifest


ENGLISH_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "seeds" / "english_monologue_seeds_v1.json"
ITALIAN_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "seeds" / "italian_monologue_seeds_v1.json"
BENCHMARKS_DIR = Path(__file__).parent / "fixtures" / "benchmarks"


class SyntheticBenchmarkGenerationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_seed_manifest(ENGLISH_FIXTURE_PATH)
        self.italian_manifest = load_seed_manifest(ITALIAN_FIXTURE_PATH)

    def test_resolve_render_config_merges_seed_overrides(self):
        seed = self.manifest.seeds[0]
        config = resolve_render_config(self.manifest, seed)
        self.assertEqual(config.provider, "macos_say")
        self.assertEqual(config.voice, "Samantha")
        self.assertEqual(config.rate_wpm, 150)
        self.assertEqual(config.sample_rate_hz, 16000)
        self.assertEqual(config.channels, 1)

    def test_estimate_render_duration_counts_words_and_pauses(self):
        estimate = estimate_render_duration("One short sentence. [[slnc 500]] Another short sentence.", 120)
        self.assertEqual(estimate.pause_count, 1)
        self.assertEqual(estimate.pause_total_sec, 0.5)
        self.assertEqual(estimate.speech_word_count, 6)
        self.assertAlmostEqual(estimate.estimated_speech_duration_sec, 3.0, places=2)
        self.assertAlmostEqual(estimate.estimated_total_duration_sec, 3.5, places=2)

    def test_render_seed_manifest_writes_audio_transcripts_and_manifest(self):
        calls: list[tuple[list[str], str | None]] = []

        def fake_run(command: list[str], *, input_text: str | None = None) -> None:
            calls.append((command, input_text))
            if command[0] == "say":
                Path(command[5]).write_bytes(b"AIFF")
            elif command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(b"WAV")

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "benchmarking.synthetic_benchmark_generation._run_subprocess",
            side_effect=fake_run,
        ):
            result = render_seed_manifest(
                self.manifest,
                tmp_dir,
                selected_seed_ids=["en_b1_favorite_place"],
            )

            self.assertEqual(len(result["items"]), 1)
            item = result["items"][0]
            bundle_dir = Path(tmp_dir) / self.manifest.manifest_id
            self.assertTrue((bundle_dir / item["audio_path"]).exists())
            self.assertTrue((bundle_dir / item["transcript_path"]).exists())
            self.assertIn("[[slnc", item["render_text_used"])
            self.assertTrue(item["source_seed_fingerprint"])
            self.assertEqual(item["rate_wpm"], 150)
            self.assertEqual(item["target_duration_sec"], 120.0)
            self.assertGreater(item["estimated_render_duration_sec"], 100.0)
            self.assertGreater(item["duration_alignment_ratio"], 0.85)
            self.assertTrue(result["seed_manifest_fingerprint"])
            self.assertIn("platform", result)
            self.assertIn("macos_version", result)

            transcript_text = (bundle_dir / item["transcript_path"]).read_text(encoding="utf-8")
            self.assertIn("My favorite place is a small lake", transcript_text)

            render_manifest_path = bundle_dir / "render_manifest.json"
            self.assertTrue(render_manifest_path.exists())
            self.assertEqual(calls[0][0][0], "say")
            self.assertEqual(calls[1][0][0], "ffmpeg")
            self.assertEqual(calls[0][0][4], "150")
            self.assertIn("[[slnc", calls[0][1] or "")
            self.assertEqual(calls[1][0][1:4], ["-hide_banner", "-loglevel", "error"])

    def test_render_seed_manifest_rejects_unknown_seed_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                render_seed_manifest(self.manifest, tmp_dir, selected_seed_ids=["missing"])

    def test_render_seed_manifest_rejects_unsupported_provider(self):
        bad_manifest = replace(
            self.manifest,
            render_defaults=replace(self.manifest.render_defaults, provider="other_tts"),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                render_seed_manifest(bad_manifest, tmp_dir)

    def test_render_seed_manifest_can_validate_against_benchmark_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                render_seed_manifest(
                    self.manifest,
                    tmp_dir,
                    selected_seed_ids=["en_b1_favorite_place"],
                    benchmark_root=BENCHMARKS_DIR,
                )

    def test_render_italian_seed_manifest_can_validate_against_benchmark_metrics(self):
        def fake_run(command: list[str], *, input_text: str | None = None) -> None:
            if command[0] == "say":
                Path(command[5]).write_bytes(b"AIFF")
            elif command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(b"WAV")

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "benchmarking.synthetic_benchmark_generation._run_subprocess",
            side_effect=fake_run,
        ):
            result = render_seed_manifest(
                self.italian_manifest,
                tmp_dir,
                benchmark_root=BENCHMARKS_DIR,
            )
            self.assertEqual(len(result["items"]), 4)
            self.assertEqual(result["manifest_id"], "italian_monologue_seeds_v1")

    def test_render_seed_manifest_refuses_to_overwrite_existing_audio(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / self.manifest.manifest_id / "audio"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            existing_audio = bundle_dir / "en_b1_favorite_place.wav"
            existing_audio.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                render_seed_manifest(
                    self.manifest,
                    tmp_dir,
                    selected_seed_ids=["en_b1_favorite_place"],
                )

    def test_render_seed_manifest_can_include_inactive_seed_when_requested(self):
        inactive_seed = replace(self.manifest.seeds[0], active=False)
        manifest = replace(self.manifest, seeds=(inactive_seed, *self.manifest.seeds[1:]))

        def fake_run(command: list[str], *, input_text: str | None = None) -> None:
            if command[0] == "say":
                Path(command[5]).write_bytes(b"AIFF")
            elif command[0] == "ffmpeg":
                Path(command[-1]).write_bytes(b"WAV")

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "benchmarking.synthetic_benchmark_generation._run_subprocess",
            side_effect=fake_run,
        ):
            result = render_seed_manifest(
                manifest,
                tmp_dir,
                include_inactive=True,
                selected_seed_ids=["en_b1_favorite_place"],
            )
            self.assertEqual(result["items"][0]["seed_id"], "en_b1_favorite_place")


if __name__ == "__main__":
    unittest.main()
