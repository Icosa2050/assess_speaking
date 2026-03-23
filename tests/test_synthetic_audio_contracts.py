from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmarking.synthetic_audio_contracts import build_rendered_audio_contract_suite, load_render_manifest
from benchmarking.synthetic_benchmark_generation import render_seed_manifest
from benchmarking.synthetic_seed_manifests import (
    load_seed_manifest,
    seed_manifest_fingerprint,
    synthetic_seed_fingerprint,
)


SEED_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "seeds" / "english_monologue_seeds_v1.json"


class SyntheticAudioContractsTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_seed_manifest(SEED_FIXTURE_PATH)

    def _fake_run(self, command: list[str], *, input_text: str | None = None) -> None:
        if command[0] == "say":
            Path(command[7]).write_bytes(b"AIFF")
        elif command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"WAV")

    def test_load_render_manifest_rejects_non_object_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.json"
            path.write_text('["bad"]', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_render_manifest(path)

    def test_build_rendered_audio_contract_suite_resolves_paths_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "benchmarking.synthetic_benchmark_generation._run_subprocess",
            side_effect=self._fake_run,
        ):
            render_seed_manifest(
                self.manifest,
                tmp_dir,
                selected_seed_ids=["en_b1_favorite_place", "en_b2_remote_work"],
            )
            render_manifest_path = (
                Path(tmp_dir) / self.manifest.manifest_id / "render_manifest.json"
            )
            suite = build_rendered_audio_contract_suite(self.manifest, render_manifest_path)
            self.assertEqual(suite.language_code, "en")
            self.assertEqual(suite.task_family, "opinion_monologue")
            self.assertEqual(len(suite.cases), 2)

            first = suite.cases[0]
            self.assertTrue(first.audio_path.exists())
            self.assertTrue(first.transcript_path.exists())
            self.assertEqual(first.case_id, "en_b1_favorite_place_rendered")
            self.assertEqual(first.expected_language, "en")
            self.assertEqual(first.target_cefr, "B1")
            self.assertEqual(first.target_duration_sec, 120.0)
            self.assertEqual(first.benchmark_suite_id, "english_monologue_cefr_v1")
            self.assertEqual(first.benchmark_case_id, "en_b1_simple_narrative")
            self.assertGreater(first.estimated_render_duration_sec, 100.0)
            self.assertGreater(first.duration_alignment_ratio, 0.85)
            self.assertIn("rendered-audio", first.tags)

    def test_build_rendered_audio_contract_suite_rejects_seed_content_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "benchmarking.synthetic_benchmark_generation._run_subprocess",
            side_effect=self._fake_run,
        ):
            render_seed_manifest(
                self.manifest,
                tmp_dir,
                selected_seed_ids=["en_b1_favorite_place"],
            )
            render_manifest_path = (
                Path(tmp_dir) / self.manifest.manifest_id / "render_manifest.json"
            )
            drifted_seed = replace(
                self.manifest.seeds[0],
                transcript=self.manifest.seeds[0].transcript + " Extra sentence.",
            )
            drifted_manifest = replace(self.manifest, seeds=(drifted_seed, *self.manifest.seeds[1:]))
            with self.assertRaises(ValueError):
                build_rendered_audio_contract_suite(drifted_manifest, render_manifest_path)

    def test_build_rendered_audio_contract_suite_rejects_mismatched_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "render_manifest.json"
            bad_path.write_text(
                """
                {
                  "manifest_id": "other_manifest",
                  "seed_manifest_version": "seed_manifest_en_v1",
                  "seed_manifest_fingerprint": "bad",
                  "renderer_version": "macos_say_ffmpeg_v1",
                  "generated_at_utc": "2026-03-14T12:00:00+00:00",
                  "items": []
                }
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_rendered_audio_contract_suite(self.manifest, bad_path)

    def test_build_rendered_audio_contract_suite_rejects_missing_rendered_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / self.manifest.manifest_id
            bundle_dir.mkdir(parents=True, exist_ok=True)
            bad_path = bundle_dir / "render_manifest.json"
            fingerprint = seed_manifest_fingerprint(self.manifest)
            seed_fingerprint = synthetic_seed_fingerprint(self.manifest.seeds[0])
            bad_path.write_text(
                json.dumps(
                    {
                        "manifest_id": "english_monologue_seeds_v1",
                        "seed_manifest_version": "seed_manifest_en_v1",
                        "seed_manifest_fingerprint": fingerprint,
                        "renderer_version": "macos_say_ffmpeg_v1",
                        "generated_at_utc": "2026-03-14T12:00:00+00:00",
                        "items": [
                            {
                                "seed_id": "en_b1_favorite_place",
                                "target_cefr": "B1",
                                "target_duration_sec": 120,
                                "topic_tag": "travel",
                                "audio_path": "audio/missing.wav",
                                "transcript_path": "transcripts/missing.txt",
                                "source_seed_fingerprint": seed_fingerprint,
                                "provider": "macos_say",
                                "voice": "Samantha",
                                "rate_wpm": 165,
                                "output_format": "wav",
                                "sample_rate_hz": 16000,
                                "channels": 1,
                                "render_text_used": "hello",
                                "seed_tags": ["baseline"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                build_rendered_audio_contract_suite(self.manifest, bad_path)


if __name__ == "__main__":
    unittest.main()
