from pathlib import Path
import tempfile
import unittest

from benchmarking.synthetic_benchmark_generation import estimate_render_duration, text_to_render
from benchmarking.synthetic_seed_manifests import discover_seed_manifests, load_seed_manifest


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "seeds"


class SyntheticSeedManifestTests(unittest.TestCase):
    def test_load_seed_manifest_parses_english_seed_corpus(self):
        manifest = load_seed_manifest(FIXTURES_DIR / "english_monologue_seeds_v1.json")
        self.assertEqual(manifest.manifest_id, "english_monologue_seeds_v1")
        self.assertEqual(manifest.language_code, "en")
        self.assertEqual(manifest.task_family, "opinion_monologue")
        self.assertEqual(manifest.render_defaults.provider, "macos_say")
        self.assertEqual(manifest.render_defaults.voice, "Samantha")
        self.assertEqual(manifest.render_defaults.rate_wpm, 160)
        self.assertEqual(len(manifest.seeds), 4)
        self.assertEqual(manifest.seeds[0].target_cefr, "B1")
        self.assertEqual(manifest.seeds[0].target_duration_sec, 120.0)
        self.assertEqual(manifest.seeds[0].benchmark_suite_id, "english_monologue_cefr_v1")
        self.assertEqual(manifest.seeds[0].benchmark_case_id, "en_b1_simple_narrative")
        self.assertEqual(manifest.seeds[0].render_overrides.rate_wpm, 150)
        self.assertIn("favorite place", manifest.seeds[0].transcript.lower())
        self.assertEqual(len(manifest.active_seeds), 4)

    def test_load_seed_manifest_parses_italian_seed_corpus(self):
        manifest = load_seed_manifest(FIXTURES_DIR / "italian_monologue_seeds_v1.json")
        self.assertEqual(manifest.manifest_id, "italian_monologue_seeds_v1")
        self.assertEqual(manifest.language_code, "it")
        self.assertEqual(manifest.task_family, "opinion_monologue")
        self.assertEqual(manifest.render_defaults.provider, "macos_say")
        self.assertEqual(manifest.render_defaults.voice, "Alice")
        self.assertEqual(len(manifest.seeds), 4)
        self.assertEqual(manifest.seeds[0].target_cefr, "B1")
        self.assertEqual(manifest.seeds[0].benchmark_suite_id, "italian_monologue_cefr_v1")
        self.assertEqual(manifest.seeds[0].benchmark_case_id, "it_b1_personal_story")
        self.assertEqual(manifest.seeds[0].render_overrides.rate_wpm, 82)
        self.assertIn("piccolo parco", manifest.seeds[0].transcript.lower())
        self.assertEqual(len(manifest.active_seeds), 4)

    def test_english_seed_corpus_is_duration_aligned(self):
        manifest = load_seed_manifest(FIXTURES_DIR / "english_monologue_seeds_v1.json")
        for seed in manifest.active_seeds:
            config_rate = seed.render_overrides.rate_wpm if seed.render_overrides else manifest.render_defaults.rate_wpm
            estimate = estimate_render_duration(text_to_render(seed), int(config_rate))
            self.assertIsNotNone(seed.target_duration_sec, seed.seed_id)
            ratio = estimate.estimated_total_duration_sec / float(seed.target_duration_sec)
            self.assertGreaterEqual(ratio, 0.85, seed.seed_id)
            self.assertLessEqual(ratio, 1.15, seed.seed_id)

    def test_italian_seed_corpus_is_duration_aligned(self):
        manifest = load_seed_manifest(FIXTURES_DIR / "italian_monologue_seeds_v1.json")
        for seed in manifest.active_seeds:
            config_rate = seed.render_overrides.rate_wpm if seed.render_overrides else manifest.render_defaults.rate_wpm
            estimate = estimate_render_duration(text_to_render(seed), int(config_rate))
            self.assertIsNotNone(seed.target_duration_sec, seed.seed_id)
            ratio = estimate.estimated_total_duration_sec / float(seed.target_duration_sec)
            self.assertGreaterEqual(ratio, 0.85, seed.seed_id)
            self.assertLessEqual(ratio, 1.15, seed.seed_id)

    def test_discovery_filters_by_language_and_tags(self):
        manifests = discover_seed_manifests(FIXTURES_DIR, language_codes={"en"})
        italian = discover_seed_manifests(FIXTURES_DIR, language_codes={"it"})
        baseline = discover_seed_manifests(FIXTURES_DIR, tags={"baseline"})
        strict = discover_seed_manifests(
            FIXTURES_DIR,
            tags={"english", "seed-corpus"},
            tag_match="all",
        )
        self.assertIn("english_monologue_seeds_v1", [item.manifest_id for item in manifests])
        self.assertIn("italian_monologue_seeds_v1", [item.manifest_id for item in italian])
        self.assertIn("english_monologue_seeds_v1", [item.manifest_id for item in baseline])
        self.assertIn("italian_monologue_seeds_v1", [item.manifest_id for item in baseline])
        self.assertIn("english_monologue_seeds_v1", [item.manifest_id for item in strict])

    def test_discovery_rejects_invalid_tag_match_mode(self):
        with self.assertRaises(ValueError):
            discover_seed_manifests(FIXTURES_DIR, tags={"english"}, tag_match="invalid")

    def test_load_seed_manifest_rejects_empty_transcript(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_seed_manifest.json"
            bad_path.write_text(
                """
                {
                  "manifest_id": "bad",
                  "language_code": "en",
                  "task_family": "opinion_monologue",
                  "version": "x",
                  "render_defaults": {},
                  "seeds": [
                    {
                      "seed_id": "bad_seed",
                      "language_code": "en",
                      "task_family": "opinion_monologue",
                      "target_cefr": "B1",
                      "topic_tag": "travel",
                      "transcript": "   ",
                      "source_type": "synthetic_seed"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_seed_manifest(bad_path)

    def test_load_seed_manifest_rejects_non_object_root_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_seed_manifest.json"
            bad_path.write_text('["not", "a", "manifest"]', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_seed_manifest(bad_path)

    def test_load_seed_manifest_rejects_duplicate_seed_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_seed_manifest.json"
            bad_path.write_text(
                """
                {
                  "manifest_id": "bad",
                  "language_code": "en",
                  "task_family": "opinion_monologue",
                  "version": "x",
                  "render_defaults": {},
                  "seeds": [
                    {
                      "seed_id": "dup",
                      "language_code": "en",
                      "task_family": "opinion_monologue",
                      "target_cefr": "B1",
                      "topic_tag": "travel",
                      "transcript": "One valid transcript.",
                      "source_type": "synthetic_seed"
                    },
                    {
                      "seed_id": "dup",
                      "language_code": "en",
                      "task_family": "opinion_monologue",
                      "target_cefr": "B1",
                      "topic_tag": "travel",
                      "transcript": "Another valid transcript.",
                      "source_type": "synthetic_seed"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_seed_manifest(bad_path)

    def test_load_seed_manifest_rejects_invalid_cefr_and_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_seed_manifest.json"
            bad_path.write_text(
                """
                {
                  "manifest_id": "bad",
                  "language_code": "en",
                  "task_family": "opinion_monologue",
                  "version": "x",
                  "render_defaults": {},
                  "seeds": [
                    {
                      "seed_id": "wrong_scope",
                      "language_code": "it",
                      "task_family": "opinion_monologue",
                      "target_cefr": "Z9",
                      "topic_tag": "travel",
                      "transcript": "Valid text but invalid scope.",
                      "source_type": "synthetic_seed"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_seed_manifest(bad_path)

    def test_load_seed_manifest_rejects_non_positive_target_duration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_seed_manifest.json"
            bad_path.write_text(
                """
                {
                  "manifest_id": "bad",
                  "language_code": "en",
                  "task_family": "opinion_monologue",
                  "version": "x",
                  "render_defaults": {},
                  "seeds": [
                    {
                      "seed_id": "bad_duration",
                      "language_code": "en",
                      "task_family": "opinion_monologue",
                      "target_cefr": "B1",
                      "target_duration_sec": 0,
                      "topic_tag": "travel",
                      "transcript": "Valid text.",
                      "source_type": "synthetic_seed"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_seed_manifest(bad_path)


if __name__ == "__main__":
    unittest.main()
