import json
from pathlib import Path
import tempfile
import unittest

from benchmarking.calibration_manifests import discover_calibration_manifests, load_calibration_manifest


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "calibration"


class CalibrationManifestTests(unittest.TestCase):
    def test_load_calibration_manifest_parses_italian_shadow_fixture(self):
        manifest = load_calibration_manifest(FIXTURES_DIR / "italian_real_audio_shadow_v1.json")
        self.assertEqual(manifest.manifest_id, "italian_real_audio_shadow_v1")
        self.assertEqual(manifest.language_code, "it")
        self.assertEqual(manifest.language_profile_key, "it_live_shadow")
        self.assertEqual(manifest.task_family, "real_audio_grading_compare")
        self.assertEqual(len(manifest.cases), 2)
        self.assertIsNone(manifest.cases[0].expected_cefr)
        self.assertTrue(manifest.cases[0].audio_path.exists())
        self.assertEqual(len(manifest.pair_expectations), 1)
        self.assertEqual(manifest.pair_expectations[0].higher_case_id, "it_real_audio_candidate_2")

    def test_discovery_filters_by_language_and_tags(self):
        manifests = discover_calibration_manifests(FIXTURES_DIR, language_codes={"it"})
        shadow = discover_calibration_manifests(FIXTURES_DIR, tags={"shadow"})
        strict = discover_calibration_manifests(
            FIXTURES_DIR,
            tags={"italian", "ordering"},
            tag_match="all",
        )
        self.assertEqual([item.manifest_id for item in manifests], ["italian_real_audio_shadow_v1"])
        self.assertEqual([item.manifest_id for item in shadow], ["italian_real_audio_shadow_v1"])
        self.assertEqual([item.manifest_id for item in strict], ["italian_real_audio_shadow_v1"])

    def test_load_calibration_manifest_rejects_unknown_pair_reference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_calibration_manifest.json"
            audio_path = (Path(__file__).parent / "audio" / "test1.m4a").resolve()
            bad_path.write_text(
                json.dumps(
                    {
                        "manifest_id": "bad",
                        "language_code": "it",
                        "task_family": "real_audio_grading_compare",
                        "version": "v1",
                        "cases": [
                            {
                                "case_id": "only_case",
                                "audio_path": audio_path.as_posix(),
                                "theme": "tema libero",
                                "speaker_id": "speaker-1",
                            }
                        ],
                        "pair_expectations": [
                            {
                                "higher_case_id": "missing",
                                "lower_case_id": "only_case",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_calibration_manifest(bad_path)


if __name__ == "__main__":
    unittest.main()
