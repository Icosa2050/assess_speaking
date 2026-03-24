from pathlib import Path
import tempfile
import unittest

from benchmarking.calibration_evaluation import (
    CalibrationRunConfig,
    evaluate_calibration_manifest,
    load_calibration_evaluation_manifest,
    write_calibration_evaluation_manifest,
)
from benchmarking.calibration_manifests import load_calibration_manifest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "calibration" / "italian_real_audio_shadow_v1.json"


class CalibrationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_calibration_manifest(FIXTURE_PATH)

    def test_evaluate_calibration_manifest_uses_live_shadow_profile_and_passes_pair_expectation(self):
        seen_calls: list[dict] = []

        def fake_runner(audio, **kwargs):
            seen_calls.append(
                {
                    "audio": Path(audio).name,
                    "expected_language": kwargs.get("expected_language"),
                    "language_profile_key": kwargs.get("language_profile_key"),
                    "theme": kwargs.get("theme"),
                    "speaker_id": kwargs.get("speaker_id"),
                    "task_family": kwargs.get("task_family"),
                }
            )
            score_by_name = {
                "test1.m4a": {"final": 3.1, "deterministic": 3.0, "continuous": 3.05, "level": "B1"},
                "test2.m4a": {"final": 4.2, "deterministic": 4.0, "continuous": 4.15, "level": "B2"},
            }[Path(audio).name]
            return {
                "report": {
                    "input": {
                        "provider": kwargs.get("provider"),
                        "llm_model": kwargs.get("llm_model"),
                        "whisper_model": kwargs.get("whisper_model"),
                        "language_profile_key": kwargs.get("language_profile_key"),
                        "language_profile_version": "language_profile_it_v1_live_shadow",
                        "language_profile": "Italian",
                    },
                    "scores": {
                        "final": score_by_name["final"],
                        "llm": score_by_name["final"] - 0.1,
                        "deterministic": score_by_name["deterministic"],
                        "band": 4,
                        "mode": "hybrid",
                        "dimensions": {"fluency": {"score": score_by_name["deterministic"]}},
                        "cefr_estimate": {
                            "level": score_by_name["level"],
                            "continuous": score_by_name["continuous"],
                        },
                    },
                    "checks": {
                        "language_pass": True,
                        "topic_pass": True,
                        "duration_pass": True,
                    },
                    "timings_ms": {"total": 250},
                    "warnings": [],
                    "errors": [],
                },
                "llm_rubric": {"overall": 4},
            }

        evaluation = evaluate_calibration_manifest(
            self.manifest,
            config=CalibrationRunConfig(
                whisper_model="tiny",
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                feedback_language=None,
                dry_run=False,
                include_raw_llm=False,
                include_full_report=False,
            ),
            runner=fake_runner,
        )

        self.assertEqual(len(evaluation.cases), 2)
        self.assertEqual(len(evaluation.pair_expectations), 1)
        self.assertTrue(evaluation.pair_expectations[0].passed)
        self.assertEqual(evaluation.pair_expectations[0].metric, "final_score")
        self.assertEqual(evaluation.language_profile_key, "it_live_shadow")
        self.assertEqual({call["expected_language"] for call in seen_calls}, {"it"})
        self.assertEqual({call["language_profile_key"] for call in seen_calls}, {"it_live_shadow"})
        self.assertEqual({call["theme"] for call in seen_calls}, {"tema libero"})
        self.assertEqual({call["task_family"] for call in seen_calls}, {"real_audio_grading_compare"})

    def test_write_and_load_calibration_evaluation_manifest_round_trip(self):
        def fake_runner(audio, **kwargs):
            case_name = Path(audio).name
            final_score = 3.0 if case_name == "test1.m4a" else 4.0
            return {
                "report": {
                    "input": {
                        "provider": kwargs.get("provider"),
                        "llm_model": kwargs.get("llm_model"),
                        "whisper_model": kwargs.get("whisper_model"),
                        "language_profile_key": kwargs.get("language_profile_key"),
                        "language_profile_version": "language_profile_it_v1_live_shadow",
                    },
                    "scores": {
                        "final": final_score,
                        "llm": final_score - 0.2,
                        "deterministic": final_score - 0.1,
                        "band": 4,
                        "mode": "hybrid",
                        "dimensions": {},
                        "cefr_estimate": {
                            "level": "B1" if case_name == "test1.m4a" else "B2",
                            "continuous": final_score,
                        },
                    },
                    "checks": {},
                    "timings_ms": {},
                }
            }

        evaluation = evaluate_calibration_manifest(
            self.manifest,
            config=CalibrationRunConfig(
                whisper_model="tiny",
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                feedback_language=None,
                dry_run=False,
                include_raw_llm=False,
                include_full_report=False,
            ),
            runner=fake_runner,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "calibration_evaluation.json"
            write_calibration_evaluation_manifest(evaluation, output_path)
            loaded = load_calibration_evaluation_manifest(output_path)

        self.assertEqual(loaded.manifest_id, evaluation.manifest_id)
        self.assertEqual(loaded.language_profile_key, "it_live_shadow")
        self.assertEqual(len(loaded.cases), 2)
        self.assertEqual(len(loaded.pair_expectations), 1)
        self.assertTrue(loaded.pair_expectations[0].passed)


if __name__ == "__main__":
    unittest.main()
