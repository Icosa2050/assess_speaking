from pathlib import Path
from dataclasses import fields, replace
import json
import tempfile
import unittest
from unittest import mock

from benchmarking.synthetic_audio_contracts import build_rendered_audio_contract_suite
from benchmarking.synthetic_benchmark_evaluation import (
    EvaluationRunConfig,
    EvaluatedRenderedAudioSuite,
    _case_to_dict,
    append_evaluation_checkpoint,
    checkpoint_lock,
    compare_cefr_levels,
    evaluate_rendered_audio_contract_suite,
    load_evaluation_manifest,
    load_evaluation_checkpoint_cases,
    write_evaluation_manifest,
)
from benchmarking.synthetic_benchmark_generation import render_seed_manifest
from benchmarking.synthetic_seed_manifests import load_seed_manifest


SEED_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "seeds" / "english_monologue_seeds_v1.json"


class SyntheticBenchmarkEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.seed_manifest = load_seed_manifest(SEED_FIXTURE_PATH)
        self.config = EvaluationRunConfig(
            whisper_model="small",
            provider="openrouter",
            llm_model="google/gemini-3.1-pro-preview",
            feedback_language="en",
            target_duration_sec=120.0,
            speaker_id="synthetic-benchmark",
            dry_run=True,
            include_raw_llm=True,
            include_full_report=True,
        )

    def _fake_render_run(self, command: list[str], *, input_text: str | None = None) -> None:
        if command[0] == "say":
            Path(command[7]).write_bytes(b"AIFF")
        elif command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"WAV")

    def _build_contract_suite(self, tmp_dir: str):
        with mock.patch(
            "benchmarking.synthetic_benchmark_generation._run_subprocess",
            side_effect=self._fake_render_run,
        ):
            render_seed_manifest(
                self.seed_manifest,
                tmp_dir,
                selected_seed_ids=["en_b2_remote_work"],
            )
        render_manifest_path = Path(tmp_dir) / self.seed_manifest.manifest_id / "render_manifest.json"
        return build_rendered_audio_contract_suite(self.seed_manifest, render_manifest_path)

    def _fake_runner(self, **kwargs):
        expected_language = kwargs["expected_language"]
        theme = kwargs["theme"]
        return {
            "llm_rubric": '{"fluency": 4}',
            "report": {
                "input": {
                    "provider": kwargs["provider"],
                    "llm_model": kwargs["llm_model"],
                    "whisper_model": kwargs["whisper_model"],
                    "expected_language": expected_language,
                    "feedback_language": kwargs["feedback_language"],
                    "theme": theme,
                    "task_family": kwargs["task_family"],
                    "prompt_version": "rubric_multilingual_v1",
                    "rubric_prompt_version": "rubric_multilingual_v1",
                    "coaching_prompt_version": "coaching_multilingual_v1",
                    "scoring_model_version": "legacy_hybrid_v1",
                    "language_profile": expected_language,
                    "language_profile_key": expected_language,
                    "language_profile_version": f"language_profile_{expected_language}_v1",
                },
                "scores": {
                    "final": 4.2,
                    "llm": 4.4,
                    "deterministic": 3.9,
                    "band": 5,
                    "mode": "hybrid",
                    "dimensions": {
                        "fluency": 4.0,
                        "grammar": 4.1,
                    },
                    "cefr_estimate": {
                        "level": "B2",
                        "continuous": 4.05,
                    },
                },
                "checks": {
                    "duration_pass": True,
                    "topic_pass": True,
                    "language_pass": True,
                },
                "warnings": [],
                "errors": [],
                "timings_ms": {
                    "asr": 10.0,
                    "llm": 20.0,
                },
            },
        }

    def test_compare_cefr_levels_supports_ordered_comparison(self):
        self.assertEqual(compare_cefr_levels("B2", "C1"), 1)
        self.assertEqual(compare_cefr_levels("C1", "B2"), -1)
        self.assertIsNone(compare_cefr_levels("B2", "unknown"))

    def test_evaluate_rendered_audio_contract_suite_emits_case_results_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=self._fake_runner,
            )
            self.assertEqual(len(evaluated.cases), 1)
            case = evaluated.cases[0]
            self.assertEqual(case.status, "ok")
            self.assertEqual(case.estimated_cefr, "B2")
            self.assertTrue(case.cefr_match)
            self.assertEqual(case.continuous_score, 4.05)
            self.assertEqual(case.feedback_language, "en")
            self.assertEqual(evaluated.run_status, "ok")
            self.assertEqual(evaluated.success_ratio, 1.0)
            self.assertEqual(case.llm_contract.response_parser, "extract_json_object")
            self.assertEqual(case.llm_contract.rubric_schema, "RubricResult")
            self.assertEqual(case.llm_contract.language_profile_key, "en")
            self.assertEqual(case.raw_llm_rubric, '{"fluency": 4}')
            self.assertIn("openrouter", evaluated.suite_id)
            output_path = Path(tmp_dir) / "evaluation_manifest.json"
            written = write_evaluation_manifest(evaluated, output_path)
            payload = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["run_status"], "ok")
            self.assertEqual(payload["success_ratio"], 1.0)
            self.assertEqual(payload["summary"]["total_cases"], 1)
            self.assertEqual(payload["summary"]["ok_cases"], 1)
            self.assertEqual(payload["summary"]["run_status"], "ok")
            self.assertEqual(payload["cases"][0]["estimated_cefr"], "B2")
            self.assertEqual(payload["cases"][0]["feedback_language"], "en")
            self.assertEqual(payload["cases"][0]["llm_contract"]["response_parser"], "extract_json_object")
            self.assertEqual(payload["cases"][0]["llm_contract"]["language_profile_key"], "en")

    def test_evaluate_rendered_audio_contract_suite_supports_legacy_cefr_estimate_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)

            def legacy_runner(**kwargs):
                payload = self._fake_runner(**kwargs)
                payload["report"]["scores"]["cefr_estimate"] = {
                    "cefr": "B2",
                    "continuous_score": 4.05,
                }
                return payload

            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=legacy_runner,
            )
            case = evaluated.cases[0]
            self.assertEqual(case.estimated_cefr, "B2")
            self.assertEqual(case.continuous_score, 4.05)

    def test_evaluate_rendered_audio_contract_suite_prefers_case_target_duration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            custom_case = replace(contract_suite.cases[0], target_duration_sec=75.0)
            contract_suite = replace(contract_suite, cases=(custom_case,))
            seen: dict[str, float] = {}

            def capturing_runner(**kwargs):
                seen["target_duration_sec"] = kwargs["target_duration_sec"]
                return self._fake_runner(**kwargs)

            evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=capturing_runner,
            )
            self.assertEqual(seen["target_duration_sec"], 75.0)

    def test_evaluate_rendered_audio_contract_suite_captures_runner_errors_without_aborting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)

            def failing_runner(**kwargs):
                raise RuntimeError("synthetic provider failure")

            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=failing_runner,
            )
            self.assertEqual(len(evaluated.cases), 1)
            self.assertEqual(evaluated.run_status, "failed")
            self.assertEqual(evaluated.success_ratio, 0.0)
            case = evaluated.cases[0]
            self.assertEqual(case.status, "runner_error")
            self.assertEqual(case.error_type, "RuntimeError")
            self.assertIn("synthetic provider failure", case.errors[0])
            self.assertIn("RuntimeError: synthetic provider failure", case.execution_traceback)
            self.assertIsNone(case.report)
            self.assertIsNone(case.raw_llm_rubric)

    def test_evaluate_rendered_audio_contract_suite_marks_partial_failures_as_degraded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "benchmarking.synthetic_benchmark_generation._run_subprocess",
                side_effect=self._fake_render_run,
            ):
                render_seed_manifest(
                    self.seed_manifest,
                    tmp_dir,
                    selected_seed_ids=["en_b1_favorite_place", "en_b2_remote_work"],
                )
            render_manifest_path = Path(tmp_dir) / self.seed_manifest.manifest_id / "render_manifest.json"
            contract_suite = build_rendered_audio_contract_suite(self.seed_manifest, render_manifest_path)

            def mixed_runner(**kwargs):
                if kwargs["audio"].name.endswith("en_b1_favorite_place.wav"):
                    raise RuntimeError("first case failed")
                return self._fake_runner(**kwargs)

            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=mixed_runner,
            )
            self.assertEqual(evaluated.run_status, "degraded")
            self.assertEqual(evaluated.success_ratio, 0.5)

    def test_evaluate_rendered_audio_contract_suite_can_omit_raw_and_full_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            config = EvaluationRunConfig(
                whisper_model="small",
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                feedback_language="en",
                target_duration_sec=120.0,
                speaker_id="synthetic-benchmark",
                dry_run=True,
                include_raw_llm=False,
                include_full_report=False,
            )
            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=config,
                runner=self._fake_runner,
            )
            case = evaluated.cases[0]
            self.assertIsNone(case.raw_llm_rubric)
            self.assertIsNone(case.report)

    def test_case_serializer_covers_all_top_level_dataclass_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=self._fake_runner,
            )
            case = evaluated.cases[0]
            payload = _case_to_dict(case)
            self.assertEqual(set(payload.keys()), {field.name for field in fields(type(case))})

    def test_load_evaluation_manifest_backfills_cefr_from_embedded_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = EvaluatedRenderedAudioSuite(
                suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
                manifest_id="english_monologue_seeds_v1",
                language_code="en",
                task_family="opinion_monologue",
                generated_at_utc="2026-03-14T16:00:00+00:00",
                run_status="ok",
                success_ratio=1.0,
                config=self.config,
                cases=(),
            )
            manifest_path = Path(tmp_dir) / "evaluation_manifest.json"
            write_evaluation_manifest(suite, manifest_path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["cases"] = [
                {
                    "case_id": "en_b2_remote_work_rendered",
                    "source_seed_id": "en_b2_remote_work",
                    "status": "ok",
                    "audio_path": "/tmp/en_b2_remote_work.wav",
                    "expected_language": "en",
                    "feedback_language": "en",
                    "target_cefr": "B2",
                    "benchmark_suite_id": "english_monologue_cefr_v1",
                    "benchmark_case_id": "en_b2_supported_argument",
                    "estimated_cefr": None,
                    "cefr_delta": None,
                    "cefr_match": None,
                    "final_score": 4.2,
                    "llm_score": 4.4,
                    "deterministic_score": 3.9,
                    "continuous_score": None,
                    "band": 5,
                    "mode": "hybrid",
                    "warnings": [],
                    "errors": [],
                    "error_type": None,
                    "execution_error": None,
                    "execution_traceback": None,
                    "checks": {"topic_pass": True},
                    "dimensions": {},
                    "timings_ms": {},
                    "llm_contract": {
                        "provider": "openrouter",
                        "llm_model": "google/gemini-3.1-pro-preview",
                        "whisper_model": "small",
                        "response_parser": "extract_json_object",
                        "rubric_schema": "RubricResult",
                    },
                    "raw_llm_rubric": None,
                    "report": {
                        "scores": {
                            "cefr_estimate": {
                                "level": "B2",
                                "continuous": 4.05,
                            }
                        }
                    },
                }
            ]
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            loaded = load_evaluation_manifest(manifest_path)
            case = loaded.cases[0]
            self.assertEqual(case.estimated_cefr, "B2")
            self.assertEqual(case.continuous_score, 4.05)

    def test_checkpoint_round_trip_restores_serialized_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=self._fake_runner,
            )
            case = evaluated.cases[0]
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"
            append_evaluation_checkpoint(
                checkpoint_path,
                manifest_id=contract_suite.manifest_id,
                suite_id=evaluated.suite_id,
                case=case,
            )
            restored = load_evaluation_checkpoint_cases(
                checkpoint_path,
                manifest_id=contract_suite.manifest_id,
                suite_id=evaluated.suite_id,
            )
            restored_case = restored[case.case_id]
            self.assertEqual(restored_case.case_id, case.case_id)
            self.assertEqual(restored_case.feedback_language, "en")
            self.assertEqual(restored_case.llm_contract.llm_model, "google/gemini-3.1-pro-preview")

    def test_resume_from_checkpoint_skips_already_evaluated_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "benchmarking.synthetic_benchmark_generation._run_subprocess",
                side_effect=self._fake_render_run,
            ):
                render_seed_manifest(
                    self.seed_manifest,
                    tmp_dir,
                    selected_seed_ids=["en_b1_favorite_place", "en_b2_remote_work"],
                )
            render_manifest_path = Path(tmp_dir) / self.seed_manifest.manifest_id / "render_manifest.json"
            contract_suite = build_rendered_audio_contract_suite(self.seed_manifest, render_manifest_path)
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"

            first_only = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                checkpoint_path=checkpoint_path,
                runner=lambda **kwargs: self._fake_runner(**kwargs)
                if kwargs["audio"].name.endswith("en_b1_favorite_place.wav")
                else (_ for _ in ()).throw(RuntimeError("second case should not run yet")),
            )
            self.assertEqual(first_only.run_status, "degraded")

            call_counter = {"count": 0}

            def resume_runner(**kwargs):
                call_counter["count"] += 1
                if kwargs["audio"].name.endswith("en_b1_favorite_place.wav"):
                    raise RuntimeError("restored checkpoint case should not rerun")
                return self._fake_runner(**kwargs)

            resumed = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                checkpoint_path=checkpoint_path,
                resume_from_checkpoint=True,
                runner=resume_runner,
            )
            self.assertEqual(call_counter["count"], 1)
            self.assertEqual(resumed.run_status, "ok")
            self.assertEqual(len(resumed.cases), 2)
            self.assertTrue(all(case.status == "ok" for case in resumed.cases))

    def test_resume_from_checkpoint_rejects_mismatched_suite_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "manifest_id": "other_manifest",
                        "suite_id": "other_suite",
                        "case": {
                            "case_id": "x",
                            "source_seed_id": "x",
                            "status": "ok",
                            "audio_path": "/tmp/x.wav",
                            "expected_language": "en",
                            "feedback_language": "en",
                            "target_cefr": "B2",
                            "estimated_cefr": "B2",
                            "cefr_delta": 0,
                            "cefr_match": True,
                            "final_score": 4.0,
                            "llm_score": 4.0,
                            "deterministic_score": 4.0,
                            "continuous_score": 4.0,
                            "band": 5,
                            "mode": "hybrid",
                            "warnings": [],
                            "errors": [],
                            "error_type": None,
                            "execution_error": None,
                            "execution_traceback": None,
                            "checks": {},
                            "dimensions": {},
                            "timings_ms": {},
                            "llm_contract": {
                                "provider": "openrouter",
                                "llm_model": "google/gemini-3.1-pro-preview",
                                "whisper_model": "small",
                                "response_parser": "extract_json_object",
                                "rubric_schema": "RubricResult",
                                "prompt_version": None,
                                "rubric_prompt_version": None,
                                "coaching_prompt_version": None,
                                "scoring_model_version": None,
                                "language_profile": None,
                                "language_profile_key": None,
                                "language_profile_version": None
                            },
                            "raw_llm_rubric": None,
                            "report": None
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_evaluation_checkpoint_cases(
                    checkpoint_path,
                    manifest_id="english_monologue_seeds_v1",
                    suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
                )

    def test_checkpoint_loader_ignores_truncated_last_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=self._fake_runner,
            )
            case = evaluated.cases[0]
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"
            append_evaluation_checkpoint(
                checkpoint_path,
                manifest_id=contract_suite.manifest_id,
                suite_id=evaluated.suite_id,
                case=case,
            )
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write('{"schema_version": 1, "manifest_id": ')
            restored = load_evaluation_checkpoint_cases(
                checkpoint_path,
                manifest_id=contract_suite.manifest_id,
                suite_id=evaluated.suite_id,
            )
            self.assertEqual(set(restored), {case.case_id})

    def test_checkpoint_loader_rejects_schema_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 999,
                        "manifest_id": "english_monologue_seeds_v1",
                        "suite_id": "english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
                        "case": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_evaluation_checkpoint_cases(
                    checkpoint_path,
                    manifest_id="english_monologue_seeds_v1",
                    suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
                )

    def test_checkpoint_loader_uses_last_write_wins_for_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=self.config,
                runner=self._fake_runner,
            )
            case = evaluated.cases[0]
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"
            append_evaluation_checkpoint(
                checkpoint_path,
                manifest_id=contract_suite.manifest_id,
                suite_id=evaluated.suite_id,
                case=case,
            )
            mutated_payload = _case_to_dict(case)
            mutated_payload["status"] = "runner_error"
            mutated_payload["errors"] = ["rerun failed"]
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "manifest_id": contract_suite.manifest_id,
                            "suite_id": evaluated.suite_id,
                            "case": mutated_payload,
                        }
                    )
                    + "\n"
                )
            restored = load_evaluation_checkpoint_cases(
                checkpoint_path,
                manifest_id=contract_suite.manifest_id,
                suite_id=evaluated.suite_id,
            )
            self.assertEqual(restored[case.case_id].status, "runner_error")

    def test_checkpoint_lock_raises_when_file_is_already_locked(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "evaluation_checkpoint.jsonl"
            with mock.patch(
                "benchmarking.synthetic_benchmark_evaluation.fcntl.flock",
                side_effect=[None, BlockingIOError(), None],
            ):
                with checkpoint_lock(checkpoint_path):
                    with self.assertRaises(RuntimeError):
                        with checkpoint_lock(checkpoint_path):
                            pass

    def test_evaluate_rendered_audio_contract_suite_forwards_llm_timeout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            seen = {}

            def runner_with_timeout(**kwargs):
                seen["llm_timeout_sec"] = kwargs.get("llm_timeout_sec")
                return self._fake_runner(**kwargs)

            config = EvaluationRunConfig(
                whisper_model="small",
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                feedback_language="en",
                target_duration_sec=120.0,
                speaker_id="synthetic-benchmark",
                dry_run=True,
                include_raw_llm=True,
                include_full_report=True,
                llm_timeout_sec=12.5,
            )
            evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=config,
                runner=runner_with_timeout,
            )
            self.assertEqual(seen["llm_timeout_sec"], 12.5)

    def test_evaluate_rendered_audio_contract_suite_forwards_language_profile_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract_suite = self._build_contract_suite(tmp_dir)
            seen = {}

            def runner_with_profile_key(**kwargs):
                seen["language_profile_key"] = kwargs.get("language_profile_key")
                return self._fake_runner(**kwargs)

            config = EvaluationRunConfig(
                whisper_model="small",
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                feedback_language="en",
                target_duration_sec=120.0,
                speaker_id="synthetic-benchmark",
                dry_run=True,
                include_raw_llm=True,
                include_full_report=True,
                language_profile_key="en",
            )
            evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=config,
                runner=runner_with_profile_key,
            )
            self.assertEqual(seen["language_profile_key"], "en")

    def test_evaluate_rendered_audio_contract_suite_aborts_after_consecutive_runner_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "benchmarking.synthetic_benchmark_generation._run_subprocess",
                side_effect=self._fake_render_run,
            ):
                render_seed_manifest(
                    self.seed_manifest,
                    tmp_dir,
                    selected_seed_ids=[
                        "en_b1_favorite_place",
                        "en_b2_remote_work",
                        "en_c1_city_change",
                    ],
                )
            render_manifest_path = Path(tmp_dir) / self.seed_manifest.manifest_id / "render_manifest.json"
            contract_suite = build_rendered_audio_contract_suite(self.seed_manifest, render_manifest_path)
            config = EvaluationRunConfig(
                whisper_model="small",
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                feedback_language="en",
                target_duration_sec=120.0,
                speaker_id="synthetic-benchmark",
                dry_run=True,
                include_raw_llm=True,
                include_full_report=True,
                max_consecutive_runner_errors=1,
            )

            def always_fail(**kwargs):
                raise RuntimeError("provider outage")

            evaluated = evaluate_rendered_audio_contract_suite(
                contract_suite,
                config=config,
                runner=always_fail,
            )
            self.assertEqual(evaluated.run_status, "aborted")
            self.assertEqual([case.status for case in evaluated.cases], ["runner_error", "skipped", "skipped"])
            self.assertEqual(evaluated.cases[1].error_type, "CircuitBreaker")


if __name__ == "__main__":
    unittest.main()
