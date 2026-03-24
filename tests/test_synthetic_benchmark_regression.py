from pathlib import Path
import json
import tempfile
import unittest

from benchmarking.benchmark_suites import load_benchmark_suite
from benchmarking.synthetic_benchmark_evaluation import (
    EvaluationLLMContract,
    EvaluationRunConfig,
    EvaluatedRenderedAudioSuite,
    EvaluatedRenderedCase,
    load_evaluation_manifest,
    write_evaluation_manifest,
)
from benchmarking.synthetic_benchmark_regression import (
    compare_evaluation_against_benchmark,
    write_regression_report,
)


BENCHMARK_FIXTURE = (
    Path(__file__).parent / "fixtures" / "benchmarks" / "english_monologue_cefr_v1.json"
)


class SyntheticBenchmarkRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark_suite = load_benchmark_suite(BENCHMARK_FIXTURE)

    def _base_config(self) -> EvaluationRunConfig:
        return EvaluationRunConfig(
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

    def _matching_contract(self) -> EvaluationLLMContract:
        return EvaluationLLMContract(
            provider="openrouter",
            llm_model="google/gemini-3.1-pro-preview",
            whisper_model="small",
            response_parser="extract_json_object",
            rubric_schema="RubricResult",
            prompt_version="rubric_multilingual_v1",
            rubric_prompt_version="rubric_multilingual_v1",
            coaching_prompt_version="coaching_multilingual_v1",
            scoring_model_version="legacy_hybrid_v1",
            language_profile="en",
            language_profile_key="en",
            language_profile_version="language_profile_en_v2",
        )

    def _passing_case(self) -> EvaluatedRenderedCase:
        return EvaluatedRenderedCase(
            case_id="en_b2_remote_work_rendered",
            source_seed_id="en_b2_remote_work",
            status="ok",
            audio_path=Path("/tmp/en_b2_remote_work.wav"),
            expected_language="en",
            feedback_language="en",
            target_cefr="B2",
            benchmark_suite_id="english_monologue_cefr_v1",
            benchmark_case_id="en_b2_supported_argument",
            estimated_cefr="B1",
            cefr_delta=-1,
            cefr_match=False,
            final_score=4.1,
            llm_score=4.2,
            deterministic_score=3.9,
            continuous_score=4.0,
            band=5,
            mode="hybrid",
            warnings=(),
            errors=(),
            error_type=None,
            execution_error=None,
            execution_traceback=None,
            checks={"topic_pass": True},
            dimensions={
                "fluency": {"score": 3.1},
                "pronunciation_intelligibility": {"score": 4.9},
                "grammar": {"score": 3.9},
                "lexicon": {"score": 4.05},
                "coherence": {"score": 3.9},
                "task_fulfillment": {"score": 5.0},
            },
            timings_ms={"llm": 20.0},
            llm_contract=self._matching_contract(),
            raw_llm_rubric=None,
            report=None,
        )

    def test_load_evaluation_manifest_round_trips_benchmark_refs(self):
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(self._passing_case(),),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "evaluation_manifest.json"
            write_evaluation_manifest(suite, manifest_path)
            loaded = load_evaluation_manifest(manifest_path)
            self.assertEqual(loaded.cases[0].benchmark_suite_id, "english_monologue_cefr_v1")
            self.assertEqual(loaded.cases[0].benchmark_case_id, "en_b2_supported_argument")

    def test_compare_evaluation_against_benchmark_passes_when_ranges_and_contract_match(self):
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(self._passing_case(),),
        )
        result = compare_evaluation_against_benchmark(self.benchmark_suite, suite)
        self.assertEqual(result.passed_cases, 1)
        self.assertEqual(result.failed_cases, 0)
        self.assertEqual(result.case_results[0].issues, ())

    def test_compare_evaluation_against_benchmark_surfaces_range_and_contract_drift(self):
        failing_case = EvaluatedRenderedCase(
            case_id="en_b2_remote_work_rendered",
            source_seed_id="en_b2_remote_work",
            status="ok",
            audio_path=Path("/tmp/en_b2_remote_work.wav"),
            expected_language="en",
            feedback_language="en",
            target_cefr="B2",
            benchmark_suite_id="english_monologue_cefr_v1",
            benchmark_case_id="en_b2_supported_argument",
            estimated_cefr="C1",
            cefr_delta=1,
            cefr_match=False,
            final_score=4.6,
            llm_score=4.7,
            deterministic_score=4.1,
            continuous_score=4.6,
            band=5,
            mode="hybrid",
            warnings=(),
            errors=(),
            error_type=None,
            execution_error=None,
            execution_traceback=None,
            checks={"topic_pass": True},
            dimensions={
                "fluency": {"score": 3.8},
                "pronunciation_intelligibility": {"score": 4.9},
                "grammar": {"score": 3.0},
                "lexicon": {"score": 4.8},
                "coherence": {"score": 4.3},
                "task_fulfillment": {"score": 5.0},
            },
            timings_ms={"llm": 20.0},
            llm_contract=EvaluationLLMContract(
                provider="openrouter",
                llm_model="google/gemini-3.1-pro-preview",
                whisper_model="small",
                response_parser="wrong_parser",
                rubric_schema="RubricResult",
                prompt_version="rubric_multilingual_v1",
                rubric_prompt_version="wrong_prompt",
                coaching_prompt_version="coaching_multilingual_v1",
                scoring_model_version="legacy_hybrid_v1",
                language_profile="en",
                language_profile_key="en",
                language_profile_version="language_profile_en_v2",
            ),
            raw_llm_rubric=None,
            report=None,
        )
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(failing_case,),
        )
        result = compare_evaluation_against_benchmark(self.benchmark_suite, suite)
        self.assertEqual(result.passed_cases, 0)
        self.assertEqual(result.failed_cases, 1)
        case_result = result.case_results[0]
        self.assertIn("cefr_mismatch", case_result.issues)
        self.assertIn("continuous_out_of_range", case_result.issues)
        self.assertIn("dimension_out_of_range:grammar", case_result.issues)
        self.assertIn("llm_contract_mismatch", case_result.issues)

    def test_compare_evaluation_against_benchmark_flags_malformed_dimensions(self):
        malformed_case = self._passing_case()
        malformed_case = EvaluatedRenderedCase(
            **{
                **malformed_case.__dict__,
                "dimensions": {
                    **malformed_case.dimensions,
                    "grammar": "bad-score",
                },
            }
        )
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(malformed_case,),
        )
        result = compare_evaluation_against_benchmark(self.benchmark_suite, suite)
        self.assertIn("dimension_malformed:grammar", result.case_results[0].issues)

    def test_compare_evaluation_against_benchmark_accepts_numeric_dimension_scores(self):
        numeric_case = self._passing_case()
        numeric_case = EvaluatedRenderedCase(
            **{
                **numeric_case.__dict__,
                "dimensions": {
                    **numeric_case.dimensions,
                    "grammar": 3.9,
                },
            }
        )
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(numeric_case,),
        )
        result = compare_evaluation_against_benchmark(self.benchmark_suite, suite)
        self.assertNotIn("dimension_malformed:grammar", result.case_results[0].issues)

    def test_compare_evaluation_against_benchmark_marks_contract_as_not_applicable_when_unconstrained(self):
        unconstrained_suite = load_benchmark_suite(BENCHMARK_FIXTURE)
        unconstrained_suite = type(unconstrained_suite)(
            suite_id=unconstrained_suite.suite_id,
            language_code=unconstrained_suite.language_code,
            language_profile_key=None,
            task_family=unconstrained_suite.task_family,
            suite_type=unconstrained_suite.suite_type,
            scorer_version=None,
            llm_contract=type(unconstrained_suite.llm_contract)(
                rubric_prompt_version=None,
                coaching_prompt_version=None,
                response_parser=None,
                rubric_schema=None,
                notes=unconstrained_suite.llm_contract.notes,
            ),
            active=unconstrained_suite.active,
            tags=unconstrained_suite.tags,
            notes=unconstrained_suite.notes,
            cases=unconstrained_suite.cases,
        )
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(self._passing_case(),),
        )
        result = compare_evaluation_against_benchmark(unconstrained_suite, suite)
        self.assertIsNone(result.case_results[0].contract_match)

    def test_compare_evaluation_against_benchmark_tolerates_unconstrained_cefr_and_continuous(self):
        base_case = self.benchmark_suite.cases[1]
        unconstrained_case = type(base_case)(
            case_id=base_case.case_id,
            target_level=base_case.target_level,
            metrics=base_case.metrics,
            checks=base_case.checks,
            rubric=base_case.rubric,
            detected_language_probability=base_case.detected_language_probability,
            expected=type(base_case.expected)(
                cefr_level=None,
                continuous_range=None,
                dimension_ranges=base_case.expected.dimension_ranges,
            ),
            active=base_case.active,
            tags=base_case.tags,
            notes=base_case.notes,
        )
        unconstrained_suite = type(self.benchmark_suite)(
            suite_id=self.benchmark_suite.suite_id,
            language_code=self.benchmark_suite.language_code,
            language_profile_key=self.benchmark_suite.language_profile_key,
            task_family=self.benchmark_suite.task_family,
            suite_type=self.benchmark_suite.suite_type,
            scorer_version=self.benchmark_suite.scorer_version,
            llm_contract=self.benchmark_suite.llm_contract,
            active=self.benchmark_suite.active,
            tags=self.benchmark_suite.tags,
            notes=self.benchmark_suite.notes,
            cases=(self.benchmark_suite.cases[0], unconstrained_case, *self.benchmark_suite.cases[2:]),
        )
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(self._passing_case(),),
        )
        result = compare_evaluation_against_benchmark(unconstrained_suite, suite)
        self.assertIsNone(result.case_results[0].cefr_match)
        self.assertIsNone(result.case_results[0].continuous_passed)

    def test_write_regression_report_emits_summary_and_case_details(self):
        suite = EvaluatedRenderedAudioSuite(
            suite_id="english_monologue_seeds_v1_openrouter_google-gemini-3-1-pro-preview_evaluation_v1",
            manifest_id="english_monologue_seeds_v1",
            language_code="en",
            task_family="opinion_monologue",
            generated_at_utc="2026-03-14T16:00:00+00:00",
            run_status="ok",
            success_ratio=1.0,
            config=self._base_config(),
            cases=(self._passing_case(),),
        )
        result = compare_evaluation_against_benchmark(self.benchmark_suite, suite)
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "benchmark_regression_report.json"
            written = write_regression_report(result, report_path)
            payload = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_cases"], 1)
            self.assertEqual(payload["summary"]["passed_cases"], 1)
            self.assertEqual(payload["cases"][0]["benchmark_case_id"], "en_b2_supported_argument")


if __name__ == "__main__":
    unittest.main()
