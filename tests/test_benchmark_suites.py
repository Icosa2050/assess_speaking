import unittest
from pathlib import Path

from assess_core.language_profiles import require_resolved_language_profile
from benchmarking.benchmark_suites import discover_benchmark_suites, evaluate_benchmark_case


LEVEL_ORDER = {"B1": 1, "B2": 2, "C1": 3, "C2": 4}


class BenchmarkSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = Path(__file__).parent / "fixtures" / "benchmarks"
        cls.suites = discover_benchmark_suites(cls.fixtures_dir)

    def test_discovery_finds_multiple_active_suites(self):
        suite_ids = {suite.suite_id for suite in self.suites}
        self.assertTrue(
            {
                "english_monologue_cefr_v1",
                "english_monologue_quality_gates_v1",
                "italian_monologue_cefr_v1",
            }.issubset(suite_ids)
        )

    def test_discovery_filters_by_suite_type_and_tag(self):
        progression = discover_benchmark_suites(self.fixtures_dir, suite_types={"progression"})
        italian_progression = discover_benchmark_suites(
            self.fixtures_dir,
            language_codes={"it"},
            suite_types={"progression"},
        )
        quality_gates = discover_benchmark_suites(self.fixtures_dir, tags={"quality-gates"})
        quality_gates_all = discover_benchmark_suites(
            self.fixtures_dir,
            tags={"english", "quality-gates"},
            tag_match="all",
        )
        progression_ids = {suite.suite_id for suite in progression}
        self.assertIn("english_monologue_cefr_v1", progression_ids)
        self.assertIn("italian_monologue_cefr_v1", progression_ids)
        self.assertEqual([suite.suite_id for suite in italian_progression], ["italian_monologue_cefr_v1"])
        self.assertIn("english_monologue_quality_gates_v1", [suite.suite_id for suite in quality_gates])
        self.assertIn("english_monologue_quality_gates_v1", [suite.suite_id for suite in quality_gates_all])

    def test_suite_metadata_matches_profile_version(self):
        for suite in self.suites:
            with self.subTest(suite=suite.suite_id):
                profile = require_resolved_language_profile(
                    suite.language_code,
                    profile_key=suite.language_profile_key,
                )
                self.assertEqual(suite.scorer_version, profile.scorer_version)
                self.assertIsNotNone(suite.language_profile_key)
                self.assertTrue(suite.active)
                self.assertGreaterEqual(len(suite.cases), 1)
                self.assertEqual(suite.llm_contract.response_parser, "extract_json_object")
                self.assertEqual(suite.llm_contract.rubric_schema, "RubricResult")

    def test_each_active_case_maps_to_expected_ranges(self):
        for suite in self.suites:
            for case in suite.cases:
                if not case.active:
                    continue
                with self.subTest(suite=suite.suite_id, case=case.case_id):
                    result = evaluate_benchmark_case(
                        case,
                        language_code=suite.language_code,
                        language_profile_key=suite.language_profile_key,
                    )
                    cefr = result["cefr_estimate"]
                    self.assertEqual(cefr["level"], case.expected.cefr_level)
                    low, high = case.expected.continuous_range
                    self.assertGreaterEqual(cefr["continuous"], low)
                    self.assertLessEqual(cefr["continuous"], high)

                    for dimension, expected_range in case.expected.dimension_ranges.items():
                        dim_score = result["dimensions"][dimension]["score"]
                        dim_low, dim_high = expected_range
                        self.assertGreaterEqual(dim_score, dim_low, msg=f"{case.case_id}:{dimension}")
                        self.assertLessEqual(dim_score, dim_high, msg=f"{case.case_id}:{dimension}")

    def test_progression_suites_are_monotonic(self):
        progression_suites = discover_benchmark_suites(self.fixtures_dir, suite_types={"progression"})
        for suite in progression_suites:
            with self.subTest(suite=suite.suite_id):
                ordered_cases = sorted(
                    (case for case in suite.cases if case.active),
                    key=lambda case: LEVEL_ORDER[case.expected.cefr_level],
                )
                continuous_scores = [
                    evaluate_benchmark_case(
                        case,
                        language_code=suite.language_code,
                        language_profile_key=suite.language_profile_key,
                    )["cefr_estimate"]["continuous"]
                    for case in ordered_cases
                ]
                self.assertEqual(sorted(continuous_scores), continuous_scores)

    def test_discovery_rejects_invalid_tag_match_mode(self):
        with self.assertRaises(ValueError):
            discover_benchmark_suites(self.fixtures_dir, tags={"english"}, tag_match="invalid")


if __name__ == "__main__":
    unittest.main()
