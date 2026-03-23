import unittest
from pathlib import Path

from benchmarking.benchmark_suites import discover_benchmark_suites, evaluate_benchmark_case


class ItalianBenchmarkSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures_dir = Path(__file__).parent / "fixtures" / "benchmarks"
        suites = discover_benchmark_suites(fixtures_dir, language_codes={"it"})
        cls.suites = tuple(suite for suite in suites if suite.suite_id == "italian_monologue_cefr_v1")

    def test_italian_progression_suite_is_present_and_pinned(self):
        self.assertEqual(len(self.suites), 1)
        suite = self.suites[0]
        self.assertEqual(suite.language_code, "it")
        self.assertEqual(suite.language_profile_key, "it_benchmark")
        self.assertEqual(suite.scorer_version, "language_profile_it_v1")
        self.assertEqual(suite.suite_type, "progression")

    def test_italian_cases_map_to_expected_ranges(self):
        suite = self.suites[0]
        for case in suite.cases:
            with self.subTest(case=case.case_id):
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


if __name__ == "__main__":
    unittest.main()
