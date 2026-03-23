import unittest
from dataclasses import replace

from assess_core.language_profiles import DEFAULT_LANGUAGE_PROFILE, require_language_profile
from assess_core.schemas import RubricResult
from assessment_runtime.dimension_scoring import aggregate_dimension_scores, score_dimensions
from assessment_runtime.scoring import compute_checks, deterministic_score, final_scores, rubric_score


class ScoringTests(unittest.TestCase):
    def test_deterministic_score_stays_in_range(self):
        metrics = {
            "duration_sec": 60.0,
            "pause_total_sec": 5.0,
            "word_count": 100,
            "wpm": 109.0,
            "fillers": 2,
            "cohesion_markers": 2,
            "complexity_index": 2,
        }
        score = deterministic_score(metrics)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 5.0)

    def test_rubric_score_averages_values(self):
        rubric = RubricResult(
            fluency=4,
            cohesion=4,
            accuracy=3,
            range=5,
            overall=4,
            comments_fluency="ok",
            comments_cohesion="ok",
            comments_accuracy="ok",
            comments_range="ok",
            overall_comment="ok",
            on_topic=True,
        )
        self.assertEqual(rubric_score(rubric), 4.0)

    def test_compute_checks_includes_language_gate(self):
        checks = compute_checks(
            metrics={"speaking_time_sec": 55.0, "word_count": 20},
            rubric=None,
            target_duration_sec=60.0,
            min_word_count=10,
            duration_pass_ratio=0.8,
            language_pass=False,
        )
        self.assertFalse(checks["language_pass"])
        self.assertTrue(checks["duration_pass"])

    def test_final_scores_caps_off_topic(self):
        scores = final_scores(deterministic=4.5, llm=4.5, topic_pass=False, topic_fail_cap_score=2.5)
        self.assertEqual(scores["final"], 2.5)

    def test_dimension_scoring_produces_provisional_english_cefr_estimate(self):
        profile = require_language_profile("en")
        rubric = RubricResult(
            fluency=4,
            cohesion=4,
            accuracy=4,
            range=5,
            overall=4,
            comments_fluency="ok",
            comments_cohesion="ok",
            comments_accuracy="ok",
            comments_range="ok",
            overall_comment="ok",
            on_topic=True,
            topic_relevance_score=5,
            language_ok=True,
            recurring_grammar_errors=[],
            coherence_issues=[],
            lexical_gaps=[],
            evidence_quotes=[],
            confidence="high",
        )
        checks = {
            "duration_pass": True,
            "topic_pass": True,
            "min_words_pass": True,
            "language_pass": True,
            "asr_pause_consistent": True,
        }
        metrics = {
            "duration_sec": 120.0,
            "pause_total_sec": 2.0,
            "word_count": 220,
            "wpm": 150.0,
            "fillers": 1,
            "cohesion_markers": 3,
            "complexity_index": 4,
        }
        dimensions = score_dimensions(
            metrics=metrics,
            rubric=rubric,
            checks=checks,
            profile=profile,
            detected_language_probability=0.97,
        )
        cefr = aggregate_dimension_scores(dimensions, profile=profile)
        self.assertIn("fluency", dimensions)
        self.assertIn("grammar", dimensions)
        self.assertGreaterEqual(dimensions["fluency"]["score"], 4.0)
        self.assertIn(cefr["level"], {"B2", "C1", "C2"})
        self.assertFalse(cefr["calibrated"])

    def test_aggregate_dimension_scores_marks_generic_fallback_as_low_confidence(self):
        cefr = aggregate_dimension_scores({}, profile=DEFAULT_LANGUAGE_PROFILE)
        self.assertIsNone(cefr["continuous"])
        self.assertIsNone(cefr["level"])
        self.assertEqual(cefr["confidence"], "low")

    def test_aggregate_dimension_scores_handles_partial_cut_scores(self):
        profile = replace(
            require_language_profile("en"),
            cefr_cut_scores={"B2": 3.5},
        )
        dimensions = {
            "fluency": {"score": 4.0, "confidence": "medium"},
            "pronunciation_intelligibility": {"score": 4.0, "confidence": "medium"},
            "grammar": {"score": 4.0, "confidence": "medium"},
            "lexicon": {"score": 4.0, "confidence": "medium"},
            "coherence": {"score": 4.0, "confidence": "medium"},
            "task_fulfillment": {"score": 4.0, "confidence": "medium"},
        }
        cefr = aggregate_dimension_scores(dimensions, profile=profile)
        self.assertEqual(cefr["continuous"], 4.0)
        self.assertEqual(cefr["level"], "B2")

    def test_aggregate_dimension_scores_ignores_none_cut_scores(self):
        profile = replace(
            require_language_profile("en"),
            cefr_cut_scores={"B2": 3.5, "C1": None, "C2": None},
        )
        dimensions = {
            "fluency": {"score": 4.0, "confidence": "medium"},
            "pronunciation_intelligibility": {"score": 4.0, "confidence": "medium"},
            "grammar": {"score": 4.0, "confidence": "medium"},
            "lexicon": {"score": 4.0, "confidence": "medium"},
            "coherence": {"score": 4.0, "confidence": "medium"},
            "task_fulfillment": {"score": 4.0, "confidence": "medium"},
        }
        cefr = aggregate_dimension_scores(dimensions, profile=profile)
        self.assertEqual(cefr["level"], "B2")

    def test_score_dimensions_handles_zero_tolerances_in_profile(self):
        profile = replace(
            require_language_profile("en"),
            pace_tolerance_wpm=0.0,
            pause_ratio_ceiling=0.0,
            filler_ratio_ceiling=0.0,
        )
        dimensions = score_dimensions(
            metrics={
                "duration_sec": 120.0,
                "pause_total_sec": 2.0,
                "word_count": 220,
                "wpm": 150.0,
                "fillers": 1,
                "cohesion_markers": 3,
                "complexity_index": 4,
            },
            rubric=None,
            checks={
                "duration_pass": True,
                "topic_pass": True,
                "min_words_pass": True,
                "language_pass": True,
                "asr_pause_consistent": True,
            },
            profile=profile,
            detected_language_probability=0.97,
        )
        self.assertGreaterEqual(dimensions["fluency"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
