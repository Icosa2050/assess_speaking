import unittest

from schemas import RubricResult
from scoring import compute_checks, deterministic_score, final_scores, rubric_score


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


if __name__ == "__main__":
    unittest.main()
