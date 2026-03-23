import unittest

from app_shell.scoring_guide import build_scoring_guide_data


class ScoringGuideDataTests(unittest.TestCase):
    def test_scoring_guide_data_matches_live_defaults(self):
        guide = build_scoring_guide_data()

        self.assertEqual(guide["score_scale"]["min"], 1.0)
        self.assertEqual(guide["score_scale"]["max"], 5.0)
        self.assertEqual(guide["formula"]["deterministic_weight_pct"], 40)
        self.assertEqual(guide["formula"]["rubric_weight_pct"], 60)
        self.assertEqual(guide["formula"]["topic_fail_cap_score"], 2.5)

        signal_weights = {
            row["id"]: row["weight_pct"]
            for row in guide["deterministic_signals"]
        }
        self.assertEqual(signal_weights["wpm"], 35)
        self.assertEqual(signal_weights["pause_ratio"], 25)
        self.assertEqual(signal_weights["filler_ratio"], 20)
        self.assertEqual(signal_weights["cohesion_markers"], 10)
        self.assertEqual(signal_weights["complexity_index"], 10)

        gate_rows = {row["id"]: row for row in guide["gates"]}
        self.assertEqual(gate_rows["duration_pass"]["duration_pass_ratio_pct"], 80)
        self.assertEqual(gate_rows["min_words_pass"]["min_word_count"], 5)

        cefr_rows = {
            row["code"]: row
            for row in guide["cefr_thresholds"]
        }
        self.assertEqual(cefr_rows["en"]["B2"], 4.05)
        self.assertEqual(cefr_rows["en"]["C1"], 4.65)
        self.assertEqual(cefr_rows["en"]["C2"], 4.85)
        self.assertEqual(cefr_rows["it"]["B2"], 3.45)
        self.assertEqual(cefr_rows["it"]["C1"], 4.10)
        self.assertEqual(cefr_rows["it"]["C2"], 4.65)


if __name__ == "__main__":
    unittest.main()
