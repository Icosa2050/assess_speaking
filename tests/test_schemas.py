import unittest

from schemas import AssessmentReport, RubricResult, SchemaValidationError


def _rubric_data() -> dict:
    return {
        "fluency": 4,
        "cohesion": 4,
        "accuracy": 3,
        "range": 4,
        "overall": 4,
        "comments_fluency": "ok",
        "comments_cohesion": "ok",
        "comments_accuracy": "ok",
        "comments_range": "ok",
        "overall_comment": "ok",
        "on_topic": True,
    }


def _report_data() -> dict:
    return {
        "timestamp_utc": "2026-03-07T10:00:00+00:00",
        "input": {"provider": "openrouter", "llm_model": "x"},
        "metrics": {
            "duration_sec": 10.0,
            "pause_count": 1,
            "pause_total_sec": 1.0,
            "speaking_time_sec": 9.0,
            "word_count": 12,
            "wpm": 80.0,
            "fillers": 1,
            "cohesion_markers": 1,
            "complexity_index": 1,
        },
        "checks": {"duration_pass": True, "topic_pass": True, "min_words_pass": True, "language_pass": True},
        "scores": {"deterministic": 3.2, "llm": 4.0, "final": 3.7, "band": 4, "mode": "hybrid"},
        "requires_human_review": False,
        "transcript_preview": "ciao mondo",
        "warnings": [],
        "errors": [],
        "rubric": _rubric_data(),
        "timings_ms": {"audio_features": 12.1, "asr": 43.8, "llm": 215.4, "total": 276.2},
    }


class RubricSchemaTests(unittest.TestCase):
    def test_from_dict_accepts_valid_payload(self):
        rubric = RubricResult.from_dict(_rubric_data())
        self.assertEqual(rubric.overall, 4)

    def test_from_dict_accepts_integer_valued_float_scores(self):
        data = _rubric_data()
        data["overall"] = 4.0
        rubric = RubricResult.from_dict(data)
        self.assertEqual(rubric.overall, 4)

    def test_from_dict_rejects_missing_key(self):
        data = _rubric_data()
        del data["on_topic"]
        with self.assertRaises(SchemaValidationError):
            RubricResult.from_dict(data)

    def test_from_dict_rejects_fractional_scores(self):
        data = _rubric_data()
        data["fluency"] = 3.9
        with self.assertRaises(SchemaValidationError):
            RubricResult.from_dict(data)

    def test_from_dict_rejects_boolean_scores(self):
        data = _rubric_data()
        data["fluency"] = True
        with self.assertRaises(SchemaValidationError):
            RubricResult.from_dict(data)


class AssessmentSchemaTests(unittest.TestCase):
    def test_report_accepts_valid_payload(self):
        report = AssessmentReport.from_dict(_report_data())
        self.assertFalse(report.requires_human_review)

    def test_report_rejects_invalid_requires_human_review(self):
        data = _report_data()
        data["requires_human_review"] = "yes"
        with self.assertRaises(SchemaValidationError):
            AssessmentReport.from_dict(data)


if __name__ == "__main__":
    unittest.main()
