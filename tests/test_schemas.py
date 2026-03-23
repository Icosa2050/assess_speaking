import unittest

from assess_core.schemas import AssessmentReport, RubricResult, SchemaValidationError


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
        "topic_relevance_score": 4,
        "language_ok": True,
        "recurring_grammar_errors": [
            {
                "category": "preposition_choice",
                "explanation": "Confonde in/a con le destinazioni.",
                "examples": ["sono andato a Spagna"],
            }
        ],
        "coherence_issues": [
            {
                "category": "missing_sequence_markers",
                "explanation": "Mancano connettivi temporali.",
                "examples": ["poi", "alla fine"],
            }
        ],
        "lexical_gaps": [
            {
                "category": "travel_vocabulary_gap",
                "explanation": "Lessico di viaggio limitato.",
                "examples": ["ostello", "coincidenza"],
            }
        ],
        "evidence_quotes": ["sono andato a Spagna"],
        "confidence": "medium",
    }


def _report_data() -> dict:
    return {
        "schema_version": 2,
        "session_id": "sess-123",
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
        "coaching": {
            "strengths": ["Resti sul tema."],
            "top_3_priorities": ["Più connettivi", "Meno filler", "Passato più stabile"],
            "next_focus": "Ordina meglio gli eventi",
            "next_exercise": "Racconta di nuovo il viaggio usando prima/poi/alla fine.",
            "coach_summary": "Buona base, ma la sequenza narrativa va resa più chiara.",
        },
        "progress_delta": {
            "comparison_scope": {"speaker_id": "bern", "task_family": "travel_narrative"},
            "previous_session_id": "sess-122",
            "previous_timestamp": "2026-03-01T10:00:00+00:00",
        },
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

    def test_from_dict_rejects_legacy_v1_payload_without_new_fields(self):
        data = _rubric_data()
        for key in [
            "topic_relevance_score",
            "language_ok",
            "recurring_grammar_errors",
            "coherence_issues",
            "lexical_gaps",
            "evidence_quotes",
            "confidence",
        ]:
            del data[key]
        with self.assertRaises(SchemaValidationError):
            RubricResult.from_dict(data)

    def test_from_dict_rejects_invalid_taxonomy_category(self):
        data = _rubric_data()
        data["recurring_grammar_errors"][0]["category"] = "verb_error"
        with self.assertRaises(SchemaValidationError):
            RubricResult.from_dict(data)


class AssessmentSchemaTests(unittest.TestCase):
    def test_report_accepts_valid_payload(self):
        report = AssessmentReport.from_dict(_report_data())
        self.assertFalse(report.requires_human_review)
        self.assertEqual(report.schema_version, 2)
        self.assertEqual(report.session_id, "sess-123")

    def test_report_preserves_extra_score_metadata(self):
        data = _report_data()
        data["scores"]["language_profile_version"] = "language_profile_en_v2"
        data["scores"]["cefr_estimate"] = {"level": "B2", "calibrated": False}
        report = AssessmentReport.from_dict(data)
        self.assertEqual(report.scores["language_profile_version"], "language_profile_en_v2")
        self.assertEqual(report.scores["cefr_estimate"]["level"], "B2")

    def test_report_rejects_invalid_requires_human_review(self):
        data = _report_data()
        data["requires_human_review"] = "yes"
        with self.assertRaises(SchemaValidationError):
            AssessmentReport.from_dict(data)

    def test_report_requires_schema_version(self):
        data = _report_data()
        del data["schema_version"]
        with self.assertRaises(SchemaValidationError):
            AssessmentReport.from_dict(data)

    def test_report_requires_session_id(self):
        data = _report_data()
        del data["session_id"]
        with self.assertRaises(SchemaValidationError):
            AssessmentReport.from_dict(data)

    def test_report_rejects_invalid_progress_delta(self):
        data = _report_data()
        data["progress_delta"] = "bad"
        with self.assertRaises(SchemaValidationError):
            AssessmentReport.from_dict(data)

    def test_report_rejects_invalid_coaching_shape(self):
        data = _report_data()
        data["coaching"]["top_3_priorities"] = ["solo uno"]
        with self.assertRaises(SchemaValidationError):
            AssessmentReport.from_dict(data)


if __name__ == "__main__":
    unittest.main()
