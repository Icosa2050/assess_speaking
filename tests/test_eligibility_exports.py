"""Grades cannot reappear through journal, CLI HTML or legacy summaries."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app_core.services import review_summary
from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run
from scripts import progress_dashboard


def payload(check_changes=None):
    checks = {key: True for key in (
        "duration_pass", "topic_pass", "language_pass", "min_words_pass", "content_validity_pass"
    )}
    checks.update(check_changes or {})
    return {
        "metrics": {"duration_sec": 31, "word_count": 40, "wpm": 90},
        "transcript_full": "Synthetic connected speech for this persistence regression.",
        "transcript_preview": "Synthetic connected speech.", "llm_rubric": "{}",
        "report": {
            "session_id": "export-case", "schema_version": 2,
            "input": {"learning_language": "it", "speaker_id": "fixture"},
            "checks": checks, "scores": {"final": 4.75, "band": 5},
            "rubric": {key: 4.75 for key in ("overall", "fluency", "cohesion", "accuracy", "range")},
            "progress_delta": {"comparison_verified": True, "score_delta": {"final": 1.5}},
            "baseline_comparison": {"level": "B2"},
        },
    }


@pytest.mark.parametrize("changes,state", [
    ({"min_words_pass": False}, "insufficient_speech"),
    ({"content_validity_pass": False}, "invalid_content"),
    ({"language_pass": None}, "content_unverified"),
    ({}, "assessable"),
])
def test_new_csv_withholds_grades_but_annotated_json_retains_observations(tmp_path, changes, state):
    assessment = payload(changes)
    with patch("assess_speaking.run_assessment", return_value=assessment), patch("assess_speaking.build_progress_delta", return_value=None):
        result = execute_assessment_run(AssessmentRunRequest(audio=tmp_path / "take.wav", log_dir=tmp_path))
    saved = json.loads(result.report_path.read_text())
    assert saved["report"]["eligibility"]["state"] == state
    assert saved["report"]["scores"]["final"] == 4.75
    assert saved["report"]["scores"]["status"] == ("assessable" if state == "assessable" else "provisional_observations")
    with (tmp_path / "history.csv").open() as handle:
        row = next(csv.DictReader(handle))
    for key in ("overall", "fluency", "cohesion", "accuracy", "range", "final_score", "band"):
        assert bool(row[key]) == (state == "assessable")


@pytest.mark.parametrize("changes,state", [
    ({"min_words_pass": False}, "insufficient_speech"),
    ({"content_validity_pass": False}, "invalid_content"),
    ({"language_pass": None}, "content_unverified"),
    ({}, "assessable"),
])
def test_legacy_grades_and_deltas_are_masked_without_rewriting_evidence(tmp_path, changes, state):
    report = tmp_path / "legacy.json"
    report.write_text(json.dumps(payload(changes)))
    before = report.read_bytes()
    history = tmp_path / "history.csv"
    history.write_text(
        "timestamp,session_id,audio,overall,final_score,band,report_path\n"
        f"2026-10-08T10:00:00,export-case,take.wav,4.75,4.75,5,{report}\n"
    )
    records = progress_dashboard.load_history(history)
    summary = progress_dashboard.summarise(records)
    expected = 4.75 if state == "assessable" else None
    assert records[0].overall == records[0].final_score == summary["avg_final"] == expected
    html = progress_dashboard.render_html(records, summary)
    if state != "assessable":
        assert "grades withheld" in html and "4.75" not in html
        assert progress_dashboard.load_progress_delta(str(report)) is None
    else:
        assert "4.75" in html and progress_dashboard.load_progress_delta(str(report))
    legacy_summary = review_summary(payload(changes))
    assert legacy_summary["score_overall"] == expected
    assert bool(legacy_summary["progress_items"]) == (state == "assessable")
    assert report.read_bytes() == before


def test_missing_report_cannot_validate_csv_grade(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text("timestamp,overall,final_score,band,report_path\n2026-10-08T10:00:00,4.75,4.75,5,\n")
    record = progress_dashboard.load_history(history)[0]
    assert record.eligibility_state == "content_unverified"
    assert record.overall is record.final_score is record.band is None
