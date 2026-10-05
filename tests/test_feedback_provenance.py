"""Authored regression contracts; these fixtures do not establish CEFR validity."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import assess_speaking
from assessment_runtime.asr import _build_transcription_result, _merge_chunk_transcriptions
from assessment_runtime.comparison import comparison_key
from assessment_runtime.transcript_quality import transcript_quality
from test_assess_speaking import _sample_report


def context():
    return dict(version=1, goal="B1", target_duration_sec=90, scoring_version="v1",
                scoring_mode="hybrid", analysis_signature="signature", provider="ollama",
                model="small", whisper_model="large-v3", asr_provider="faster_whisper",
                dry_run=False, prompt_text="Describe your city", retry_of_session_id="")


def save_prior(tmp_path, report, practice):
    report = deepcopy(report)
    report["session_id"] = "parent"
    path = tmp_path / "parent.json"
    path.write_text(json.dumps({"report": report, "meta": {"practice": practice}}))
    with (tmp_path / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=assess_speaking.HISTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(dict(session_id="parent", timestamp="2026-10-01T00:00:00",
            speaker_id="bern", learning_language="it", task_family="travel_narrative",
            final_score=3, overall=3, wpm=90, report_path=str(path)))


def current_report():
    report = _sample_report()
    report["input"]["learning_language"] = "it"
    return report


def test_matching_provenance_is_verified(tmp_path):
    report, practice = current_report(), context()
    save_prior(tmp_path, report, practice)
    delta = assess_speaking.build_progress_delta(tmp_path / "history.csv", report, practice=practice)
    assert delta["comparison_verified"] is True
    assert delta["previous_session_id"] == "parent"


@pytest.mark.parametrize("field,value", [("model", "different"), ("whisper_model", "tiny"),
    ("target_duration_sec", 120), ("goal", "C1"), ("scoring_mode", "deterministic_only"),
    ("analysis_signature", "changed"), ("scoring_version", "v2"), ("asr_provider", "chunked")])
def test_changed_conditions_do_not_compare(tmp_path, field, value):
    report, practice = current_report(), context()
    save_prior(tmp_path, report, practice)
    practice[field] = value
    assert assess_speaking.build_progress_delta(tmp_path / "history.csv", report, practice=practice) is None


def test_explicit_retry_does_not_substitute_or_change_prompt(tmp_path):
    report, practice = current_report(), context()
    save_prior(tmp_path, report, practice)
    for parent, prompt in [("absent", practice["prompt_text"]), ("parent", "Changed prompt")]:
        changed = {**practice, "retry_of_session_id": parent, "prompt_text": prompt}
        assert assess_speaking.build_progress_delta(tmp_path / "history.csv", report, practice=changed) is None
    assert assess_speaking.build_progress_delta(tmp_path / "history.csv", report,
        practice={**practice, "retry_of_session_id": "parent"})["previous_session_id"] == "parent"


def test_missing_provenance_and_invalid_duration_are_unknown():
    assert comparison_key(current_report(), None) is None
    assert comparison_key(current_report(), {**context(), "target_duration_sec": float("nan")}) is None


def test_asr_diagnostics_survive_native_and_chunked_results():
    word = SimpleNamespace(word="Ciao", start=0., end=.5, probability=.2)
    segment = SimpleNamespace(text="Ciao", words=[word], avg_logprob=-1.5, no_speech_prob=.1)
    result = _build_transcription_result([segment], {"language": "it", "language_probability": .99},
        compute_type_used="int8", compute_fallback_used=False)
    assert result["words"][0]["probability"] == .2
    assert transcript_quality(result)["status"] == "uncertain"
    merged = _merge_chunk_transcriptions([result, result])
    assert len(merged["segment_diagnostics"]) == 2
    assert transcript_quality(merged)["weak_segment_count"] == 2


def test_language_confidence_is_not_transcript_confidence():
    assert transcript_quality({"words": [], "language_probability": .999})["status"] == "unknown"
    result = {"words": [{"probability": p} for p in [.2, .3, .9, .9]], "language_probability": .999}
    assert transcript_quality(result)["status"] == "uncertain"
    assert transcript_quality({"words": [{"probability": .9}]})["status"] == "no_review_trigger"


def test_observed_weak_transcript_skips_criticism_even_with_confident_language():
    audio = dict(duration_sec=30., pause_count=1, pause_total_sec=1., speaking_time_sec=29., pauses=[(1., 2., 1.)])
    words = [dict(text="ciao", t0=i*.5, t1=i*.5+.4, probability=.2) for i in range(40)]
    result = dict(text=" ".join(w["text"] for w in words), words=words,
                  detected_language="it", language_probability=.999)
    with mock.patch.object(assess_speaking, "load_audio_features", return_value=audio), \
         mock.patch.object(assess_speaking, "transcribe", return_value=result), \
         mock.patch.object(assess_speaking, "generate_rubric") as rubric, \
         mock.patch.object(assess_speaking, "generate_coaching_summary") as coach:
        output = assess_speaking.run_assessment(Path("sample.wav"), provider="ollama", llm_model="small", expected_language="it")
    rubric.assert_not_called()
    coach.assert_not_called()
    report = output["report"]
    assert report["rubric"] is None
    assert report["scores"]["mode"] == "deterministic_only"
    assert report["requires_human_review"] is True
    assert "transcript_uncertain" in report["warnings"]
    assert report["input"]["transcript_quality"]["status"] == "uncertain"


def test_saved_runner_comparisons_match_history_provenance(tmp_path):
    from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run
    report = _sample_report()
    report["input"]["scoring_model_version"] = "v1"
    def output(session, model):
        current = deepcopy(report)
        current["session_id"] = session
        current["input"]["llm_model"] = model
        return dict(report=current, metrics=current["metrics"], transcript_full="ciao mondo",
            transcript_preview="ciao mondo", llm_rubric=json.dumps(current["rubric"]))
    request = AssessmentRunRequest(audio=tmp_path / "audio.wav", log_dir=tmp_path,
        expected_language="it", speaker_id="bern", task_family="travel_narrative",
        target_cefr="B1", prompt_text="Describe your city", target_duration_sec=60,
        provider="openrouter", llm_model="small")
    with mock.patch.object(assess_speaking, "run_assessment", return_value=output("parent", "small")):
        first = execute_assessment_run(request)
    with mock.patch.object(assess_speaking, "run_assessment", return_value=output("child", "small")):
        second = execute_assessment_run(request)
    assert second.report["progress_delta"]["previous_session_id"] == "parent"
    assert second.report["progress_delta"]["comparison_verified"] is True
    with mock.patch.object(assess_speaking, "run_assessment", return_value=output("different", "other")):
        third = execute_assessment_run(request)
    assert third.report.get("progress_delta") is None
    assert json.loads(first.report_path.read_text())["report"]["session_id"] == "parent"


def test_empty_silence_segments_do_not_gate_usable_transcript():
    segment = SimpleNamespace(text="   ", words=[], avg_logprob=-5., no_speech_prob=.99)
    result = _build_transcription_result([segment], {}, compute_type_used="int8", compute_fallback_used=False)
    assert result["segment_diagnostics"] == []
    assert transcript_quality(result)["status"] == "unknown"


def test_isolated_weak_segment_uses_ratio_threshold():
    result = {"segment_diagnostics": [{"avg_logprob": -1.5}] + [{"avg_logprob": -.1}] * 9}
    assert transcript_quality(result)["status"] == "no_review_trigger"
    result["segment_diagnostics"] = result["segment_diagnostics"][:4]
    assert transcript_quality(result)["status"] == "uncertain"


def test_ungrounded_rubric_exhaustion_discards_scores_and_history_categories():
    from assessment_runtime import llm_client
    audio = dict(duration_sec=30., pauses=[(1., 2., 1.)])
    words = [dict(text="parola", t0=i*.5, t1=i*.5+.4) for i in range(40)]
    transcription = dict(text="Sono andato a casa.", words=words,
                         detected_language="it", language_probability=.99)
    fabricated = _sample_report()["rubric"]
    fabricated["evidence_quotes"] = ["Invented learner words"]
    with mock.patch.object(assess_speaking, "load_audio_features", return_value=audio), \
         mock.patch.object(assess_speaking, "transcribe", return_value=transcription), \
         mock.patch.object(llm_client, "_chat_completion", return_value=json.dumps(fabricated)) as chat:
        output = assess_speaking.run_assessment(Path("sample.wav"), provider="ollama", llm_model="small", expected_language="it")
    assert chat.call_count == 2
    report = output["report"]
    assert report["rubric"] is None
    assert report["scores"]["llm"] is None
    assert report["scores"]["mode"] == "deterministic_only"
    assert report["requires_human_review"] is True
    assert "llm_invalid_schema" in report["warnings"]
    assert assess_speaking._extract_issue_categories(report["rubric"], "recurring_grammar_errors") == ""


def test_issue_on_uncertain_asr_span_fails_closed_below_global_threshold():
    from assessment_runtime import llm_client
    tokens = ["Sono", "andato", "a", "casa.", "Poi", "ho", "visitato", "mia", "sorella.", "Tutto", "bene."]
    words = [dict(text=text, t0=i*.3, t1=i*.3+.2, probability=.2 if i == 1 else .95) for i, text in enumerate(tokens)]
    transcription = dict(text=" ".join(tokens), words=words, detected_language="it", language_probability=.99)
    assert transcript_quality(transcription)["status"] == "no_review_trigger"
    rubric = _sample_report()["rubric"]
    rubric["evidence_quotes"] = ["Sono andato a casa."]
    rubric["recurring_grammar_errors"] = [dict(category="verb_conjugation_past", explanation="Uncertain diagnosis", examples=["Sono andato"])]
    with mock.patch.object(assess_speaking, "load_audio_features", return_value=dict(duration_sec=30., pauses=[])), \
         mock.patch.object(assess_speaking, "transcribe", return_value=transcription), \
         mock.patch.object(llm_client, "_chat_completion", return_value=json.dumps(rubric)) as chat:
        output = assess_speaking.run_assessment(Path("sample.wav"), provider="ollama", llm_model="small", expected_language="it", min_word_count=1)
    assert chat.call_count == 2
    report = output["report"]
    assert report["rubric"] is None
    assert report["scores"]["mode"] == "deterministic_only"
    assert report["requires_human_review"] is True
    assert "transcript_uncertain" in report["warnings"]
    assert report["input"]["transcript_quality"]["trigger"] == "quoted_asr_evidence_uncertain"
    assert report["input"]["transcript_quality"]["low_confidence_spans"][0]["text"] == "andato"


@pytest.mark.parametrize("score,expected_delta", [(2, -1), (3, 0), (4, 1)])
def test_retry_comparison_preserves_decrease_flat_and_increase(tmp_path, score, expected_delta):
    report, practice = current_report(), context()
    save_prior(tmp_path, report, practice)
    report["scores"]["final"] = score
    report["scores"]["llm"] = score
    report["metrics"]["wpm"] = 80
    delta = assess_speaking.build_progress_delta(tmp_path / "history.csv", report,
        practice={**practice, "retry_of_session_id": "parent"})
    assert delta["comparison_verified"] is True
    assert delta["previous_session_id"] == "parent"
    assert delta["score_delta"] == {"final": expected_delta, "overall": expected_delta, "wpm": -10}


@pytest.mark.parametrize("corrupt", [None, [], {"report": []}, {"report": {}, "meta": []},
    {"report": {"input": []}, "meta": {"practice": "bad"}}])
def test_corrupt_history_report_is_skipped_without_losing_valid_comparison(tmp_path, corrupt):
    from assessment_runtime.comparison import comparable_history_rows
    report, practice = current_report(), context()
    save_prior(tmp_path, report, practice)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(corrupt))
    rows = [None, {"session_id": "bad", "report_path": str(bad)},
            {"session_id": "parent", "report_path": str(tmp_path / "parent.json"), "timestamp": None}]
    assert comparable_history_rows(tmp_path / "history.csv", report, practice, rows) == [rows[-1]]
