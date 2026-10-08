from pathlib import Path
from unittest import mock

import pytest

import assess_speaking
from assessment_runtime.feedback import build_fallback_coaching
from assessment_runtime.llm_client import LLMClientError
from assessment_runtime.recording_policy import RecordingTooShortError, require_review_duration
from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run


@pytest.mark.parametrize("duration", [0, 0.1, 12, 29, 29.999])
def test_rejects_short_recordings_before_any_transcription_review_or_history(tmp_path, duration):
    cloud_asr = mock.Mock()
    with mock.patch.object(assess_speaking, "load_audio_features", return_value={"duration_sec": duration, "pauses": []}), \
         mock.patch.object(assess_speaking, "transcribe") as transcribe, \
         mock.patch.object(assess_speaking, "generate_rubric") as rubric, \
         mock.patch.object(assess_speaking, "generate_coaching_summary") as coaching:
        with pytest.raises(RecordingTooShortError, match="at least 30 seconds"):
            execute_assessment_run(AssessmentRunRequest(
                audio=tmp_path / "short.wav", provider="ollama", llm_model="synthetic",
                cloud_asr=cloud_asr, log_dir=tmp_path / "reports",
            ))
    transcribe.assert_not_called()
    cloud_asr.assert_not_called()
    rubric.assert_not_called()
    coaching.assert_not_called()
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("duration", [30, 30.001, 120])
def test_minimum_boundary_accepts_thirty_seconds(duration):
    require_review_duration(duration)


@pytest.mark.parametrize("detected,confidence", [("en", .3), ("it", .3), ("it", None)])
def test_uncertain_blah_transcript_cannot_claim_required_language(detected, confidence):
    tokens = ["Blah,"] * 10 + ["please."]
    transcription = {"text": " ".join(tokens), "detected_language": detected,
        "language_probability": confidence,
        "words": [{"text": token, "t0": index, "t1": index + .5} for index, token in enumerate(tokens)]}
    with mock.patch.object(assess_speaking, "load_audio_features", return_value={"duration_sec": 30, "pauses": []}), \
         mock.patch.object(assess_speaking, "transcribe", return_value=transcription), \
         mock.patch.object(assess_speaking, "generate_rubric", side_effect=LLMClientError("Synthetic provider unavailable")), \
         mock.patch.object(assess_speaking, "generate_coaching_summary") as coaching:
        result = assess_speaking.run_assessment(Path("synthetic.wav"), provider="ollama", llm_model="synthetic",
            expected_language="it", feedback_language="en", target_cefr="B1")
    report = result["report"]
    assert "language_detection_uncertain" in report["warnings"]
    assert report["checks"]["language_pass"] is None
    assert "You spoke in the required language." not in report["coaching"]["strengths"]
    assert "Complete the full task in Italian" not in " ".join(report["coaching"]["top_3_priorities"])
    assert result["baseline_comparison"]["valid"] is False
    coaching.assert_not_called()


@pytest.mark.parametrize("confirmed,content,expected", [(False, True, False), (True, None, False), (True, False, False), (True, True, True)])
def test_language_praise_requires_positive_content_and_language_evidence(confirmed, content, expected):
    feedback = build_fallback_coaching(metrics={"word_count": 11, "wpm": 80, "fillers": 0},
        checks={"language_pass": True, "language_detection_confident": confirmed, "content_validity_pass": content},
        theme="A synthetic topic", target_duration_sec=60, ui_locale="en", learning_language="it")
    assert ("You spoke in the required language." in feedback["strengths"]) is expected


def test_unknown_language_on_sparse_speech_is_not_a_wrong_language_diagnosis():
    feedback = build_fallback_coaching(metrics={"word_count": 3, "wpm": 80, "fillers": 0},
        checks={"language_pass": None, "content_validity_pass": None}, transcript="bla bla bla",
        theme="A synthetic topic", target_duration_sec=60, ui_locale="en", learning_language="it", detected_language="en")
    assert "detected as English" not in feedback["coach_summary"]
    assert "transcript is very short" in feedback["coach_summary"]
