"""Small live adapter checks; opt-in failures must never become silent skips."""
import os
import re

import pytest

from assessment_runtime.assessment_prompts import coaching_prompt, rubric_prompt
from assessment_runtime.llm_client import generate_coaching_summary, generate_rubric
from assess_core.schemas import RubricResult


CASES = {
    "it": ("Il mio ultimo viaggio", "Lo scorso anno ho visitato il mare con mia sorella. Abbiamo viaggiato in treno e abbiamo visitato il centro storico. Ricordo il tramonto sulla spiaggia."),
    "en": ("My last trip", "Last summer I visited the coast with my sister. We travelled by train and explored the old town. I remember watching the sunset on the beach."),
}
METRICS = dict(duration_sec=30, speaking_time_sec=25, pause_total_sec=5, pause_count=3,
               word_count=30, wpm=72, fillers=0, cohesion_markers=2, complexity_index=0)


def assert_feedback_language(text, language):
    # A wrong-language smoke check, not a semantic quality classifier.
    markers = r"\b(?:il|la|le|un|una|che|di|per|con|nel|tuo|tua|hai|puoi|frasi)\b" if language == "it" else r"\b(?:the|your|you|and|with|use|try|to|for|this|that|was|were)\b"
    assert len(re.findall(markers, text.casefold())) >= 3, f"Expected {language} feedback: {text}"


def require_openrouter():
    if os.getenv("RUN_OPENROUTER_INTEGRATION") != "1":
        pytest.skip("Set RUN_OPENROUTER_INTEGRATION=1 for live provider checks")
    assert os.getenv("OPENROUTER_API_KEY"), "Enabled OpenRouter tests require OPENROUTER_API_KEY"
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    assert model, "Set OPENROUTER_MODEL to the model to verify (no implicit preview-model dependency)"
    return dict(provider="openrouter", model=model, openrouter_api_key=os.environ["OPENROUTER_API_KEY"], timeout_sec=180)


@pytest.mark.parametrize("language", CASES)
def test_generate_rubric_round_trip(language):
    connection = require_openrouter()
    theme, transcript = CASES[language]
    rubric, raw = generate_rubric(
        **connection, transcript=transcript,
        prompt=rubric_prompt(transcript, METRICS, theme, expected_language=language, feedback_language=language),
    )
    # Generation validates schema, claim policy and quote membership in the real source.
    for field in ("overall", "fluency", "accuracy", "range", "cohesion"):
        assert 1 <= getattr(rubric, field) <= 5
    assert rubric.overall_comment.strip()
    assert_feedback_language(" ".join(getattr(rubric, field) for field in (
        "comments_fluency", "comments_cohesion", "comments_accuracy", "comments_range", "overall_comment")), language)
    assert '"overall"' in raw


@pytest.mark.parametrize("language", CASES)
def test_generate_coaching_summary_round_trip(language):
    connection = require_openrouter()
    theme, transcript = CASES[language]
    comment = "La sequenza del racconto è comprensibile." if language == "it" else "The sequence of the story is understandable."
    rubric = RubricResult(
        fluency=3, cohesion=3, accuracy=3, range=3, overall=3,
        comments_fluency=comment, comments_cohesion=comment, comments_accuracy=comment,
        comments_range=comment, overall_comment=comment, on_topic=True,
        evidence_quotes=[transcript.split(".")[0]],
    ).to_dict()
    coaching, raw = generate_coaching_summary(
        **connection, rubric=rubric, target_duration_sec=90,
        prompt=coaching_prompt(METRICS, rubric, theme, 90, expected_language=language,
                               feedback_language=language, checks={"duration_pass": False}),
    )
    assert len(coaching.top_3_priorities) == 3
    assert all(priority.strip() for priority in coaching.top_3_priorities)
    assert coaching.next_exercise.strip()
    assert coaching.retry_duration_sec == 90
    assert coaching.next_attempt_instruction.strip()
    assert_feedback_language(" ".join((coaching.coach_summary, coaching.next_focus, coaching.next_exercise)), language)
    assert '"coach_summary"' in raw
