import json
from unittest import mock

import pytest

from assess_core.schemas import CompactFeedback, SchemaValidationError
from assessment_runtime.llm_client import LLMSchemaError, generate_coaching_summary
from assessment_runtime.output_validation import generation_json_schema, validate_compact_generation


TRANSCRIPT = "I goes to Rome and I visit a museum."
RUBRIC = {"recurring_grammar_errors": [{"examples": ["I goes to Rome"]}]}


def payload():
    return {
        "summary": "Your answer has a clear topic.",
        "fixes": [{"you_said": "I goes to Rome", "better": "I go to Rome.",
                   "why": "Use the base verb with I."}],
        "next_step": "Record a 90 seconds retry and use one clear opening sentence.",
        "feedback_language": "en",
        "practice_language": "en",
    }


def validate(data):
    validate_compact_generation(CompactFeedback.from_dict(data), transcript=TRANSCRIPT,
                                feedback_language="en", practice_language="en",
                                target_duration_sec=90, rubric=RUBRIC)


def test_valid_compact_feedback_and_provider_schema():
    validate(payload())
    schema = generation_json_schema("coaching", feedback_language="en", practice_language="en")
    assert schema["additionalProperties"] is False
    assert schema["properties"]["fixes"]["maxItems"] == 3
    assert schema["properties"]["summary"]["maxLength"] == 160


@pytest.mark.parametrize("field", ["summary", "next_step"])
def test_overlong_text_is_rejected(field):
    data = payload()
    data[field] = "word " * 60
    with pytest.raises(SchemaValidationError, match="exceeds"):
        validate(data)


def test_overlong_fix_and_fourth_fix_are_rejected():
    data = payload()
    data["fixes"][0]["why"] = "because " * 30
    with pytest.raises(SchemaValidationError, match=r"fixes\[0\].why: exceeds"):
        validate(data)
    data = payload()
    data["fixes"] *= 4
    with pytest.raises(SchemaValidationError, match="at most 3"):
        validate(data)


@pytest.mark.parametrize("field", ["summary", "next_step", "fixes", "feedback_language"])
def test_missing_field_is_rejected(field):
    data = payload()
    del data[field]
    with pytest.raises(SchemaValidationError, match="missing keys"):
        validate(data)


def test_extra_fields_are_rejected_at_both_levels():
    data = payload()
    data["extra"] = "rambling"
    with pytest.raises(SchemaValidationError, match="extra keys"):
        validate(data)
    data = payload()
    data["fixes"][0]["extra"] = "rambling"
    with pytest.raises(SchemaValidationError, match="extra keys"):
        validate(data)


def test_wrong_explanation_and_example_languages_are_rejected():
    data = payload()
    data["summary"] = "Die Antwort ist klar und gut."
    with pytest.raises(SchemaValidationError, match="wrong language"):
        validate(data)
    data = payload()
    data["fixes"][0]["better"] = "Du gehst nach Rom mit dem Zug."
    with pytest.raises(SchemaValidationError, match="wrong language"):
        validate(data)
    data = payload()
    data["feedback_language"] = "de"
    with pytest.raises(SchemaValidationError, match="expected en"):
        validate(data)


@pytest.mark.parametrize("locale, summary, next_step, why", [
    ("en", "Your answer scores 3.5 on this scale.", "Record a 90 seconds retry with a clear opening.", "Use the base verb with I."),
    ("de", "Die Antwort hat z. B. eine klare Reihenfolge.", "Sprich 90 Sekunden und beginne mit einem Beispiel.", "Verwende die Grundform nach I."),
    ("it", "La risposta del Sig. Rossi è chiara.", "Registra un nuovo tentativo di 90 secondi con un'apertura chiara.", "Usa la forma base dopo I."),
    ("fr", "La réponse est claire et vaut 3.5 points.", "Enregistre un nouvel essai de 90 secondes avec une ouverture claire.", "Utilise la forme de base après I."),
    ("es", "La respuesta conecta ideas, p. es. con ejemplos.", "Graba otro intento de 90 segundos con una frase inicial clara.", "Usa la forma base después de I."),
])
def test_valid_localized_feedback_and_sentence_abbreviations(locale, summary, next_step, why):
    data = payload()
    data.update(summary=summary, next_step=next_step, feedback_language=locale)
    data["fixes"][0]["why"] = why
    validate_compact_generation(CompactFeedback.from_dict(data), transcript=TRANSCRIPT,
                                feedback_language=locale, practice_language="en",
                                target_duration_sec=90, rubric=RUBRIC)


def test_practice_words_quoted_in_localized_explanation_are_not_language_evidence():
    data = payload()
    data.update(summary="La risposta è chiara.",
                next_step="Registra un nuovo tentativo di 90 secondi.",
                feedback_language="it")
    data["fixes"] = [{"you_said": "your the story", "better": "I like your story.",
                      "why": "Non usare 'the' dopo 'your'."}]
    transcript = "I like your the story."
    rubric = {"recurring_grammar_errors": [{"examples": ["your the story"]}]}
    validate_compact_generation(CompactFeedback.from_dict(data), transcript=transcript,
                                feedback_language="it", practice_language="en",
                                target_duration_sec=90, rubric=rubric)
    data["fixes"][0]["why"] = "Use 'the' with 'your' because the phrase is clear."
    with pytest.raises(SchemaValidationError, match="wrong language"):
        validate_compact_generation(CompactFeedback.from_dict(data), transcript=transcript,
                                    feedback_language="it", practice_language="en",
                                    target_duration_sec=90, rubric=rubric)


def test_second_sentence_is_rejected():
    data = payload()
    data["summary"] = "Your answer has a clear topic. Add detail next."
    with pytest.raises(SchemaValidationError, match="one sentence"):
        validate(data)


def test_summary_cannot_contradict_retained_grammar_finding():
    data = payload()
    data["fixes"] = []
    data["summary"] = "Your grammar is error-free."
    with pytest.raises(SchemaValidationError, match="contradicts the retained grammar"):
        validate(data)


def test_compact_fix_cannot_promote_optional_style_to_error():
    data = payload()
    data["fixes"][0]["why"] = "Replace 'I visit' with 'I toured' to correct your style."
    rubric = {**RUBRIC, "style_suggestions": [{"original": "I visit", "suggestion": "I toured"}]}
    with pytest.raises(SchemaValidationError, match="optional style suggestion"):
        validate_compact_generation(CompactFeedback.from_dict(data), transcript=TRANSCRIPT,
                                    feedback_language="en", practice_language="en",
                                    target_duration_sec=90, rubric=rubric)


def test_unsupported_practice_language_skips_provider_before_call():
    with mock.patch("assessment_runtime.llm_client._chat_completion") as completion:
        with pytest.raises(LLMSchemaError, match="unavailable for practice language de"):
            generate_coaching_summary("ollama", "fixture", "prompt", rubric=RUBRIC,
                                      transcript=TRANSCRIPT, target_duration_sec=90,
                                      feedback_language="en", practice_language="de")
    completion.assert_not_called()


def test_quote_and_duration_are_grounded():
    data = payload()
    data["fixes"][0]["you_said"] = "I went to Rome"
    with pytest.raises(SchemaValidationError, match="quote the transcript"):
        validate(data)
    data = payload()
    data["next_step"] = "Record a 30 seconds retry."
    with pytest.raises(SchemaValidationError, match="configured full spoken retry"):
        validate(data)


def test_compact_fix_may_quote_a_grounded_phrase_from_a_long_rubric_example():
    data = payload()
    transcript = "Every summer I goes to Rome with my family, and we always visit the same small museum near the station before dinner."
    rubric = {"recurring_grammar_errors": [{"examples": [transcript]}]}
    validate_compact_generation(CompactFeedback.from_dict(data), transcript=transcript,
                                feedback_language="en", practice_language="en",
                                target_duration_sec=90, rubric=rubric)
    data["fixes"][0]["you_said"] = "I"
    with pytest.raises(SchemaValidationError, match="no supported rubric finding"):
        validate_compact_generation(CompactFeedback.from_dict(data), transcript=transcript,
                                    feedback_language="en", practice_language="en",
                                    target_duration_sec=90, rubric=rubric)


def test_unsupported_fix_and_nonspoken_step_are_rejected():
    data = payload()
    with pytest.raises(SchemaValidationError, match="no supported rubric finding"):
        validate_compact_generation(CompactFeedback.from_dict(data), transcript=TRANSCRIPT,
                                    feedback_language="en", practice_language="en",
                                    target_duration_sec=90, rubric={})
    data["next_step"] = "Write a 90 seconds retry with one clear opening sentence."
    with pytest.raises(SchemaValidationError, match="spoken action"):
        validate(data)


def test_one_repair_only_then_fail_closed():
    bad = {**payload(), "summary": "x" * 200}
    with mock.patch("assessment_runtime.llm_client._chat_completion",
                    side_effect=[json.dumps(bad), json.dumps(payload())]) as completion:
        report, _ = generate_coaching_summary("ollama", "fixture", "prompt", rubric=RUBRIC,
                                              transcript=TRANSCRIPT, target_duration_sec=90,
                                              feedback_language="en", practice_language="en")
    assert report.summary == payload()["summary"]
    assert completion.call_count == 2
    assert "exceeds 160" in completion.call_args.args[2]
    with mock.patch("assessment_runtime.llm_client._chat_completion", return_value=json.dumps(bad)) as completion:
        with pytest.raises(LLMSchemaError, match="exceeds 160"):
            generate_coaching_summary("ollama", "fixture", "prompt", rubric=RUBRIC,
                                      transcript=TRANSCRIPT, target_duration_sec=90,
                                      feedback_language="en", practice_language="en")
    assert completion.call_count == 2
