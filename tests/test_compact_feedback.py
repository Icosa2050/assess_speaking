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


def test_quote_and_duration_are_grounded():
    data = payload()
    data["fixes"][0]["you_said"] = "I went to Rome"
    with pytest.raises(SchemaValidationError, match="quote the transcript"):
        validate(data)
    data = payload()
    data["next_step"] = "Record a 30 seconds retry."
    with pytest.raises(SchemaValidationError, match="configured full spoken retry"):
        validate(data)


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
