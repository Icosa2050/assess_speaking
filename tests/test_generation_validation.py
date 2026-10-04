"""Regression cases for output grounding; no live provider or quality oracle."""

import http.client
import json
from unittest import mock

import pytest

from assess_core.schemas import CoachingSummary, RubricResult, SchemaValidationError
from assessment_runtime import llm_client
from assessment_runtime.output_validation import UncertainTranscriptEvidenceError, validate_coaching_generation, validate_rubric_generation


def rubric_payload():
    return {
        **{key: 4 for key in ("fluency", "cohesion", "accuracy", "range", "overall", "topic_relevance_score")},
        **{key: "Assess text construction." for key in ("comments_fluency", "comments_cohesion", "comments_accuracy", "comments_range", "overall_comment")},
        "on_topic": True, "language_ok": True, "recurring_grammar_errors": [],
        "coherence_issues": [], "lexical_gaps": [], "evidence_quotes": [], "confidence": "medium",
    }


def coaching_payload():
    return {"strengths": [], "top_3_priorities": ["Add details", "Use examples", "Organize ideas"],
            "next_focus": "Add details", "next_exercise": "Prepare aloud with a 30-second drill.",
            "coach_summary": "Repeat the full task after preparation.",
            "next_attempt_instruction": "Record a 90-second answer with 3 examples.", "retry_duration_sec": 90}


@pytest.mark.parametrize("quote", ["", "   ", "sono andati", "I went home"])
def test_rejects_blank_fabricated_and_translated_evidence(quote):
    payload = rubric_payload()
    payload["evidence_quotes"] = [quote]
    with pytest.raises(SchemaValidationError):
        validate_rubric_generation(RubricResult.from_dict(payload), "Sono andato a casa.")


def test_grounding_handles_defined_normalization_and_prompt_escaping():
    payload = rubric_payload()
    payload["evidence_quotes"] = ["L'ULTIMA   SERA", "città", "'''ciao'''"]
    validate_rubric_generation(RubricResult.from_dict(payload), 'L’ultima sera nella citta\u0300 ho detto """ciao""".')


def test_grounding_does_not_remove_punctuation_or_fuzzily_match():
    payload = rubric_payload()
    payload["evidence_quotes"] = ["came home"]
    with pytest.raises(SchemaValidationError):
        validate_rubric_generation(RubricResult.from_dict(payload), "came, home")


def test_clean_no_error_output_is_valid():
    validate_rubric_generation(RubricResult.from_dict(rubric_payload()), "A correct sentence.")


@pytest.mark.parametrize("field,category", [("recurring_grammar_errors", "article_usage"), ("lexical_gaps", "travel_vocabulary_gap")])
def test_claimed_grammar_and_lexical_issues_need_examples(field, category):
    payload = rubric_payload()
    payload[field] = [{"category": category, "explanation": "Claimed error", "examples": []}]
    with pytest.raises(SchemaValidationError):
        validate_rubric_generation(RubricResult.from_dict(payload), "A correct sentence.")


def test_coherence_absence_can_have_no_example_but_not_blank_example():
    payload = rubric_payload()
    payload["coherence_issues"] = [{"category": "underdeveloped_detail", "explanation": "No supporting detail", "examples": []}]
    validate_rubric_generation(RubricResult.from_dict(payload), "I agree.")
    payload["coherence_issues"][0]["examples"] = [""]
    with pytest.raises(SchemaValidationError):
        validate_rubric_generation(RubricResult.from_dict(payload), "I agree.")


def test_grounding_retries_then_fails_closed_without_returning_scores():
    payload = rubric_payload()
    payload["evidence_quotes"] = ["invented phrase"]
    with mock.patch.object(llm_client, "_chat_completion", return_value=json.dumps(payload)) as chat:
        with pytest.raises(llm_client.LLMSchemaError):
            llm_client.generate_rubric("ollama", "model", "prompt", transcript="Actual transcript")
    assert chat.call_count == 2
    assert "ATTENZIONE" not in chat.call_args.args[2]


def test_grounding_repair_can_return_corrected_second_response():
    invalid = rubric_payload()
    invalid["evidence_quotes"] = ["fabricated"]
    with mock.patch.object(llm_client, "_chat_completion", side_effect=[json.dumps(invalid), json.dumps(rubric_payload())]):
        rubric, _ = llm_client.generate_rubric("ollama", "model", "prompt", transcript="Actual transcript")
    assert rubric.evidence_quotes == []


@pytest.mark.parametrize("field", ["next_focus", "next_exercise", "coach_summary"])
def test_blank_generated_coaching_rejected_but_legacy_payload_loads(field):
    payload = coaching_payload()
    payload[field] = "  "
    coaching = CoachingSummary.from_dict(payload)
    with pytest.raises(SchemaValidationError):
        validate_coaching_generation(coaching, 90)


def test_historical_coaching_has_no_new_required_fields():
    payload = coaching_payload()
    payload.pop("next_attempt_instruction")
    payload.pop("retry_duration_sec")
    coaching = CoachingSummary.from_dict(payload)
    assert coaching.next_attempt_instruction is None
    assert coaching.to_dict() == payload


@pytest.mark.parametrize("instruction", [
    "Record for 90 seconds with 3 connectors.", "Registra per 90 secondi.",
    "Sprich 90 Sekunden.", "Record for 1.5 minutes.", "Parla per 1,5 minuti.",
    "Record for 90s.", "Record for 90 s.", "Record for 90 sec.", "Sprich 90 Sek.",
    "Habla durante 90 segundos.", "Parlez pendant 90 secondes.",
    "Record for 1 minute 30 seconds.", "Registra per 1 minuto e 30 secondi.",
    "Sprich 1 Minute und 30 Sekunden.", "Habla durante 1 minuto y 30 segundos.",
    "Parlez pendant 1 minute et 30 secondes.",
])
def test_retry_duration_accepts_languages_equivalent_minutes_and_short_scaffolds(instruction):
    payload = coaching_payload()
    payload["next_attempt_instruction"] = instruction
    validate_coaching_generation(CoachingSummary.from_dict(payload), 90)


@pytest.mark.parametrize("instruction", ["Record a 90-second answer; aim for 15 seconds.", "Record for 30 seconds.", "Try again with 3 connectors.", ""])
def test_retry_instruction_rejects_missing_or_conflicting_duration(instruction):
    payload = coaching_payload()
    payload["next_attempt_instruction"] = instruction
    with pytest.raises(SchemaValidationError):
        validate_coaching_generation(CoachingSummary.from_dict(payload), 90)


@pytest.mark.parametrize("duration", [None, 30, True, float("nan"), float("inf")])
def test_retry_structured_duration_rejects_wrong_or_invalid_values(duration):
    payload = coaching_payload()
    payload["retry_duration_sec"] = duration
    with pytest.raises(SchemaValidationError):
        validate_coaching_generation(CoachingSummary.from_dict(payload), 90)


@pytest.mark.parametrize("method", ["_post_json", "_get_json"])
@pytest.mark.parametrize("failure", [ConnectionResetError(), http.client.IncompleteRead(b"partial"), UnicodeDecodeError("utf8", b"\xff", 0, 1, "invalid")])
def test_read_and_decode_failures_become_provider_errors(method, failure):
    response = mock.Mock()
    response.read.side_effect = failure
    with mock.patch.object(llm_client.request, "urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = response
        args = ("https://provider.invalid", {}, {}, 1) if method == "_post_json" else ("https://provider.invalid", {}, 1)
        with pytest.raises(llm_client.LLMClientError, match="could not be read"):
            getattr(llm_client, method)(*args)


def test_actual_invalid_utf8_body_and_connection_reset_are_normalized():
    response = mock.Mock()
    response.read.return_value = b"\xff"
    with mock.patch.object(llm_client.request, "urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = response
        with pytest.raises(llm_client.LLMClientError, match="UnicodeDecodeError"):
            llm_client._get_json("https://provider.invalid", {}, 1)
        urlopen.side_effect = ConnectionResetError()
        with pytest.raises(llm_client.LLMClientError, match="ConnectionResetError"):
            llm_client._get_json("https://provider.invalid", {}, 1)


def test_openrouter_generation_uses_strict_schema_and_required_parameter_routing():
    with mock.patch.object(llm_client, "_post_json", return_value={"choices": [{"message": {"content": json.dumps(rubric_payload())}}]}) as post:
        llm_client.generate_rubric("openrouter", "model", "prompt", openrouter_api_key="test-placeholder", transcript="Speech")
    payload = post.call_args.args[1]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["provider"] == {"require_parameters": True}


def test_strict_generation_cannot_silently_downgrade_schema():
    with mock.patch.object(llm_client, "_post_json", side_effect=llm_client.LLMClientError("response_format unsupported")) as post:
        with pytest.raises(llm_client.LLMClientError):
            llm_client.generate_rubric("openrouter", "model", "prompt", openrouter_api_key="test-placeholder")
    assert post.call_count == 1


def test_openrouter_connection_probe_cannot_accept_plain_ok():
    with mock.patch.object(llm_client, "_chat_completion", return_value="OK"):
        with pytest.raises(llm_client.LLMSchemaError, match="capability probe"):
            llm_client.test_connection(provider="openrouter", model="model", api_key="test-placeholder")


def test_openrouter_connection_probe_preserves_strict_failure_and_timeout():
    with mock.patch.object(llm_client, "_post_json", side_effect=llm_client.LLMClientError("json_schema unsupported")) as post:
        with pytest.raises(llm_client.LLMClientError, match="unsupported"):
            llm_client.test_connection(provider="openrouter", model="model", api_key="test-placeholder", timeout_sec=7)
    assert post.call_count == 1
    _, payload, _, timeout = post.call_args.args
    assert timeout == 7
    assert payload["response_format"]["json_schema"]["schema"] == llm_client.generation_json_schema("rubric")


@pytest.mark.parametrize("instruction", ["Record for 1 minute. Later speak 30 seconds.", "Record for 1 minute or 30 seconds.", "Record for 1 minute 90 seconds."])
def test_separate_or_invalid_compound_durations_are_still_rejected(instruction):
    payload = coaching_payload()
    payload["next_attempt_instruction"] = instruction
    with pytest.raises(SchemaValidationError):
        validate_coaching_generation(CoachingSummary.from_dict(payload), 90)


def test_spanish_minute_unit_supports_configured_two_minute_retry():
    payload = coaching_payload()
    payload.update(next_attempt_instruction="Habla durante 2 minutos.", retry_duration_sec=120)
    validate_coaching_generation(CoachingSummary.from_dict(payload), 120)


def _issue_payload(field="recurring_grammar_errors", quote="hanno"):
    payload = rubric_payload()
    category = {"recurring_grammar_errors": "verb_conjugation_past",
                "lexical_gaps": "travel_vocabulary_gap", "coherence_issues": "unclear_reference"}[field]
    payload[field] = [{"category": category, "explanation": "A claimed issue", "examples": [quote]}]
    return payload


def _asr_words(low=False):
    return [{"text": "Lo", "probability": 0.99}, {"text": "scorso", "probability": 0.99},
            {"text": "hanno", "probability": 0.2 if low else 0.99},
            {"text": "siamo", "probability": 0.99}, {"text": "andati.", "probability": 0.99}]


@pytest.mark.parametrize("field", ["recurring_grammar_errors", "lexical_gaps", "coherence_issues"])
def test_issue_quote_cannot_assign_low_confidence_asr_word_to_learner(field):
    with pytest.raises(UncertainTranscriptEvidenceError, match="low-confidence"):
        validate_rubric_generation(RubricResult.from_dict(_issue_payload(field)), "Lo scorso hanno siamo andati.", _asr_words(low=True))


def test_confident_quote_and_unquoted_low_confidence_word_are_distinct():
    validate_rubric_generation(RubricResult.from_dict(_issue_payload()), "Lo scorso hanno siamo andati.", _asr_words())
    validate_rubric_generation(RubricResult.from_dict(_issue_payload(quote="siamo andati")), "Lo scorso hanno siamo andati.", _asr_words(low=True))


@pytest.mark.parametrize("words", [None, [{"text": "not aligned"}], [{"text": "not aligned", "probability": None}]])
def test_missing_word_confidence_remains_unknown_and_legacy_compatible(words):
    validate_rubric_generation(RubricResult.from_dict(_issue_payload()), "Lo scorso hanno siamo andati.", words)


def test_observed_confidence_with_failed_alignment_is_uncertain():
    with pytest.raises(UncertainTranscriptEvidenceError, match="alignment"):
        validate_rubric_generation(RubricResult.from_dict(_issue_payload()), "Lo scorso hanno siamo andati.", [{"text": "unmatched", "probability": 0.99}])


def test_quote_mapping_gap_is_uncertain_even_when_other_words_align():
    with pytest.raises(UncertainTranscriptEvidenceError, match="alignment"):
        validate_rubric_generation(RubricResult.from_dict(_issue_payload()), "Lo scorso hanno siamo andati.", [{"text": "Lo", "probability": 0.99}])


def test_asr_alignment_normalizes_case_spaces_and_apostrophes_without_fuzzy_matching():
    validate_rubric_generation(RubricResult.from_dict(_issue_payload(quote="l'ultima sera")), "L’ultima   sera.",
                               [{"text": "l'ultima", "probability": 0.99}, {"text": "sera.", "probability": 0.99}])


def test_repeated_quote_cannot_choose_only_confident_occurrence():
    with pytest.raises(UncertainTranscriptEvidenceError, match="low-confidence"):
        validate_rubric_generation(RubricResult.from_dict(_issue_payload(quote="hanno")), "hanno hanno",
                                   [{"text": "hanno", "probability": 0.99}, {"text": "hanno", "probability": 0.2}])


def test_uncertain_issue_generation_retries_and_has_distinct_final_exception():
    with mock.patch.object(llm_client, "_chat_completion", return_value=json.dumps(_issue_payload())) as chat:
        with pytest.raises(llm_client.LLMTranscriptUncertaintyError):
            llm_client.generate_rubric("ollama", "model", "prompt", transcript="Lo scorso hanno siamo andati.", asr_words=_asr_words(low=True))
    assert chat.call_count == 2


def test_retry_can_omit_uncertain_claim_without_removing_its_score_after_acceptance():
    with mock.patch.object(llm_client, "_chat_completion", side_effect=[json.dumps(_issue_payload()), json.dumps(rubric_payload())]):
        rubric, _ = llm_client.generate_rubric("ollama", "model", "prompt", transcript="Lo scorso hanno siamo andati.", asr_words=_asr_words(low=True))
    assert rubric.recurring_grammar_errors == []


def test_transcript_evidence_without_error_diagnosis_is_not_a_global_word_gate():
    payload = rubric_payload()
    payload["evidence_quotes"] = ["hanno"]
    validate_rubric_generation(RubricResult.from_dict(payload), "Lo scorso hanno siamo andati.", _asr_words(low=True))


def test_short_issue_word_does_not_match_inside_unrelated_uncertain_word():
    payload = _issue_payload(quote="a")
    words = [{"text": "Sono", "probability": 0.99}, {"text": "andato", "probability": 0.2},
             {"text": "a", "probability": 0.99}, {"text": "casa.", "probability": 0.99}]
    validate_rubric_generation(RubricResult.from_dict(payload), "Sono andato a casa.", words)


def test_midword_issue_fragment_cannot_skip_confidence_validation():
    with pytest.raises(SchemaValidationError, match="complete-word"):
        validate_rubric_generation(RubricResult.from_dict(_issue_payload(quote="ndat")), "Sono andato.",
                                   [{"text": "Sono", "probability": 0.99}, {"text": "andato.", "probability": 0.99}])


def test_unrelated_unmatched_and_empty_asr_words_do_not_invalidate_aligned_quote():
    words = [{"text": "", "probability": 0.99}, {"text": "unmatched", "probability": 0.1},
             {"text": "Sono", "probability": 0.99}, {"text": "andato.", "probability": 0.99}]
    validate_rubric_generation(RubricResult.from_dict(_issue_payload(quote="andato")), "Sono andato.", words)


@pytest.mark.parametrize("second_response", ['{"invalid":true}', json.dumps(rubric_payload() | {"evidence_quotes": ["fabricated"]})])
def test_uncertainty_survives_different_final_validation_failure(second_response):
    with mock.patch.object(llm_client, "_chat_completion", side_effect=[json.dumps(_issue_payload()), second_response]):
        with pytest.raises(llm_client.LLMTranscriptUncertaintyError, match="low-confidence"):
            llm_client.generate_rubric("ollama", "model", "prompt", transcript="Lo scorso hanno siamo andati.", asr_words=_asr_words(low=True))


def test_overlapping_repeated_quotes_include_uncertain_last_occurrence():
    with pytest.raises(UncertainTranscriptEvidenceError, match="low-confidence"):
        validate_rubric_generation(RubricResult.from_dict(_issue_payload(quote="a a")), "a a a",
                                   [{"text": "a", "probability": 0.99}, {"text": "a", "probability": 0.99}, {"text": "a", "probability": 0.2}])


@pytest.mark.parametrize("provider", ["ollama", "lmstudio", "openai_compatible"])
def test_cloud_key_is_never_used_for_local_or_compatible_provider(provider):
    with mock.patch.object(llm_client, "_post_json", return_value={"choices": [{"message": {"content": "OK"}}]}) as post:
        llm_client._chat_completion(provider, "model", "prompt", 1, "cloud-placeholder", base_url="http://localhost:1234/v1")
    assert "Authorization" not in post.call_args.args[2]


@pytest.mark.parametrize("quote", ["parl", "arlo", "arl"])
def test_evidence_cannot_use_partial_words_without_asr_confidence(quote):
    payload = rubric_payload() | {"evidence_quotes": [quote]}
    with pytest.raises(SchemaValidationError, match="complete-word"):
        validate_rubric_generation(RubricResult.from_dict(payload), "parlo")


@pytest.mark.parametrize("choice,match", [
    ({"finish_reason": "length", "message": {"content": "{}"}}, "truncated"),
    ({"message": {"content": "{}", "refusal": "Declined"}}, "refused"),
    ({"message": {"reasoning": '{"draft":true}'}}, "no assistant text"),
])
def test_incomplete_or_refused_provider_output_is_not_an_answer(choice, match):
    with pytest.raises(llm_client.LLMClientError, match=match):
        llm_client._extract_assistant_message_text({"choices": [choice]})


def test_feedback_prompts_keep_acoustic_metrics_out_of_text_diagnosis():
    from assessment_runtime.assessment_prompts import rubric_prompt, coaching_prompt
    metrics = {"duration_sec": 70, "pause_count": 9001, "wpm": 9002, "speaking_time_sec": 9003,
               "pause_total_sec": 9004, "fillers": 9005, "cohesion_markers": 9006, "complexity_index": 9007, "word_count": 9008}
    for prompt in [rubric_prompt("A transcript", metrics), coaching_prompt(metrics, rubric_payload(), "city", 90)]:
        assert all(str(number) not in prompt for number in range(9001, 9009))
    assert "continuity of ideas" in rubric_prompt("A transcript", metrics)


@pytest.mark.parametrize("kind", ["reasoning", "thinking", "refusal", "unknown"])
def test_non_answer_content_blocks_are_not_parsed_as_json(kind):
    with pytest.raises(llm_client.LLMClientError, match="no assistant text"):
        llm_client._extract_assistant_message_text({"choices": [{"message": {"content": [{"type": kind, "text": '{"draft": true}'}]}}]})


def test_coaching_may_qualify_a_rubric_diagnosis_without_prescribing_a_false_correction():
    rubric = rubric_payload()
    rubric['recurring_grammar_errors'] = [{
        'category': 'verb_conjugation_present', 'explanation': 'Alleged agreement error.',
        'examples': ['can save time'],
    }]
    coaching = coaching_payload()
    coaching['top_3_priorities'][0] = "The diagnosis for 'can save time' needs checking; practise explaining the benefit aloud."
    validate_coaching_generation(CoachingSummary.from_dict(coaching), 90, rubric)
