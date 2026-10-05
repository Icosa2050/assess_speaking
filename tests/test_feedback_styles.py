"""Separate optional alternatives from error evidence, preserving historical reports."""
import json
from unittest import mock

import pytest

from assess_core.schemas import CoachingSummary, RubricResult, SchemaValidationError
from assessment_runtime import llm_client
from assessment_runtime.assessment_prompts import rubric_prompt, coaching_prompt
from assessment_runtime.output_validation import generation_json_schema, validate_coaching_generation, validate_rubric_generation
from tests.test_generation_validation import rubric_payload, coaching_payload


def style_payload():
    return rubric_payload() | {"style_suggestions": [{
        "original": "The trip was good", "suggestion": "The trip was enjoyable",
        "explanation": "An optional more specific adjective; the original is grammatical.",
    }]}


def test_optional_style_roundtrips_without_becoming_grammar_or_changing_scores():
    rubric = RubricResult.from_dict(style_payload())
    validate_rubric_generation(rubric, "The trip was good.")
    saved = json.loads(json.dumps(rubric.to_dict()))
    assert RubricResult.from_dict(saved).to_dict() == saved
    assert saved["recurring_grammar_errors"] == []
    assert saved["accuracy"] == 4
    assert saved["style_suggestions"][0]["suggestion"] == "The trip was enjoyable"
    assert "style_suggestions" not in RubricResult.from_dict(rubric_payload()).to_dict()


@pytest.mark.parametrize("field", ["original", "suggestion", "explanation"])
def test_blank_optional_fields_rejected(field):
    payload = style_payload()
    payload["style_suggestions"][0][field] = " "
    with pytest.raises(SchemaValidationError):
        validate_rubric_generation(RubricResult.from_dict(payload), "The trip was good.")


def test_fabricated_original_and_unchanged_alternative_rejected():
    for changes, error in [({"original": "An invented sentence"}, "not present"),
                           ({"suggestion": "THE TRIP WAS GOOD"}, "different optional")]:
        payload = style_payload()
        payload["style_suggestions"][0].update(changes)
        with pytest.raises(SchemaValidationError, match=error):
            validate_rubric_generation(RubricResult.from_dict(payload), "The trip was good.")


def test_same_quote_cannot_be_both_error_and_optional_style():
    payload = style_payload()
    payload["recurring_grammar_errors"] = [{"category": "word_order", "explanation": "An alleged error", "examples": ["The trip was good"]}]
    with pytest.raises(SchemaValidationError, match="both a grammar error or lexical gap and optional style"):
        validate_rubric_generation(RubricResult.from_dict(payload), "The trip was good.")


def test_optional_style_cannot_claim_audio_quality():
    payload = style_payload()
    payload["style_suggestions"][0]["explanation"] = "Your pronunciation is unclear."
    with pytest.raises(SchemaValidationError):
        validate_rubric_generation(RubricResult.from_dict(payload), "The trip was good.")


def test_optional_style_does_not_require_error_coaching_and_cannot_be_a_correction_priority():
    coaching = coaching_payload()
    validate_coaching_generation(CoachingSummary.from_dict(coaching), 90, style_payload())
    coaching["top_3_priorities"][0] = "Correct 'The trip was good'."
    with pytest.raises(SchemaValidationError, match="dedicated section"):
        validate_coaching_generation(CoachingSummary.from_dict(coaching), 90, style_payload())


def test_provider_schema_requires_separate_style_array():
    schema = generation_json_schema("rubric")
    assert "style_suggestions" in schema["required"]
    assert schema["properties"]["style_suggestions"]["items"]["required"] == ["original", "suggestion", "explanation"]


def test_invalid_style_is_repaired_through_production_client():
    invalid = style_payload()
    invalid["style_suggestions"][0]["original"] = "Invented text"
    with mock.patch.object(llm_client, "_chat_completion", side_effect=[json.dumps(invalid), json.dumps(style_payload())]) as chat:
        rubric, _ = llm_client.generate_rubric("ollama", "model", "prompt", transcript="The trip was good.")
    assert chat.call_count == 2
    assert rubric.style_suggestions[0].original == "The trip was good"


def test_prompt_preserves_spoken_register_and_separates_optional_advice():
    metrics = dict(duration_sec=90, speaking_time_sec=80, pause_total_sec=10, pause_count=4,
                   word_count=100, wpm=75, fillers=0, cohesion_markers=3, complexity_index=2)
    prompt = rubric_prompt("can save", metrics, expected_language="en")
    assert 'modal + base verb' in prompt
    assert 'indicative after volere che' not in prompt
    assert 'indicative after volere che' in rubric_prompt('vogliamo che', metrics, expected_language='it')
    assert 'never in recurring_grammar_errors or lexical_gaps' in prompt
    assert 'do not translate either original or suggestion' in prompt.lower()
    prompt = coaching_prompt(metrics, style_payload(), "Travel", 90)
    assert 'dedicated optional section' in prompt

@pytest.mark.parametrize('grammar_quote,style_quote', [('The trip was good', 'trip was good'), ('trip was good', 'The trip was good'), ('The trip was', 'trip was good')])
def test_overlapping_error_style_evidence_rejected(grammar_quote, style_quote):
    payload = style_payload()
    payload['style_suggestions'][0]['original'] = style_quote
    payload['recurring_grammar_errors'] = [{'category': 'word_order', 'explanation': 'An alleged error', 'examples': [grammar_quote]}]
    with pytest.raises(SchemaValidationError, match='overlapping evidence'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


def test_style_and_lexical_gap_cannot_share_evidence():
    payload = style_payload()
    payload['lexical_gaps'] = [{'category': 'travel_vocabulary_gap', 'explanation': 'An alleged gap', 'examples': ['The trip was good']}]
    with pytest.raises(SchemaValidationError, match='overlapping evidence'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


@pytest.mark.parametrize('explanation', ['The original is incorrect.', 'You made a grammar error here.', 'Correggi questo errore.', 'La frase originale è sbagliata.'])
def test_style_explanation_cannot_relabel_acceptable_original_as_error(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    with pytest.raises(SchemaValidationError, match='optional style'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


@pytest.mark.parametrize('field', ['next_focus', 'next_exercise', 'coach_summary', 'next_attempt_instruction'])
def test_style_example_stays_out_of_all_correction_coaching_fields(field):
    payload = coaching_payload()
    payload[field] = "Correct 'The trip was good'."
    with pytest.raises(SchemaValidationError, match='dedicated section'):
        validate_coaching_generation(CoachingSummary.from_dict(payload), 90, style_payload())


def test_local_generation_deliberately_accepts_legacy_omission_of_optional_style():
    # Remote strict schemas require the new array, but historical/local responses
    # without it carry no style claim and retain compatibility.
    with mock.patch.object(llm_client, '_chat_completion', return_value=json.dumps(rubric_payload())):
        rubric, _ = llm_client.generate_rubric('ollama', 'model', 'prompt', transcript='The trip was good.')
    assert rubric.style_suggestions is None


@pytest.mark.parametrize('explanation', ['The original is correct; this is an optional alternative.', 'This is not a grammar error.', 'La frase non è sbagliata; alternativa facoltativa.'])
def test_optional_style_allows_reassurance(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


def test_blank_error_example_with_style_fails_validation_not_alignment_code():
    payload = style_payload()
    payload['recurring_grammar_errors'] = [{'category': 'word_order', 'explanation': 'An alleged error', 'examples': ['']}]
    with pytest.raises(SchemaValidationError, match='nonblank'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


def test_error_vocabulary_inside_known_learner_quote_is_not_a_style_diagnosis():
    payload = style_payload()
    payload['style_suggestions'][0] = {'original': 'I made a mistake', 'suggestion': 'I got it wrong',
                                     'explanation': "'I made a mistake' is acceptable; this is an informal alternative."}
    validate_rubric_generation(RubricResult.from_dict(payload), 'I made a mistake yesterday.')


@pytest.mark.parametrize('explanation', [
    "Correct 'The trip was good' to 'The trip was enjoyable'.", 'This should be corrected.',
    'Fix this.', 'The original is incorrectly formed.', 'La frase è errata.', 'È uno sbaglio.',
    'Das Original ist ein Fehler.', 'Die Formulierung ist fehlerhaft.', 'Eine Korrektur ist nötig.',
])
def test_reviewed_correction_explanations_rejected(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    with pytest.raises(SchemaValidationError, match='optional style'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


@pytest.mark.parametrize('explanation', [
    'The original needs no correction; this is optional.', "This isn't an error.",
    'There is nothing wrong with the original.', 'The alternative is also error-free.',
    'The original does not need correction.', 'Nessun errore; alternativa facoltativa.',
    "Non c'è nessun errore.", 'La frase non è affatto sbagliata.',
    'Das Original ist nicht falsch; dies ist eine optionale Alternative.',
    'Das Original ist kein Fehler.', 'Keine Korrektur nötig.',
])
def test_reviewed_scoped_reassurance_allowed(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


@pytest.mark.parametrize('explanation', [
    "This isn't an error, but the verb is wrong.",
    'Nessun errore; tuttavia la frase è errata.',
    'Das Original ist nicht falsch. Korrigiere die Grammatik.',
])
def test_reassurance_does_not_hide_subsequent_criticism(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    with pytest.raises(SchemaValidationError, match='optional style'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


@pytest.mark.parametrize('original,suggestion', [
    ('I like the rhythm of this song', 'I enjoy the rhythm of this song'),
    ('Abbiamo fatto una sosta', 'Abbiamo fatto una pausa'),
])
def test_topic_vocabulary_in_replacement_speech_is_not_an_acoustic_assessment(original, suggestion):
    payload = style_payload()
    payload['style_suggestions'][0] = dict(original=original, suggestion=suggestion,
        explanation=f'Optional alternative: “{suggestion}”. The original is grammatical.')
    validate_rubric_generation(RubricResult.from_dict(payload), original)
    payload['style_suggestions'][0]['explanation'] = 'Your pronunciation is unclear.'
    with pytest.raises(SchemaValidationError, match='acoustic'):
        validate_rubric_generation(RubricResult.from_dict(payload), original)


def test_single_word_optional_item_does_not_reject_unrelated_praise():
    payload = style_payload()
    payload['style_suggestions'][0]['original'] = 'good'
    payload['style_suggestions'][0]['suggestion'] = 'enjoyable'
    coaching = coaching_payload() | {'strengths': ['Good supporting examples.']}
    validate_coaching_generation(CoachingSummary.from_dict(coaching), 90, payload)


@pytest.mark.parametrize('text', [
    "Replace 'good' with 'enjoyable' in your opening.", "Say 'The trip was enjoyable' instead.",
    'Use The trip was enjoyable in the opening.', 'Correct The trip was good.',
])
def test_partial_and_replacement_references_cannot_be_correction_priorities(text):
    coaching = coaching_payload()
    coaching['top_3_priorities'][0] = text
    with pytest.raises(SchemaValidationError, match='dedicated section'):
        validate_coaching_generation(CoachingSummary.from_dict(coaching), 90, style_payload())


def test_acceptable_optional_expression_can_be_a_strength():
    coaching = coaching_payload() | {'strengths': ["Clear opening: 'The trip was good'."]}
    validate_coaching_generation(CoachingSummary.from_dict(coaching), 90, style_payload())


def test_optional_source_content_uses_typographic_single_quotes():
    payload = style_payload()
    payload['style_suggestions'][0] = dict(original='I made a mistake', suggestion='I got it wrong',
        explanation='‘I made a mistake’ is acceptable; this is an informal alternative.')
    validate_rubric_generation(RubricResult.from_dict(payload), 'I made a mistake.')


def test_probe_exact_example_satisfies_the_required_schema_fields():
    with mock.patch.object(llm_client, '_chat_completion', return_value=json.dumps(rubric_payload())) as chat:
        llm_client.test_connection(provider='openrouter', model='fixture', api_key='test-placeholder')
    example = json.loads(chat.call_args.args[2].split('exactly: ', 1)[1])
    schema = chat.call_args.kwargs['extra_payload']['response_format']['json_schema']['schema']
    assert set(schema['required']) <= set(example)
    assert example['style_suggestions'] == []


@pytest.mark.parametrize('explanation', ['The original is not error-free.', 'The original is not free of errors.',
    "The original isn't mistake-free.", 'This cannot be used without correction.',
    'Ohne Korrektur ist der Satz unverständlich.', 'Senza correzione la frase non funziona.'])
def test_negated_reassurance_and_required_correction_are_not_optional(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    with pytest.raises(SchemaValidationError, match='optional style'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


@pytest.mark.parametrize('explanation', ['Both versions would be correct.', 'Correct as written.',
    'Correct and natural; the alternative is more vivid.', 'This does not change the meaning.',
    'Non cambia il significato.', "You could replace 'good' with 'enjoyable' as an optional choice.",
    'Das Original ist fehlerfrei.', 'Kein Grammatikfehler.', 'No corrections needed.', 'Non sono errori.'])
def test_style_explanation_can_describe_acceptable_replacement(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')


def test_content_quote_with_internal_apostrophe_is_exempt():
    payload = style_payload()
    payload['style_suggestions'][0] = dict(original="I don't make mistakes", suggestion='I avoid mistakes',
        explanation="'I don't make mistakes' is acceptable; an optional alternative.")
    validate_rubric_generation(RubricResult.from_dict(payload), "I don't make mistakes.")


def test_unchanged_fragment_in_style_does_not_block_unrelated_grammar_priority():
    payload = style_payload()
    payload['style_suggestions'][0] = dict(original='I went to a good beach', suggestion='I went to an enjoyable beach',
        explanation='Optional specificity.')
    payload['recurring_grammar_errors'] = [dict(category='verb_conjugation_past', explanation='Use the past tense went.', examples=['I goed'])]
    validate_rubric_generation(RubricResult.from_dict(payload), 'I went to a good beach. Yesterday I goed home.')
    coach = coaching_payload()
    coach['top_3_priorities'][0] = "Correct 'I goed' to 'I went'."
    validate_coaching_generation(CoachingSummary.from_dict(coach), 90, payload)



def test_style_word_inside_distinct_grammar_quote_does_not_block_priority():
    payload = style_payload()
    payload['style_suggestions'][0] = dict(original='è bello', suggestion='è splendido', explanation='Alternativa facoltativa.')
    payload['recurring_grammar_errors'] = [dict(category='adjective_agreement', explanation='Concorda il genere.', examples=['la casa bello'])]
    coach = coaching_payload()
    coach['top_3_priorities'][0] = "Correggi 'la casa bello': usa il femminile."
    validate_coaching_generation(CoachingSummary.from_dict(coach), 90, payload)


@pytest.mark.parametrize('explanation', ['This is not correct.', 'Non è corretto.', 'Nicht richtig.', 'This is inaccurate.'])
def test_negated_correctness_is_a_diagnosis_not_optional_style(explanation):
    payload = style_payload()
    payload['style_suggestions'][0]['explanation'] = explanation
    with pytest.raises(SchemaValidationError, match='optional style'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'The trip was good.')
