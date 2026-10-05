"""Known claim/consistency regressions, not a semantic language evaluator."""
import json
from unittest import mock

import pytest

from assess_core.schemas import CoachingSummary, RubricResult, SchemaValidationError
from assessment_runtime.feedback_claims import validate_claim_text
from assessment_runtime.output_validation import validate_coaching_generation, validate_rubric_generation
from assessment_runtime import llm_client
from tests.test_generation_validation import rubric_payload, coaching_payload


@pytest.mark.parametrize('text', [
    'The pronunciation is clear.', 'The speaker has minimal hesitation.',
    'No noticeable pauses.', 'Pronunciation is clear and rhythm is unknown.', 'The rhythm is natural.',
    'La pronuncia è chiara.', 'Non ci sono esitazioni evidenti.',
    'La velocità di eloquio è adeguata.', 'La velocità di produzione è adeguata.', 'La velocità di emissione è adeguata.', 'Das Sprechtempo ist natürlich.',
    'Cannot assess pronunciation from text, but the accent is clear.',
])
def test_known_acoustic_claims_fail(text):
    with pytest.raises(SchemaValidationError, match='acoustic'):
        validate_claim_text(text, 'comments_fluency')


@pytest.mark.parametrize('text', [
    'The sentences are logically connected.',
    'Cannot assess pronunciation from text alone.',
    'Pronunciation cannot be assessed without audio.',
    'Non è possibile valutare la pronuncia dal testo.',
    'Die Aussprache kann nicht bewertet werden.',
])
def test_text_structure_and_explicit_limitations_are_allowed(text):
    validate_claim_text(text, 'comment')


def error_rubric():
    payload = rubric_payload()
    payload['recurring_grammar_errors'] = [{'category': 'verb_conjugation_present',
        'explanation': 'Use the base verb after can.', 'examples': ['can saves']}]
    return payload


@pytest.mark.parametrize('text', ['No grammatical errors observed.', 'La grammatica è corretta.', 'Non ci sono errori grammaticali.', 'Non ci sono errori.', 'No grammar errors and other vocabulary is rich.'])
def test_accuracy_cannot_deny_retained_errors(text):
    payload = error_rubric()
    payload['comments_accuracy'] = text
    with pytest.raises(SchemaValidationError, match='contradicts'):
        validate_rubric_generation(RubricResult.from_dict(payload), 'Working from home can saves time.')


def test_learner_quote_is_not_an_assessor_acoustic_claim():
    payload = rubric_payload()
    payload['evidence_quotes'] = ['My pronunciation is clear']
    validate_rubric_generation(RubricResult.from_dict(payload), 'My pronunciation is clear')


def test_coaching_requires_every_grammar_finding_in_priorities():
    payload = coaching_payload()
    with pytest.raises(SchemaValidationError, match='reference a quoted example'):
        validate_coaching_generation(CoachingSummary.from_dict(payload), 90, error_rubric())
    payload['top_3_priorities'][0] = "Correct 'can saves' to 'can save'."
    validate_coaching_generation(CoachingSummary.from_dict(payload), 90, error_rubric())
    payload['coach_summary'] = 'No grammar errors observed.'
    with pytest.raises(SchemaValidationError, match='contradicts'):
        validate_coaching_generation(CoachingSummary.from_dict(payload), 90, error_rubric())


def test_rubric_claim_rejection_repairs_then_fails_closed():
    payload = rubric_payload()
    payload['comments_fluency'] = 'The rhythm is natural.'
    with mock.patch.object(llm_client, '_chat_completion', return_value=json.dumps(payload)) as chat:
        with pytest.raises(llm_client.LLMSchemaError, match='acoustic'):
            llm_client.generate_rubric('ollama', 'model', 'prompt', transcript='Correct text.')
    assert chat.call_count == 2


def test_coaching_client_receives_rubric_and_repairs_missing_priority():
    missing = coaching_payload()
    fixed = coaching_payload()
    fixed['top_3_priorities'][0] = "Correct 'can saves' to 'can save'."
    with mock.patch.object(llm_client, '_chat_completion', side_effect=[json.dumps(missing), json.dumps(fixed)]) as chat:
        result, _ = llm_client.generate_coaching_summary('ollama', 'model', 'prompt', target_duration_sec=90, rubric=error_rubric())
    assert chat.call_count == 2
    assert 'can saves' in result.top_3_priorities[0]


@pytest.mark.parametrize('text', [
    'Pronunciation cannot be assessed from text alone. However, it sounds natural.',
    'Well-paced and accent-free.', 'Akzentfrei.', 'Non c’è nessun errore.',
])
def test_reviewed_bypasses_fail(text):
    with pytest.raises(SchemaValidationError):
        validate_claim_text(text, 'comment', grammar_errors=True)


@pytest.mark.parametrize('text', [
    'The story describes parcel delivery.', 'Discuss the pace of life.',
    'Explain the rhythm of city life.', 'Manca un accento grafico.',
    "Non è possibile valutare l'intonazione dal testo.",
    'Pronunciation, rhythm, and intonation cannot be assessed from the transcript.',
    'Die Aussprache kann anhand des Textes nicht bewertet werden.',
])
def test_topics_orthography_and_limitation_variants_are_allowed(text):
    validate_claim_text(text, 'comment')


def test_only_known_learner_quotes_are_exempt():
    quote = 'I has a strong accent'
    payload = error_rubric()
    payload['recurring_grammar_errors'][0]['examples'] = [quote]
    coach = coaching_payload()
    coach['top_3_priorities'][0] = f"Correct '{quote}' to 'I have a strong accent'."
    # The correction itself contains the topic noun too: direct quote context is
    # known only for observed evidence. Use the inflected verb as the correction.
    coach['top_3_priorities'][0] = f"Correct '{quote}': use 'have', not 'has'."
    validate_coaching_generation(CoachingSummary.from_dict(coach), 90, payload)
    with pytest.raises(SchemaValidationError, match='acoustic'):
        validate_claim_text("The 'pronunciation is clear'", 'comment', known_quotes=[quote])


def test_qualified_finding_goals_and_whole_word_priority_coverage():
    validate_claim_text("Apart from 'can saves', no errors.", 'comment', grammar_errors=True,
                        known_quotes=['can saves'], grammar_examples=['can saves'])
    validate_claim_text('Aim for no grammatical errors in your next attempt.', 'exercise', grammar_errors=True)
    with pytest.raises(SchemaValidationError, match='contradicts'):
        validate_claim_text('No errors except spelling.', 'comment', grammar_errors=True, grammar_examples=['can saves'])
    rubric = error_rubric()
    rubric['recurring_grammar_errors'][0]['examples'] = ['a']
    with pytest.raises(SchemaValidationError, match='reference'):
        validate_coaching_generation(CoachingSummary.from_dict(coaching_payload()), 90, rubric)


@pytest.mark.parametrize("text", ["Il testo parla della pace nel mondo.", "Prova ad accentuare il contrasto tra le idee."])
def test_italian_topic_words_are_not_acoustic_claims(text):
    validate_claim_text(text, "comment")


def test_goal_in_first_sentence_does_not_hide_later_contradiction():
    with pytest.raises(SchemaValidationError, match="contradicts"):
        validate_claim_text("Practice linking words. Your grammar is error-free.", "comment", grammar_errors=True)


def test_limitation_does_not_taint_distant_text_comment():
    validate_claim_text("Pronunciation cannot be assessed from text. The story has a clear structure. It uses good connectors.", "comment")


@pytest.mark.parametrize("text", ["Record it again, your grammar was correct.", "Try again: no errors this time.", "Besides good vocabulary, there are no errors. Fix 'io va'."])
def test_goal_and_exception_scoping_cannot_hide_contradictory_assertion(text):
    with pytest.raises(SchemaValidationError, match="contradicts"):
        validate_claim_text(text, "comment", grammar_errors=True, grammar_examples=["io va"])


def test_quoted_single_acoustic_word_is_not_a_learner_statement():
    with pytest.raises(SchemaValidationError, match="acoustic"):
        validate_claim_text('Il "ritmo" è naturale.', 'comment', known_quotes=['ritmo'])
