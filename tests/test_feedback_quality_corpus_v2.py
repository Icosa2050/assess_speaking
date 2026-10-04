"""Source-labelled corpus integrity, not certification of model diagnoses."""
import json
from pathlib import Path

import pytest

from assess_core.coaching_taxonomy import GRAMMAR_ERROR_CATEGORIES
from assess_core.schemas import RubricResult
from assessment_runtime.output_validation import validate_rubric_generation
from tests.test_generation_validation import rubric_payload

CORPUS = json.loads((Path(__file__).parent / 'fixtures/feedback_quality/bilingual_v2.json').read_text())
CASES = CORPUS['cases']


def test_v2_separates_pairs_register_and_optional_style():
    assert len(CASES) == 20
    assert len({case['case_id'] for case in CASES}) == 20
    assert all('scores' not in case and 'target_level' not in case for case in CASES)
    assert len([case for case in CASES if case['variant'] == 'seeded_error']) == 8
    assert len([case for case in CASES if case['variant'] == 'register_variant']) == 2
    assert len([case for case in CASES if case['variant'] == 'style_only']) == 2
    for case in CASES:
        assert 'independent_human' in case['review_status']
        assert all(source in CORPUS['sources'] for source in case['source_ids'])
        assert case['assessment_context']
        if case['variant'] != 'seeded_error':
            assert case['expected_grammar_errors'] == []
            continue
        clean = next(other for other in CASES if other['pair_id'] == case['pair_id'] and other['variant'] == 'clean')
        error, = case['expected_grammar_errors']
        assert error['category'] in GRAMMAR_ERROR_CATEGORIES
        assert case['transcript'].replace(error['quote'], error['correction'], 1) == clean['transcript'] == case['corrected_reference']
        assert error['correction'] not in case['transcript']


@pytest.mark.parametrize('case', CASES, ids=[case['case_id'] for case in CASES])
def test_authored_reference_contracts_stay_separate(case):
    payload = rubric_payload() | {
        'recurring_grammar_errors': [{'category': error['category'], 'explanation': error['reason'], 'examples': [error['quote']]}
                                     for error in case['expected_grammar_errors']],
        'style_suggestions': case['expected_style_suggestions'],
    }
    validate_rubric_generation(RubricResult.from_dict(payload), case['transcript'])
    if case['variant'] in ('register_variant', 'style_only'):
        assert not payload['recurring_grammar_errors']
