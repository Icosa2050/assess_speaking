"""Corpus integrity and guard mechanics, not model-quality certification."""

import json
from pathlib import Path

import pytest

from assess_core.coaching_taxonomy import GRAMMAR_ERROR_CATEGORIES
from assess_core.schemas import RubricResult, SchemaValidationError
from assessment_runtime.output_validation import validate_rubric_generation


CORPUS_PATH = Path(__file__).parent / "fixtures/feedback_quality/bilingual_v1.json"
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
CASES = CORPUS["cases"]


def _rubric(case):
    return RubricResult(
        fluency=4, cohesion=4, accuracy=4, range=4, overall=4,
        comments_fluency="Illustrative comment, not a calibrated score.",
        comments_cohesion="Illustrative comment, not a calibrated score.",
        comments_accuracy="No additional errors asserted.",
        comments_range="Illustrative comment, not a calibrated score.",
        overall_comment="Test the output contract independently of semantic review.",
        on_topic=True, evidence_quotes=[],
    ).to_dict() | {"recurring_grammar_errors": [
        {"category": error["category"], "explanation": error["reason"], "examples": [error["quote"]]}
        for error in case["expected_grammar_errors"]
    ]}


def test_authored_corpus_is_separate_from_cefr_scoring_suites():
    assert CORPUS["schema_version"] == 1
    assert CORPUS["corpus_type"] == "authored_transcript_feedback"
    assert set(CORPUS["languages"]) == {"en", "it"}
    assert len({case["case_id"] for case in CASES}) == len(CASES)
    assert all("target_level" not in case and "scores" not in case for case in CASES)
    for pair_id in {case["pair_id"] for case in CASES}:
        pair = [case for case in CASES if case["pair_id"] == pair_id]
        assert len(pair) == 2
        clean = next(case for case in pair if case["variant"] == "clean")
        seeded = next(case for case in pair if case["variant"] == "seeded_error")
        assert clean["language_code"] == seeded["language_code"]
        assert clean["expected_grammar_errors"] == []
        assert seeded["corrected_reference"] == clean["transcript"]
        assert len(seeded["expected_grammar_errors"]) == 1
        error = seeded["expected_grammar_errors"][0]
        assert error["category"] in GRAMMAR_ERROR_CATEGORIES
        assert error["quote"] in seeded["transcript"]
        assert error["quote"] not in clean["transcript"]
        assert error["correction"] in clean["transcript"]
        assert error["reason"].strip()
        assert seeded["transcript"].replace(error["quote"], error["correction"], 1) == clean["transcript"]


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_clean_and_seeded_reference_responses_pass_grounding(case):
    rubric = RubricResult.from_dict(_rubric(case))
    validate_rubric_generation(rubric, case["transcript"])
    if case["variant"] == "clean":
        assert rubric.recurring_grammar_errors == []
    else:
        assert len(rubric.recurring_grammar_errors) == 1


@pytest.mark.parametrize("case", [case for case in CASES if case["variant"] == "seeded_error"], ids=[case["case_id"] for case in CASES if case["variant"] == "seeded_error"])
def test_invented_correction_cannot_be_used_as_learner_error_quote(case):
    payload = _rubric(case)
    payload["recurring_grammar_errors"][0]["examples"] = [case["expected_grammar_errors"][0]["correction"]]
    with pytest.raises(SchemaValidationError, match="not present"):
        validate_rubric_generation(RubricResult.from_dict(payload), case["transcript"])


def test_grounding_does_not_claim_to_detect_false_diagnosis_of_valid_quote():
    clean = CASES[0]
    payload = _rubric(clean)
    payload["recurring_grammar_errors"] = [{
        "category": "verb_conjugation_present", "explanation": "Deliberately false diagnosis for a contract-limit test.",
        "examples": ["can save"],
    }]
    # This passes membership but is contrary to the clean case's authored labels.
    # A future semantic comparison must count it as a false correction.
    validate_rubric_generation(RubricResult.from_dict(payload), clean["transcript"])
    assert clean["expected_grammar_errors"] == []
