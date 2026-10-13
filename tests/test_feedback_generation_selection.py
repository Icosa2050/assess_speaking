import pytest
from scripts.evaluate_feedback_generation import select_cases

CORPUS={'cases':[{'case_id':'a'},{'case_id':'b'},{'case_id':'c'}]}
def test_full_corpus_is_default_and_subsets_require_recorded_exclusions():
    assert len(select_cases(CORPUS))==3
    with pytest.raises(ValueError,match='omitted'): select_cases(CORPUS,['a'])
    assert select_cases(CORPUS,['a'],{'b':'pilot held out','c':'pilot held out'})==[{'case_id':'a'}]
@pytest.mark.parametrize('requested,excluded', [(['a','a'],{}),(['unknown'],{}),([] ,{}),(['a'],{'b':'','c':'held out'}),(None,{'a':'hold','b':'hold','c':'hold'}),(None,{'unknown':'hold'})])
def test_ambiguous_selection_is_rejected(requested,excluded):
    with pytest.raises(ValueError): select_cases(CORPUS,requested,excluded)
def test_duplicate_corpus_ids_are_rejected():
    with pytest.raises(ValueError): select_cases({'cases':[{'case_id':'a'},{'case_id':'a'}]})
