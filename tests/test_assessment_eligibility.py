import copy
import pytest
from assessment_runtime.eligibility import annotate_payload, eligibility
from assessment_runtime.comparison import comparison_key


def assessable():
    return {'metrics':{'duration_sec':31,'word_count':30},'checks':{'min_words_pass':True,'language_pass':True,'content_validity_pass':True,'topic_pass':True,'duration_pass':False},'scores':{'final':3.5,'band':4}}


def test_task_completion_is_separate_from_metric_reliability():
    result=eligibility(assessable())
    assert result['state']=='assessable' and result['metrics_reliable'] and result['task_complete'] is False


@pytest.mark.parametrize('key,value,state', [('duration_sec',12,'insufficient_speech'),('min_words_pass',False,'insufficient_speech'),('content_validity_pass',False,'invalid_content'),('language_pass',False,'invalid_content'),('content_validity_pass',None,'content_unverified'),('language_pass',None,'content_unverified')])
def test_short_invalid_and_unknown_evidence_cannot_establish_grades(key,value,state):
    report=assessable()
    (report['metrics'] if key=='duration_sec' else report['checks'])[key]=value
    result=eligibility(report)
    assert result['state']==state and not result['metrics_reliable']
    assert comparison_key(report,{}) is None


def test_legacy_reports_are_annotated_without_overwriting_saved_observations():
    report=assessable();report['checks']['content_validity_pass']=None
    payload={'report':report};original=copy.deepcopy(payload)
    result=annotate_payload(payload)
    assert payload==original
    assert result['report']['scores']['final']==3.5
    assert result['report']['scores']['status']=='provisional_observations'
    assert result['report']['eligibility']['state']=='content_unverified'


def test_unknown_duration_is_not_inferred_from_an_old_target_pass():
    report=assessable();report['metrics']={};report['checks']['duration_pass']=True
    assert eligibility(report)['state']=='content_unverified'
