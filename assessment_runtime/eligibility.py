"""Versioned eligibility for grades and comparisons; retain raw observations.

Thirty seconds is an app policy, not a validated CEFR sufficiency cutoff. Task
completion is separate: a shorter-than-target, meaningful response can still
supply measurements, while unverified or invalid content cannot establish grades.
"""
from __future__ import annotations
import copy
import math


def eligibility(report: dict, metrics: dict | None = None) -> dict:
    checks=report.get('checks') or {}
    metrics=metrics or report.get('metrics') or {}
    duration=metrics.get('duration_sec')
    duration_known=isinstance(duration,(int,float)) and not isinstance(duration,bool) and math.isfinite(duration)
    reasons=[]
    if (duration_known and duration<30) or checks.get('min_words_pass') is False or 'llm_skipped_low_word_count' in (report.get('warnings') or []):
        state='insufficient_speech';reasons.append('recording_too_short' if duration_known and duration<30 else 'minimum_words')
    elif any(checks.get(key) is False for key in ('language_pass','topic_pass','content_validity_pass')):
        state='invalid_content';reasons.extend(key for key in ('language_pass','topic_pass','content_validity_pass') if checks.get(key) is False)
    elif not duration_known or any(checks.get(key) is not True for key in ('min_words_pass','language_pass','content_validity_pass')):
        state='content_unverified'
        if not duration_known: reasons.append('duration_unknown')
        reasons.extend(key for key in ('min_words_pass','language_pass','content_validity_pass') if checks.get(key) is not True)
    else: state='assessable'
    return {'version':1,'state':state,'reasons':reasons,'metrics_reliable':state=='assessable',
            'task_complete': all(checks.get(key) is True for key in ('duration_pass','topic_pass','language_pass','min_words_pass')) if all(checks.get(key) is not None for key in ('duration_pass','topic_pass','language_pass','min_words_pass')) else None}


def annotate_report(report: dict, metrics: dict | None = None) -> dict:
    result=copy.deepcopy(report)
    result['eligibility']=eligibility(result,metrics)
    if result['eligibility']['state'] != 'assessable': result['requires_human_review'] = True
    if isinstance(result.get('scores'),dict):
        result['scores']['status']='assessable' if result['eligibility']['state']=='assessable' else 'provisional_observations'
    return result


def annotate_payload(payload: dict) -> dict:
    result=copy.deepcopy(payload)
    if isinstance(result.get('report'),dict):
        result['report']=annotate_report(result['report'],result.get('metrics'))
    return result
