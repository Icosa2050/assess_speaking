"""Real API, spawned assessment, private broker, validation and report persistence."""
import csv
import json
import os
from pathlib import Path

import pytest

from tests.helpers.cloud_harness import backend


@pytest.mark.parametrize('mode', ['free', 'paid-fallback', 'disabled', 'auth', 'schema', 'unknown', 'resume', 'chatgpt'])
def test_real_cloud_pipeline(tmp_path, mode):
    with backend(tmp_path, mode) as app:
        app.configure(chatgpt=mode == 'chatgpt', fallback=mode == 'paid-fallback')
        identity = app.submit(chatgpt=mode == 'chatgpt')
        status = app.wait(identity)
        assert status['status'] == 'completed', status.get('error')
        job = json.loads((tmp_path / 'app/jobs' / (identity + '.json')).read_text())
        report = job['payload']
        calls = app.dispatches()
        assert sum(c['phase'] == 'asr' for c in calls) == 1
        guard_pids = {json.loads(line)['pid'] for line in (tmp_path / 'guards.jsonl').read_text().splitlines()}
        assert app.process.pid in guard_pids
        assert job['worker_pid'] in guard_pids
        assert report['report']['input']['asr_provider'] != 'groq'
        sid = report['report']['session_id']
        assert app.api.get(f'/v1/history/{sid}/audio').status_code == 200
        history = app.request('GET', '/v1/history')
        assert len(history['items']) == 1
        assert Path(job['report_path']).is_file()
        assert 'fixture-analysis' not in json.dumps(job)
        generations = [c for c in calls if c['phase'] in ('rubric', 'coaching')]
        if mode in ('disabled', 'auth', 'schema', 'unknown'):
            assert all(c['route'] == 'free' for c in generations)
            assert not report['report'].get('rubric') or report['report']['scores']['llm'] is None
        else:
            assert report['report'].get('rubric'), report['report'].get('warnings')
            if mode != 'resume':
                assert report['report'].get('coaching')
        if mode == 'paid-fallback':
            assert [(c['phase'], c['route']) for c in generations] == [('rubric', 'free'), ('rubric', 'paid'), ('coaching', 'paid')]
            assert all(c.get('reserved_before_post') for c in generations if c['route'] == 'paid')
            spending = app.request('GET', '/v1/runtime/cloud')['spending']
            assert spending['spent_usd'] == .02 and spending['reserved_usd'] == 0
        if mode == 'resume':
            old_path = Path(job['report_path'])
            old_timestamp = report['meta']['timestamp']
            before = [(c['phase'], c['route']) for c in calls]
            app.set_scenario('free')
            new = app.request('POST', f'/v1/assessments/{identity}/resume', json={'request_id': 'fixture-resume'})
            resumed = app.wait(new['assessment_id'])
            assert resumed['status'] == 'completed'
            final = json.loads((tmp_path / 'app/jobs' / (new['assessment_id'] + '.json')).read_text())['payload']
            assert final['report']['session_id'] == sid
            assert final['meta']['timestamp'] == old_timestamp
            assert final['report'].get('coaching')
            after = app.dispatches()[len(before):]
            assert [(c['phase'], c['route']) for c in after if c['phase'] != 'other'] == [('coaching', 'free')]
            assert old_path.is_file()
            assert len(app.request('GET', '/v1/history')['items']) == 1


def test_groq_language_uncertainty_withholds_grade_and_positive_coaching(tmp_path):
    with backend(tmp_path, 'free') as app:
        app.configure(groq_asr=True)
        identity = app.submit()
        status = app.wait(identity)
        assert status['status'] == 'completed', status.get('error')
        report = status['payload']['report']
        assert report['input']['asr_provider'] == 'groq'
        assert 'cloud_asr_preview' in report['warnings']
        assert report['requires_human_review']
        assert report['eligibility']['state'] == 'content_unverified'
        assert status['summary']['score_overall'] is None
        assert not any(c['phase'] == 'coaching' for c in app.dispatches())
        assert report['checks']['language_pass'] is None
