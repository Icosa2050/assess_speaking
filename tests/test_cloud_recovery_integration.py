"""Controlled cache publication and unknown-charge process recovery boundaries."""
import json
from pathlib import Path
import os
import signal
import time

import pytest

from tests.helpers.cloud_harness import backend


def wait_file(app, name):
    until = min(app.deadline, time.monotonic() + 15)
    while time.monotonic() < until:
        if (app.root / name).exists():
            return
        time.sleep(.02)
    raise AssertionError('Missing controlled fixture boundary: ' + name)


@pytest.mark.parametrize('loss', ['worker-loss', 'backend-loss'])
def test_published_reply_survives_process_loss_and_resume(tmp_path, loss):
    with backend(tmp_path, loss) as app:
        app.configure()
        identity = app.submit()
        wait_file(app, 'published')
        job_path = tmp_path / 'app/jobs' / (identity + '.json')
        prior = json.loads(job_path.read_text())
        cache = Path(prior['stage_cache_dir']) / 'cloud-replies'
        assert (cache / 'manifest.json').is_file()
        if loss == 'worker-loss':
            os.kill(prior['worker_pid'], signal.SIGKILL)
            assert app.wait(identity)['status'] == 'failed'
            (tmp_path / 'release').touch()
        else:
            app.stop()
        app.set_scenario('free')
        if loss == 'backend-loss':
            app.start()
            assert app.request('GET', '/v1/assessments/' + identity)['status'] == 'failed'
        resumed = app.request('POST', f'/v1/assessments/{identity}/resume', json={'request_id': 'fixture-recover'})
        final = app.wait(resumed['assessment_id'])
        assert final['status'] == 'completed'
        generations = [c for c in app.dispatches() if c['phase'] in ('asr', 'rubric', 'coaching')]
        assert [c['phase'] for c in generations] == ['asr', 'rubric', 'coaching']
        assert final['payload']['report']['rubric']
        routes = final['payload']['report']['input']['cloud_routes']
        assert any(c.get('cache_reused') for c in routes)
        assert len(app.request('GET', '/v1/history')['items']) == 1
        guard_pids = {json.loads(line)['pid'] for line in (tmp_path / 'guards.jsonl').read_text().splitlines()}
        assert app.process.pid in guard_pids


def test_unknown_paid_result_survives_restart_and_budget_blocks_post(tmp_path):
    with backend(tmp_path, 'unknown-paid') as app:
        _, paid = app.configure()
        app.request('POST', f'/v1/runtime/settings/connections/{paid}/default')
        identity = app.submit(paid=True)
        wait_file(app, 'post-entered')
        state = app.request('GET', '/v1/runtime/cloud')
        reserved = state['spending']['reserved_usd']
        assert reserved > 0 and len(state['spending']['unresolved_requests']) == 1
        app.stop()
        app.set_scenario('free')
        app.start()
        restarted = app.request('GET', '/v1/runtime/cloud')
        assert restarted['spending']['reserved_usd'] == reserved
        assert len([c for c in app.dispatches() if c['phase'] == 'rubric']) == 1
        settings = restarted['settings']
        settings['monthly_budget_usd'] = max(.01, reserved)
        app.request('PUT', '/v1/runtime/cloud', json=settings)
        resumed = app.request('POST', f'/v1/assessments/{identity}/resume', json={'request_id': 'fixture-budget-retry'})
        final = app.wait(resumed['assessment_id'])
        assert final['status'] == 'completed'
        assert final['payload']['report']['scores']['llm'] is None
        assert len([c for c in app.dispatches() if c['phase'] == 'rubric']) == 1
        reservation = restarted['spending']['unresolved_requests'][0]
        # Budget-only policy permits a sequential user retry while the old outcome is unknown.
        settings['monthly_budget_usd'] = 5
        app.request('PUT', '/v1/runtime/cloud', json=settings)
        retry = app.request('POST', f'/v1/assessments/{resumed["assessment_id"]}/resume', json={'request_id': 'fixture-confirmed-retry'})
        assert app.wait(retry['assessment_id'])['payload']['report']['rubric']
        assert len([c for c in app.dispatches() if c['phase'] == 'rubric']) == 2
        pending = app.request('GET', '/v1/runtime/cloud')['spending']
        assert pending['reserved_usd'] == reserved and pending['spent_usd'] == .02
        settled = app.request('POST', f'/v1/runtime/cloud/spending/{reservation}/reconcile', json={'actual_cost_usd': 0, 'provider_cost_confirmed': True})
        assert settled['spending']['reserved_usd'] == 0
        ledger = json.loads((tmp_path/'app/cloud-spending.json').read_text())
        assert ledger['requests'][reservation]['reconciliation_source'] == 'user_confirmed_provider_cost'
        assert ledger['requests'][reservation]['reconciled_at']
        assert len(app.request('GET', '/v1/history')['items']) == 1


@pytest.mark.parametrize('kind', ['ledger', 'cache'])
def test_cross_process_ledger_and_prompt_serialization(tmp_path, monkeypatch, kind):
    import multiprocessing
    from app_core.cloud_policy import CloudSettings, SpendingLedger, write_settings
    from tests.helpers.cloud_harness import ROOT
    from tests.helpers.cloud_concurrency import run
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(ROOT/'tests/helpers/cloud_guard'), str(ROOT)]))
    monkeypatch.setenv('VOSTAVO_FIXTURE_GUARD_LOG', str(tmp_path/'guards.jsonl'))
    (tmp_path/'scenario.json').write_text(json.dumps({'mode':'concurrent-paid'}))
    write_settings(tmp_path/'app', CloudSettings(openrouter_modes={'paid':'paid'}))
    ctx = multiprocessing.get_context('spawn')
    barrier, output = ctx.Barrier(2), ctx.Queue()
    children = [ctx.Process(target=run, args=(str(tmp_path), barrier, output, kind)) for _ in range(2)]
    try:
        for child in children:
            child.start()
        if kind == 'cache':
            until = time.monotonic() + 15
            while not (tmp_path/'post-entered').exists() and time.monotonic() < until:
                time.sleep(.02)
            assert (tmp_path/'post-entered').exists()
            assert SpendingLedger(tmp_path/'app').summary()['reserved_usd'] > 0
            (tmp_path/'release').touch()
        results = [output.get(timeout=15) for _ in children]
        for child in children:
            child.join(5)
            assert child.exitcode == 0
        guards = {json.loads(line)['pid'] for line in (tmp_path/'guards.jsonl').read_text().splitlines()}
        assert all(child.pid in guards for child in children)
        if kind == 'ledger':
            assert sorted(value[0] for value in results) == ['blocked', 'reserved']
            assert SpendingLedger(tmp_path/'app').summary()['reserved_usd'] == 3
        else:
            assert sorted(value[1] for value in results) == [0, .01]
            calls = [json.loads(line) for line in (tmp_path/'dispatch.jsonl').read_text().splitlines()]
            assert len([c for c in calls if c['phase'] == 'rubric']) == 1
            assert SpendingLedger(tmp_path/'app').summary()['spent_usd'] == .01
    finally:
        (tmp_path/'release').touch()
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join(5)
        output.close()
        output.join_thread()
