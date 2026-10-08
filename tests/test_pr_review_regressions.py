"""Regression cases for verified PR #16 review findings."""
import threading
import time
import os
import subprocess
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_backend.jobs import JobManager
from app_backend.contracts import AssessmentCreateRequest
from app_core.journal_lock import journal_guard
from app_core.openrouter_auth import OpenRouterAuth
from app_core.services import history_rows
from scripts.run_bounded_check import run


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process-group cleanup')
@pytest.mark.parametrize('second_timeout', [False, True])
def test_bounded_check_handles_already_exited_process_groups(second_timeout):
    process = mock.Mock(pid=1234)
    timeout = subprocess.TimeoutExpired(['fixture'], 1)
    process.wait.side_effect = [timeout, timeout, 0] if second_timeout else [timeout, 0]
    with mock.patch('scripts.run_bounded_check.subprocess.Popen', return_value=process), mock.patch(
        'scripts.run_bounded_check.os.killpg', side_effect=ProcessLookupError
    ) as kill:
        assert run(1, ['fixture']) == 124
    assert kill.call_count == (2 if second_timeout else 1)


@pytest.mark.parametrize('failure', ['constructor', 'start'])
def test_worker_creation_failure_closes_credentials_pipe(tmp_path, failure):
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    manager = JobManager(config)
    upload = manager.register_upload(data=b'fake-audio', filename='sample.wav')
    request = AssessmentCreateRequest(
        audio_id=upload.audio_id, request_id='pipe-failure', whisper='tiny', provider='chatgpt',
        llm_model='model', expected_language='it', feedback_language='en', speaker_id='fixture',
        task_family='free_monologue', theme='Travel', target_duration_sec=90,
    )
    parent, child, process = mock.Mock(), mock.Mock(), mock.Mock()
    if failure == 'start':
        process.start.side_effect = OSError('spawn failed')
    factory = {'side_effect': OSError('spawn failed')} if failure == 'constructor' else {'return_value': process}
    with mock.patch.object(manager._ctx, 'Pipe', return_value=(parent, child)), mock.patch.object(
        manager._ctx, 'Process', **factory
    ), pytest.raises(OSError, match='spawn failed'):
        manager.submit(request, credential_ref='fixture-ref')
    parent.close.assert_called_once_with()
    child.close.assert_called_once_with()
    assert list(config.jobs_dir.glob('*.json')) == []
    assert manager._processes == {}


@pytest.mark.parametrize('error_type', [PermissionError, BlockingIOError])
def test_guard_preserves_errors_from_guarded_work(tmp_path, error_type):
    original = error_type('recording is not writable')
    with pytest.raises(error_type) as caught:
        with journal_guard(tmp_path):
            raise original
    assert caught.value is original
    with journal_guard(tmp_path, exclusive=True):
        pass  # The failing body released its lease.


@pytest.mark.parametrize('report', [None, [], 'invalid', {'checks': []}])
def test_history_keeps_malformed_reports_unverified(report):
    record = mock.Mock(duration_sec=31, word_count=100, top_priorities=(),
                       grammar_error_categories=(), coherence_issue_categories=())
    with mock.patch('app_core.services.load_history_records', return_value=[record]), mock.patch(
        'app_core.services.load_report_payload', return_value={'report': report}
    ):
        rows = history_rows()
    assert rows[0]['eligibility']['state'] == 'content_unverified'
    assert rows[0]['content_validity_pass'] is None
    assert rows[0]['final_score'] is None


def test_browser_handoff_does_not_block_authorization_status():
    auth = OpenRouterAuth(lambda value: None)
    auth.pending['fixture'] = {'status': 'waiting', 'expires': time.time() + 60,
                               'authorization_url': 'https://openrouter.ai/auth?fixture'}
    acquired = []

    def handoff(url):
        def inspect_lock():
            available = auth.lock.acquire(timeout=0.2)
            acquired.append(available)
            if available:
                auth.lock.release()
        probe = threading.Thread(target=inspect_lock, daemon=True)
        probe.start()
        probe.join(timeout=1)
        assert url == auth.pending['fixture']['authorization_url']
        return True

    with mock.patch('app_core.openrouter_auth.webbrowser.open', side_effect=handoff):
        assert auth.open_browser('fixture') is True
        assert auth.open_browser('missing') is False
    assert acquired == [True]


def test_disk_recovery_error_applies_only_to_journal_routes(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path / 'app', cache_dir=tmp_path / 'cache', port=8871)
    app = create_app(config)

    @app.get('/v1/review-fixture/disk-failure')
    def unrelated_failure():
        raise ConnectionError('fixture provider is unavailable')

    with TestClient(app, raise_server_exceptions=False) as client:
        with mock.patch.object(app.state.journal, 'state', side_effect=OSError('fixture disk failed')):
            storage = client.get('/v1/journal/status')
        assert storage.status_code == 507
        assert storage.json()['detail']['code'] == 'storage_error'
        unrelated = client.get('/v1/review-fixture/disk-failure')
        assert unrelated.status_code == 500
        assert 'recovery in Settings' not in unrelated.text
