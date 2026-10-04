"""Offline contracts for the explicit live-check runner; no provider traffic or OS keychain."""
import json
from pathlib import Path

import httpx
import pytest

from scripts import check_provider_connections as checks
from assessment_runtime.llm_client import LLMClientError


@pytest.mark.parametrize('provider', list(checks.KEY_NAMES))
def test_saved_credentials_survive_reload_and_are_deleted(monkeypatch, provider):
    seen = []
    def health(**kwargs):
        seen.append(kwargs['api_key'])
        return {'endpoint': 'fixture', 'payload': {'data': [{'id': 'fixture-model'}]}}
    def probe(**kwargs):
        seen.append(kwargs['api_key'])
        return {'tested_at': 'test', 'content_preview': 'OK'}
    monkeypatch.setattr('app_core.services.llm_health_check', health)
    monkeypatch.setattr('app_core.services.test_llm_connection', probe)
    url = checks.default_setup_base_url(provider) or 'http://127.0.0.1:1234/v1'
    result = checks.check_key_connection(provider, 'fixture-model', 'fixture-secret', url)
    assert result['status'] == 'passed', result
    assert seen == ['fixture-secret', 'fixture-secret']
    assert 'fixture-secret' not in json.dumps(result)


def test_failed_probe_is_not_success_and_does_not_echo_provider_body(monkeypatch):
    def fail(**_):
        raise LLMClientError('provider echoed fixture-secret and private text')
    monkeypatch.setattr('app_core.services.llm_health_check', fail)
    result = checks.check_key_connection('groq', 'model', 'fixture-secret', checks.default_setup_base_url('groq'))
    assert result['status'] == 'failed'
    assert result['reason'] == 'saved-key probe: HTTP 502'
    assert 'fixture-secret' not in json.dumps(result) and 'private text' not in json.dumps(result)


def test_env_reader_never_executes_or_imports_unrelated_values(tmp_path, monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    path = tmp_path/'keys.env'
    sentinel = tmp_path/'should-not-exist'
    path.write_text(f"GROQ_API_KEY='$(touch {sentinel})' # literal\nUNRELATED_SECRET='ignore'\n")
    values = checks.read_keys(path)
    assert values['GROQ_API_KEY'] == f'$(touch {sentinel})'
    assert 'UNRELATED_SECRET' not in values and not sentinel.exists()
    monkeypatch.setenv('GROQ_API_KEY', 'environment-takes-precedence')
    assert checks.read_keys(path)['GROQ_API_KEY'] == 'environment-takes-precedence'


@pytest.mark.parametrize('url', ['https://example.org', 'http://127.0.0.1.evil.test', 'http://user:pass@localhost', 'http://localhost?token=secret'])
def test_saved_session_checker_rejects_remote_or_credential_urls(url):
    with pytest.raises(ValueError):
        checks.loopback_url(url)


def test_missing_credentials_are_blocked_not_skipped_or_passed(monkeypatch, capsys):
    monkeypatch.setattr(checks, 'read_keys', lambda _: {})
    assert checks.main(['--providers', 'groq,xai,openrouter,chatgpt']) == 1
    report = json.loads(capsys.readouterr().out)
    assert len(report['results']) == 4
    assert all(r['status'] == 'blocked' for r in report['results'])


def test_cli_runs_all_three_groq_models_and_fails_if_one_fails(monkeypatch, capsys):
    monkeypatch.setattr(checks, 'read_keys', lambda _: {'GROQ_API_KEY':'never-print'})
    def run(provider, model, key, url):
        assert key == 'never-print' and url == 'https://api.groq.com/openai/v1'
        return {'provider':provider,'model':model,'status':'failed' if model == checks.GROQ_MODELS[1] else 'passed'}
    monkeypatch.setattr(checks, 'check_key_connection', run)
    assert checks.main([]) == 1
    output = capsys.readouterr().out
    assert 'never-print' not in output
    assert [r['model'] for r in json.loads(output)['results']] == list(checks.GROQ_MODELS)


def test_chatgpt_uses_saved_backend_session_without_exporting_tokens(monkeypatch):
    requests = []
    def respond(request):
        requests.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json={'connections':[{'connection_id':'test-account','provider_key':'chatgpt','model':'plan-model','base_url':'https://api.openai.com/v1'}]})
        draft = json.loads(request.content)['connection']
        assert draft['api_key'] == '' and draft['connection_id'] == 'test-account'
        return httpx.Response(200, json={'content_preview':'synthetic'})
    original = httpx.Client
    monkeypatch.setattr(checks.httpx, 'Client', lambda **kwargs: original(**kwargs,transport=httpx.MockTransport(respond)))
    result = checks.check_saved_chatgpt('http://127.0.0.1:8819', 'test-account')
    assert result['status'] == 'passed'
    assert [r.method for r in requests] == ['GET','POST']


def test_existing_report_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path/'report.json'; path.write_text('keep')
    monkeypatch.setattr(checks, 'read_keys', lambda _: {})
    with pytest.raises(SystemExit):
        checks.main(['--output', str(path)])
    assert path.read_text() == 'keep'


def test_unexpected_runtime_error_cannot_echo_credentials(monkeypatch):
    def fail(**_):
        raise RuntimeError('provider echoed fixture-secret')
    monkeypatch.setattr('app_core.services.llm_health_check', fail)
    result = checks.check_key_connection('groq', 'model', 'fixture-secret', checks.default_setup_base_url('groq'))
    assert result['status'] == 'failed' and result['reason'] == 'saved-key probe: HTTP 500'
    assert 'fixture-secret' not in json.dumps(result)



def test_local_checks_do_not_inherit_unrelated_environment_keys(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'unrelated-secret')
    seen = []
    def health(**kwargs):
        seen.append(kwargs['api_key'])
        return {'endpoint':'fixture','payload':{'data':[{'id':'fixture-model'}]}}
    def probe(**kwargs):
        seen.append(kwargs['api_key'])
        return {'tested_at':'fixture','content_preview':'OK'}
    monkeypatch.setattr('app_core.services.llm_health_check', health)
    monkeypatch.setattr('app_core.services.test_llm_connection', probe)
    result = checks.check_key_connection('ollama_local', 'fixture-model', '', 'http://localhost:11434')
    assert result['status'] == 'passed'
    assert seen == ['', '']
    assert checks.os.environ['LLM_API_KEY'] == 'unrelated-secret'


def test_empty_env_placeholders_are_absent(tmp_path, monkeypatch):
    monkeypatch.delenv('XAI_API_KEY', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    path = tmp_path/'keys.env'; path.write_text('XAI_API_KEY=\nGROQ_API_KEY=fixture-key\n')
    values = checks.read_keys(path)
    assert 'XAI_API_KEY' not in values and values['GROQ_API_KEY'] == 'fixture-key'


def test_session_only_storage_cannot_pass_persistence_check(monkeypatch):
    from app_core import services
    from app_core.secret_store import SessionSecretStore, SERVICE_NAME, SecretStoreStatus
    def session_only(connection, api_key):
        connection.provider_metadata['persistent'] = False
        SessionSecretStore().set_secret(SERVICE_NAME, connection.secret_ref, api_key)
        return SecretStoreStatus(False,'test')
    monkeypatch.setattr(services, '_persist_connection_secret', session_only)
    result = checks.check_key_connection('groq', 'model', 'fixture-secret', checks.default_setup_base_url('groq'))
    assert result['status'] == 'failed'
    assert result['reason'] == 'Credential did not reach isolated persistent storage'


def test_probe_and_cleanup_failures_are_both_retained(monkeypatch):
    def fail(**_):
        raise LLMClientError('fixture error')
    monkeypatch.setattr('app_core.services.llm_health_check', fail)
    monkeypatch.setattr('app_backend.app.delete_provider_connection', lambda *args, **kwargs: False)
    result = checks.check_key_connection('groq', 'model', 'fixture-secret', checks.default_setup_base_url('groq'))
    assert result['status'] == 'failed'
    assert result['reason'] == 'saved-key probe: HTTP 502'
    assert result['cleanup_error'] == 'delete: HTTP 404'
    assert 'delete' not in result['checks']


def test_secret_in_non_json_cache_file_fails(monkeypatch):
    original = checks.create_app
    def app_with_leak(config):
        # Exercise scan coverage outside the app JSON directory, including bytes.
        (config.app_data.root.parent/'cache-leak.bin').write_bytes(b'prefix\x00fixture-secret')
        return original(config)
    monkeypatch.setattr(checks, 'create_app', app_with_leak)
    monkeypatch.setattr('app_core.services.llm_health_check', lambda **kwargs: {'payload':{'data':[{'id':'model'}]}})
    monkeypatch.setattr('app_core.services.test_llm_connection', lambda **kwargs: {'content_preview':'OK'})
    result = checks.check_key_connection('groq', 'model', 'fixture-secret', checks.default_setup_base_url('groq'))
    assert result['status'] == 'failed' and result['reason'] == 'Credential exposed in application files'


def test_secret_in_api_response_is_rejected_without_echo():
    with pytest.raises(checks.CheckFailure, match='Credential exposed') as caught:
        checks.ensure_no_key({'content_preview':'fixture-secret'}, 'fixture-secret')
    assert 'fixture-secret' not in str(caught.value)
