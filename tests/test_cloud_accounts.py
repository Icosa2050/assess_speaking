"""Cloud connection boundaries tested offline; these do not claim live plan eligibility."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app_core import chatgpt_auth as auth
from app_core import secret_store
from app_core.state import AppState, AppPreferences, ProviderConnection
from app_backend.app import create_app, _assessment_request_with_saved_runtime_secret
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateRequest
from assessment_runtime import responses_client


@pytest.fixture(autouse=True)
def isolated_secrets(monkeypatch):
    values = {}
    fake = SimpleNamespace(
        get_password=lambda service, account: values.get((service, account)),
        set_password=lambda service, account, value: values.__setitem__((service, account), value),
        delete_password=lambda service, account: values.pop((service, account), None),
    )
    monkeypatch.setattr(secret_store, '_load_keyring_module', lambda: (fake, secret_store.SecretStoreStatus(True, 'test')))
    monkeypatch.setattr(auth, "_secure_store_supported", lambda store: store.is_persistent_supported())
    secret_store._SESSION_SECRETS.clear()
    yield values
    secret_store._SESSION_SECRETS.clear()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, 'HTTPServer', lambda *_: SimpleNamespace(server_port=43111, server_close=lambda: None, handle_request=lambda: time.sleep(0.005)))
    instance = auth.ChatGPTAuth(tmp_path)
    yield instance
    instance.close()


def begin(manager):
    result = manager.start()
    query = parse_qs(urlparse(result['authorization_url']).query)
    return result['attempt_id'], query


def grant(monkeypatch, manager, attempt, *, subject='user-one'):
    monkeypatch.setattr(auth, 'validate_identity', lambda *_: {'sub': subject, 'email': 'unused@example.test'})
    monkeypatch.setattr(auth, 'account_models', lambda _: [{'slug': 'plan-model', 'display_name': 'Plan model'}])
    exchange = Mock(return_value={'access_token': 'access-private', 'refresh_token': 'refresh-private', 'id_token': 'id-private', 'expires_in': 3600, 'scope': auth.SCOPE})
    monkeypatch.setattr(auth, '_request', exchange)
    manager.finish(attempt, {'state': [manager.pending[attempt]['state']], 'code': ['one-use-code'], 'client_id': ['issued-client']})
    return exchange


def test_pkce_state_and_public_registration(manager):
    attempt, query = begin(manager)
    assert query['client_id'] == ['dynamic_agent_client']
    assert query['code_challenge_method'] == ['S256']
    assert query['redirect_uri'] == ['http://127.0.0.1:43111/auth/callback']
    assert query['code_challenge'][0] != manager.pending[attempt]['verifier']
    assert 'nonce' in query and 'state' in query
    assert 'verifier' not in manager.path.read_text()
    manager.cancel(attempt)
    again, query2 = begin(manager)
    assert query2['state'] != query['state']
    assert query2['ext_agent_host_id'] == query['ext_agent_host_id']


def test_bad_state_denial_replay_and_token_secrecy(manager, monkeypatch):
    attempt, _ = begin(manager)
    manager.finish(attempt, {'state': ['bad'], 'code': ['bad']})
    assert manager.status(attempt)['status'] == 'waiting'
    exchange = grant(monkeypatch, manager, attempt)
    assert manager.status(attempt)['status'] == 'connected'
    assert exchange.call_args.kwargs['data']['client_id'] == 'issued-client'
    assert 'private' not in json.dumps(manager.status(attempt))
    assert 'private' not in manager.path.read_text()
    manager.finish(attempt, {'state': [manager.pending[attempt]['state']], 'code': ['one-use-code']})
    assert exchange.call_count == 1
    assert 'verifier' not in manager.pending[attempt]
    connection_id = manager.status(attempt)['connection_id']
    again = manager.start(connection_id)
    query = parse_qs(urlparse(again['authorization_url']).query)
    assert query['client_id'] == ['issued-client']
    manager.finish(again['attempt_id'], {'state': query['state'], 'error': ['access_denied']})
    assert manager.status(again['attempt_id'])['status'] == 'failed'
    assert exchange.call_count == 1


def test_reconnect_cannot_replace_another_account(manager, monkeypatch):
    attempt, _ = begin(manager)
    grant(monkeypatch, manager, attempt)
    connection = manager.status(attempt)['connection_id']
    again = manager.start(connection)['attempt_id']
    grant(monkeypatch, manager, again, subject='different-user')
    assert manager.status(again)['status'] == 'failed'
    assert manager.records['accounts'][connection]['subject'] == 'user-one'


def test_cancel_during_exchange_never_publishes(manager, monkeypatch):
    attempt, _ = begin(manager)
    monkeypatch.setattr(auth, 'validate_identity', lambda *_: {'sub': 'user-one'})
    def cancel_then_models(_):
        manager.cancel(attempt)
        return [{'slug': 'model', 'display_name': 'Model'}]
    monkeypatch.setattr(auth, 'account_models', cancel_then_models)
    monkeypatch.setattr(auth, '_request', lambda *_a, **_kw: {'access_token': 'private', 'refresh_token': 'private', 'id_token': 'private', 'expires_in': 3600, 'scope': auth.SCOPE})
    manager.finish(attempt, {'state': [manager.pending[attempt]['state']], 'code': ['code'], 'client_id': ['issued-client']})
    assert manager.status(attempt)['status'] == 'cancelled'
    assert manager.records['accounts'] == {}


def test_id_token_verification_checks_signature_audience_nonce_and_expiry(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(auth, '_discovery', lambda: {'jwks_uri': auth.ISSUER + '/jwks'})
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    monkeypatch.setattr(auth, '_request', lambda *_a, **_k: {'keys': [{**jwk, 'kid':'test-key'}]})
    claims = {'iss': auth.ISSUER, 'sub': 'user-one', 'aud': 'issued-client', 'exp': time.time()+60, 'nonce': 'nonce'}
    token = jwt.encode(claims, key, algorithm='RS256', headers={'kid':'test-key'})
    assert auth.validate_identity(token, 'issued-client', 'nonce')['sub'] == 'user-one'
    for changed in ({'aud': 'wrong'}, {'iss': 'https://evil.test'}, {'exp': time.time()-60}, {'nonce': 'wrong'}):
        with pytest.raises(auth.ChatGPTAuthError):
            auth.validate_identity(jwt.encode({**claims, **changed}, key, algorithm='RS256', headers={'kid':'test-key'}), 'issued-client', 'nonce')
    with pytest.raises(auth.ChatGPTAuthError):
        auth.validate_identity(jwt.encode(claims, other, algorithm='RS256', headers={'kid':'test-key'}), 'issued-client', 'nonce')


def test_refresh_serialized_rotation_and_session_fallback(monkeypatch):
    auth._write_tokens('test-ref', {'access_token': 'old', 'refresh_token': 'old-refresh', 'client_id': 'issued', 'expires_at': 0, 'scope': auth.SCOPE})
    exchange = Mock(return_value={'access_token': 'new', 'refresh_token': 'rotated', 'expires_in': 3600})
    monkeypatch.setattr(auth, '_request', exchange)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(auth.access_token, ['test-ref']*4)) == ['new']*4
    assert exchange.call_count == 1
    assert 'scope' not in exchange.call_args.kwargs['data']
    assert auth._read_tokens('test-ref')['refresh_token'] == 'rotated'
    monkeypatch.setattr(secret_store.KeyringSecretStore, 'is_persistent_supported', lambda _: False)
    assert auth._write_tokens('test-ref', {'access_token': 'session'}) is False
    assert auth.cached_access_token('test-ref') == 'session'


def test_disconnect_clears_locally_when_revocation_fails(manager, monkeypatch):
    attempt, _ = begin(manager)
    grant(monkeypatch, manager, attempt)
    connection = manager.status(attempt)['connection_id']
    monkeypatch.setattr(auth, '_discovery', Mock(side_effect=auth.ChatGPTAuthError('offline')))
    ref = manager.records['accounts'][connection]['secret_ref']
    assert auth.cached_access_token(ref) == 'access-private'
    assert manager.disconnect(connection) is False
    assert not auth.cached_access_token(ref)
    assert manager.records['accounts'][connection]['client_id'] == 'issued-client'


@pytest.mark.parametrize('provider', ['chatgpt', 'xai'])
@pytest.mark.parametrize('ending', ['response.failed', 'response.incomplete', 'response.refusal.delta', None, 'response.completed'])
def test_stream_requires_terminal_success(monkeypatch, provider, ending):
    events = [SimpleNamespace(type='response.output_text.delta', delta='{"score":3}')]
    if ending:
        events.append(SimpleNamespace(type=ending, response=SimpleNamespace(status='completed')))
    stream = Mock()
    stream.__enter__ = Mock(return_value=iter(events)); stream.__exit__ = Mock(return_value=False)
    client = Mock(); client.__enter__ = Mock(return_value=client); client.__exit__ = Mock(return_value=False)
    client.responses.create.return_value = stream
    factory = Mock(return_value=client)
    monkeypatch.setattr(responses_client, 'OpenAI', factory)
    kwargs = dict(provider=provider, model='chosen', prompt='Exercise transcript', api_key='secret', timeout_sec=10, schema={'name':'rubric','strict':True,'schema':{'type':'object'}})
    if ending == 'response.completed':
        assert responses_client.complete(**kwargs) == '{"score":3}'
    else:
        with pytest.raises(responses_client.ResponsesError):
            responses_client.complete(**kwargs)
    payload = client.responses.create.call_args.kwargs
    assert payload['store'] is False and payload['stream'] is True
    assert payload['text']['format']['type'] == 'json_schema'
    assert 'temperature' not in payload
    assert ('max_output_tokens' in payload) == (provider == 'xai')
    assert factory.call_args.kwargs['max_retries'] == 0


def test_account_routes_origin_host_and_secret_boundaries(tmp_path, monkeypatch):
    app = create_app(build_backend_runtime_config(app_data_dir=tmp_path, port=8765))
    client = TestClient(app, base_url='http://127.0.0.1:8765')
    monkeypatch.setattr(app.state.chatgpt_auth, 'start', lambda _: {'attempt_id': 'public', 'authorization_url': auth.AUTHORIZE})
    assert client.post('/v1/runtime/chatgpt/sign-in', json={}).status_code == 403
    headers = {'X-Vostavo-Client': 'desktop', 'Origin': 'https://evil.test'}
    assert client.post('/v1/runtime/chatgpt/sign-in', json={}, headers=headers).status_code == 403
    headers['Origin'] = 'http://localhost:4173'
    assert client.post('/v1/runtime/chatgpt/sign-in', json={}, headers=headers).status_code == 200
    headers['Host'] = 'attacker.example'
    assert client.post('/v1/runtime/chatgpt/sign-in', json={}, headers=headers).status_code == 403
    # Cross-origin form writes must be rejected even outside the OAuth router.
    assert client.post('/v1/assessments', json={}, headers={'Origin': 'https://evil.test'}).status_code == 403


def test_assessment_uses_saved_account_and_fixed_endpoint(monkeypatch):
    connection = ProviderConnection(connection_id='saved', provider_kind='chatgpt', base_url=auth.RESOURCE, secret_ref='account-ref', default_model='plan-model', provider_metadata={'models':[{'slug':'plan-model'}]})
    state = AppState(prefs=AppPreferences(connections=[connection], active_connection_id='saved'))
    monkeypatch.setattr('app_backend.app.access_token', lambda ref: 'backend-token' if ref == 'account-ref' else 'bad')
    request = AssessmentCreateRequest(audio_id='sample', whisper='tiny', expected_language='en', feedback_language='en', speaker_id='test', task_family='monologue', theme='test', target_duration_sec=60, provider='chatgpt', llm_model='plan-model', llm_api_key='untrusted', llm_base_url='https://evil.test')
    result = _assessment_request_with_saved_runtime_secret(state, request)
    assert result.llm_api_key == 'backend-token' and result.llm_base_url == auth.RESOURCE
    with pytest.raises(auth.ChatGPTAuthError):
        _assessment_request_with_saved_runtime_secret(state, request.model_copy(update={'llm_model':'unavailable'}))


@pytest.mark.parametrize('provider', ['xai', 'groq'])
def test_api_provider_rejects_custom_endpoints_and_environment_fallback(monkeypatch, provider):
    from app_core.runtime_providers import runtime_base_url
    from app_core.runtime_resolver import resolve_connection_runtime
    monkeypatch.setenv('LLM_API_KEY', 'unrelated-key')
    connection = ProviderConnection(provider_kind=provider, default_model='chosen', secret_ref='missing')
    assert resolve_connection_runtime(connection).api_key == ''
    with pytest.raises(ValueError):
        runtime_base_url(provider, 'https://evil.test/v1')


def test_success_publishes_without_frontend_polling(tmp_path, manager, monkeypatch):
    app = create_app(build_backend_runtime_config(app_data_dir=tmp_path / 'app', port=8765))
    owner = app.state.chatgpt_auth
    attempt, _ = begin(owner)
    grant(monkeypatch, owner, attempt)
    # Never call the attempt-status endpoint: closing the browser cannot lose the account.
    client = TestClient(app, base_url='http://127.0.0.1:8765')
    result = client.get('/v1/runtime/settings').json()
    assert len(result['connections']) == 1
    connection = result['connections'][0]
    assert connection['provider_key'] == 'chatgpt' and connection['has_api_key']
    assert 'private' not in json.dumps(result)
    assert 'private' not in json.dumps(client.get('/v1/runtime').json())
    for file in (tmp_path / 'app').rglob('*.json'):
        assert 'access-private' not in file.read_text()
        assert 'refresh-private' not in file.read_text()
    saved = client.put('/v1/runtime/settings', json={'ui_locale':'it', 'whisper_model':'large-v3', 'clear_saved_secret':False, 'connection':{'connection_id':connection['connection_id'], 'provider_choice':'chatgpt', 'model':connection['model'], 'base_url':connection['base_url']}})
    assert saved.status_code == 200
    assert saved.json()['ui_locale'] == 'it' and saved.json()['whisper_model'] == 'large-v3'
    assert saved.json()['connections'][0]['has_api_key']
    cancel = Mock()
    monkeypatch.setattr(app.state.job_manager, 'cancel_provider', cancel)
    monkeypatch.setattr(auth, '_revoke', lambda *_: True)
    response = client.delete('/v1/runtime/chatgpt/connections/'+connection['connection_id'], headers={'X-Vostavo-Client':'desktop'})
    assert response.json() == {'disconnected': True, 'revocation_confirmed': True}
    cancel.assert_called_once_with('chatgpt')
    assert client.get('/v1/runtime/settings').json()['connections'] == []


def test_abandoned_grant_is_revoked_and_not_published(manager, monkeypatch):
    attempt, _ = begin(manager)
    monkeypatch.setattr(auth, '_revoke', revoke := Mock(return_value=True))
    monkeypatch.setattr(auth, 'validate_identity', Mock(side_effect=auth.ChatGPTAuthError('Wrong account')))
    monkeypatch.setattr(auth, '_request', lambda *_a, **_kw: {'refresh_token':'abandoned-refresh'})
    manager.finish(attempt, {'state':[manager.pending[attempt]['state']], 'code':['code'], 'client_id':['issued-client']})
    revoke.assert_called_once_with('issued-client', 'abandoned-refresh')
    assert manager.status(attempt)['status'] == 'failed'
    assert not manager.records['accounts']


def test_worker_broker_requests_current_token_for_each_inference(monkeypatch):
    from app_backend.jobs import JobManager
    monkeypatch.setattr(auth, 'access_token', supplier := Mock(side_effect=['first-token', 'renewed-token']))
    channel = Mock()
    channel.poll.return_value = True
    channel.recv.side_effect = ['access-token', 'access-token', EOFError()]
    process = Mock(); process.is_alive.return_value = True
    JobManager._serve_credentials(process, channel, 'selected-account')
    assert supplier.call_args_list == [(('selected-account',),), (('selected-account',),)]
    assert [c.args[0] for c in channel.send.call_args_list] == [{'token':'first-token'}, {'token':'renewed-token'}]
    channel.close.assert_called_once()


def test_generation_resolves_credentials_at_request_time(monkeypatch):
    from assessment_runtime.llm_client import _chat_completion
    monkeypatch.setattr(responses_client, 'complete', complete := Mock(return_value='{}'))
    supplier = Mock(side_effect=['first-token', 'renewed-token'])
    for _ in range(2):
        assert _chat_completion('chatgpt', 'plan-model', 'prompt', 10, None, api_key=supplier) == '{}'
    assert [c.kwargs['api_key'] for c in complete.call_args_list] == ['first-token', 'renewed-token']


def test_rotation_refuses_to_replay_when_keyring_cannot_clear(monkeypatch):
    auth._write_tokens('account', {'access_token':'expired', 'refresh_token':'one-use', 'client_id':'issued', 'scope':auth.SCOPE, 'expires_at':0})
    monkeypatch.setattr(secret_store.KeyringSecretStore, 'delete_secret', lambda *_: None)
    monkeypatch.setattr(secret_store.KeyringSecretStore, 'set_secret', Mock(side_effect=RuntimeError('locked')))
    monkeypatch.setattr(auth, '_request', exchange := Mock())
    with pytest.raises(auth.ChatGPTAuthError, match='Secure storage'):
        auth.access_token('account')
    exchange.assert_not_called()


def test_real_loopback_callback_rejects_wrong_state_and_finishes(tmp_path, monkeypatch):
    import httpx
    owner = auth.ChatGPTAuth(tmp_path)
    monkeypatch.setattr(auth, 'validate_identity', lambda *_: {'sub':'loopback-user'})
    monkeypatch.setattr(auth, 'account_models', lambda _: [{'slug':'plan-model', 'display_name':'Plan model'}])
    monkeypatch.setattr(auth, '_request', lambda *_a, **_kw: {'access_token':'private-access', 'refresh_token':'private-refresh', 'id_token':'signed-elsewhere', 'scope':auth.SCOPE, 'expires_in':3600})
    published = Mock()
    owner.on_connected = published
    try:
        attempt, query = begin(owner)
        url = query['redirect_uri'][0]
        assert httpx.get(url, params={'state':'wrong', 'code':'code'}, timeout=2).status_code == 200
        assert owner.status(attempt)['status'] == 'waiting'
        response = httpx.get(url, params={'state':query['state'][0], 'code':'code', 'client_id':'issued'}, timeout=2)
        assert response.status_code == 200 and 'private' not in response.text
        assert response.headers['cache-control'] == 'no-store'
        assert owner.status(attempt)['status'] == 'connected'
        published.assert_called_once()
    finally:
        owner.close()


@pytest.mark.parametrize('provider', ['xai', 'groq'])
def test_api_key_remains_usable_when_secure_storage_is_unavailable(monkeypatch, provider):
    from app_core.services import _persist_connection_secret
    from app_core.runtime_resolver import resolve_connection_runtime
    monkeypatch.setattr(secret_store, '_load_keyring_module', lambda: (None, secret_store.SecretStoreStatus(False, 'unavailable')))
    connection = ProviderConnection(connection_id=provider, provider_kind=provider, default_model='chosen')
    status = _persist_connection_secret(connection, 'session-provider-key')
    assert not status.persistent and connection.provider_metadata['persistent'] is False
    assert resolve_connection_runtime(connection).api_key == 'session-provider-key'
    secret_store.delete_secret(connection.secret_ref)
    assert resolve_connection_runtime(connection).api_key == ''


@pytest.mark.parametrize('status', [429, None, 500])
def test_refresh_failure_preserves_revocation_and_prevents_uncertain_replay(monkeypatch, status):
    auth._write_tokens('account', {'access_token':'expired', 'refresh_token':'old-refresh', 'client_id':'issued', 'expires_at':0, 'scope':auth.SCOPE})
    monkeypatch.setattr(auth, '_request', exchange := Mock(side_effect=auth.ChatGPTAuthError('Unavailable', status=status)))
    with pytest.raises(auth.ChatGPTAuthError):
        auth.access_token('account')
    saved = auth._read_tokens('account')
    assert saved['refresh_token'] == 'old-refresh'
    assert bool(saved.get('rotation_pending')) == (status != 429)
    if status != 429:
        # Simulate restart: the durable marker, not process memory, prevents replay.
        secret_store._SESSION_SECRETS.clear()
        with pytest.raises(auth.ChatGPTAuthError, match='interrupted'):
            auth.access_token('account')
        assert exchange.call_count == 1


def test_invalid_rotated_grant_is_revoked_and_never_replayed(monkeypatch):
    auth._write_tokens('account', {'access_token':'expired', 'refresh_token':'old', 'client_id':'issued', 'expires_at':0, 'scope':auth.SCOPE})
    monkeypatch.setattr(auth, '_request', Mock(return_value={'access_token':'new', 'refresh_token':'rotated', 'expires_in':None}))
    monkeypatch.setattr(auth, '_revoke', revoked := Mock(return_value=True))
    with pytest.raises(auth.ChatGPTAuthError):
        auth.access_token('account')
    revoked.assert_called_once_with('issued', 'rotated')
    assert not auth._read_tokens('account')


def test_mid_stream_transport_error_is_safe_and_can_degrade_to_local_report(monkeypatch):
    import httpx2
    def events():
        yield SimpleNamespace(type='response.output_text.delta', delta='partial')
        raise httpx2.ReadTimeout('private provider body')
    stream = Mock(); stream.__enter__ = Mock(return_value=events()); stream.__exit__ = Mock(return_value=False)
    client = Mock(); client.__enter__ = Mock(return_value=client); client.__exit__ = Mock(return_value=False)
    client.responses.create.return_value = stream
    monkeypatch.setattr(responses_client, 'OpenAI', Mock(return_value=client))
    from assessment_runtime.llm_client import _chat_completion, LLMClientError
    with pytest.raises(LLMClientError, match='interrupted') as error:
        _chat_completion('chatgpt', 'model', 'prompt', 10, None, api_key='secret')
    assert 'private' not in str(error.value)


def groq_sdk(monkeypatch, response=None, failure=None):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.side_effect = failure
    client.chat.completions.create.return_value = SimpleNamespace(model_dump=lambda: response)
    monkeypatch.setattr(responses_client, 'OpenAI', factory := Mock(return_value=client))
    return factory, client


def test_groq_probe_uses_strict_nonstreaming_chat_protocol(monkeypatch):
    from assessment_runtime.llm_client import test_connection
    factory, client = groq_sdk(monkeypatch)
    def echo(**payload):
        result = payload['messages'][0]['content'].split('exactly: ', 1)[1]
        return SimpleNamespace(model_dump=lambda: {'choices':[{'finish_reason':'stop','message':{'content':result}}]})
    client.chat.completions.create.side_effect = echo
    result = test_connection(provider='groq', model='openai/gpt-oss-120b', api_key='groq-secret')
    assert result['ok'] and result['provider'] == 'groq'
    assert factory.call_args.kwargs['base_url'] == 'https://api.groq.com/openai/v1'
    assert factory.call_args.kwargs['max_retries'] == 0
    payload = client.chat.completions.create.call_args.kwargs
    assert payload['stream'] is False and payload['reasoning_effort'] == 'low'
    assert payload['max_completion_tokens'] == 4096
    assert payload['response_format']['json_schema']['strict'] is True
    assert payload['response_format']['json_schema']['name'] == 'assess_speaking_rubric'
    assert 'provider' not in payload and 'store' not in payload
    client.responses.create.assert_not_called()


@pytest.mark.parametrize('status, message', [(429, 'usage limit'), (413, 'too large'), (401, 'authorization'), (400, 'request failed')])
def test_groq_errors_are_safe_and_not_retried(monkeypatch, status, message):
    import httpx2
    from openai import APIStatusError
    from assessment_runtime.llm_client import _chat_completion, LLMClientError
    response = httpx2.Response(status, request=httpx2.Request('POST', 'https://api.groq.com/openai/v1/chat/completions'))
    _, client = groq_sdk(monkeypatch, failure=APIStatusError('private content', response=response, body={'private':'secret'}))
    with pytest.raises(LLMClientError, match=message) as caught:
        _chat_completion('groq', 'openai/gpt-oss-120b', 'prompt', 10, None, api_key='groq-secret')
    assert 'private' not in str(caught.value) and 'secret' not in str(caught.value)
    assert client.chat.completions.create.call_count == 1


@pytest.mark.parametrize('choice', [
    {'finish_reason':'length','message':{'content':'{}'}},
    {'finish_reason':'stop','message':{'refusal':'refused','content':'{}'}},
    {'finish_reason':'stop','message':{'content':''}},
])
def test_groq_incomplete_feedback_is_not_accepted(monkeypatch, choice):
    from assessment_runtime.llm_client import _chat_completion, LLMClientError
    groq_sdk(monkeypatch, {'choices':[choice]})
    with pytest.raises(LLMClientError):
        _chat_completion('groq', 'openai/gpt-oss-120b', 'prompt', 10, None, api_key='groq-secret')


def test_groq_saved_key_injection_and_delete(tmp_path):
    from app_backend.app import _load_persisted_state
    config = build_backend_runtime_config(port=8817, app_data_dir=tmp_path/'app', cache_dir=tmp_path/'cache')
    app = create_app(config)
    client = TestClient(app)
    draft = {'provider_choice':'groq','model':'openai/gpt-oss-120b','base_url':'https://api.groq.com/openai/v1','api_key':'groq-private'}
    response = client.put('/v1/runtime/settings', json={'ui_locale':'it','whisper_model':'tiny','connection':draft})
    assert response.status_code == 200, response.text
    saved = response.json()['connections'][0]
    assert saved['provider_key'] == 'groq' and saved['has_api_key']
    assert 'groq-private' not in response.text
    state = _load_persisted_state(config)
    request = AssessmentCreateRequest(audio_id='sample', whisper='tiny', expected_language='it', feedback_language='en', speaker_id='test', task_family='monologue', theme='test', target_duration_sec=60, provider='groq', llm_model=draft['model'])
    assert _assessment_request_with_saved_runtime_secret(state, request).llm_api_key == 'groq-private'
    with pytest.raises(ValueError):
        _assessment_request_with_saved_runtime_secret(state, request.model_copy(update={'llm_base_url':'https://evil.test/v1'}))
    deleted = client.delete('/v1/runtime/settings/connections/' + saved['connection_id'])
    assert deleted.status_code == 200, deleted.text
    assert not secret_store.get_secret('connection:' + saved['connection_id'])
    for path in (tmp_path/'app').rglob('*.json'):
        assert 'groq-private' not in path.read_text()
