"""Permanent offline cloud boundary regressions."""

import hashlib

import json

import threading

import time

import wave

from pathlib import Path

from types import SimpleNamespace

from urllib.parse import parse_qs, urlsplit

from unittest.mock import Mock

import httpx

import numpy as np

import pytest

from app_core.cloud_policy import CloudSettings, SpendingLedger, write_settings

from app_core.state import AppPreferences, ProviderConnection

from assessment_runtime.checkpoints import StageCache

@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    import socket
    import ipaddress
    import keyring.core
    from scripts.journey_keyring import MemoryKeyring
    monkeypatch.setattr(MemoryKeyring, '_secrets', {})
    monkeypatch.setattr(keyring.core, '_keyring_backend', MemoryKeyring())
    monkeypatch.setattr('app_core.secret_store._SESSION_SECRETS', {})
    original = socket.socket.connect
    def connect(sock, address):
        if sock.family != socket.AF_UNIX:
            try:
                allowed = address[0] == 'localhost' or ipaddress.ip_address(address[0]).is_loopback
            except ValueError:
                allowed = False
            if not allowed:
                raise PermissionError('Unexpected external socket in cloud account fixture')
        return original(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', connect)


def make_audio(path, seconds=4):
    samples = np.random.default_rng(7).integers(-25000, 25000, int(16000*seconds), dtype=np.int16)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(samples.astype('<i2').tobytes())
    return path

def mock_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(handler), trust_env=False, **kw))

def test_actual_broker_asr_and_chunk_cache(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    audio = make_audio(tmp_path/'audio.wav', 2)
    speech = ProviderConnection(connection_id='speech',provider_kind='groq',secret_ref='test-only')
    state = SimpleNamespace(prefs=AppPreferences(connections=[speech],active_connection_id='speech'))
    write_settings(tmp_path, CloudSettings(asr_provider='groq',asr_connection_id='speech'))
    calls=[]
    def handler(request):
        calls.append(request.url.path)
        assert request.headers['Authorization']=='Bearer fixture'
        return httpx.Response(200,json={'text':'I I think','language':'english','words':[{'word':'I','start':.2,'end':.5},{'word':'I','start':.6,'end':.8},{'word':'think','start':1,'end':1.5}]})
    mock_transport(monkeypatch,handler)
    first = CloudRuntime(tmp_path,state,lambda _: 'fixture',audio_path=audio,cache_root=tmp_path/'stages')
    result = first.handle({'operation':'asr','language':'it'})
    restarted = CloudRuntime(tmp_path,state,lambda _: 'fixture',audio_path=audio,cache_root=tmp_path/'stages')
    assert restarted.handle({'operation':'asr'}) == result
    assert calls == ['/openai/v1/audio/transcriptions']
    assert result['detected_language']=='en'

@pytest.mark.parametrize('status',[401,403,429,500])
def test_actual_asr_http_failure_retains_input(tmp_path,monkeypatch,status):
    from assessment_runtime.groq_asr import transcribe, GroqASRError
    audio=make_audio(tmp_path/'audio.wav',2)
    digest=hashlib.file_digest(audio.open('rb'),'sha256').hexdigest()
    mock_transport(monkeypatch,lambda _: httpx.Response(status,json={'error':'fixture'}))
    with pytest.raises(GroqASRError): transcribe(audio,api_key='fixture',model='whisper-large-v3')
    assert hashlib.file_digest(audio.open('rb'),'sha256').hexdigest()==digest

@pytest.mark.parametrize('body', [b'<html>gateway response</html>', b'[]', b'null'])
def test_actual_asr_malformed_success_retains_input(tmp_path, monkeypatch, body):
    from assessment_runtime.groq_asr import transcribe, GroqASRError
    audio = make_audio(tmp_path / 'audio.wav', 2)
    original = audio.read_bytes()
    mock_transport(monkeypatch, lambda _: httpx.Response(200, content=body))
    with pytest.raises(GroqASRError, match='Retry the retained recording'):
        transcribe(audio, api_key='fixture', model='whisper-large-v3')
    assert audio.read_bytes() == original


def test_recursive_split_resumes_only_unfinished_chunk(tmp_path,monkeypatch):
    import assessment_runtime.groq_asr as groq
    audio=make_audio(tmp_path/'audio.wav')
    # Lower only the cap to exercise genuine PyAV encoding, recursive splitting
    # and resume without allocating a 20-minute file.
    monkeypatch.setattr(groq,'MAX_BYTES',105000)
    calls=[]
    fail_second=True
    def handler(request):
        name=request.content.split(b'filename="')[1].split(b'"')[0].decode()
        calls.append(name)
        if name=='0b.flac' and fail_second: return httpx.Response(429,json={'error':'fixture'})
        # The second range starts at 1 second due to its overlap.
        offset=0 if name=='0a.flac' else 1
        absolute=[(.5,.7,'I'),(1.5,1.7,'I'),(2.5,2.7,'think'),(3.5,3.7,'so')]
        words=[{'start':start-offset,'end':end-offset,'word':word} for start,end,word in absolute if start>=offset and end<=offset+3]
        return httpx.Response(200,json={'text':' '.join(w['word'] for w in words),'words':words})
    mock_transport(monkeypatch,handler)
    cache=StageCache(tmp_path/'chunks')
    with pytest.raises(groq.GroqASRError,match='quota'): groq.transcribe(audio,api_key='fixture',model='whisper-large-v3',cached=cache.run)
    fail_second=False
    result=groq.transcribe(audio,api_key='fixture',model='whisper-large-v3',cached=cache.run)
    assert calls==['0a.flac','0b.flac','0b.flac']
    assert result['text']=='I I think so'
    assert [w['t0'] for w in result['words']]==[.5,1.5,2.5,3.5]

def test_overlap_only_chunk_can_be_empty_without_losing_neighbor_speech():
    from assessment_runtime.groq_asr import normalize
    # A silent central range with legitimate speech only in the overlap.
    result=normalize({'text':'neighbor','words':[{'start':0,'end':.5,'word':'neighbor'}]},offset=1,keep_start=2,keep_end=4)
    assert result['words']==[]

def test_managed_probe_does_not_bypass_free_policy(tmp_path,monkeypatch):
    from app_core.openrouter_cloud import probe_policy
    from assessment_runtime.llm_client import _chat_completion, LLMClientError
    write_settings(tmp_path,CloudSettings(openrouter_modes={'free':'free'}))
    calls=[]
    def handler(request):
        calls.append(request.method)
        return httpx.Response(200,json={'data':[{'id':'vendor/model:free','pricing':{'prompt':'.1','completion':'0'},'supported_parameters':['response_format']}]})
    mock_transport(monkeypatch,handler)
    with probe_policy(tmp_path,'free','vendor/model:free','fixture'):
        with pytest.raises(LLMClientError,match='no longer free'):
            _chat_completion('openrouter','vendor/model:free','test',2,'fixture')
    assert calls==['GET']

def test_actual_managed_paid_broker_dispatch_and_cost(tmp_path,monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    conn=ProviderConnection(connection_id='paid',provider_kind='openrouter',default_model='vendor/model',secret_ref='fixture')
    state=SimpleNamespace(prefs=AppPreferences(connections=[conn],active_connection_id='paid'))
    write_settings(tmp_path,CloudSettings(openrouter_modes={'paid':'paid'}))
    seen=[]
    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith('/models'): return httpx.Response(200,json={'data':[{'id':'vendor/model','pricing':{'prompt':'.000001','completion':'.000002'},'supported_parameters':['response_format']}]})
        if request.url.path.endswith('/key'): return httpx.Response(200,json={'data':{'limit':5,'limit_remaining':5}})
        payload=json.loads(request.content)
        assert payload['response_format']['type']=='json_schema'
        assert payload['provider']['allow_fallbacks'] is False
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"ok":true}'}}],'usage':{'cost':.01}})
    mock_transport(monkeypatch,handler)
    runtime=CloudRuntime(tmp_path,state,lambda _:'fixture',cache_root=tmp_path/'stages')
    message={'operation':'completion','provider':'openrouter','model':'vendor/model','prompt':'test','schema':{'name':'probe','schema':{'type':'object'}},'timeout':2}
    assert runtime.handle(message)['cost_usd']==.01
    assert runtime.handle(message)['cost_usd']==0
    assert SpendingLedger(tmp_path).summary()['spent_usd']==.01
    assert seen==['/api/v1/models','/api/v1/key','/api/v1/chat/completions']

def test_cancel_during_pkce_exchange_keeps_cancelled_status(monkeypatch):
    from app_core.openrouter_auth import OpenRouterAuth
    original=httpx.Client
    entered=threading.Event(); release=threading.Event()
    def handler(_request):
        entered.set(); assert release.wait(3)
        return httpx.Response(200,json={'key':'fixture'})
    mock_transport(monkeypatch,handler)
    publish=Mock(); owner=OpenRouterAuth(publish)
    attempt=owner.start(); query=parse_qs(urlsplit(attempt['authorization_url']).query)
    errors=[]
    def callback():
        try:
            with original(timeout=5) as client:
                client.get(query['callback_url'][0],params={'state':query['state'][0],'code':'fixture'})
        except Exception as exc: errors.append(type(exc).__name__)
    worker=threading.Thread(target=callback,daemon=True); worker.start()
    try:
        assert entered.wait(2)
        owner.close(); release.set(); worker.join(timeout=5)
        publish.assert_not_called()
        assert not worker.is_alive()
        assert owner.status(attempt['attempt_id'])['status']=='cancelled'
    finally:
        release.set();owner.close();worker.join(timeout=5)

def test_pkce_expiry_stops_listener():
    from app_core.openrouter_auth import OpenRouterAuth
    owner=OpenRouterAuth(Mock()); attempt=owner.start(); server=owner.server
    try:
        owner._expire(attempt['attempt_id'],server)
        assert owner.status(attempt['attempt_id'])=={'status':'expired'}
        assert owner.server is None and owner.timer is None
        owner.publish.assert_not_called()
    finally: owner.close()

@pytest.mark.parametrize('storage',['persistent','unavailable','write-failure','readback-failure'])
def test_actual_pkce_routes_publish_saved_model_less_connection(tmp_path,monkeypatch,storage):
    persistent = storage == 'persistent'
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    from app_core.secret_store import SessionSecretStore,SERVICE_NAME,set_secret,SecretStoreStatus
    config=build_backend_runtime_config(app_data_dir=tmp_path/'app',cache_dir=tmp_path/'cache',port=8860)
    published={}
    if storage == 'unavailable':
        monkeypatch.setattr('app_core.secret_store._load_keyring_module',lambda:(None,SecretStoreStatus(persistent=False,backend_name='fixture-unavailable')))
    if storage == 'write-failure':
        monkeypatch.setattr('scripts.journey_keyring.MemoryKeyring.set_password', Mock(side_effect=OSError('fixture keyring write failure')))
    if storage == 'readback-failure':
        monkeypatch.setattr('scripts.journey_keyring.MemoryKeyring.get_password', lambda *_: '')
    def save_secret(ref,key,**kwargs):
        published[ref]=key
        return set_secret(ref,key,**kwargs)
    monkeypatch.setattr('app_core.services.set_secret',save_secret)
    app=create_app(config)
    original=httpx.Client
    with TestClient(app,base_url='http://127.0.0.1:8860') as api:
        mock_transport(monkeypatch,lambda _:httpx.Response(200,json={'key':'fixture-pkce-key'}))
        assert api.get('/v1/health').status_code==200
        start=api.post('/v1/runtime/cloud/openrouter/sign-in',headers={'X-Vostavo-Client':'desktop'})
        assert start.status_code==200
        attempt=start.json();params=parse_qs(urlsplit(attempt['authorization_url']).query)
        with original(timeout=3) as callback:
            assert callback.get(params['callback_url'][0],params={'state':params['state'][0],'code':'fixture'}).status_code==200
        status=api.get('/v1/runtime/cloud/openrouter/attempts/'+attempt['attempt_id'],headers={'X-Vostavo-Client':'desktop'})
        assert status.json()['status']=='connected'
        settings=api.get('/v1/runtime/settings').json()
        record=settings['connections'][0]
        assert record['provider_key']=='openrouter' and record['model']==''
        assert record['has_api_key'], 'Connected callback must leave a usable saved or session-only key'
        assert 'fixture-pkce-key' not in json.dumps(settings)
        if not persistent:
            ref=next(iter(published))
            assert SessionSecretStore().get_secret(SERVICE_NAME,ref)=='fixture-pkce-key'
            SessionSecretStore().delete_secret(SERVICE_NAME,ref)
            with TestClient(create_app(config), base_url='http://127.0.0.1:8860') as restarted:
                assert not restarted.get('/v1/runtime/settings').json()['connections'][0]['has_api_key']
        for path in config.app_data.root.rglob('*.json'):
            assert 'fixture-pkce-key' not in path.read_text()


@pytest.mark.parametrize('terminal', ['expired', 'replacement'])
def test_late_exchange_preserves_terminal_and_replacement(monkeypatch, terminal):
    from app_core.openrouter_auth import OpenRouterAuth
    original = httpx.Client
    entered, release = threading.Event(), threading.Event()
    def exchange(_):
        entered.set()
        assert release.wait(5)
        return httpx.Response(200, json={'key': 'fixture'})
    mock_transport(monkeypatch, exchange)
    owner = OpenRouterAuth(Mock())
    attempt = owner.start()
    query = parse_qs(urlsplit(attempt['authorization_url']).query)
    def callback():
        with original(timeout=5, trust_env=False) as client:
            assert client.get(query['callback_url'][0], params={'state': query['state'][0], 'code': 'fixture'}).status_code == 400
    thread = threading.Thread(target=callback)
    thread.start()
    try:
        assert entered.wait(3)
        if terminal == 'expired':
            owner._expire(attempt['attempt_id'], owner.server)
        else:
            replacement = owner.start()
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert owner.status(attempt['attempt_id'])['status'] == ('expired' if terminal == 'expired' else 'cancelled')
        if terminal == 'replacement':
            assert owner.status(replacement['attempt_id'])['status'] == 'waiting'
            assert owner.timer is not None and not owner.timer.finished.is_set()
        owner.publish.assert_not_called()
    finally:
        release.set()
        owner.close()
        thread.join(5)


@pytest.mark.parametrize('start,end', [(float('nan'), 1), (0, float('inf')), (2, 1), ('bad', 1)])
def test_invalid_overlap_timing_is_not_discarded(start, end):
    from assessment_runtime.groq_asr import normalize, GroqASRError
    with pytest.raises(GroqASRError, match='timestamps'):
        normalize({'text': 'overlap', 'words': [{'word': 'overlap', 'start': start, 'end': end}]}, keep_start=20)


def test_half_open_partition_keeps_repetitions_and_rejects_missing_timing():
    from assessment_runtime.groq_asr import normalize, GroqASRError
    raw = {'text': 'I I', 'words': [{'word': 'I', 'start': .5, 'end': 1.5}, {'word': 'I', 'start': 2, 'end': 2.5}]}
    assert normalize(raw, keep_end=1)['words'] == []
    assert [w['text'] for w in normalize(raw, keep_start=1)['words']] == ['I', 'I']
    assert normalize({'text': '', 'words': []})['words'] == []
    with pytest.raises(GroqASRError, match='omitted'):
        normalize({'text': 'untimed', 'words': []})


@pytest.mark.parametrize('persistent', [True, False])
def test_pkce_preserves_prior_key_then_blank_model_update_and_inference(tmp_path, monkeypatch, persistent):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app, _load_persisted_state, _saved_connection_api_key
    from app_backend.config import build_backend_runtime_config
    from app_backend.cloud_runtime import CloudRuntime
    from app_core.runtime_resolver import resolve_connection_runtime
    from app_core.secret_store import SecretStoreStatus
    if not persistent:
        monkeypatch.setattr('app_core.secret_store._load_keyring_module', lambda: (None, SecretStoreStatus(False, 'fixture-unavailable')))
    config = build_backend_runtime_config(app_data_dir=tmp_path/'app', cache_dir=tmp_path/'cache', port=8860)
    original = httpx.Client
    app = create_app(config)
    with TestClient(app, base_url='http://127.0.0.1:8860') as api:
        previous = api.put('/v1/runtime/settings', json={'connection': {'provider_choice': 'openrouter', 'model': 'vendor/old:free', 'api_key': 'fixture-old'}}).json()['active_connection_id']
        seen = []
        def handler(request):
            if request.url.path.endswith('/auth/keys'):
                return httpx.Response(200, json={'key': 'fixture-new'})
            if request.url.path.endswith('/models'):
                return httpx.Response(200, json={'data': [{'id': 'vendor/new:free', 'pricing': {'prompt': '0', 'completion': '0'}, 'supported_parameters': ['response_format']}]})
            assert request.headers['Authorization'] == 'Bearer fixture-new'
            seen.append(request.url.path)
            return httpx.Response(200, json={'choices': [{'message': {'content': '{"ok":true}'}}]})
        mock_transport(monkeypatch, handler)
        attempt = api.post('/v1/runtime/cloud/openrouter/sign-in', headers={'X-Vostavo-Client': 'desktop'}).json()
        query = parse_qs(urlsplit(attempt['authorization_url']).query)
        with original(timeout=3, trust_env=False) as callback:
            assert callback.get(query['callback_url'][0], params={'state': query['state'][0], 'code': 'fixture'}).status_code == 200
        settings = api.get('/v1/runtime/settings').json()
        assert settings['active_connection_id'] == previous
        new = next(c for c in settings['connections'] if c['connection_id'] != previous)
        assert new['model'] == '' and new['has_api_key']
        assert new['provider_metadata']['persistent'] is persistent
        assert new['provider_metadata']['requires_model_selection']
        with TestClient(create_app(config), base_url='http://127.0.0.1:8860') as reloaded:
            assert next(c for c in reloaded.get('/v1/runtime/settings').json()['connections'] if c['connection_id'] == new['connection_id'])['has_api_key']
        state = _load_persisted_state(config)
        assert resolve_connection_runtime(next(c for c in state.prefs.connections if c.connection_id == previous)).api_key == 'fixture-old'
        response = api.put('/v1/runtime/settings', json={'connection': {'connection_id': new['connection_id'], 'provider_choice': 'openrouter', 'base_url': new['base_url'], 'model': 'vendor/new:free', 'api_key': ''}})
        assert response.status_code == 200 and response.json()['active_connection_id'] == new['connection_id']
        write_settings(config.app_data.root, CloudSettings(openrouter_modes={new['connection_id']: 'free'}))
        runtime = CloudRuntime(config.app_data.root, _load_persisted_state(config), _saved_connection_api_key)
        assert runtime.handle({'operation': 'completion', 'provider': 'openrouter', 'model': 'vendor/new:free', 'prompt': 'fixture', 'timeout': 2})['text'] == '{"ok":true}'
        assert seen == ['/api/v1/chat/completions']
        assert api.delete('/v1/runtime/settings/connections/' + new['connection_id']).status_code == 200
        state = _load_persisted_state(config)
        assert resolve_connection_runtime(state.prefs.connections[0]).api_key == 'fixture-old'
        for path in config.app_data.root.rglob('*.json'):
            assert 'fixture-new' not in path.read_text() and 'fixture-old' not in path.read_text()


def test_failed_publication_cleans_only_new_secret(tmp_path, monkeypatch):
    from app_backend.app import create_app, _load_persisted_state
    from app_backend.config import build_backend_runtime_config
    from app_core.secret_store import set_secret, get_secret
    config = build_backend_runtime_config(app_data_dir=tmp_path/'app', cache_dir=tmp_path/'cache', port=8860)
    app = create_app(config)
    set_secret('fixture-prior-ref', 'fixture-prior')
    stored = []
    from app_core.services import _persist_connection_secret
    def store(conn, key):
        stored.append(conn.secret_ref)
        return _persist_connection_secret(conn, key)
    monkeypatch.setattr('app_core.services._persist_connection_secret', store)
    monkeypatch.setattr('app_core.services.save_state_preferences', Mock(side_effect=OSError('fixture write failure')))
    with pytest.raises(OSError):
        app.state.openrouter_auth.publish('fixture-new')
    assert get_secret(stored[0]) == ''
    assert get_secret('fixture-prior-ref') == 'fixture-prior'
    app.state.openrouter_auth.close()


def test_cancel_between_chunks_retains_success_and_cleans_temporary_files(tmp_path, monkeypatch):
    import assessment_runtime.groq_asr as groq
    audio = make_audio(tmp_path/'audio.wav')
    monkeypatch.setattr(groq, 'MAX_BYTES', 105000)
    paths, calls = [], []
    original_encode = groq.write_flac_range
    def encode(source, target, *args, **kwargs):
        paths.append(target)
        return original_encode(source, target, *args, **kwargs)
    monkeypatch.setattr(groq, 'write_flac_range', encode)
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={'text':'I', 'words':[{'word':'I','start':.5,'end':.8}]})
    mock_transport(monkeypatch, handler)
    cache = StageCache(tmp_path/'chunks')
    with pytest.raises(groq.GroqASRError, match='stopped'):
        groq.transcribe(audio, api_key='fixture', model='whisper-large-v3-turbo', cached=cache.run, cancelled=lambda: bool(calls))
    assert len(calls) == 1 and audio.is_file()
    assert 'chunk-0a' in json.loads(cache.manifest.read_text())['stages']
    assert all(not path.exists() for path in paths)


def test_low_disk_fails_before_upload_and_retains_input(tmp_path, monkeypatch):
    import assessment_runtime.groq_asr as groq
    audio = make_audio(tmp_path/'audio.wav')
    monkeypatch.setattr(groq.shutil, 'disk_usage', lambda _: SimpleNamespace(free=groq.DISK_RESERVE))
    handler = Mock(side_effect=AssertionError('No upload permitted'))
    mock_transport(monkeypatch, handler)
    with pytest.raises(groq.GroqASRError, match='disk'):
        groq.transcribe(audio, api_key='fixture', model='whisper-large-v3')
    handler.assert_not_called()
    assert audio.is_file()


def test_restart_fallback_checks_free_again_and_reuses_paid_reply(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    free = ProviderConnection(connection_id='free',provider_kind='openrouter',default_model='vendor/free:free',secret_ref='fixture')
    paid = ProviderConnection(connection_id='paid',provider_kind='openrouter',default_model='vendor/paid',secret_ref='fixture')
    state = SimpleNamespace(prefs=AppPreferences(connections=[free,paid],active_connection_id='free'))
    write_settings(tmp_path, CloudSettings(openrouter_modes={'free':'free','paid':'paid'},fallback_connection_id='paid',paid_fallback_enabled=True))
    dispatched=[]
    def handler(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'data':[{'id':c.default_model,'pricing':{'prompt':'0' if c is free else '.000001','completion':'0' if c is free else '.000001'},'supported_parameters':['response_format']} for c in (free,paid)]})
        if request.url.path.endswith('/key'):
            return httpx.Response(200,json={'data':{'limit':5,'limit_remaining':5}})
        model=json.loads(request.content)['model'];dispatched.append(model)
        return httpx.Response(429,json={'error':'quota'}) if model==free.default_model else httpx.Response(200,json={'choices':[{'message':{'content':'{"ok":true}'}}],'usage':{'cost':.01}})
    mock_transport(monkeypatch,handler)
    message={'operation':'completion','provider':'openrouter','model':free.default_model,'prompt':'fixture','timeout':2}
    first=CloudRuntime(tmp_path,state,lambda _:'fixture',cache_root=tmp_path/'stages')
    assert first.handle(message)['cost_usd']==.01
    restarted=CloudRuntime(tmp_path,state,lambda _:'fixture',cache_root=tmp_path/'stages')
    assert restarted.handle(message)['cache_reused']
    assert dispatched==['vendor/free:free','vendor/paid','vendor/free:free']
    assert SpendingLedger(tmp_path).summary()['spent_usd']==.01
