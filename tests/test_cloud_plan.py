"""Cloud routing, recovery, spending and real callback contract regressions."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app_core.cloud_policy import CloudSettings, CloudPolicyError, SpendingLedger, read_settings, write_settings
from app_core.state import ProviderConnection, AppPreferences
from app_core.openrouter_cloud import complete, OpenRouterError
from assessment_runtime.checkpoints import StageCache
from assessment_runtime.groq_asr import normalize, GroqASRError


def transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))


def connection(model='vendor/model:free', identity='source'):
    return ProviderConnection(connection_id=identity, provider_kind='openrouter', default_model=model, secret_ref='reference')


def catalog(model, prompt='0', completion='0'):
    return {'data': [{'id': model, 'pricing': {'prompt': prompt, 'completion': completion, 'request': '0'}, 'supported_parameters': ['response_format']}]}


def test_stage_recovery_hash_invalidation_and_tamper(tmp_path):
    stage = StageCache(tmp_path)
    produce = Mock(return_value={'text': 'original transcript'})
    first = stage.run('transcript', {'audio_hash': 'one'}, produce)
    assert stage.run('transcript', {'audio_hash': 'one'}, produce) == first
    produce.assert_called_once()
    identifier = stage.attempt_id()
    assert StageCache(tmp_path).attempt_id() == identifier
    assert stage.run('transcript', {'audio_hash': 'two'}, produce) == first
    assert produce.call_count == 2
    manifest = json.loads(stage.manifest.read_text())
    (tmp_path / manifest['stages']['transcript']['file']).write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        stage.run('transcript', {'audio_hash': 'two'}, produce)
    assert produce.call_count == 2


def test_concurrent_reservations_and_restart_never_release_unknown_cost(tmp_path):
    ledger = SpendingLedger(tmp_path)
    ids, errors = [], []
    def reserve():
        try:
            ids.append(ledger.reserve(Decimal(3), 5, 'paid'))
        except CloudPolicyError as exc:
            errors.append(exc)
    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=3)
    assert len(ids) == len(errors) == 1
    ledger = SpendingLedger(tmp_path)
    ledger.reconcile(ids[0], None)
    assert ledger.summary()['reserved_usd'] == 3
    with pytest.raises(CloudPolicyError, match='budget'):
        ledger.reserve(Decimal(3), 5, 'paid')
    ledger.reconcile(ids[0], Decimal('0.2'))
    assert ledger.summary()['spent_usd'] == .2
    assert ledger.summary()['reserved_usd'] == 0


@pytest.mark.parametrize('value', [-1, 'NaN', 'Infinity', None])
def test_invalid_provider_cost_is_not_a_free_request(tmp_path, value):
    ledger = SpendingLedger(tmp_path)
    with pytest.raises(CloudPolicyError):
        ledger.reserve(value, 5, 'paid')


def test_unresolved_previous_month_is_still_reserved(tmp_path):
    ledger = SpendingLedger(tmp_path)
    identity = ledger.reserve(Decimal(4), 5, 'paid')
    value = json.loads(ledger.path.read_text())
    value['requests'][identity]['month'] = '2020-01'
    ledger.path.write_text(json.dumps(value))
    with pytest.raises(CloudPolicyError): ledger.reserve(Decimal(2), 5, 'paid')


def test_free_route_has_output_and_price_caps_and_never_reads_paid_key_limit(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json=catalog('vendor/model:free'))
        assert request.url.path.endswith('/chat/completions')
        body = json.loads(request.content)
        assert body['max_tokens'] == 4096
        assert body['provider']['max_price']['prompt'] == 0
        assert body['provider']['require_parameters'] and not body['provider']['allow_fallbacks']
        assert 'tools' not in body and 'plugins' not in body
        assert request.headers['authorization'] == 'Bearer test-key'
        return httpx.Response(200, json={'choices':[{'message':{'content':'{}'}, 'finish_reason':'stop'}]})
    transport(monkeypatch, handler)
    result = complete(root=tmp_path, connection=connection(), api_key='test-key', mode='free', settings=CloudSettings(), prompt='test', schema=None, timeout=2)
    assert result['model'] == 'vendor/model:free' and len(calls) == 2
    assert not (tmp_path / 'cloud-spending.json').exists()


@pytest.mark.parametrize('model', ['', 'paid/model', 'openrouter/free', 'openrouter/auto:free'])
def test_free_mode_rejects_missing_paid_or_automatic_models(tmp_path, model, monkeypatch):
    transport(monkeypatch, lambda _: pytest.fail('Must not dispatch'))
    with pytest.raises(OpenRouterError):
        complete(root=tmp_path, connection=connection(model), api_key='test-key', mode='free', settings=CloudSettings(), prompt='test', schema=None, timeout=2)


def test_price_change_disables_free_dispatch(tmp_path, monkeypatch):
    def handler(request):
        assert request.method == 'GET'
        return httpx.Response(200, json=catalog('vendor/model:free', '0.01'))
    transport(monkeypatch, handler)
    with pytest.raises(OpenRouterError, match='no longer free'):
        complete(root=tmp_path, connection=connection(), api_key='test-key', mode='free', settings=CloudSettings(), prompt='test', schema=None, timeout=2)


@pytest.mark.parametrize('remaining', [None, 0])
def test_paid_mode_requires_verified_remaining_key_limit(tmp_path, monkeypatch, remaining):
    def handler(request):
        assert request.method == 'GET'
        if request.url.path.endswith('/models'): return httpx.Response(200, json=catalog('vendor/model', '.000001', '.000002'))
        return httpx.Response(200, json={'data':{'limit':None if remaining is None else 5,'limit_remaining':remaining}})
    transport(monkeypatch, handler)
    with pytest.raises(OpenRouterError):
        complete(root=tmp_path, connection=connection('vendor/model'), api_key='test-key', mode='paid', settings=CloudSettings(), prompt='test', schema=None, timeout=2)


def test_paid_timeout_reservation_survives_and_success_reconciles(tmp_path, monkeypatch):
    failed = True
    def handler(request):
        if request.url.path.endswith('/models'): return httpx.Response(200, json=catalog('vendor/model', '.000001', '.000002'))
        if request.url.path.endswith('/key'): return httpx.Response(200, json={'data':{'limit':5,'limit_remaining':5}})
        assert SpendingLedger(tmp_path).summary()['reserved_usd'] > 0
        if failed: raise httpx.ReadTimeout('private provider error')
        return httpx.Response(200, json={'usage':{'cost':.001},'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]})
    transport(monkeypatch, handler)
    args = dict(root=tmp_path, connection=connection('vendor/model'), api_key='test-key', mode='paid', settings=CloudSettings(), prompt='test', schema=None, timeout=2)
    with pytest.raises(OpenRouterError): complete(**args)
    retained = SpendingLedger(tmp_path).summary()['reserved_usd']
    failed = False
    complete(**args)
    summary = SpendingLedger(tmp_path).summary()
    assert summary['reserved_usd'] == retained and summary['spent_usd'] == .001


def test_asr_partition_preserves_real_repetition_and_unknown_confidence():
    value = normalize({'text':'I I think', 'words':[{'start':0,'end':.2,'word':'I'}, {'start':.3,'end':.5,'word':'I'}, {'start':.6,'end':1,'word':'think'}]}, offset=10, keep_start=10.25, keep_end=11)
    assert [word['text'] for word in value['words']] == ['I', 'think']
    assert value['words'][0]['t0'] == 10.3
    assert value['compute_type_used'] is None and value['segment_diagnostics'] == []
    assert 'probability' not in value['words'][0]
    with pytest.raises(GroqASRError, match='timestamps'): normalize({'text':'speech', 'words':[]})


def test_fallback_only_after_known_free_quota_and_keeps_destination(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    primary = connection()
    paid = connection('vendor/stronger', 'paid')
    state = SimpleNamespace(prefs=AppPreferences(connections=[primary, paid], active_connection_id=primary.connection_id))
    write_settings(tmp_path, CloudSettings(openrouter_modes={'source':'free','paid':'paid'}, fallback_connection_id='paid', paid_fallback_enabled=True))
    runtime = CloudRuntime(tmp_path, state, lambda c: 'key-' + c.connection_id)
    calls = []
    def dispatch(conn, message):
        calls.append(conn.connection_id)
        if conn.connection_id == 'source': raise OpenRouterError('quota', status=429)
        return {'text':'accepted','provider':'openrouter','model':conn.default_model}
    monkeypatch.setattr(runtime, '_complete', dispatch)
    message = {'operation':'completion','provider':'openrouter','model':primary.default_model}
    assert runtime.handle(message)['fallback_reason']
    assert runtime.handle(message)['model'] == paid.default_model
    assert calls == ['source','paid','paid']
    runtime.fallback = False
    monkeypatch.setattr(runtime, '_complete', lambda *_: (_ for _ in ()).throw(OpenRouterError('unknown remote outcome')))
    with pytest.raises(OpenRouterError): runtime.handle(message)
    assert not runtime.fallback


def test_settings_refuse_implicit_paid_fallback_and_unknown_schema(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError): CloudSettings(paid_fallback_enabled=True)
    with pytest.raises(ValidationError): CloudSettings(version=2)
    settings = CloudSettings(asr_provider='groq', asr_connection_id='groq')
    write_settings(tmp_path, settings)
    assert read_settings(tmp_path) == settings


def test_loopback_openrouter_wrong_state_replay_and_backend_only_key(monkeypatch):
    from app_core.openrouter_auth import OpenRouterAuth
    published = Mock()
    original_client = httpx.Client
    def client(**kwargs):
        def handler(request):
            assert request.url == 'https://openrouter.ai/api/v1/auth/keys'
            body = json.loads(request.content)
            assert body['code_challenge_method'] == 'S256' and body['code_verifier']
            return httpx.Response(200, json={'key':'fixture-key'})
        return original_client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr('app_core.openrouter_auth.httpx.Client', client)
    owner = OpenRouterAuth(published)
    try:
        attempt = owner.start()
        query = parse_qs(urlsplit(attempt['authorization_url']).query)
        callback = query['callback_url'][0]
        with original_client(timeout=2) as real:
            assert real.get(callback, params={'state':'wrong','code':'fixture-code'}).status_code == 400
            assert owner.status(attempt['attempt_id'])['status'] == 'waiting'
            assert real.get(callback.replace('/callback', '/wrong'), params={'state':query['state'][0], 'code':'fixture-code'}).status_code == 400
            assert real.get(callback, params={'state':query['state'][0], 'code':'fixture-code'}, headers={'Host':'wrong.invalid'}).status_code == 400
            assert owner.status(attempt['attempt_id'])['status'] == 'waiting'
            params = {'state':query['state'][0],'code':'fixture-code'}
            assert real.get(callback, params=params).status_code == 200
            # Successful callbacks close their listener; a racing replay may
            # receive 400 or encounter the already-closed socket.
            try:
                assert real.get(callback, params=params).status_code == 400
            except httpx.TransportError:
                assert owner.server is None
        published.assert_called_once_with('fixture-key')
        assert owner.status(attempt['attempt_id']) == {'status':'connected'}
        assert 'fixture-key' not in repr(owner.pending)
    finally:
        owner.close()


def test_submission_id_is_idempotent_even_while_first_job_runs(tmp_path, monkeypatch):
    from app_backend.config import build_backend_runtime_config
    from app_backend.contracts import AssessmentCreateRequest
    from app_backend.jobs import JobManager
    config = build_backend_runtime_config(app_data_dir=tmp_path/'app', cache_dir=tmp_path/'cache', port=8860)
    manager = JobManager(config)
    upload = manager.register_upload(data=b'fixture audio', filename='sample.wav')
    process = Mock()
    process.is_alive.return_value = True
    factory = Mock(return_value=process)
    monkeypatch.setattr(manager._ctx, 'Process', factory)
    request = AssessmentCreateRequest(request_id='stable-uuid', audio_id=upload.audio_id, whisper='small', provider='openrouter', llm_model='model', expected_language='it', feedback_language='en', speaker_id='fixture', task_family='free_monologue', theme='test', target_duration_sec=90)
    created = manager.submit(request)
    assert manager.submit(request).assessment_id == created.assessment_id
    assert factory.call_count == 1
    with pytest.raises(ValueError, match='different request'):
        manager.submit(request.model_copy(update={'theme':'different'}))
    manager._processes.clear()


def test_report_revisions_keep_one_take_and_original_files(tmp_path, monkeypatch):
    import csv
    import assess_speaking
    from test_assessment_runner import _assessment_payload
    from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run
    monkeypatch.setattr(assess_speaking, 'run_assessment', lambda *_a, **_kw: deepcopy(_assessment_payload()))
    monkeypatch.setattr(assess_speaking, 'build_progress_delta', lambda *_a, **_kw: None)
    request = AssessmentRunRequest(audio=tmp_path/'sample.wav', stage_cache_dir=tmp_path/'stages', log_dir=tmp_path/'reports')
    first = execute_assessment_run(request)
    second = execute_assessment_run(request)
    assert first.report['session_id'] == second.report['session_id']
    assert first.report_path != second.report_path
    assert first.report_path.exists() and second.report_path.exists()
    with (request.log_dir/'history.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]['report_path'] == str(second.report_path)


def test_score_delta_compares_raw_to_raw_and_never_claims_paraphrase_resolution(tmp_path, monkeypatch):
    import assess_speaking
    from test_assess_speaking import _sample_report
    current = _sample_report(overall=4)
    current['scores']['llm'] = 4.2
    current['coaching']['top_3_priorities'] = ['Explain the destination']
    row = {'speaker_id':current['input']['speaker_id'], 'task_family':current['input']['task_family'], 'learning_language':current['input'].get('learning_language', current['input']['expected_language']), 'session_id':'prior', 'overall':4, 'final_score':3.68, 'wpm':82.8, 'top_priority_1':'Add the place you visited'}
    path = tmp_path/'history.csv'
    assess_speaking.append_history(path, row)
    monkeypatch.setattr('assessment_runtime.comparison.comparable_history_rows', lambda *_: [row])
    result = assess_speaking.build_progress_delta(path, current, practice={})
    assert result['score_delta']['overall'] == 0
    assert result['new_priorities'] == result['resolved_priorities'] == []
    assert result['previous_priorities'] != result['latest_priorities']


def test_cloud_broker_never_passes_key_to_worker_or_persisted_request(tmp_path, monkeypatch):
    from app_backend.config import build_backend_runtime_config
    from app_backend.contracts import AssessmentCreateRequest
    from app_backend.jobs import JobManager
    config = build_backend_runtime_config(app_data_dir=tmp_path/'app', cache_dir=tmp_path/'cache', port=8860)
    manager = JobManager(config)
    upload = manager.register_upload(data=b'fixture audio', filename='sample.wav')
    process = Mock()
    process.is_alive.return_value = False
    factory = Mock(return_value=process)
    monkeypatch.setattr(manager._ctx, 'Process', factory)
    runtime = SimpleNamespace(settings=CloudSettings(), primary=None)
    request = AssessmentCreateRequest(audio_id=upload.audio_id, whisper='small', provider='groq', llm_model='model', expected_language='it', feedback_language='en', speaker_id='fixture', task_family='free_monologue', theme='test', target_duration_sec=90, llm_api_key='must-not-leak')
    created = manager.submit(request, cloud_runtime=runtime)
    worker_request = factory.call_args.kwargs['args'][1]
    assert worker_request['cloud_broker'] and not worker_request['llm_api_key']
    assert 'must-not-leak' not in (config.jobs_dir/(created.assessment_id+'.json')).read_text()
    manager._processes.clear()


def test_groq_quota_uses_only_explicit_paid_fallback(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    from assessment_runtime.responses_client import ResponsesError
    primary = ProviderConnection(connection_id='groq', provider_kind='groq', default_model='text-model')
    paid = connection('vendor/stronger', 'paid')
    state = SimpleNamespace(prefs=AppPreferences(connections=[primary, paid], active_connection_id='groq'))
    write_settings(tmp_path, CloudSettings(openrouter_modes={'paid':'paid'}, fallback_connection_id='paid', paid_fallback_enabled=True))
    runtime = CloudRuntime(tmp_path, state, lambda _: 'fixture')
    def dispatch(conn, _message):
        if conn.provider_kind == 'groq': raise ResponsesError('quota', status=429)
        return {'text':'{}','provider':'openrouter','model':conn.default_model}
    monkeypatch.setattr(runtime, '_complete', dispatch)
    message = {'operation':'completion','provider':'groq','model':'text-model'}
    assert runtime.handle(message)['model'] == paid.default_model
    runtime.fallback = False
    runtime.cancelled = lambda: True
    with pytest.raises(CloudPolicyError, match='cancelled'): runtime.handle(message)
    assert not runtime.fallback


def test_groq_encodes_with_packaged_pyav_and_reuses_verified_chunks(tmp_path, monkeypatch):
    import io
    import wave
    import av
    import numpy as np
    from assessment_runtime.groq_asr import transcribe
    source = tmp_path / 'synthetic.wav'
    with wave.open(str(source), 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(np.zeros(32000, dtype='<i2').tobytes())
    calls = []
    def handler(request):
        calls.append(request)
        body = request.read()
        start = body.index(b'fLaC')
        payload = body[start:body.index(b'\r\n--', start)]
        with av.open(io.BytesIO(payload)) as audio:
            assert audio.streams.audio[0].codec_context.sample_rate == 16000
            assert audio.streams.audio[0].codec_context.channels == 1
            assert sum(frame.samples for frame in audio.decode(audio=0)) == 32000
        return httpx.Response(200, json={'text':'I I think','words':[{'start':0,'end':.3,'word':'I'},{'start':.4,'end':.7,'word':'I'},{'start':.8,'end':1.1,'word':'think'}]})
    transport(monkeypatch, handler)
    monkeypatch.setenv('PATH', '')
    cache = StageCache(tmp_path / 'chunks')
    args = dict(api_key='fixture-key', model='whisper-large-v3', cached=cache.run)
    first = transcribe(source, **args)
    assert transcribe(source, **args) == first and len(calls) == 1
    assert first['text'] == 'I I think' and len(first['words']) == 3
    transcribe(source, **{**args, 'model':'whisper-large-v3-turbo'})
    assert len(calls) == 2


def test_reconciliation_preserves_audit_and_rejects_duplicate_or_unknown(tmp_path):
    ledger = SpendingLedger(tmp_path)
    identity = ledger.reserve(Decimal('0.1'), 5, 'paid')
    ledger.reconcile(identity, 0, source='user_confirmed_provider_cost')
    row = json.loads(ledger.path.read_text())['requests'][identity]
    assert row['reconciliation_source'] == 'user_confirmed_provider_cost' and row['reconciled_at']
    assert ledger.summary()['reserved_usd'] == 0
    with pytest.raises(CloudPolicyError): ledger.reconcile(identity, .2)
    with pytest.raises(CloudPolicyError): ledger.reconcile('missing', 0)


def test_cloud_routes_validate_connections_cost_confirmation_and_local_access(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app_backend.app import create_app
    from app_backend.config import build_backend_runtime_config
    from test_app_backend_api import persist_runtime_settings_seed
    from app_core.state import AppState
    config = build_backend_runtime_config(app_data_dir=tmp_path/'app', cache_dir=tmp_path/'cache', port=8860)
    groq = ProviderConnection(connection_id='speech', provider_kind='groq', default_model='openai/gpt-oss-120b')
    state = AppState(prefs=AppPreferences(connections=[groq, connection()], active_connection_id='source'))
    persist_runtime_settings_seed(config, state)
    with TestClient(create_app(config), base_url='http://127.0.0.1:8860') as client:
        headers = {'X-Vostavo-Client':'desktop'}
        assert client.get('/v1/runtime/cloud').status_code == 403
        settings = CloudSettings(asr_provider='groq', asr_connection_id='speech', openrouter_modes={'source':'free'})
        assert client.put('/v1/runtime/cloud', headers=headers, json=settings.model_dump()).status_code == 200
        assert client.get('/v1/runtime/settings').json()['asr_provider'] == 'groq'
        bad = settings.model_copy(update={'asr_connection_id':'source'})
        assert client.put('/v1/runtime/cloud', headers=headers, json=bad.model_dump()).status_code == 400
        ledger = SpendingLedger(config.app_data.root)
        identity = ledger.reserve(Decimal('.1'), 5, 'source')
        path = f'/v1/runtime/cloud/spending/{identity}/reconcile'
        assert client.post(path, headers=headers, json={'actual_cost_usd':0,'provider_cost_confirmed':False}).status_code == 422
        assert ledger.summary()['reserved_usd'] == .1
        assert client.post(path, headers=headers, json={'actual_cost_usd':.02,'provider_cost_confirmed':True}).status_code == 200
        assert ledger.summary()['spent_usd'] == .02 and ledger.summary()['reserved_usd'] == 0
        assert client.post(path, headers=headers, json={'actual_cost_usd':0,'provider_cost_confirmed':True}).status_code == 400


def test_backend_reply_cache_retains_paid_success_after_worker_disappears(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    primary = connection('vendor/model')
    state = SimpleNamespace(prefs=AppPreferences(connections=[primary], active_connection_id=primary.connection_id))
    write_settings(tmp_path, CloudSettings(openrouter_modes={'source':'paid'}))
    message = {'operation':'completion','provider':'openrouter','model':primary.default_model,'prompt':'same prompt','timeout':500,'schema':None}
    first = CloudRuntime(tmp_path, state, lambda _: 'fixture-key', cache_root=tmp_path/'stages')
    produce = Mock(return_value={'text':'valid paid output','provider':'openrouter','model':primary.default_model,'cost_usd':.01,'reservation_id':'original-charge'})
    monkeypatch.setattr(first, '_complete', produce)
    assert first.handle(message)['cost_usd'] == .01
    assert produce.call_args[0][1]['timeout'] == 120
    restarted = CloudRuntime(tmp_path, state, lambda _: 'fixture-key', cache_root=tmp_path/'stages')
    monkeypatch.setattr(restarted, '_complete', lambda *_: pytest.fail('A completed provider reply must not be paid for again'))
    result = restarted.handle(message)
    assert result['text'] == 'valid paid output' and result['cache_reused']
    assert result['cost_usd'] == 0 and result['original_cost_usd'] == .01


def test_cloud_asr_with_keyed_compatible_analysis_uses_backend_secret(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    primary = ProviderConnection(connection_id='compatible', provider_kind='openai_compatible', default_model='chosen', base_url='https://example.invalid/v1', auth_mode='bearer')
    state = SimpleNamespace(prefs=AppPreferences(connections=[primary], active_connection_id='compatible'))
    runtime = CloudRuntime(tmp_path, state, lambda _: 'saved-compatible-key')
    complete = Mock(return_value='accepted')
    monkeypatch.setattr('assessment_runtime.llm_client._chat_completion', complete)
    assert runtime.handle({'operation':'completion','provider':'openai_compatible','model':'chosen','prompt':'test','timeout':30})['text'] == 'accepted'
    assert complete.call_args.kwargs['api_key'] == 'saved-compatible-key'
    assert complete.call_args.kwargs['base_url'] == 'https://example.invalid/v1'


def test_real_acoustic_stage_and_unknown_asr_survive_cache_roundtrip(tmp_path, monkeypatch):
    import assess_speaking
    import wave
    import numpy as np
    source = tmp_path / 'synthetic.wav'
    with wave.open(str(source), 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(np.zeros(16000*31, dtype='<i2').tobytes())
    speech = Mock(return_value={'text':'hello', 'words':[{'text':'hello','t0':0,'t1':1}], 'asr_model':'whisper-large-v3', 'diagnostic_policy':'groq_uncalibrated_v1'})
    args = dict(asr_provider='groq', cloud_asr=speech, stage_cache_dir=tmp_path/'stages', min_word_count=20, expected_language='en')
    first = assess_speaking.run_assessment(source, 'whisper-large-v3', **args)
    second = assess_speaking.run_assessment(source, 'whisper-large-v3', **args)
    assert first['metrics'] == second['metrics'] and speech.call_count == 1
    assert second['report']['requires_human_review'] and 'cloud_asr_preview' in second['report']['warnings']


@pytest.mark.parametrize('failure', ['timeout', 'wrong_reply'])
def test_rpc_failure_stops_further_dispatch_and_never_consumes_stale_reply(tmp_path, monkeypatch, failure):
    from app_backend.jobs import _job_worker
    from assessment_runtime.cloud_transport import completion
    from assessment_runtime.runner import AssessmentRunResult
    from assessment_runtime.llm_client import LLMClientError
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'assessment_id':'fixture','status':'queued'}))
    channel = Mock()
    channel.poll.return_value = failure != 'timeout'
    channel.recv.return_value = {'rpc_id':'old-rubric-reply','value':{'text':'old paid result'}}
    def execute(_request, **_kwargs):
        hook = completion.get()
        with pytest.raises(LLMClientError):
            hook({'operation':'completion','provider':'openrouter','model':'chosen','prompt':'rubric'})
        with pytest.raises(LLMClientError, match='transport stopped'):
            hook({'operation':'completion','provider':'openrouter','model':'chosen','prompt':'coaching'})
        return AssessmentRunResult(meta={},output={},stdout_json='{}',report={},report_path=None,saved_payload={})
    monkeypatch.setattr('app_backend.jobs.validate_audio_duration', lambda _: None)
    monkeypatch.setattr('app_backend.jobs.execute_assessment_run', execute)
    request = {'cloud_broker':True,'whisper':'tiny','provider':'openrouter','llm_model':'chosen',
               'theme':'test','task_family':'free_monologue','speaker_id':'fixture','target_duration_sec':90,'log_dir':str(tmp_path)}
    _job_worker(str(job), request, str(tmp_path/'synthetic.wav'), channel)
    assert channel.send.call_count == 1
    assert channel.send.call_args[0][0]['rpc_id']
    assert json.loads(job.read_text())['status'] == 'completed'


def test_pre_checkpoint_take_retains_identity_and_practice_date(tmp_path):
    cache = StageCache(tmp_path/'stages')
    cache.seed_attempt('old-take', '2026-10-01T09:00:00')
    assert cache.attempt_id() == 'old-take' and cache.created_at() == '2026-10-01T09:00:00'
    cache.seed_attempt('different-take')
    assert cache.attempt_id() == 'old-take'


def test_recovery_reuses_validated_rubric_but_retries_failed_coaching(tmp_path, monkeypatch):
    import assess_speaking
    from assess_core.schemas import RubricResult, CoachingSummary
    from assessment_runtime.llm_client import LLMClientError
    from test_assess_speaking import _sample_report
    source = tmp_path/'synthetic.wav'; source.write_bytes(b'fixture')
    payload = _sample_report()
    audio = Mock(return_value={'duration_sec':30.,'pauses':[]})
    speech = Mock(return_value={'detected_language':'it','language_probability':.99,'text':'ciao mondo','words':[{'text':'ciao','t0':0.,'t1':.5},{'text':'mondo','t0':.6,'t1':1.}]})
    rubric = Mock(return_value=(RubricResult.from_dict(payload['rubric']), json.dumps(payload['rubric'])))
    coaching = Mock(side_effect=[LLMClientError('quota'), (CoachingSummary.from_dict(payload['coaching']), json.dumps(payload['coaching']))])
    monkeypatch.setattr(assess_speaking, 'load_audio_features', audio)
    monkeypatch.setattr(assess_speaking, 'transcribe', speech)
    monkeypatch.setattr(assess_speaking, 'generate_rubric', rubric)
    monkeypatch.setattr(assess_speaking, 'generate_coaching_summary', coaching)
    args = dict(provider='ollama',llm_model='chosen',expected_language='it',min_word_count=1,stage_cache_dir=tmp_path/'stages')
    first = assess_speaking.run_assessment(source, **args)
    assert 'coaching_unavailable' in first['report']['warnings']
    second = assess_speaking.run_assessment(source, **args)
    assert 'coaching_unavailable' not in second['report']['warnings']
    assert audio.call_count == speech.call_count == rubric.call_count == 1 and coaching.call_count == 2


def test_cloud_language_detection_never_fabricates_confidence():
    value = normalize({'text':'Ciao','words':[{'word':'Ciao','start':0,'end':.4}], 'language':'italian'})
    assert value['detected_language'] == 'it' and value['language_probability'] is None


def test_invalid_feedback_reply_is_unpublished_and_explicit_resume_can_regenerate(tmp_path, monkeypatch):
    from app_backend.cloud_runtime import CloudRuntime
    from assessment_runtime.cloud_transport import completion as hook
    from assessment_runtime.llm_client import generate_rubric, LLMSchemaError
    from test_assess_speaking import _sample_report
    primary = connection()
    state = SimpleNamespace(prefs=AppPreferences(connections=[primary],active_connection_id='source'))
    runtime = CloudRuntime(tmp_path, state, lambda _: 'fixture-key',cache_root=tmp_path/'stages')
    valid = json.dumps(_sample_report()['rubric'])
    provider = Mock(side_effect=[{'text':'{}','provider':'openrouter','model':primary.default_model},
                                 {'text':valid,'provider':'openrouter','model':primary.default_model}])
    monkeypatch.setattr(runtime,'_complete',provider)
    last = {}
    def dispatch(message):
        nonlocal last
        if message['operation'] == 'reject_reply': message = {**message, **last}
        result = runtime.handle(message)
        if message['operation'] == 'completion': last = {key:result[key] for key in ('reply_cache_id','reply_text_sha256')}
        return result
    token = hook.set(dispatch)
    try:
        args = dict(provider='openrouter',model=primary.default_model,prompt='same prompt',max_validation_retries=0)
        with pytest.raises(LLMSchemaError): generate_rubric(**args)
        assert generate_rubric(**args)[0].overall == 4
        assert generate_rubric(**args)[0].overall == 4 and provider.call_count == 2
    finally:
        hook.reset(token)


@pytest.mark.parametrize('status', [400,401,402,403,422,429,408,503])
def test_definite_paid_rejection_releases_estimate_but_uncertain_failure_retains_it(tmp_path, monkeypatch, status):
    def handler(request):
        if request.url.path.endswith('/models'): return httpx.Response(200,json=catalog('vendor/model','.000001','.000002'))
        if request.url.path.endswith('/key'): return httpx.Response(200,json={'data':{'limit':5,'limit_remaining':5}})
        return httpx.Response(status,json={'error':{'code':status}})
    transport(monkeypatch,handler)
    with pytest.raises(OpenRouterError):
        complete(root=tmp_path,connection=connection('vendor/model'),api_key='fixture-key',mode='paid',settings=CloudSettings(),prompt='test',schema=None,timeout=2)
    summary = SpendingLedger(tmp_path).summary()
    assert bool(summary['unresolved_requests']) == (status in {408,503})
    assert summary['spent_usd'] == 0
    if status not in {408,503}:
        row = next(iter(json.loads((tmp_path/'cloud-spending.json').read_text())['requests'].values()))
        assert row['reconciliation_source'] == f'provider_rejected_http_{status}'


def test_stale_validation_rejection_cannot_discard_a_replacement_reply(tmp_path):
    cache = StageCache(tmp_path/'replies')
    name = 'reply-'+'a'*64
    cache.run(name, {'prompt':'same'}, lambda: {'text':'new valid result'})
    assert not cache.reject_reply(name, hashlib.sha256(b'old invalid result').hexdigest())
    assert cache.run(name, {'prompt':'same'}, lambda: pytest.fail('New valid result must remain published'))['text'] == 'new valid result'
