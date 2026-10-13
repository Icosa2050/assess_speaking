"""Isolated real backend with provider HTTP fixtures. Never imported by production."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# This launcher is re-imported by multiprocessing spawn. Keep local ASR
# deterministic for coaching/recovery tests, independently of the real Groq
# HTTP normalizer's deliberately unknown language confidence.
import assess_speaking


def local_fixture_asr(audio, *args, **kwargs):
    root = Path(os.environ['VOSTAVO_CLOUD_FIXTURE_ROOT'])
    scenario = json.loads((root / 'scenario.json').read_text())
    cases = json.loads((ROOT / 'tests/fixtures/asr/sample_references.json').read_text())['cases']
    case = next(c for c in cases if c['language'] == scenario.get('language', 'en') and c['goal'] == 'B1')
    tokens = case['text'].split()
    step = assess_speaking.load_audio_features(audio)['duration_sec'] / len(tokens)
    with (root / 'dispatch.jsonl').open('a') as log:
        log.write(json.dumps({'phase': 'asr', 'route': 'local', 'pid': os.getpid()}) + '\n')
    return {'text': case['text'], 'detected_language': case['language'], 'language_probability': .99,
            'compute_type_used': 'fixture', 'compute_fallback_used': False,
            'words': [{'text': word, 't0': i * step, 't1': (i + .8) * step} for i, word in enumerate(tokens)]}


if os.environ.get('VOSTAVO_CLOUD_FIXTURE_ROOT'):
    assess_speaking.transcribe = local_fixture_asr


def install_transport(root):
    import httpx
    import httpx2
    from assessment_runtime import responses_client
    from app_core.cloud_policy import SpendingLedger
    original = httpx.Client
    original2 = httpx2.Client
    def handler(request):
        scenario = json.loads((root / 'scenario.json').read_text())
        path = request.url.path
        assert request.url.host in {'api.groq.com', 'openrouter.ai', 'api.openai.com', 'auth.openai.com'}
        phase = 'other'
        body = None
        if path.endswith('/transcriptions'):
            phase = 'asr'
            assert request.headers['Authorization'] == 'Bearer fixture-groq'
            assert b'fLaC' in request.content and b'whisper-large-v3' in request.content
        elif path.endswith('/chat/completions') or path.endswith('/responses'):
            body = json.loads(request.content)
            schema = body.get('response_format', {}).get('json_schema', body.get('text', {}).get('format', {}))
            phase = 'coaching' if 'coaching' in schema.get('name', '').lower() else 'rubric'
        route = 'paid' if body and body.get('model') == 'vendor/paid' else 'free'
        entry = {'phase': phase, 'route': route, 'path': path, 'pid': os.getpid()}
        if route == 'paid' and phase in ('rubric', 'coaching'):
            assert SpendingLedger(root / 'app').summary()['reserved_usd'] > 0
            entry['reserved_before_post'] = True
        with (root / 'dispatch.jsonl').open('a') as log:
            log.write(json.dumps(entry) + '\n')
        if path.endswith('/auth/keys'):
            return httpx.Response(200, json={'key': 'fixture-pkce-key'})
        if path.endswith('/oauth/token'):
            from urllib.parse import parse_qs
            assert parse_qs(request.content.decode())['refresh_token'] == ['fixture-refresh']
            return httpx.Response(200, json={'access_token': 'fixture-renewed', 'refresh_token': 'fixture-refresh-next', 'expires_in': 3600})
        if path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': model, 'pricing': {'prompt': price, 'completion': price}, 'supported_parameters': ['response_format']} for model, price in [('vendor/free:free', '0'), ('vendor/paid', '.000001')]]})
        if path.endswith('/key'):
            return httpx.Response(200, json={'data': {'limit': 5, 'limit_remaining': 5}})
        if phase == 'asr':
            cases = json.loads((ROOT / 'tests/fixtures/asr/sample_references.json').read_text())['cases']
            case = next(c for c in cases if c['language'] == scenario.get('language', 'en') and c['goal'] == 'B1')
            words = case['text'].split()
            return httpx.Response(200, json={'text': case['text'], 'language': 'english' if case['language'] == 'en' else 'italian', 'words': [{'word': word, 'start': i * .35, 'end': i * .35 + .3} for i, word in enumerate(words)]})
        mode = scenario['mode']
        if phase in ('rubric', 'coaching'):
            if mode in ('paid-fallback', 'disabled') and route == 'free':
                return httpx.Response(429, json={'error': 'fixture quota'})
            if mode == 'auth':
                return httpx.Response(401, json={'error': 'fixture auth'})
            if mode == 'concurrent-paid':
                (root / 'post-entered').touch()
                until = time.monotonic() + 15
                while not (root / 'release').exists() and time.monotonic() < until:
                    time.sleep(.02)
            if mode == 'unknown-paid':
                (root / 'post-entered').touch()
                until = time.monotonic() + 15
                while not (root / 'release').exists() and time.monotonic() < until:
                    time.sleep(.02)
                raise httpx.ReadTimeout('Fixture ambiguous paid outcome')
            if mode == 'unknown':
                raise httpx.ReadTimeout('Fixture ambiguous outcome')
            if mode == 'resume' and phase == 'coaching':
                return httpx.Response(503, json={'error': 'fixture interruption'})
            from tests.test_generation_validation import rubric_payload, coaching_payload
            value = coaching_payload() if phase == 'coaching' else rubric_payload()
            if mode == 'schema':
                value = {'invalid': True}
            text = json.dumps(value)
            if path.endswith('/responses'):
                assert request.headers['Authorization'] == 'Bearer fixture-renewed'
                events = [{'type': 'response.output_text.delta', 'delta': text}, {'type': 'response.completed', 'response': {'id': 'fixture', 'status': 'completed'}}]
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())
            assert body['provider']['allow_fallbacks'] is False
            if route == 'free':
                assert body['provider']['max_price']['prompt'] == 0
            return httpx.Response(200, json={'choices': [{'message': {'content': text}}], 'usage': {'cost': .01 if route == 'paid' else 0}})
        raise AssertionError('Unexpected fixture provider request')
    class FixtureClient(original):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(handler), trust_env=False, **kw)
    httpx.Client = FixtureClient
    def handler2(request):
        value = handler(request)
        return httpx2.Response(value.status_code, headers=dict(value.headers), content=value.content)
    responses_client.DefaultHttpxClient = lambda **kw: original2(transport=httpx2.MockTransport(handler2), trust_env=False, **kw)
    from assessment_runtime.checkpoints import StageCache
    original_run = StageCache.run
    def published(cache, name, identity, produce):
        value = original_run(cache, name, identity, produce)
        scenario = json.loads((root / 'scenario.json').read_text())
        if name.startswith('reply-') and scenario['mode'] in ('worker-loss', 'backend-loss') and not (root / 'published').exists():
            (root / 'published').write_text(name)
            until = time.monotonic() + 15
            while not (root / 'release').exists() and time.monotonic() < until:
                time.sleep(.02)
        return value
    StageCache.run = published
    import webbrowser
    webbrowser.open = lambda _url: False


def main():
    import uvicorn
    from app_backend.app import create_app, _load_persisted_state
    from app_backend.config import build_backend_runtime_config
    from app_core.secret_store import set_secret, SessionSecretStore, SERVICE_NAME
    from app_core.services import save_provider_connection
    from app_core.state import ProviderConnection
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert os.environ.get('PYTHON_KEYRING_BACKEND') == 'scripts.journey_keyring.MemoryKeyring'
    assert os.environ.get('VOSTAVO_FIXTURE_GUARD_LOG')
    config = build_backend_runtime_config(app_data_dir=args.root / 'app', cache_dir=args.root / 'cache', port=args.port)
    install_transport(args.root)
    app = create_app(config)
    state = _load_persisted_state(config)
    for conn in state.prefs.connections:
        if conn.provider_kind != 'chatgpt':
            set_secret(conn.secret_ref, 'fixture-groq' if conn.provider_kind == 'groq' else 'fixture-analysis')
    scenario = json.loads((args.root / 'scenario.json').read_text())
    if scenario['mode'] == 'chatgpt':
        conn = ProviderConnection(connection_id='fixture-chatgpt', provider_kind='chatgpt', default_model='fixture-chatgpt-model', secret_ref='fixture-chatgpt-ref')
        SessionSecretStore().set_secret(SERVICE_NAME, conn.secret_ref, json.dumps({'access_token': 'fixture-expired', 'refresh_token': 'fixture-refresh', 'expires_at': 0, 'client_id': 'fixture-client', 'scope': 'chatgpt.tokens.use.direct'}))
        save_provider_connection(state, conn, persist_draft=False)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
