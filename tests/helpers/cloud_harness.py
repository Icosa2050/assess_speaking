"""Bounded process harness shared by cloud integration and recovery tests."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]


class Backend:
    def __init__(self, root, mode, language='en'):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.set_scenario(mode, language)
        self.deadline = time.monotonic() + 60
        self.process = None
        self.log = None
        self.start()

    def set_scenario(self, mode, language='en'):
        (self.root / 'scenario.json').write_text(json.dumps({'mode': mode, 'language': language}))

    def start(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = {key: value for key, value in os.environ.items() if not ('PROXY' in key.upper() or any(part in key.upper() for part in ('API_KEY', 'ACCESS_TOKEN', 'REFRESH_TOKEN', 'SESSION_TOKEN', 'MEDIA_TOKEN')))}
        env.pop('VOSTAVO_SKIP_BOOTSTRAP', None)
        env.update(PYTHON_KEYRING_BACKEND='scripts.journey_keyring.MemoryKeyring', VOSTAVO_HOME=str(self.root / 'app'), HF_HUB_OFFLINE='1', PYTHONPATH=os.pathsep.join([str(ROOT / 'tests/helpers/cloud_guard'), str(ROOT)]), VOSTAVO_FIXTURE_GUARD_LOG=str(self.root / 'guards.jsonl'), VOSTAVO_CLOUD_FIXTURE_ROOT=str(self.root))
        self.log = (self.root / 'backend.log').open('a')
        self.process = subprocess.Popen([sys.executable, str(ROOT / 'tests/helpers/cloud_backend.py'), '--root', str(self.root), '--port', str(port)], cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.api = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5, trust_env=False, headers={'X-Vostavo-Client': 'desktop'})
        until = min(self.deadline, time.monotonic() + 20)
        while time.monotonic() < until:
            if self.process.poll() is not None:
                raise AssertionError('Fixture backend exited: ' + (self.root / 'backend.log').read_text()[-3000:])
            try:
                if self.api.get('/v1/health').status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.05)
        raise AssertionError('Fixture backend startup timeout')

    def stop(self):
        if self.process:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
            self.api.close()
            self.log.close()
            self.process = None

    def request(self, method, path, **kwargs):
        if method == 'POST' and (path == '/v1/assessments' or path.endswith('/resume')):
            body = kwargs.get('json', {})
            route = (self.api.post('/v1/assessment-route', json=body) if path == '/v1/assessments' else self.api.get(path + '-route')).json()
            kwargs['json'] = {**body, 'sharing_fingerprint': route['fingerprint']}
        response = self.api.request(method, path, **kwargs)
        assert response.status_code < 300, f'{method} {path}: {response.status_code} {response.text[:500]}'
        return response.json()

    def configure(self, chatgpt=False, fallback=False, groq_asr=False):
        def save(provider, model, key):
            data = self.request('PUT', '/v1/runtime/settings', json={'connection': {'provider_choice': provider, 'model': model, 'label': provider + model, 'api_key': key}})
            return data['active_connection_id']
        speech = save('groq', 'openai/gpt-oss-20b', 'fixture-groq')
        paid = save('openrouter', 'vendor/paid', 'fixture-analysis')
        free = save('openrouter', 'vendor/free:free', 'fixture-analysis')
        if chatgpt:
            free = 'fixture-chatgpt'
            self.request('POST', f'/v1/runtime/settings/connections/{free}/default')
        from app_core.cloud_policy import CloudSettings
        settings = CloudSettings(asr_provider='groq' if groq_asr else 'local', asr_connection_id=speech if groq_asr else '', openrouter_modes={paid: 'paid', **({} if chatgpt else {free: 'free'})}, fallback_connection_id=paid, paid_fallback_enabled=fallback)
        self.request('PUT', '/v1/runtime/cloud', json=settings.model_dump())
        return free, paid

    def submit(self, chatgpt=False, language='en', paid=False):
        from scripts.prepare_journey_audio import prepare
        prepare(self.root / 'audio')
        path = self.root / f'audio/{language}/B1/travel_story.wav'
        upload = self.request('POST', '/v1/uploads', files={'file': ('sample.wav', path.read_bytes(), 'audio/wav')})
        body = {'request_id': 'fixture-stable-submit', 'audio_id': upload['audio_id'], 'whisper': 'tiny', 'provider': 'chatgpt' if chatgpt else 'openrouter', 'llm_model': 'fixture-chatgpt-model' if chatgpt else 'vendor/paid' if paid else 'vendor/free:free', 'expected_language': language, 'feedback_language': language, 'speaker_id': 'fixture', 'task_family': 'narration', 'theme': 'Travel', 'target_duration_sec': 90}
        first = self.request('POST', '/v1/assessments', json=body)
        # Simulate losing the first accepted HTTP response: repeat the same API request.
        second = self.request('POST', '/v1/assessments', json=body)
        assert first['assessment_id'] == second['assessment_id']
        return first['assessment_id']

    def wait(self, identity):
        while time.monotonic() < self.deadline:
            value = self.request('GET', f'/v1/assessments/{identity}')
            if value['status'] in ('completed', 'failed', 'cancelled'):
                job = json.loads((self.root / 'app/jobs' / (identity + '.json')).read_text())
                try:
                    os.kill(job['worker_pid'], 0)
                except (ProcessLookupError, KeyError):
                    return value
            time.sleep(.05)
        raise AssertionError('60-second fixture deadline; dispatch phases: ' + repr(self.dispatches()))

    def dispatches(self):
        path = self.root / 'dispatch.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@contextmanager
def backend(root, mode, language='en'):
    value = Backend.__new__(Backend)
    value.process = None
    try:
        value.__init__(root, mode, language)
        yield value
    finally:
        value.stop()
