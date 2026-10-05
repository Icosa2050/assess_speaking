#!/usr/bin/env python3
"""Explicit live connection checks using Vostavo's real save/reload/probe/delete API."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from fastapi.testclient import TestClient
from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_core.runtime_providers import default_setup_base_url
from app_core.secret_store import SecretStoreStatus, SERVICE_NAME
from app_core import secret_store

GROQ_MODELS = ('openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b')
KEY_NAMES = {'groq': 'GROQ_API_KEY', 'xai': 'XAI_API_KEY', 'openrouter': 'OPENROUTER_API_KEY',
             'ollama_cloud': 'OLLAMA_API_KEY', 'ollama_local': 'OLLAMA_LOCAL_API_KEY',
             'lmstudio_local': 'LMSTUDIO_API_KEY', 'openai_compatible': 'COMPATIBLE_API_KEY'}
REQUIRED_KEYS = {'groq', 'xai', 'openrouter', 'ollama_cloud'}


class CheckFailure(RuntimeError):
    """Fixed test-stage diagnostic, safe for machine-readable evidence."""



def read_keys(env_file: Path | None) -> dict[str, str]:
    """Read only named credentials. No shell evaluation, interpolation or env mutation."""
    values = {}
    if env_file:
        for line in env_file.read_text().splitlines():
            candidate = line.strip().removeprefix('export ').strip()
            name, sep, raw = candidate.partition('=')
            if sep and name.strip() in KEY_NAMES.values():
                try:
                    parts = shlex.split(raw, comments=True)
                except ValueError:
                    raise ValueError('Invalid quoted credential assignment in env file') from None
                if not parts:
                    continue
                if len(parts) != 1:
                    raise ValueError('Expected a single credential value in env file')
                values[name.strip()] = parts[0]
    for name in KEY_NAMES.values():
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


def loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Expected a plain HTTP loopback URL without credentials or query')
    return value.rstrip('/')


@contextmanager
def isolated_keyring():
    """Substitute only storage. Provider HTTP and application routes remain real."""
    values = {}
    fake = SimpleNamespace(get_password=lambda service, ref: values.get((service, ref)),
        set_password=lambda service, ref, value: values.__setitem__((service, ref), value),
        delete_password=lambda service, ref: values.pop((service, ref), None))
    # Supplied keys enter through the save API only. Never inherit another
    # provider's environment fallback while testing a keyless local connection.
    cleared_env = {name: '' for name in {*KEY_NAMES.values(), 'LLM_API_KEY'}}
    with patch.dict(os.environ, cleared_env), patch('app_core.secret_store._SESSION_SECRETS', {}), patch('app_core.secret_store._load_keyring_module', return_value=(fake, SecretStoreStatus(True, 'isolated-test'))):
        try:
            yield values
        finally:
            values.clear()


def checked(response, stage: str):
    # Never emit response bodies: another provider may echo a request/credential.
    if response.status_code != 200:
        raise CheckFailure(f'{stage}: HTTP {response.status_code}')
    return response.json()


def safe_failure(exc: Exception) -> str:
    return str(exc) if type(exc) is CheckFailure else type(exc).__name__


def ensure_no_key(value: object, key: str):
    if key and key in json.dumps(value):
        raise CheckFailure('Credential exposed in API response')


def check_key_connection(provider: str, model: str, key: str, base_url: str) -> dict:
    result = {'provider': provider, 'model': model, 'status': 'failed', 'checks': []}
    with TemporaryDirectory(prefix='vostavo-connection-') as temporary, isolated_keyring() as secrets:
        root = Path(temporary)
        config = build_backend_runtime_config(port=8819, app_data_dir=root/'app', cache_dir=root/'cache')
        draft = {'provider_choice': provider, 'label': 'Automated connection test', 'model': model,
                 'base_url': base_url, 'api_key': key}
        try:
            with TestClient(create_app(config), raise_server_exceptions=False) as client:
                saved = checked(client.put('/v1/runtime/settings', json={
                    'ui_locale': 'en', 'whisper_model': 'tiny', 'connection': draft}), 'save')
                ensure_no_key(saved, key)
                connection = saved['connections'][0]
                ref = connection['connection_id']
                if key and (secrets.get((SERVICE_NAME, 'connection:' + ref)) != key
                            or connection['secret_state'] != 'present'):
                    raise CheckFailure('Credential did not reach isolated persistent storage')
                if key and provider in {'groq', 'xai'} and connection['provider_metadata'].get('persistent') is not True:
                    raise CheckFailure('Unexpected session-only credential fallback')
                result['checks'].append('save')
            secret_store._SESSION_SECRETS.clear()
            # Fresh application + cleared session storage forces a stored-key read.
            with TestClient(create_app(config), raise_server_exceptions=False) as client:
                try:
                    loaded = checked(client.get('/v1/runtime/settings'), 'reload')
                    ensure_no_key(loaded, key)
                    if loaded['connections'][0]['model'] != model:
                        raise CheckFailure('Saved model changed on reload')
                    result['checks'].append('reload')
                    stored_draft = {**draft, 'connection_id': ref, 'api_key': ''}
                    probe = checked(client.post('/v1/runtime/settings/test-connection', json={
                        'connection': stored_draft}), 'saved-key probe')
                    ensure_no_key(probe, key)
                    if probe.get('provider') != connection['provider_key']:
                        raise CheckFailure('Wrong provider answered the probe')
                    result['checks'].append('saved-connection inference')
                except Exception as exc:  # quality: allow[broad-except] live-test boundary must never leak unexpected provider exception text
                    result['reason'] = safe_failure(exc)
                finally:
                    try:
                        checked(client.delete('/v1/runtime/settings/connections/' + ref), 'delete')
                        after = checked(client.get('/v1/runtime/settings'), 'verify delete')
                        ensure_no_key(after, key)
                        if after['connections'] or secrets or secret_store._SESSION_SECRETS:
                            raise CheckFailure('Connection or secret remains after deletion')
                        result['checks'].append('delete')
                    except Exception as exc:  # quality: allow[broad-except] preserve primary failure and record cleanup failure without secret-bearing text
                        result['cleanup_error'] = safe_failure(exc)
            if key and any(key.encode() in path.read_bytes() for path in root.rglob('*') if path.is_file()):
                raise CheckFailure('Credential exposed in application files')
            if 'reason' not in result and 'cleanup_error' not in result:
                result['status'] = 'passed'
        except Exception as exc:  # quality: allow[broad-except] sanitized test boundary produces a failed result instead of logging provider exception bodies
            result['reason'] = safe_failure(exc)
    return result

def check_saved_chatgpt(backend: str, connection_id: str) -> dict:
    """Exercise the running app's token broker; never export tokens or revoke a session."""
    endpoint = loopback_url(backend)
    result = {'provider': 'chatgpt', 'status': 'failed'}
    try:
        with httpx.Client(base_url=endpoint, timeout=150, follow_redirects=False, trust_env=False) as client:
            settings = checked(client.get('/v1/runtime/settings'), 'read signed-in connection')
            connection = next((c for c in settings['connections'] if c['connection_id'] == connection_id
                               and c['provider_key'] == 'chatgpt'), None)
            if connection is None:
                return {**result, 'status': 'blocked', 'reason': 'Sign in through this backend first; specify its ChatGPT connection ID'}
            draft = {'connection_id': connection_id, 'provider_choice': 'chatgpt', 'model': connection['model'],
                     'base_url': connection['base_url'], 'label': 'Existing ChatGPT test connection', 'api_key': ''}
            checked(client.post('/v1/runtime/settings/test-connection', json={'connection': draft}), 'ChatGPT saved-session probe')
            return {**result, 'status': 'passed', 'checks': ['saved session', 'model discovery', 'inference'],
                    'scope': 'Existing authorization only; initial consent and forced expiry/revocation not exercised'}
    except Exception as exc:  # quality: allow[broad-except] live-test boundary only reports fixed stages or exception types
        return {**result, 'reason': safe_failure(exc)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--providers', default='groq', help='Comma-separated setup choices, or chatgpt')
    parser.add_argument('--env-file', type=Path, help='Optional .env; only provider-specific key names are read')
    parser.add_argument('--model', action='append', default=[], metavar='PROVIDER=MODEL', help='Repeat to select models; explicit Groq selections replace its three defaults')
    parser.add_argument('--base-url', help='Local OpenAI-compatible endpoint (loopback only)')
    parser.add_argument('--chatgpt-backend', help='Running local backend where you manually signed in')
    parser.add_argument('--chatgpt-connection-id', help='Explicit saved ChatGPT test connection ID')
    parser.add_argument('--output', type=Path, help='Optional new JSON report; never overwrites an existing file')
    args = parser.parse_args(argv)
    providers = args.providers.split(',')
    if len(set(providers)) != len(providers) or any(p not in {*KEY_NAMES, 'chatgpt'} for p in providers):
        parser.error('Unknown or duplicate provider choice')
    selected = {}
    for entry in args.model:
        provider, sep, model = entry.partition('=')
        if not sep or provider not in providers or provider == 'chatgpt' or not model.strip():
            parser.error('Each model must name a selected API-key/local provider')
        selected.setdefault(provider, []).append(model.strip())
    try:
        keys = read_keys(args.env_file)
        if args.base_url:
            loopback_url(args.base_url)
        if args.chatgpt_backend:
            loopback_url(args.chatgpt_backend)
    except (ValueError, OSError):
        parser.error('Could not read credential file or validate loopback endpoint')
    if args.base_url and 'openai_compatible' not in providers:
        parser.error('--base-url applies only to openai_compatible')
    if args.output and args.output.exists():
        parser.error('Output file already exists')
    results = []
    for provider in providers:
        if provider == 'chatgpt':
            if not args.chatgpt_backend or not args.chatgpt_connection_id:
                results.append({'provider': provider, 'status': 'blocked', 'reason': 'Manual browser sign-in and explicit backend/connection ID required'})
            else:
                results.append(check_saved_chatgpt(args.chatgpt_backend, args.chatgpt_connection_id))
            continue
        key = keys.get(KEY_NAMES[provider], '')
        models = selected.get(provider, list(GROQ_MODELS) if provider == 'groq' else [])
        if (provider in REQUIRED_KEYS and not key) or not models or (provider == 'openai_compatible' and not args.base_url):
            results.append({'provider': provider, 'status': 'blocked', 'reason': 'Required provider key, explicit model, or local endpoint is missing'})
            continue
        url = args.base_url if provider == 'openai_compatible' else default_setup_base_url(provider)
        for model in models:
            results.append(check_key_connection(provider, model, key, url))
    report = {'created_at': datetime.now(UTC).isoformat(), 'scope': 'Connection lifecycle and small synthetic capability probes; no learner audio', 'results': results}
    content = json.dumps(report, indent=2)
    print(content)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as output:
                output.write(content + '\n')
        except OSError:
            print('Report file could not be created; results are printed above.', file=sys.stderr)
            return 1
    return 0 if results and all(r['status'] == 'passed' for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
