"""Local ChatGPT plan authorization. Only the backend owns renewable tokens."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

import httpx
import jwt

from app_core.secret_store import KeyringSecretStore, SessionSecretStore, SERVICE_NAME

ISSUER = 'https://auth.openai.com'
AUTHORIZE = ISSUER + '/api/accounts/authorize'
TOKEN = ISSUER + '/api/accounts/oauth/token'
RESOURCE = 'https://api.openai.com/v1'
SCOPE = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct'
_LOCK = threading.RLock()


class ChatGPTAuthError(ValueError):
    """Safe, user-facing connection failure; never includes provider bodies/tokens."""
    def __init__(self, message: str, *, status: int | None = None, invalid_grant: bool = False):
        super().__init__(message)
        self.status = status
        self.invalid_grant = invalid_grant


def _request(method: str, url: str, **kwargs) -> dict:
    try:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            r = client.request(method, url, **kwargs)
            if not 200 <= r.status_code < 300:
                if r.status_code == 400:
                    try:
                        invalid_grant = r.json().get('error') == 'invalid_grant'
                    except (ValueError, AttributeError):
                        invalid_grant = False
                    if invalid_grant:
                        raise ChatGPTAuthError('Your ChatGPT authorization expired or was revoked. Reconnect your account.', status=r.status_code, invalid_grant=True)
                if r.status_code in (401, 403):
                    raise ChatGPTAuthError('ChatGPT authorization was rejected. Reconnect your account.', status=r.status_code)
                if r.status_code == 429:
                    raise ChatGPTAuthError('ChatGPT allowance is unavailable. Check your account usage and retry later.', status=r.status_code)
                raise ChatGPTAuthError('ChatGPT could not complete this request. Please retry the connection.', status=r.status_code)
            value = r.json() if r.content else {}
            if not isinstance(value, dict):
                raise ChatGPTAuthError('Unexpected ChatGPT response. Please retry.')
            return value
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise ChatGPTAuthError('Could not reach ChatGPT. Check your connection and retry.') from None


def _read_tokens(ref: str) -> dict:
    raw = SessionSecretStore().get_secret(SERVICE_NAME, ref) or KeyringSecretStore().get_secret(SERVICE_NAME, ref)
    try:
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}



def _secure_store_supported(store) -> bool:
    if not store.is_persistent_supported():
        return False
    try:
        backend = store._keyring.get_keyring()
        return type(backend).__module__ in {
            'keyring.backends.macOS', 'keyring.backends.Windows',
            'keyring.backends.SecretService', 'keyring.backends.kwallet', 'keyring.backends.libsecret',
        }
    except Exception:  # quality: allow[broad-except] unknown keyring plugins must use session-only storage
        return False


def _write_tokens(ref: str, value: dict) -> bool:
    # A single keyring item replaces the complete rotating token set together.
    raw = json.dumps(value)
    store = KeyringSecretStore()
    if _secure_store_supported(store):
        try:
            store.set_secret(SERVICE_NAME, ref, raw)
            if store.get_secret(SERVICE_NAME, ref) == raw:
                SessionSecretStore().delete_secret(SERVICE_NAME, ref)
                return True
        except Exception:  # quality: allow[broad-except] keyring plugins may fail; keep credentials in backend memory
            pass
    # Never leave a rotated-out credential available after a restart.
    store.delete_secret(SERVICE_NAME, ref)
    SessionSecretStore().set_secret(SERVICE_NAME, ref, raw)
    return False



def _lifetime(result: dict) -> float:
    import math
    try:
        value = float(result.get('expires_in', 0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise ChatGPTAuthError('ChatGPT returned an invalid session lifetime. Reconnect your account.') from None


def _clear_tokens(ref: str) -> bool:
    store = KeyringSecretStore()
    store.delete_secret(SERVICE_NAME, ref)
    SessionSecretStore().set_secret(SERVICE_NAME, ref, '{}')
    return not bool(store.get_secret(SERVICE_NAME, ref))


def _revoke(client_id: str, refresh_token: str) -> bool:
    if not client_id or not refresh_token:
        return False
    try:
        endpoint = _trusted_endpoint(_discovery()['revocation_endpoint'])
        _request('POST', endpoint, data={'client_id': client_id, 'token': refresh_token, 'token_type_hint': 'refresh_token'})
        return True
    except (ChatGPTAuthError, KeyError):
        return False


def cached_access_token(ref: str) -> str:
    saved = _read_tokens(ref)
    return '' if saved.get('rotation_pending') else str(saved.get('access_token') or '')


def access_token(ref: str) -> str:
    with _LOCK:
        saved = _read_tokens(ref)
        if not saved.get('access_token'):
            raise ChatGPTAuthError('Reconnect your ChatGPT account.')
        if saved.get('rotation_pending'):
            raise ChatGPTAuthError('ChatGPT session renewal was interrupted. Reconnect your account.')
        try:
            fresh = float(saved.get('expires_at', 0)) > time.time() + 300
        except (ValueError, TypeError):
            raise ChatGPTAuthError('Reconnect your ChatGPT account.') from None
        if fresh:
            return saved['access_token']
        if not saved.get('refresh_token') or not saved.get('client_id'):
            raise ChatGPTAuthError('Your ChatGPT session expired. Reconnect your account.')
        # Retain a revocable token, but never replay it after an uncertain rotation.
        _write_tokens(ref, {**saved, 'rotation_pending': True})
        disk = KeyringSecretStore().get_secret(SERVICE_NAME, ref)
        try:
            pending_on_disk = not disk or bool(json.loads(disk).get('rotation_pending'))
        except (ValueError, AttributeError):
            pending_on_disk = False
        if not pending_on_disk:
            raise ChatGPTAuthError('Secure storage could not rotate this session. Reconnect your ChatGPT account.')
        try:
            result = _request('POST', TOKEN, data={
                'grant_type': 'refresh_token', 'client_id': saved['client_id'],
                'refresh_token': saved['refresh_token'], 'resource': RESOURCE,
            })
        except ChatGPTAuthError as exc:
            if exc.invalid_grant or exc.status in (401, 403):
                _clear_tokens(ref)
            elif exc.status is not None and 400 <= exc.status < 500:
                _write_tokens(ref, saved)  # Explicit rejection: no rotation took place.
            # A timeout/5xx keeps the marker and token for reconnect/revocation.
            raise
        try:
            if not result.get('access_token'):
                raise ChatGPTAuthError('ChatGPT did not renew the session. Reconnect your account.')
            lifetime = _lifetime(result)
            scopes = str(result.get('scope') or saved.get('scope', '')).split()
            if 'chatgpt.tokens.use.direct' not in scopes:
                raise ChatGPTAuthError('This account has not authorized ChatGPT plan usage.')
        except ChatGPTAuthError:
            _revoke(saved['client_id'], result.get('refresh_token') or saved['refresh_token'])
            _clear_tokens(ref)
            raise
        updated = {**saved, 'access_token': result['access_token'],
                   'refresh_token': result.get('refresh_token') or saved['refresh_token'],
                   'scope': ' '.join(scopes),
                   'expires_at': time.time() + lifetime, 'rotation_pending': False}
        _write_tokens(ref, updated)
        return updated['access_token']


def account_models(token: str) -> list[dict[str, str]]:
    data = _request('GET', RESOURCE + '/models', headers={'Authorization': 'Bearer ' + token})
    items = data.get('models')
    if not isinstance(items, list):
        raise ChatGPTAuthError('ChatGPT returned an invalid model catalog.')
    return [{'slug': str(m['slug']), 'display_name': str(m.get('display_name') or m['slug'])}
            for m in items if isinstance(m, dict) and m.get('visibility') == 'list' and m.get('slug')]


def _discovery() -> dict:
    data = _request('GET', ISSUER + '/.well-known/openid-configuration')
    if data.get('issuer') != ISSUER:
        raise ChatGPTAuthError('Unexpected ChatGPT identity issuer.')
    return data


def _trusted_endpoint(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'auth.openai.com' or parsed.username or parsed.fragment:
        raise ChatGPTAuthError('Unexpected ChatGPT identity endpoint.')
    return url


def validate_identity(token: str, client_id: str, nonce: str) -> dict:
    discovery = _discovery()
    try:
        keys = jwt.PyJWKSet.from_dict(_request('GET', _trusted_endpoint(discovery['jwks_uri'])))
        kid = jwt.get_unverified_header(token).get('kid')
        matches = [key for key in keys.keys if key.key_id == kid]
        if not kid or len(matches) != 1:
            raise ChatGPTAuthError('Could not verify the ChatGPT signing key.')
        claims = jwt.decode(token, matches[0].key, algorithms=['RS256'], audience=client_id,
                            issuer=ISSUER, options={'require': ['exp', 'iss', 'aud', 'sub', 'nonce']})
        if not secrets.compare_digest(str(claims['nonce']), nonce) or not claims['sub']:
            raise ChatGPTAuthError('ChatGPT sign-in identity did not match. Please reconnect.')
        return claims
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        raise ChatGPTAuthError('Could not verify ChatGPT sign-in. Please reconnect.') from None


class ChatGPTAuth:
    def __init__(self, root: Path):
        self.root = root
        self.pending: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.on_connected = lambda _account, _models, _persistent: None
        self.path = root / 'chatgpt-registrations.json'
        try:
            self.records = json.loads(self.path.read_text()) if self.path.exists() else {'host_id': 'urn:uuid:' + str(uuid4()), 'accounts': {}}
            if not isinstance(self.records, dict) or not isinstance(self.records.get('host_id'), str) or not isinstance(self.records.get('accounts'), dict):
                raise ValueError('Invalid registration storage')
        except (ValueError, UnicodeError):
            # Preserve damaged metadata for recovery without preventing local practice.
            self.path.rename(self.path.with_suffix(f'.corrupt-{time.time_ns()}.json'))
            self.records = {'host_id': 'urn:uuid:' + str(uuid4()), 'accounts': {}}
        self._save()

    def _save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            os.chmod(tmp, 0o600)
            json.dump(self.records, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def start(self, connection_id: str = '') -> dict:
        with self.lock:
            record = self.records['accounts'].get(connection_id) if connection_id else None
            if connection_id and record is None:
                raise ChatGPTAuthError('This ChatGPT account registration was not found.')
            if any(p['status'] in ('waiting', 'processing') and p['expires'] > time.time() for p in self.pending.values()):
                raise ChatGPTAuthError('A ChatGPT sign-in is already open. Finish or cancel it first.')
            self.pending = {k: p for k, p in self.pending.items() if p['expires'] > time.time()}
            attempt = secrets.token_urlsafe(24)
            state, verifier, nonce = (secrets.token_urlsafe(32) for _ in range(3))
            p = {'status': 'waiting', 'expires': time.time() + 600, 'state': state, 'verifier': verifier,
                 'nonce': nonce, 'record': record, 'connection_id': connection_id or uuid4().hex, 'detail': ''}
            owner = self

            class Callback(BaseHTTPRequestHandler):
                timeout = 10

                def log_message(self, *_args):
                    pass  # OAuth callback query contains a code; never log it.

                def do_GET(self):
                    parsed = urlparse(self.path)
                    if parsed.path != '/auth/callback':
                        self.send_error(404)
                        return
                    owner.finish(attempt, parse_qs(parsed.query))
                    body = b'ChatGPT sign-in processed. Return to Vostavo to see the result.'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Referrer-Policy', 'no-referrer')
                    self.end_headers()
                    self.wfile.write(body)

            server = HTTPServer(('127.0.0.1', 0), Callback)
            server.timeout = 0.5
            p['redirect_uri'] = f'http://127.0.0.1:{server.server_port}/auth/callback'
            self.pending[attempt] = p

            def listen():
                try:
                    while p['status'] == 'waiting' and time.time() < p['expires']:
                        server.handle_request()
                    if p['status'] == 'waiting':
                        p.update(status='expired', detail='Sign-in timed out. Please try again.')
                finally:
                    server.server_close()

            threading.Thread(target=listen, daemon=True).start()
            args = {'client_id': record['client_id'] if record else 'dynamic_agent_client',
                    'ext_agent_host_id': self.records['host_id'], 'response_type': 'code',
                    'redirect_uri': p['redirect_uri'], 'scope': SCOPE, 'resource': RESOURCE,
                    'state': state, 'nonce': nonce, 'code_challenge_method': 'S256',
                    'code_challenge': base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')}
            if not record:
                args['agent_name_hint'] = 'Vostavo'
            # Omit id_token_hint so credentials never travel through our frontend.
            p['authorization_url'] = AUTHORIZE + '?' + urlencode(args)
            return {'attempt_id': attempt, 'authorization_url': p['authorization_url']}

    def finish(self, attempt: str, query: dict):
        with self.lock:
            p = self.pending.get(attempt)
            if not p or p['status'] != 'waiting' or time.time() >= p['expires']:
                return
            if query.get('state') != [p['state']]:
                return  # An unrelated browser request cannot cancel the pending login.
            p['status'] = 'processing'
        result = {}
        client_id = ''
        committed = False
        ref = ''
        previous_session = {}
        try:
            if query.get('error'):
                raise ChatGPTAuthError('ChatGPT sign-in was cancelled or permission was denied.')
            record = p['record']
            client_id = (query.get('client_id') or [record['client_id'] if record else ''])[0]
            if not client_id or client_id == 'dynamic_agent_client' or (record and client_id != record['client_id']):
                raise ChatGPTAuthError('ChatGPT returned an incomplete or different account registration.')
            if len(query.get('code', [])) != 1 or len(query.get('client_id', [client_id])) != 1:
                raise ChatGPTAuthError('ChatGPT did not return a valid authorization code.')
            result = _request('POST', TOKEN, data={'grant_type': 'authorization_code', 'client_id': client_id,
                'code': query['code'][0], 'code_verifier': p['verifier'], 'redirect_uri': p['redirect_uri'], 'resource': RESOURCE})
            claims = validate_identity(str(result.get('id_token') or ''), client_id, p['nonce'])
            if record and claims['sub'] != record['subject']:
                raise ChatGPTAuthError('Choose the original ChatGPT account or add a new connection.')
            if 'chatgpt.tokens.use.direct' not in str(result.get('scope', '')).split():
                raise ChatGPTAuthError('Enable ChatGPT plan usage for Vostavo when signing in.')
            if not result.get('access_token') or not result.get('refresh_token') or _lifetime(result) <= 0:
                raise ChatGPTAuthError('ChatGPT did not grant a renewable inference session.')
            models = account_models(result['access_token'])
            if not models:
                raise ChatGPTAuthError('This ChatGPT account has no available inference models.')
            ref = 'chatgpt:' + p['connection_id'] + ':' + uuid4().hex
            stored = {k: result[k] for k in ('access_token', 'refresh_token', 'scope')}
            stored.update(client_id=client_id, subject=claims['sub'], expires_at=time.time()+_lifetime(result))
            with self.lock, _LOCK:
                if p['status'] != 'processing':
                    return
                persistent = _write_tokens(ref, stored)
                account = {'connection_id': p['connection_id'], 'client_id': client_id, 'subject': claims['sub'],
                           'secret_ref': ref, 'models': models, 'persistent': persistent, 'pending_publish': True}
                self.records['accounts'][p['connection_id']] = account
                self._save()
                self.on_connected(account, models, persistent)
                committed = True
                account['pending_publish'] = False
                p.update(status='connected', account=account, models=models, persistent=persistent)
                self._save()
                if record and record['secret_ref'] != ref:
                    previous_session = _read_tokens(record['secret_ref'])
                    _clear_tokens(record['secret_ref'])
        except Exception as exc:  # quality: allow[broad-except] callback must reach a terminal state without exposing upstream payloads
            if p['status'] == 'processing':
                p.update(status='failed', detail=str(exc) if isinstance(exc, ChatGPTAuthError) else 'Could not save ChatGPT connection. Please reconnect.')
        finally:
            if committed and previous_session.get('refresh_token') and previous_session['refresh_token'] != result.get('refresh_token'):
                if not _revoke(client_id, previous_session['refresh_token']):
                    p['detail'] = 'Connected. Previous-session revocation was not confirmed; check Vostavo access in ChatGPT settings.'
            if not committed and result.get('refresh_token'):
                if ref:
                    _clear_tokens(ref)
                    with self.lock:
                        if p.get('record'):
                            self.records['accounts'][p['connection_id']] = p['record']
                        else:
                            self.records['accounts'].pop(p['connection_id'], None)
                        self._save()
                if not _revoke(client_id, result['refresh_token']):
                    p['detail'] += ' Remote revocation was not confirmed; remove Vostavo access in ChatGPT account settings.'
            p.pop('verifier', None)
            p.pop('nonce', None)

    def status(self, attempt: str) -> dict:
        with self.lock:
            p = self.pending.get(attempt)
            if p is None:
                raise ChatGPTAuthError('Sign-in attempt was not found. Please try again.')
            return {k: p[k] for k in ('status', 'detail', 'connection_id', 'models', 'persistent') if k in p}

    def cancel(self, attempt: str):
        with self.lock:
            p = self.pending.get(attempt)
            if p and p['status'] in ('waiting', 'processing'):
                p.update(status='cancelled', detail='Sign-in cancelled.')

    def disconnect(self, connection_id: str, before_clear=None) -> bool:
        with self.lock, _LOCK:
            if before_clear:
                before_clear()
            for pending in self.pending.values():
                if pending['connection_id'] == connection_id:
                    pending.update(status='cancelled', detail='Account disconnected.')
            account = self.records['accounts'].get(connection_id)
            if not account:
                return False
            ref = account['secret_ref']
            saved = _read_tokens(ref)
            account['pending_publish'] = False
            self._save()
            cleared = _clear_tokens(ref)
        # Revocation can be slow; local status and other accounts remain responsive.
        return _revoke(account['client_id'], saved.get('refresh_token', '')) and cleared

    def close(self):
        with self.lock:
            for p in self.pending.values():
                if p['status'] in ('waiting', 'processing'):
                    p.update(status='cancelled', detail='App closed.')

    def open_browser(self, attempt: str) -> bool:
        with self.lock:
            p = self.pending.get(attempt)
            if not p or p['status'] != 'waiting' or p['expires'] <= time.time():
                raise ChatGPTAuthError('Start a new sign-in before opening the browser.')
            url = p['authorization_url']
        return webbrowser.open(url, new=2, autoraise=True)
