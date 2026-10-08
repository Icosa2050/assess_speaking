"""Single-attempt OpenRouter PKCE login with a one-time localhost callback."""
from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import secrets
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import webbrowser

import httpx


class OpenRouterAuth:
    def __init__(self, publish):
        self.publish = publish
        self.lock = threading.RLock()
        self.pending = {}
        self.server = None
        self.timer = None

    def start(self):
        with self.lock:
            self.close()
            verifier = secrets.token_urlsafe(64)
            state = secrets.token_urlsafe(32)
            attempt_id = secrets.token_hex(16)
            owner = self
            class Callback(BaseHTTPRequestHandler):
                def do_GET(self):
                    url = urlsplit(self.path)
                    values = parse_qs(url.query)
                    success = False
                    with owner.lock:
                        item = owner.pending.get(attempt_id)
                        valid = (url.path == '/callback' and self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'
                                 and item and item['status'] == 'waiting' and item['expires'] > time.time()
                                 and secrets.compare_digest(values.get('state', [''])[0], state) and values.get('code'))
                        if valid:
                            item['status'] = 'processing'
                    if valid:
                        try:
                            with httpx.Client(timeout=20, follow_redirects=False) as client:
                                response = client.post('https://openrouter.ai/api/v1/auth/keys', json={
                                    'code': values['code'][0], 'code_verifier': verifier, 'code_challenge_method': 'S256'})
                                response.raise_for_status()
                                key = response.json()['key']
                            if not isinstance(key, str) or not key:
                                raise ValueError('Missing key')
                            with owner.lock:
                                if item['status'] != 'processing':
                                    raise ValueError('Authorization was cancelled')
                                owner.publish(key)
                                item['status'] = 'connected'
                            success = True
                        except Exception:  # quality: allow[broad-except] login boundary never exposes exchange or storage errors containing credentials
                            with owner.lock:
                                if item['status'] == 'processing':
                                    item['status'] = 'failed'
                                    item['detail'] = 'OpenRouter login did not complete. Try again or enter a key manually.'
                    self.send_response(200 if success else 400)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(b'Connected. Return to Vostavo.' if success else b'Authorization could not be accepted. Return to Vostavo.')
                    if valid:
                        threading.Thread(target=owner._stop_server, args=(self.server,), daemon=True).start()
                def log_message(self, *_args):
                    pass
                def setup(self):
                    super().setup()
                    self.connection.settimeout(5)
            server = ThreadingHTTPServer(('127.0.0.1', 0), Callback)
            server.daemon_threads = True
            self.server = server
            callback = f'http://127.0.0.1:{server.server_port}/callback'
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
            url = 'https://openrouter.ai/auth?' + urlencode({'callback_url': callback, 'code_challenge': challenge,
                                                          'code_challenge_method': 'S256', 'state': state})
            self.pending[attempt_id] = {'status': 'waiting', 'expires': time.time() + 600, 'authorization_url': url}
            threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True).start()
            self.timer = threading.Timer(600, self._expire, args=(attempt_id, server))
            self.timer.daemon = True
            self.timer.start()
            return {'attempt_id': attempt_id, 'authorization_url': url}

    def status(self, attempt_id):
        with self.lock:
            item = self.pending.get(attempt_id)
            if not item:
                return {'status': 'failed', 'detail': 'Authorization attempt not found.'}
            if item['status'] == 'waiting' and item['expires'] <= time.time():
                item['status'] = 'expired'
            return {key: value for key, value in item.items() if key in {'status', 'detail'}}

    def open_browser(self, attempt_id):
        with self.lock:
            item = self.pending.get(attempt_id)
            if not item or self.status(attempt_id)['status'] != 'waiting':
                return False
            url = item['authorization_url']
        return webbrowser.open(url)

    def _expire(self, attempt_id, server):
        with self.lock:
            item = self.pending.get(attempt_id)
            if item and item['status'] in {'waiting', 'processing'}:
                item['status'] = 'expired'
        self._stop_server(server)

    def _stop_server(self, server):
        with self.lock:
            if self.server is not server:
                return
            self.server = None
            if self.timer:
                self.timer.cancel()
                self.timer = None
        server.shutdown()
        server.server_close()

    def close(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            for item in self.pending.values():
                if item['status'] in {'waiting', 'processing'}:
                    item['status'] = 'cancelled'
            if self.server:
                self._stop_server(self.server)
