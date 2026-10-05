from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app_backend.desktop_session import install_desktop_session


def client():
    app = FastAPI()
    @app.get('/v1/health')
    def health():
        return {'ok': True}
    @app.get('/v1/history/example/audio')
    def audio():
        return {'audio': True}
    install_desktop_session(app, 'a' * 64, 8123, 'b' * 64)
    return TestClient(app, base_url='http://127.0.0.1:8123')


def test_desktop_api_requires_launch_token_and_rejects_foreign_hosts():
    c = client()
    assert c.get('/v1/health').status_code == 401
    assert c.get('/v1/health', headers={'X-Vostavo-Session': 'wrong'}).status_code == 401
    good = c.get('/v1/health', headers={'X-Vostavo-Session': 'a' * 64})
    assert good.status_code == 200
    assert good.headers['Referrer-Policy'] == 'no-referrer'
    assert c.get('/v1/health', headers={'Host': 'attacker.example', 'X-Vostavo-Session': 'a' * 64}).status_code == 403


def test_query_capability_only_authorizes_read_only_audio():
    c = client()
    assert c.get('/v1/history/example/audio?session=' + 'b' * 64).status_code == 200
    assert c.get('/v1/health?session=' + 'a' * 64).status_code == 401
    assert c.post('/v1/history/example/audio?session=' + 'a' * 64).status_code == 401
    assert c.options('/v1/health').status_code != 401
    assert c.get('/v1/history/example/audio?session=%C3%A9').status_code == 401


def test_media_token_cannot_authorize_api_requests():
    c = client()
    assert c.get('/v1/health', headers={'X-Vostavo-Session': 'b' * 64}).status_code == 401
    assert c.get('/v1/history/example/audio?session=' + 'a' * 64).status_code == 401
