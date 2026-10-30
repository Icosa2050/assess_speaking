"""Preferences writes must remain independent of provider connections."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest import mock

from fastapi.testclient import TestClient

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_core.services import load_workspace_prefs, save_workspace_prefs


def test_preferences_patch_without_connection_and_validates_both_fields(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8765)
    with mock.patch.dict("os.environ", {"VOSTAVO_SESSION_TOKEN": ""}), TestClient(create_app(config)) as client:
        for locale in ("en", "de", "es", "fr", "it"):
            response = client.patch("/v1/runtime/preferences", json={"ui_locale": locale, "whisper_model": "large-v3"})
            assert response.status_code == 200
            assert response.json() == {"ui_locale": locale, "whisper_model": "large-v3"}
        assert client.get("/v1/runtime/settings").json()["connections"] == []
        for body in ({}, {"ui_locale": "pt"}, {"ui_locale": ""}, {"whisper_model": "unknown"},
                     {"whisper_model": ""}, {"connection": {"provider_choice": "openrouter"}}):
            assert client.patch("/v1/runtime/preferences", json=body).status_code in {400, 422}
        assert load_workspace_prefs(config.app_data.reports_dir)["ui_locale"] == "it"


def test_preferences_patch_requires_desktop_session(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8765)
    with mock.patch.dict("os.environ", {"VOSTAVO_SESSION_TOKEN": "a" * 64}), \
            TestClient(create_app(config), base_url="http://127.0.0.1:8765") as client:
        assert client.patch("/v1/runtime/preferences", json={"ui_locale": "fr"}).status_code == 401
        assert client.patch("/v1/runtime/preferences", json={"ui_locale": "fr"}, headers={"X-Vostavo-Session": "wrong"}).status_code == 401
        assert client.patch("/v1/runtime/preferences", json={"ui_locale": "fr"}, headers={"X-Vostavo-Session": "a" * 64}).status_code == 200


def test_preferences_patch_preserves_connection_and_never_accesses_secret(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8765)
    connection = {"connection_id": "free", "provider_kind": "openrouter", "default_model": "vendor/paid", "secret_ref": "secret-1"}
    original = {"ui_locale": "en", "whisper_model": "small", "active_connection_id": "free",
                "connections": [connection], "custom": {"keep": True}}
    save_workspace_prefs(config.app_data.reports_dir, original)
    with mock.patch.dict("os.environ", {"VOSTAVO_SESSION_TOKEN": ""}), \
            mock.patch("app_backend.app.delete_secret", side_effect=AssertionError("secret deleted")), \
            mock.patch("app_backend.app._saved_connection_api_key", side_effect=AssertionError("secret read")), \
            mock.patch("app_backend.app.save_provider_connection", side_effect=AssertionError("connection saved")), \
            mock.patch("app_core.cloud_policy.read_settings", side_effect=AssertionError("connection policy checked")), \
            TestClient(create_app(config)) as client:
        response = client.patch("/v1/runtime/preferences", json={"ui_locale": "de"})
        assert response.status_code == 200
        assert response.json() == {"ui_locale": "de", "whisper_model": "small"}
    assert load_workspace_prefs(config.app_data.reports_dir) == {**original, "ui_locale": "de"}


def test_preferences_patch_serializes_read_modify_write(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8765)
    entered_save = Event()
    release_save = Event()
    second_read = Event()
    real_load = load_workspace_prefs
    real_save = save_workspace_prefs
    reads = 0

    def observed_load(path):
        nonlocal reads
        reads += 1
        if reads == 2:
            second_read.set()
        return real_load(path)

    def delayed_save(path, prefs):
        if prefs.get("ui_locale") == "fr":
            entered_save.set()
            assert release_save.wait(5)
        return real_save(path, prefs)

    with mock.patch.dict("os.environ", {"VOSTAVO_SESSION_TOKEN": ""}), \
            mock.patch("app_backend.app.load_workspace_prefs", side_effect=observed_load), \
            mock.patch("app_backend.app.save_workspace_prefs", side_effect=delayed_save), \
            TestClient(create_app(config)) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.patch, "/v1/runtime/preferences", json={"ui_locale": "fr"})
        assert entered_save.wait(5)
        second = pool.submit(client.patch, "/v1/runtime/preferences", json={"whisper_model": "tiny"})
        assert not second_read.wait(0.2)
        release_save.set()
        assert first.result(timeout=5).status_code == 200
        assert second.result(timeout=5).status_code == 200
    assert load_workspace_prefs(config.app_data.reports_dir) == {"ui_locale": "fr", "whisper_model": "tiny"}
