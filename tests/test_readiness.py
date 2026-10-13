"""Acceptance coverage for the learner readiness contract; probes never use the network."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_core.cloud_policy import CloudSettings, write_settings
from app_core.runtime_connections import serialize_connections
from app_core.state import ProviderConnection
from assessment_runtime.theme_library import save_workspace_prefs


@pytest.fixture
def readiness_env(tmp_path, monkeypatch):
    # Bootstrap mutates environment variables: keep each scenario isolated.
    monkeypatch.setattr("os.environ", dict())
    config = build_backend_runtime_config(app_data_dir=tmp_path, cache_dir=tmp_path / "cache")
    secrets = {"ai-secret": "private-ai-key", "asr-secret": "private-asr-key"}
    monkeypatch.setattr("app_core.runtime_resolver.get_secret", lambda ref: secrets.get(ref, ""))
    monkeypatch.setattr("app_core.runtime_resolver.SessionSecretStore", lambda: Mock(get_secret=Mock(return_value="")))
    monkeypatch.setattr("app_core.services.secret_store_status", lambda **kwargs: Mock(persistent=True))
    health = Mock(return_value={"endpoint": "http://private-ai-key@localhost:11434/api/tags"})
    monkeypatch.setattr("app_core.diagnostics.llm_health_check", health)
    cached = Mock(return_value={"cached": True, "cached_path": "/private/cache"})
    monkeypatch.setattr("app_core.diagnostics.describe_model_availability", cached)
    monkeypatch.setattr("app_core.diagnostics._media_available", lambda: True)
    # Fail loudly if an implementation adds a network call outside the diagnostic probe.
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("No network in readiness tests")))
    with TestClient(create_app(config)) as client:
        yield config, client, secrets, health, cached


def save_selection(config, *, provider="ollama", model="tutor-model", asr="local", asr_connection="speech"):
    connections = [ProviderConnection(
        connection_id="tutor", provider_kind=provider, default_model=model,
        base_url="http://localhost:11434/v1" if provider in {"ollama", "lmstudio"} else "https://example.test/v1",
        is_default=True, is_local=provider in {"ollama", "lmstudio"}, secret_ref="ai-secret",
    )] if provider else []
    connections.append(ProviderConnection(
        connection_id="speech", provider_kind="groq", default_model="unused-chat-model", secret_ref="asr-secret",
    ))
    save_workspace_prefs(config.app_data.reports_dir, {
        "connections": serialize_connections(connections), "active_connection_id": "tutor",
        "whisper_model": "medium", "setup_complete": True,
    })
    write_settings(config.app_data.root, CloudSettings(
        asr_provider=asr, asr_connection_id=asr_connection if asr == "groq" else "",
        asr_model="whisper-large-v3-turbo",
    ))


@pytest.mark.parametrize("scenario,overrides", [
    ("everything_available", {}),
    ("ollama_stopped", {"ai_tutor": ("unavailable", "local_service_unreachable")}),
    ("lmstudio_stopped", {"ai_tutor": ("unavailable", "local_service_unreachable")}),
    ("lmstudio_available", {}),
    ("not_configured", {"ai_tutor": ("needs_setup", "not_configured")}),
    ("cloud_key_missing", {"ai_tutor": ("needs_setup", "credential_missing")}),
    ("cloud_available", {}),
    ("cloud_model_missing", {"ai_tutor": ("needs_setup", "not_configured")}),
    ("decoder_missing", {"media_decoder": ("unavailable", "media_decoder_missing")}),
    ("storage_unwritable", {"storage": ("unavailable", "storage_not_writable")}),
    ("saved_whisper_not_cached", {"speech_recognition": ("needs_setup", "model_not_cached")}),
    ("groq_available", {}),
    ("groq_key_missing", {"speech_recognition": ("needs_setup", "credential_missing")}),
    ("groq_connection_missing", {"speech_recognition": ("needs_setup", "not_configured")}),
    ("groq_wrong_provider", {"speech_recognition": ("needs_setup", "not_configured")}),
    ("multiple_failures", {
        "ai_tutor": ("unavailable", "local_service_unreachable"),
        "speech_recognition": ("needs_setup", "model_not_cached"),
        "media_decoder": ("unavailable", "media_decoder_missing"),
        "storage": ("unavailable", "storage_not_writable"),
    }),
])
def test_readiness_acceptance(readiness_env, monkeypatch, scenario, overrides):
    config, client, secrets, health, cached = readiness_env
    provider = "openrouter" if scenario.startswith("cloud_") else "ollama"
    if scenario.startswith("lmstudio_"):
        provider = "lmstudio"
    if scenario == "not_configured":
        provider = ""
    asr = "groq" if scenario.startswith("groq_") else "local"
    asr_connection = {"groq_connection_missing": "removed", "groq_wrong_provider": "tutor"}.get(scenario, "speech")
    save_selection(config, provider=provider, model="" if scenario == "cloud_model_missing" else "tutor-model",
                   asr=asr, asr_connection=asr_connection)
    if scenario in {"ollama_stopped", "lmstudio_stopped", "multiple_failures"}:
        health.side_effect = TimeoutError("private-ai-key: connection timed out")
    if scenario == "cloud_key_missing":
        secrets.pop("ai-secret")
    if scenario == "groq_key_missing":
        secrets.pop("asr-secret")
    if scenario in {"decoder_missing", "multiple_failures"}:
        monkeypatch.setattr("app_core.diagnostics._media_available", lambda: False)
    if scenario in {"storage_unwritable", "multiple_failures"}:
        monkeypatch.setattr("app_core.diagnostics.tempfile.NamedTemporaryFile", Mock(side_effect=PermissionError("private-storage-detail")))
    if scenario == "saved_whisper_not_cached" or scenario.startswith("groq_") or scenario == "multiple_failures":
        cached.return_value = {"cached": False}

    # Query/draft values must never override the saved model or provider.
    response = client.get("/v1/readiness?whisper_model=tiny&asr_provider=local")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"checks", "ready"}
    assert [check["key"] for check in payload["checks"]] == ["ai_tutor", "speech_recognition", "media_decoder", "storage"]
    for check in payload["checks"]:
        assert (check["state"], check["reason"]) == overrides.get(check["key"], ("available", "available"))
        assert check["blocking"] is True
        assert set(check) == {"key", "state", "blocking", "reason", "detail_args"}
    assert payload["ready"] is (not overrides)
    assert payload["ready"] == all(check["state"] == "available" for check in payload["checks"] if check["blocking"])
    speech = payload["checks"][1]
    if asr == "local":
        cached.assert_called_once_with("medium")
        assert speech["detail_args"] == {"provider": "local", "model": "medium"}
    else:
        cached.assert_not_called()
        assert speech["detail_args"] == {"provider": "groq", "model": "whisper-large-v3-turbo"}
    if provider in {"ollama", "lmstudio"}:
        health.assert_called_once()
        assert health.call_args.kwargs["timeout_sec"] == 1.5
    else:
        health.assert_not_called()
    for private in ("private-ai-key", "private-asr-key", "private-storage-detail", "ai-secret", "asr-secret", "http://", "/private/cache"):
        assert private not in response.text


def test_readiness_is_typed_and_advertised(readiness_env):
    _, client, _, _, _ = readiness_env
    schema = client.get("/openapi.json").json()
    route = schema["paths"].get("/v1/readiness")
    assert route is not None
    assert route["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ReadinessResponse"}
    check = schema["components"]["schemas"]["ReadinessCheck"]["properties"]
    assert set(check["state"]["enum"]) == {"available", "needs_setup", "unavailable"}
    assert set(check["key"]["enum"]) == {"ai_tutor", "speech_recognition", "media_decoder", "storage"}
    assert set(check["reason"]["enum"]) == {"available", "not_configured", "credential_missing", "local_service_unreachable", "model_not_cached", "media_decoder_missing", "storage_not_writable"}
    assert "/v1/readiness" in client.get("/v1/contract").json()["product_routes"]


def test_diagnostics_contract_is_unchanged(readiness_env):
    config, client, _, health, _ = readiness_env
    save_selection(config)
    response = client.get("/v1/diagnostics")
    assert response.status_code == 200
    assert set(response.json()) == {"items"}
    by_key = {item["key"]: item for item in response.json()["items"]}
    assert {"app_data", "media_decoder", "whisper", "runtime", "microphone"} <= by_key.keys()
    assert "runtime_local_health" not in by_key
    assert by_key["microphone"]["status"] == "info"
    for item in by_key.values():
        assert set(item) == {"key", "status", "title_key", "detail_key", "detail_args"}
    health.assert_not_called()
