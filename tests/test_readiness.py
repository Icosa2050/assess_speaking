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
    health = Mock(return_value={"endpoint": "http://private-ai-key@localhost:11434/api/tags", "payload": {"models": [{"name": "tutor-model"}]}})
    monkeypatch.setattr("app_core.diagnostics.llm_health_check", health)
    cached = Mock(return_value={"cached": True, "cached_path": "/private/cache"})
    monkeypatch.setattr("app_core.diagnostics.describe_model_availability", cached)
    monkeypatch.setattr("app_core.diagnostics._media_available", lambda: True)
    # Fail loudly if an implementation adds a network call outside the diagnostic probe.
    def reject_network(*args, **kwargs):
        pytest.fail("No network in readiness tests")

    monkeypatch.setattr("urllib.request.urlopen", reject_network)
    # TestClient uses its own in-process transport, so native HTTP is safe to block.
    monkeypatch.setattr("httpx.HTTPTransport.handle_request", reject_network)
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
    if not provider:
        connections = []
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
    assert set(check["reason"]["enum"]) == {"available", "not_configured", "credential_missing", "local_service_unreachable", "model_not_cached", "model_unavailable", "media_decoder_missing", "storage_not_writable"}
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


@pytest.mark.parametrize("provider,models", [
    ("ollama", {"models": [{"name": "other-model"}]}),
    ("lmstudio", {"data": [{"id": "other-model"}]}),
    ("ollama", {"models": []}),
])
def test_reachable_local_service_requires_saved_model(readiness_env, provider, models):
    config, client, _, health, _ = readiness_env
    save_selection(config, provider=provider)
    health.return_value["payload"] = models
    response = client.get("/v1/readiness")
    assert response.status_code == 200
    check = response.json()["checks"][0]
    assert (check["state"], check["reason"], check["blocking"]) == ("needs_setup", "model_unavailable", True)
    assert response.json()["ready"] is False
    health.assert_called_once()


@pytest.mark.parametrize("has_key", [True, False])
def test_ollama_cloud_checks_credential_without_network(readiness_env, has_key):
    config, client, secrets, health, _ = readiness_env
    connection = ProviderConnection(
        connection_id="tutor", provider_kind="ollama", default_model="cloud-model",
        base_url="https://ollama.com/v1", is_local=False, secret_ref="ai-secret",
        provider_metadata={"deployment": "cloud"},
    )
    save_workspace_prefs(config.app_data.reports_dir, {
        "connections": serialize_connections([connection]), "active_connection_id": "tutor", "whisper_model": "medium",
    })
    if not has_key:
        secrets.pop("ai-secret")
    health.side_effect = AssertionError("Cloud services must not be probed")
    response = client.get("/v1/readiness")
    assert response.status_code == 200
    check = response.json()["checks"][0]
    expected = ("available", "available") if has_key else ("needs_setup", "credential_missing")
    assert (check["state"], check["reason"]) == expected
    assert check["blocking"] is True
    assert response.json()["ready"] is has_key
    health.assert_not_called()


@pytest.mark.parametrize("provider,saved_model,catalog_model,available", [
    ("ollama", "tutor-model", "tutor-model:latest", True),
    ("ollama", "tutor-model", "tutor-model:7b", False),
    ("ollama", "registry.local:5000/team/tutor-model", "registry.local:5000/team/tutor-model:latest", True),
    ("ollama", "tutor-model:7b", "tutor-model:latest", False),
    ("lmstudio", "tutor-model", "tutor-model:latest", False),
])
def test_local_model_aliases(readiness_env, provider, saved_model, catalog_model, available):
    config, client, _, health, _ = readiness_env
    save_selection(config, provider=provider, model=saved_model)
    health.return_value["payload"] = {"models": [{"name": catalog_model}]}
    response = client.get("/v1/readiness")
    assert response.status_code == 200
    check = response.json()["checks"][0]
    expected = ("available", "available") if available else ("needs_setup", "model_unavailable")
    assert (check["state"], check["reason"]) == expected
    assert check["blocking"] is True
    assert check["detail_args"]["model"] == saved_model
    assert response.json()["ready"] is available
    health.assert_called_once()


@pytest.mark.parametrize("transport", ["urllib", "httpx"])
def test_readiness_network_guard_covers_product_transports(readiness_env, transport):
    # Neither URI can reach a socket, even when the guard is absent during the red run.
    # Exercise the product's actual urllib helper, and httpx's native HTTP transport.
    with pytest.raises(pytest.fail.Exception, match="No network in readiness tests"):
        if transport == "urllib":
            from assessment_runtime.llm_client import _get_json
            _get_json("data:application/json,%7B%7D", {}, 0.1)
        else:
            import httpx
            with httpx.Client() as client:
                client.get("offline://readiness-test", timeout=0.1)
