"""Availability checks must report missing providers without exposing credentials."""
from unittest.mock import Mock, patch
import json

import pytest

from scripts.run_live_journeys import discover, main


def response(data, status=200):
    value = Mock(status_code=status)
    value.json.return_value = data
    return value


def test_openrouter_without_credential_does_not_contact_service(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("scripts.run_live_journeys.requests.get") as request:
        result = discover("openrouter")
    assert not result["available"]
    assert "absent" in result["reason"]
    request.assert_not_called()


def test_openrouter_auth_failure_retains_no_credential(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-credential")
    with patch("scripts.run_live_journeys.requests.get", return_value=response({}, 401)):
        result = discover("openrouter")
    assert not result["available"]
    assert "401" in result["reason"]
    assert "private-test-credential" not in str(result)


def test_lmstudio_chooses_available_alternate_model(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_E2E_ALTERNATE_MODEL", raising=False)
    monkeypatch.delenv("LMSTUDIO_E2E_MODEL", raising=False)
    with patch("scripts.run_live_journeys.requests.get", return_value=response({"data": [
        {"id": "qwen2.5-3b-instruct"}, {"id": "qwen2.5-7b-instruct"},
    ]})):
        result = discover("lmstudio")
    assert result["available"]
    assert result["alternate_model"] == "qwen2.5-7b-instruct"


def test_requested_missing_alternate_model_is_not_silently_ignored(monkeypatch):
    monkeypatch.setenv("OLLAMA_E2E_ALTERNATE_MODEL", "missing-model")
    monkeypatch.delenv("OLLAMA_E2E_MODEL", raising=False)
    with patch("scripts.run_live_journeys.requests.get", return_value=response({"data": [{"id": "qwen3.5:4b"}]})):
        result = discover("ollama")
    assert not result["available"]
    assert "alternate" in result["reason"]


def test_openrouter_endpoint_override_cannot_receive_credential(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-credential")
    monkeypatch.setenv("OPENROUTER_E2E_BASE_URL", "https://example.org/api/v1")
    with patch("scripts.run_live_journeys.requests.get") as request:
        result = discover("openrouter")
    assert not result["available"]
    assert "Unsafe" in result["reason"]
    request.assert_not_called()


def test_isolated_keyring_does_not_use_user_storage():
    from scripts.journey_keyring import MemoryKeyring
    first, second = MemoryKeyring(), MemoryKeyring()
    first.set_password("live-test", "isolated-account", "test-only")
    assert second.get_password("live-test", "isolated-account") == "test-only"
    second.delete_password("live-test", "isolated-account")
    assert first.get_password("live-test", "isolated-account") is None


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("executed", [0, 1])
@pytest.mark.parametrize("has_assessment", [False, True])
def test_guarded_fallback_is_explicit_and_recorded(monkeypatch, tmp_path, guarded, exit_code, executed, has_assessment):
    output = tmp_path / "evidence"
    arguments = ["run_live_journeys.py", "--providers", "lmstudio", "--output", str(output)]
    if guarded:
        arguments.append("--allow-guarded-fallback")
    monkeypatch.setattr("sys.argv", arguments)
    # An inherited variable must never relax the strict CLI default.
    monkeypatch.setenv("LIVE_E2E_ALLOW_GUARDED_FALLBACK", "1" if not guarded else "0")
    available = {
        "provider": "lmstudio", "available": True, "endpoint": "http://127.0.0.1:1234/v1",
        "model": "test-model", "alternate_model": "test-model",
    }
    def execute(*args, **kwargs):
        from pathlib import Path
        directory = Path(kwargs["env"]["VOSTAVO_OLLAMA_TEST_ROOT"])
        (directory / "results.json").write_text(json.dumps({"stats": {"expected": executed, "skipped": 1-executed}}))
        return Mock(returncode=exit_code)
    with (
        patch("scripts.run_live_journeys.discover", return_value=available),
        patch("scripts.run_live_journeys.version", return_value="test-version"),
        patch("scripts.run_live_journeys.subprocess.check_output", return_value="test-revision"),
        patch("assessment_runtime.asr.describe_model_availability", return_value={"cached": True}),
        patch("scripts.run_live_journeys.shutil.which", return_value="/test/node"),
        patch("scripts.run_live_journeys.subprocess.run", side_effect=execute) as run,
        patch("scripts.inspect_journey_evidence.inspect", return_value={"assessments": [
            {"status": "completed", "session_id": "test-session", "transcript": "test speech",
             "coaching": {"next_exercise": "retry"}, "recordings": [{"path": "fixture.wav"}]},
        ] if has_assessment else []}),
    ):
        assert main() == (0 if exit_code == 0 and executed and has_assessment else 1)
    assert run.call_args.kwargs["env"]["LIVE_E2E_ALLOW_GUARDED_FALLBACK"] == ("1" if guarded else "0")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["allow_guarded_fallback"] is guarded
    assert manifest["runs"][0]["allow_guarded_fallback"] is guarded
    expected = "workflow_only_with_guarded_validation_fallback" if guarded else "strict_model_output_and_workflow"
    assert manifest["acceptance_scope"] == expected
    assert manifest["runs"][0]["acceptance_scope"] == expected
    assert manifest["runs"][0]["status"] == ("passed" if exit_code == 0 and executed and has_assessment else "failed")
    assert manifest["runs"][0]["executed_tests"] == executed
    assert "scripts/run_live_journeys.py" in manifest["source_hashes"]
    assert "assessment_runtime/feedback_claims.py" in manifest["source_hashes"]
