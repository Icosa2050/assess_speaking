"""Fast regression tests for live-test gates, independent of installed AI services."""
import json
import unittest
from unittest.mock import patch

import pytest

from test_integration_openrouter import require_openrouter
from test_real_audio_assessment import _require_real_audio_env, _assert_feedback_contract
from scripts.run_live_journeys import main


@pytest.mark.parametrize("provider", ["ollama", "lmstudio"])
def test_local_audio_opt_in_needs_no_cloud_key(monkeypatch, provider):
    monkeypatch.setenv("RUN_REAL_AUDIO_ASSESSMENT", "1")
    monkeypatch.setenv("ASSESS_SPEAKING_REAL_PROVIDER", provider)
    monkeypatch.setenv("ASSESS_SPEAKING_REAL_LLM_MODEL", "test-model")
    monkeypatch.delenv("ASSESS_SPEAKING_REAL_AUDIO_PATH", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _require_real_audio_env().is_file()


def test_enabled_missing_audio_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_REAL_AUDIO_ASSESSMENT", "1")
    monkeypatch.setenv("ASSESS_SPEAKING_REAL_PROVIDER", "ollama")
    monkeypatch.setenv("ASSESS_SPEAKING_REAL_LLM_MODEL", "test-model")
    monkeypatch.setenv("ASSESS_SPEAKING_REAL_AUDIO_PATH", str(tmp_path / "absent.wav"))
    with pytest.raises(AssertionError, match="recording"):
        _require_real_audio_env()


def test_disabled_audio_skips_before_checking_prerequisites(monkeypatch):
    monkeypatch.delenv("RUN_REAL_AUDIO_ASSESSMENT", raising=False)
    with pytest.raises(unittest.SkipTest):
        _require_real_audio_env()


def test_enabled_openrouter_without_key_fails(monkeypatch):
    monkeypatch.setenv("RUN_OPENROUTER_INTEGRATION", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AssertionError, match="OPENROUTER_API_KEY"):
        require_openrouter()


def test_enabled_openrouter_requires_explicit_model(monkeypatch):
    monkeypatch.setenv("RUN_OPENROUTER_INTEGRATION", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(AssertionError, match="OPENROUTER_MODEL"):
        require_openrouter()


def test_guarded_result_cannot_pass_as_ai_feedback_or_hide_transport_failure():
    report = {"warnings": ["transcript_uncertain"], "requires_human_review": True,
              "rubric": None, "scores": {"mode": "deterministic_only", "final": 2, "llm": None},
              "coaching": {"top_3_priorities": ["a", "b", "c"], "next_exercise": "Speak for 60 seconds"}}
    with pytest.raises(AssertionError, match="not accepted"):
        _assert_feedback_contract(report, allow_guarded=False)
    _assert_feedback_contract(report, allow_guarded=True)
    report["warnings"].append("llm_unavailable")
    with pytest.raises(AssertionError, match="Provider failure"):
        _assert_feedback_contract(report, allow_guarded=True)


@pytest.mark.parametrize("selection", ["lmstudio", "openrouter", "ollama,lmstudio"])
def test_unavailable_requested_provider_fails_before_running_journeys(monkeypatch, tmp_path, selection):
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["run_live_journeys.py", "--providers", selection, "--output", str(output)])
    def unavailable(provider):
        return {"provider": provider, "available": provider == "ollama", "reason": "test prerequisite absent"}
    with (
        patch("scripts.run_live_journeys.discover", side_effect=unavailable),
        patch("scripts.run_live_journeys.version", return_value="test"),
        patch("scripts.run_live_journeys.subprocess.check_output", return_value="test-revision"),
        patch("scripts.run_live_journeys.subprocess.run") as run,
    ):
        assert main() == 1
    run.assert_not_called()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runs"] and all(r["status"] == "unavailable" for r in manifest["runs"])
