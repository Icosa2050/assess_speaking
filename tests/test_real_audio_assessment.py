"""Provider-neutral long-audio checks; AI quality and safe fallback are distinct modes."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest

import assess_speaking
from assessment_runtime.asr import describe_model_availability
from assessment_runtime.llm_client import list_models

DEFAULT_REAL_AUDIO_PATH = Path(__file__).resolve().parent / "audio" / "test1.m4a"
SECOND_REAL_AUDIO_PATH = Path(__file__).resolve().parent / "audio" / "test2.m4a"


def _require_fixture(path: Path) -> Path:
    assert path.is_file(), f"Enabled real-audio test needs its recording: {path}"
    return path


def _require_real_audio_env() -> Path:
    if os.getenv("RUN_REAL_AUDIO_ASSESSMENT") != "1":
        raise unittest.SkipTest("Set RUN_REAL_AUDIO_ASSESSMENT=1 to run the real audio assessment test.")
    provider = os.getenv("ASSESS_SPEAKING_REAL_PROVIDER", "ollama")
    assert provider in {"ollama", "lmstudio", "openrouter"}, "Choose ollama, lmstudio or openrouter"
    if provider == "openrouter":
        assert os.getenv("OPENROUTER_API_KEY"), "OpenRouter requires OPENROUTER_API_KEY"
    assert os.getenv("ASSESS_SPEAKING_REAL_LLM_MODEL"), "Set ASSESS_SPEAKING_REAL_LLM_MODEL to an installed/available model"
    audio = os.getenv("ASSESS_SPEAKING_REAL_AUDIO_PATH")
    return _require_fixture(Path(audio).expanduser() if audio else DEFAULT_REAL_AUDIO_PATH)


def _decoded_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            check=True, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def _assert_feedback_contract(report: dict, *, allow_guarded: bool) -> None:
    warnings = set(report.get("warnings", []))
    assert "llm_unavailable" not in warnings, "Provider failure is not a successful live test"
    scores = report["scores"]
    assert 1 <= scores["final"] <= 5
    if scores["mode"] == "deterministic_only":
        assert allow_guarded, f"AI feedback was not accepted ({sorted(warnings)}); opt into guarded workflow coverage explicitly"
        assert warnings & {"transcript_uncertain", "llm_invalid_schema"}
        assert report["requires_human_review"] is True
        assert report["rubric"] is None and scores["llm"] is None
    else:
        assert scores["mode"] == "hybrid"
        assert report["rubric"] and scores["llm"] is not None
    # Coaching transport and validation errors share a warning: require usable coaching.
    assert "coaching_unavailable" not in warnings
    assert len(report["coaching"]["top_3_priorities"]) == 3
    assert report["coaching"]["next_exercise"].strip()


@pytest.mark.parametrize("recording", [None] if os.getenv("ASSESS_SPEAKING_REAL_AUDIO_PATH") else [None, SECOND_REAL_AUDIO_PATH],
                         ids=["custom"] if os.getenv("ASSESS_SPEAKING_REAL_AUDIO_PATH") else ["test1", "test2"])
def test_recording_preserves_duration_provider_and_feedback_contract(recording, monkeypatch, record_property):
    first = _require_real_audio_env()
    audio = _require_fixture(recording) if recording else first
    whisper = os.getenv("ASSESS_SPEAKING_REAL_WHISPER_MODEL", "tiny")
    assert describe_model_availability(whisper)["cached"], f"Download Whisper {whisper} before this test"
    assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "Install ffmpeg (including ffprobe) before this test"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    language = os.getenv("ASSESS_SPEAKING_REAL_LANGUAGE", "it")
    provider = os.getenv("ASSESS_SPEAKING_REAL_PROVIDER", "ollama")
    model = os.environ["ASSESS_SPEAKING_REAL_LLM_MODEL"]
    base_url = os.getenv("ASSESS_SPEAKING_REAL_BASE_URL")
    # Guarded ASR rejection may skip generation, but it must not hide an absent provider/model.
    available = list_models(provider=provider, base_url=base_url, timeout_sec=15)
    assert model in [entry["id"] for entry in available["data"]], "Requested provider model must be available"
    with tempfile.TemporaryDirectory() as tmpdir:
        result = assess_speaking.run_assessment(
            audio, provider=provider, llm_model=model, whisper_model=whisper,
            llm_base_url=base_url, llm_timeout_sec=180,
            expected_language=language, feedback_language=language,
            theme="tema libero" if language == "it" else "free topic",
            task_family="real_audio_smoke", speaker_id="real-audio-test",
            target_duration_sec=60, train_dir=Path(tmpdir),
        )
    report = result["report"]
    record_property("acceptance_scope", "guarded_workflow" if os.getenv("REAL_AUDIO_ALLOW_GUARDED_FALLBACK") == "1" else "accepted_ai_feedback")
    record_property("ai_output_accepted", report["scores"]["mode"] == "hybrid" and "coaching_unavailable" not in report.get("warnings", []))
    record_property("generation_attempted", report.get("timings_ms", {}).get("llm", 0) > 0)
    record_property("warnings", json.dumps(report.get("warnings", [])))
    record_property("feedback_quality_review", "unreviewed")
    assert result["transcript_full"].strip()
    assert report["metrics"]["word_count"] > 10
    assert report["metrics"]["duration_sec"] == pytest.approx(_decoded_duration(audio), abs=.5)
    for field, expected in dict(provider=provider, llm_model=model, whisper_model=whisper,
                                expected_language=language, speaker_id="real-audio-test", task_family="real_audio_smoke").items():
        assert report["input"][field] == expected
    _assert_feedback_contract(report, allow_guarded=os.getenv("REAL_AUDIO_ALLOW_GUARDED_FALLBACK") == "1")
    # Reports must remain usable after a JSON round trip, with finite measurements.
    saved = json.loads(json.dumps(report, allow_nan=False))
    assert saved["metrics"] == report["metrics"]
    assert saved["coaching"] == report["coaching"]
