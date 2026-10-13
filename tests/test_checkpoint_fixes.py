import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import assess_speaking
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateRequest, CleanupTarget
from app_backend.jobs import JobManager
from app_backend.maintenance import execute_cleanup
from app_core.cloud_policy import locked_file
from assess_core.schemas import CoachingSummary, RubricResult
from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run
from test_assess_speaking import _sample_report


@pytest.mark.parametrize("changed", ["endpoint", "connection", "unchanged"])
def test_feedback_cache_tracks_connection_and_endpoint(tmp_path, monkeypatch, changed):
    source = tmp_path / "sample.wav"
    source.write_bytes(b"fixture")
    payload = _sample_report()
    audio = Mock(return_value={"duration_sec": 30.0, "pauses": []})
    speech = Mock(return_value={"detected_language": "it", "language_probability": .99, "text": "ciao mondo", "words": [
        {"text": "ciao", "t0": 0.0, "t1": 0.5}, {"text": "mondo", "t0": 0.6, "t1": 1.0},
    ]})
    rubric = Mock(return_value=(RubricResult.from_dict(payload["rubric"]), json.dumps(payload["rubric"])))
    coaching = Mock(return_value=(CoachingSummary.from_dict(payload["coaching"]), json.dumps(payload["coaching"])))
    monkeypatch.setattr(assess_speaking, "load_audio_features", audio)
    monkeypatch.setattr(assess_speaking, "transcribe", speech)
    monkeypatch.setattr(assess_speaking, "generate_rubric", rubric)
    monkeypatch.setattr(assess_speaking, "generate_coaching_summary", coaching)
    args = dict(provider="openai_compatible", llm_model="chosen", expected_language="it", min_word_count=1,
                stage_cache_dir=tmp_path / "stages", llm_base_url="https://first.invalid/v1", llm_connection_id="first")
    request = AssessmentRunRequest(audio=source, no_log=True, **args)
    first = execute_assessment_run(request)
    if changed == "endpoint":
        args["llm_base_url"] = "https://second.invalid/v1"
    elif changed == "connection":
        args["llm_connection_id"] = "second"
    second = execute_assessment_run(AssessmentRunRequest(audio=source, no_log=True, **args))
    assert first.report["session_id"] == second.report["session_id"]
    assert audio.call_count == speech.call_count == 1
    assert rubric.call_count == coaching.call_count == (1 if changed == "unchanged" else 2)
    manifest = (tmp_path / "stages" / "manifest.json").read_text()
    assert "first.invalid" not in manifest  # Identity is hashed, not published with replies.


def test_worker_receives_backend_connection_snapshot(tmp_path, monkeypatch):
    config = build_backend_runtime_config(app_data_dir=tmp_path / "app", cache_dir=tmp_path / "cache", port=8764)
    manager = JobManager(config)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    monkeypatch.setattr(manager, "_resolve_audio_path", lambda _: audio)
    process = Mock()
    factory = Mock(return_value=process)
    monkeypatch.setattr(manager._ctx, "Process", factory)
    request = AssessmentCreateRequest(audio_id="aud_fixture", provider="openai_compatible", llm_model="chosen",
                                      whisper="small", expected_language="it", feedback_language="en",
                                      speaker_id="fixture", task_family="free_monologue", theme="test", target_duration_sec=90)
    manager.submit(request, llm_connection_id="saved-local")
    worker = factory.call_args.kwargs["args"][1]
    assert worker["llm_connection_id"] == "saved-local"
    manager._processes.clear()
    runtime = SimpleNamespace(primary=SimpleNamespace(connection_id="saved-cloud", base_url="https://actual.invalid/v1"),
                              settings=SimpleNamespace(asr_provider="local", asr_model="large-v3"))
    # The broker uses its saved snapshot, even if a client supplied a different URL.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(manager._ctx, "Pipe", lambda: (Mock(), Mock()))
    manager.submit(request, cloud_runtime=runtime)
    worker = factory.call_args.kwargs["args"][1]
    assert worker["llm_connection_id"] == "saved-cloud"
    assert worker["llm_base_url"] == "https://actual.invalid/v1"


def _job(config, name, status="completed", *, age=31, cache=None):
    path = config.jobs_dir / f"asmt_{name}.json"
    path.write_text(json.dumps({"status": status, "completed_at": (datetime.now(UTC) - timedelta(days=age)).isoformat(),
                                "stage_cache_dir": str(cache or config.jobs_dir / f"asmt_{name}-stages")}))
    return path


def _cache(config, name):
    root = config.jobs_dir / f"asmt_{name}-stages"
    (root / "cloud-replies").mkdir(parents=True)
    (root / "manifest.json").write_text("manifest")
    (root / "cloud-replies" / "reply.json").write_text("provider reply")
    return root


@pytest.mark.parametrize("target", [CleanupTarget.JOBS, CleanupTarget.ALL_SAFE])
def test_cleanup_preview_and_execution_prune_only_unreferenced_checkpoints(tmp_path, target):
    config = build_backend_runtime_config(app_data_dir=tmp_path / "app", cache_dir=tmp_path / "cache", port=8764)
    expired = _cache(config, "expired")
    old_job = _job(config, "expired")
    shared = _cache(config, "parent")
    parent_job = _job(config, "parent")
    recent_job = _job(config, "recent", age=1, cache=shared)
    active = _cache(config, "active")
    active_job = _job(config, "active", status="running", age=90, cache=active)
    orphan = _cache(config, "orphan")  # Metadata may have expired before this fix.
    unrelated = config.jobs_dir / "keep-other-directory"
    unrelated.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.json").write_text("keep")
    (config.jobs_dir / "asmt_link-stages").symlink_to(outside, target_is_directory=True)
    (orphan / "outside-link").symlink_to(outside, target_is_directory=True)
    preview = execute_cleanup(config, target, dry_run=True)
    assert old_job.exists() and parent_job.exists() and expired.exists() and orphan.exists()
    result = execute_cleanup(config, target)
    assert result.deleted_file_count == preview.deleted_file_count
    assert result.freed_bytes == preview.freed_bytes
    assert not result.warnings
    assert not old_job.exists() and not parent_job.exists()
    assert not expired.exists() and not orphan.exists()
    assert shared.exists() and recent_job.exists() and active.exists() and active_job.exists()
    assert unrelated.exists() and (outside / "keep.json").exists()
    assert (config.jobs_dir / "asmt_link-stages").is_symlink()
    _job(config, "recent", cache=shared)
    execute_cleanup(config, target)
    assert not shared.exists()


def test_failed_metadata_deletion_preserves_checkpoint(tmp_path, monkeypatch):
    config = build_backend_runtime_config(app_data_dir=tmp_path / "app", cache_dir=tmp_path / "cache", port=8764)
    cache = _cache(config, "locked")
    job = _job(config, "locked", cache=cache)
    unlink = Path.unlink
    def fail_metadata(path, *args, **kwargs):
        if path == job:
            raise PermissionError("fixture locked metadata")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_metadata)
    result = execute_cleanup(config, CleanupTarget.JOBS)
    assert result.warnings and result.deleted_file_count == 0
    assert job.exists() and (cache / "cloud-replies" / "reply.json").exists()


def test_cleanup_keeps_checkpoints_when_retained_metadata_is_unreadable(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path / "app", cache_dir=tmp_path / "cache", port=8764)
    cache = _cache(config, "unknown")
    (config.jobs_dir / "asmt_unknown.json").write_text("incomplete JSON")
    execute_cleanup(config, CleanupTarget.JOBS)
    assert cache.exists()


def test_cleanup_waits_for_resume_publication(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path / "app", cache_dir=tmp_path / "cache", port=8764)
    shared = _cache(config, "parent")
    _job(config, "parent", cache=shared)
    started = threading.Event()
    results = []
    def clean():
        started.set()
        results.append(execute_cleanup(config, CleanupTarget.JOBS))
    with locked_file(config.jobs_dir / "retention.lock"):
        thread = threading.Thread(target=clean)
        thread.start()
        assert started.wait(2)
        assert not results
        _job(config, "resumed", status="queued", age=0, cache=shared)
    thread.join(timeout=3)
    assert not thread.is_alive() and results
    assert shared.exists()
