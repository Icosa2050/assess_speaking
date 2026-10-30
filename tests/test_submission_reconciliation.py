from datetime import UTC, datetime, timedelta
import io
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateRequest, CleanupTarget
from app_backend.jobs import JobManager, _request_index_file, _sanitize_request_metadata, _upload_request_index_file
from app_backend.maintenance import execute_cleanup


def test_request_lookup_survives_helper_restart_and_requires_desktop_session(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8870)
    token = "a" * 64
    with patch.dict("os.environ", {"VOSTAVO_SESSION_TOKEN": token}):
        app = create_app(config)
        client = TestClient(app, base_url="http://127.0.0.1:8870")
        headers = {"X-Vostavo-Session": token, "X-Upload-Request-Id": "upload-1"}
        uploaded = client.post("/v1/uploads", headers=headers, files={"file": ("take.wav", b"audio")})
        assert uploaded.status_code == 200
        audio_id = uploaded.json()["audio_id"]
        job = config.jobs_dir / "asmt_one.json"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text(json.dumps({"assessment_id": "asmt_one", "status": "completed", "phase": "done",
                                   "request": {"request_id": "assessment-1", "audio_id": audio_id,
                                               "speaker_id": "Alex", "sharing_fingerprint": "b" * 64},
                                   "payload": {"report": {"session_id": "report-1"}}}), encoding="utf-8")
        restarted = TestClient(create_app(config), base_url="http://127.0.0.1:8870")
        upload_params = {"sha256": hashlib.sha256(b"audio").hexdigest(), "original_name": "take.wav"}
        assessment_params = {"audio_id": audio_id, "speaker_id": "alex", "sharing_fingerprint": "b" * 64}
        assert restarted.get("/v1/uploads/by-request/upload-1").status_code == 401
        assert restarted.get("/v1/assessments/by-request/assessment-1").status_code == 401
        assert restarted.get("/v1/uploads/by-request/upload-1", headers=headers, params=upload_params).json() == {"audio_id": audio_id}
        assert restarted.get("/v1/assessments/by-request/assessment-1", headers=headers, params=assessment_params).json() == {
            "assessment_id": "asmt_one", "status": "completed", "report_id": "report-1"}
        assert restarted.get("/v1/uploads/by-request/upload-1", headers=headers, params={**upload_params, "sha256": "0" * 64}).status_code == 404
        assert restarted.get("/v1/uploads/by-request/upload-1", headers=headers, params={**upload_params, "original_name": "other.wav"}).status_code == 404
        assert restarted.get("/v1/assessments/by-request/assessment-1", headers=headers, params={**assessment_params, "speaker_id": "Bea"}).status_code == 404
        assert restarted.get("/v1/assessments/by-request/assessment-1", headers=headers, params={**assessment_params, "audio_id": "aud_" + "a" * 32}).status_code == 404
        assert restarted.get("/v1/uploads/by-request/absent", headers=headers, params=upload_params).status_code == 404
        assert restarted.get("/v1/assessments/by-request/absent", headers=headers, params=assessment_params).status_code == 404


def test_assessment_request_lookup_is_read_only_and_indexed(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8874)
    manager = JobManager(config)
    request_id = "indexed-request"
    audio_id = "aud_" + "a" * 32
    for index in range(300):
        (config.jobs_dir / f"asmt_decoy_{index}.json").write_text('{"status":"queued"}', encoding="utf-8")
    job = config.jobs_dir / "asmt_indexed.json"
    job.write_text(json.dumps({"assessment_id": "asmt_indexed", "status": "running", "request": {
        "request_id": request_id, "audio_id": audio_id, "speaker_id": "Alex", "sharing_fingerprint": "c" * 64},
        "payload": {"report": {"session_id": "report-indexed", "large": "x" * 100_000}}}), encoding="utf-8")
    index_file = _request_index_file(config.jobs_dir, request_id)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps({"request_id": request_id, "assessment_id": "asmt_indexed"}), encoding="utf-8")
    original = job.read_bytes()
    from app_backend import jobs as jobs_module
    with patch.object(manager, "_get_status_locked", side_effect=AssertionError("status reconciliation is a write")), \
         patch.object(jobs_module, "_read_json", wraps=jobs_module._read_json) as read:
        found = manager.lookup_assessment_request(request_id, audio_id=audio_id, speaker_id="alex", sharing_fingerprint="c" * 64)
    assert found == {"assessment_id": "asmt_indexed", "status": "running", "report_id": "report-indexed"}
    assert read.call_count == 2
    assert job.read_bytes() == original


def test_accepted_submission_persists_bounded_request_index(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8875)
    manager = JobManager(config)
    upload = manager.register_upload_stream(source=io.BytesIO(b"audio"), filename="take.wav", request_id="upload-indexed")
    request = AssessmentCreateRequest.model_validate({"request_id": "assessment-indexed", "audio_id": upload.audio_id,
        "sharing_fingerprint": "a" * 64, "whisper": "small", "provider": "ollama", "llm_model": "test",
        "expected_language": "en", "feedback_language": "en", "speaker_id": "Alex",
        "task_family": "free_monologue", "theme": "A topic", "target_duration_sec": 90})
    with patch.object(manager._ctx, "Process") as process_factory:
        accepted = manager._submit(request)
        process_factory.return_value.start.assert_called_once()
    assert _request_index_file(config.jobs_dir, request.request_id).is_file()
    assert _upload_request_index_file(config.app_data.uploads_dir, "upload-indexed").is_file()
    assert manager.lookup_assessment_request(request.request_id, audio_id=upload.audio_id,
        speaker_id="alex", sharing_fingerprint="a" * 64) == {
            "assessment_id": accepted.assessment_id, "status": "queued", "report_id": None}


def test_runtime_headers_are_excluded_from_assessment_dedupe_identity():
    request = {"request_id": "same", "audio_id": "aud_" + "a" * 32,
               "openrouter_http_referer": "https://first.example", "openrouter_app_title": "First",
               "llm_base_url": "https://first.example/v1", "llm_api_key": "secret"}
    safe = _sanitize_request_metadata(request)
    assert safe == {"request_id": request["request_id"], "audio_id": request["audio_id"]}
    changed = {**request, "openrouter_http_referer": "https://second.example", "openrouter_app_title": "Second",
               "llm_base_url": "https://second.example/v1", "llm_api_key": "other"}
    assert _sanitize_request_metadata(changed) == safe


def test_malformed_matching_upload_sidecar_cannot_claim_or_reconcile_a_take(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8876)
    manager = JobManager(config)
    foreign = tmp_path / "foreign.wav"
    foreign.write_bytes(b"different")
    bogus = config.app_data.uploads_dir / ("aud_" + "0" * 32 + ".json")
    bogus.write_text(json.dumps({"request_id": "upload-safe", "audio_id": "../../foreign",
        "stored_path": str(foreign), "sha1": hashlib.sha1(b"audio").hexdigest(),
        "sha256": hashlib.sha256(b"audio").hexdigest(), "original_name": "take.wav"}), encoding="utf-8")
    uploaded = manager.register_upload_stream(source=io.BytesIO(b"audio"), filename="take.wav", request_id="upload-safe")
    assert uploaded.audio_id.startswith("aud_")
    assert manager.lookup_upload_request("upload-safe", sha256=hashlib.sha256(b"audio").hexdigest(),
        original_name="take.wav") == uploaded.audio_id


def test_orphan_upload_cleanup_spares_jobs_journal_recent_and_unknown_age(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8871)
    manager = JobManager(config)
    names = ("orphan", "job", "journal", "report", "archive", "recent", "receipt_window", "legacy")
    uploads = {name: manager.register_upload_stream(source=io.BytesIO(name.encode()),
                                                    filename=f"{name}.wav", request_id=name) for name in names}
    old = (datetime.now(UTC) - timedelta(days=9)).isoformat()
    for name, upload in uploads.items():
        sidecar = config.app_data.uploads_dir / f"{upload.audio_id}.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if name == "legacy":
            payload.pop("created_at")
        elif name == "receipt_window":
            payload["created_at"] = (datetime.now(UTC) - timedelta(days=7, hours=12)).isoformat()
        elif name != "recent":
            payload["created_at"] = old
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    (config.jobs_dir / "asmt_job.json").write_text(json.dumps({"status": "failed", "request": {
        "audio_id": uploads["job"].audio_id}}), encoding="utf-8")
    config.app_data.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.app_data.reports_dir / "history.csv").write_text(str(uploads["journal"].stored_path), encoding="utf-8")
    (config.app_data.reports_dir / "report.json").write_text(json.dumps({"audio_id": uploads["report"].audio_id}), encoding="utf-8")
    archive = config.app_data.reports_dir / "archive" / "row.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(json.dumps({"audio_id": uploads["archive"].audio_id}), encoding="utf-8")
    preview = execute_cleanup(config, CleanupTarget.JOBS, dry_run=True)
    assert preview.deleted_file_count == 3
    assert all((config.app_data.uploads_dir / f"{upload.audio_id}.json").exists() for upload in uploads.values())
    assert all(Path(upload.stored_path).exists() for upload in uploads.values())
    cleaned = execute_cleanup(config, CleanupTarget.JOBS)
    assert cleaned.deleted_file_count == 3
    assert not (config.app_data.uploads_dir / f"{uploads['orphan'].audio_id}.json").exists()
    assert not _upload_request_index_file(config.app_data.uploads_dir, "orphan").exists()
    assert not Path(uploads["orphan"].stored_path).exists()
    for name in names[1:]:
        assert (config.app_data.uploads_dir / f"{uploads[name].audio_id}.json").exists()
        assert Path(uploads[name].stored_path).exists()


def test_orphan_cleanup_keeps_uploads_if_job_metadata_is_unreadable(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8872)
    upload = JobManager(config).register_upload(data=b"audio", filename="take.wav")
    sidecar = config.app_data.uploads_dir / f"{upload.audio_id}.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["created_at"] = (datetime.now(UTC) - timedelta(days=9)).isoformat()
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    (config.jobs_dir / "corrupt.json").write_text("{", encoding="utf-8")
    assert execute_cleanup(config, CleanupTarget.JOBS, dry_run=True).deleted_file_count == 0
    assert sidecar.exists()


def test_old_unreferenced_upload_without_sidecar_and_sidecar_without_media_are_cleaned(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8873)
    config.app_data.uploads_dir.mkdir(parents=True, exist_ok=True)
    media = config.app_data.uploads_dir / ("aud_" + "a" * 32 + ".wav")
    media.write_bytes(b"orphan")
    old = datetime.now(UTC) - timedelta(days=9)
    os.utime(media, (old.timestamp(), old.timestamp()))
    missing = config.app_data.uploads_dir / ("aud_" + "b" * 32 + ".json")
    missing.write_text(json.dumps({"audio_id": missing.stem,
        "stored_path": str(missing.with_suffix(".wav")), "created_at": old.isoformat()}), encoding="utf-8")
    preview = execute_cleanup(config, CleanupTarget.ALL_SAFE, dry_run=True)
    assert preview.deleted_file_count == 2
    assert media.exists() and missing.exists()
    assert execute_cleanup(config, CleanupTarget.ALL_SAFE).deleted_file_count == 2
    assert not media.exists() and not missing.exists()


def test_assessment_metadata_limits_match_replayable_receipt_capacity():
    request = {"audio_id": "aud_one", "whisper": "small", "provider": "ollama", "llm_model": "test",
               "expected_language": "en", "feedback_language": "en", "speaker_id": "Alex",
               "task_family": "free_monologue", "theme": "A topic", "target_duration_sec": 90,
               "prompt_text": "p" * 12000, "label": "L" * 500, "notes": "😀" * 12000}
    accepted = AssessmentCreateRequest.model_validate(request)
    assert accepted.notes == request["notes"]
    for field in ("label", "notes"):
        with pytest.raises(ValidationError):
            AssessmentCreateRequest.model_validate({**request, field: request[field] + "x"})
