from datetime import UTC, datetime, timedelta
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import CleanupTarget
from app_backend.jobs import JobManager
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
                                   "request": {"request_id": "assessment-1", "audio_id": audio_id},
                                   "payload": {"report": {"session_id": "report-1"}}}), encoding="utf-8")
        restarted = TestClient(create_app(config), base_url="http://127.0.0.1:8870")
        assert restarted.get("/v1/uploads/by-request/upload-1").status_code == 401
        assert restarted.get("/v1/assessments/by-request/assessment-1").status_code == 401
        assert restarted.get("/v1/uploads/by-request/upload-1", headers=headers).json() == {"audio_id": audio_id}
        assert restarted.get("/v1/assessments/by-request/assessment-1", headers=headers).json() == {
            "assessment_id": "asmt_one", "status": "completed", "report_id": "report-1"}
        assert restarted.get("/v1/uploads/by-request/absent", headers=headers).status_code == 404
        assert restarted.get("/v1/assessments/by-request/absent", headers=headers).status_code == 404


def test_orphan_upload_cleanup_spares_jobs_journal_recent_and_unknown_age(tmp_path):
    config = build_backend_runtime_config(app_data_dir=tmp_path, port=8871)
    manager = JobManager(config)
    names = ("orphan", "job", "journal", "recent", "receipt_window", "legacy")
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
    preview = execute_cleanup(config, CleanupTarget.JOBS, dry_run=True)
    assert preview.deleted_file_count == 2
    assert all((config.app_data.uploads_dir / f"{upload.audio_id}.json").exists() for upload in uploads.values())
    cleaned = execute_cleanup(config, CleanupTarget.JOBS)
    assert cleaned.deleted_file_count == 2
    assert not (config.app_data.uploads_dir / f"{uploads['orphan'].audio_id}.json").exists()
    assert not Path(uploads["orphan"].stored_path).exists()
    for name in names[1:]:
        assert (config.app_data.uploads_dir / f"{uploads[name].audio_id}.json").exists()


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
