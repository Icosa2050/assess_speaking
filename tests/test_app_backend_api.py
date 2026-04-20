import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app_backend.app import create_app
from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateResponse, AssessmentStatusResponse, JobStatus


class BackendApiTests(unittest.TestCase):
    def test_runtime_config_honors_app_data_and_cache_overrides(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )

        self.assertEqual(config.app_data.root, Path(app_dir).resolve())
        self.assertEqual(config.app_data.cache_root, Path(cache_dir).resolve())
        self.assertEqual(config.state_file, Path(app_dir).resolve() / "backend_state.json")
        self.assertEqual(config.jobs_dir, Path(app_dir).resolve() / "jobs")
        self.assertEqual(config.log_file, Path(app_dir).resolve() / "logs" / "backend.log")

    def test_health_and_samples_endpoints_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8765)
            client = TestClient(create_app(config))
            health = client.get("/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ready")

            samples = client.get("/v1/samples")
            self.assertEqual(samples.status_code, 200)
            self.assertTrue(samples.json()["items"])

    def test_upload_endpoint_persists_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8766)
            client = TestClient(create_app(config))
            response = client.post(
                "/v1/uploads",
                files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(Path(payload["stored_path"]).exists())
            self.assertTrue(payload["audio_id"].startswith("aud_"))

    def test_assessment_routes_delegate_to_job_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_backend_runtime_config(log_dir=tmpdir, port=8767)
            app = create_app(config)
            client = TestClient(app)
            fake_submit = AssessmentCreateResponse(assessment_id="asmt_1", status=JobStatus.QUEUED)
            fake_status = AssessmentStatusResponse(
                assessment_id="asmt_1",
                status=JobStatus.COMPLETED,
                phase="done",
                progress=1.0,
                summary=None,
                error=None,
                report_path=None,
                payload={"report": {"session_id": "sess-1"}},
            )
            with mock.patch.object(app.state.job_manager, "submit", return_value=fake_submit) as mock_submit, \
                    mock.patch.object(app.state.job_manager, "get_status", return_value=fake_status) as mock_status, \
                    mock.patch.object(app.state.job_manager, "cancel", return_value=fake_status) as mock_cancel:
                created = client.post(
                    "/v1/assessments",
                    json={
                        "audio_id": "aud_1",
                        "whisper": "small",
                        "provider": "openrouter",
                        "llm_model": "google/gemini-3.1-pro-preview",
                        "expected_language": "en",
                        "feedback_language": "en",
                        "speaker_id": "bern",
                        "task_family": "free_monologue",
                        "theme": "travel",
                        "target_duration_sec": 90,
                        "dry_run": True,
                    },
                )
                self.assertEqual(created.status_code, 200)
                self.assertEqual(created.json()["assessment_id"], "asmt_1")
                mock_submit.assert_called_once()
                self.assertTrue(mock_submit.call_args.args[0].dry_run)

                status = client.get("/v1/assessments/asmt_1")
                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json()["status"], "completed")
                mock_status.assert_called_once_with("asmt_1")

                cancelled = client.post("/v1/assessments/asmt_1/cancel")
                self.assertEqual(cancelled.status_code, 200)
                self.assertEqual(cancelled.json()["phase"], "done")
                mock_cancel.assert_called_once_with("asmt_1")


if __name__ == "__main__":
    unittest.main()
