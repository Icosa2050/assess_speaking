import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest import mock

from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateRequest, ErrorCode, JobStatus
from app_backend.jobs import JobManager, _job_worker, prunable_job_metadata_files
from assessment_runtime.runner import AssessmentRunResult


class BackendJobsTests(unittest.TestCase):
    def test_prunable_job_metadata_files_handles_mixed_case_terminal_statuses(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8766)
            now = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
            old_completed = config.jobs_dir / "old-completed.json"
            old_completed.write_text(
                json.dumps({"status": "Completed", "completed_at": (now - timedelta(days=31)).isoformat()}),
                encoding="utf-8",
            )

            candidates = prunable_job_metadata_files(config.jobs_dir, now=now)

        self.assertEqual(candidates, [old_completed])

    def test_cancel_does_not_overwrite_completed_job_after_process_join(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8767)
            manager = JobManager(config)
            assessment_id = "asmt_race"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps(
                    {
                        "assessment_id": assessment_id,
                        "status": "running",
                        "phase": "running",
                        "progress": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            process = mock.Mock()
            process.is_alive.return_value = True

            def complete_on_join(timeout=None):
                job_file.write_text(
                    json.dumps(
                        {
                            "assessment_id": assessment_id,
                            "status": "completed",
                            "phase": "done",
                            "progress": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )

            process.join.side_effect = complete_on_join
            manager._processes[assessment_id] = process

            status = manager.cancel(assessment_id)

        self.assertEqual(status.status.value, "completed")

    def test_submit_does_not_persist_llm_api_key_in_job_metadata(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8769)
            manager = JobManager(config)
            upload = manager.register_upload(data=b"fake-audio", filename="sample.wav")
            process = mock.Mock()

            with mock.patch.object(manager._ctx, "Process", return_value=process) as process_factory:
                created = manager.submit(
                    AssessmentCreateRequest(
                        audio_id=upload.audio_id,
                        whisper="tiny",
                        provider="openrouter",
                        llm_model="demo-model",
                        expected_language="it",
                        feedback_language="en",
                        speaker_id="speaker",
                        task_family="free_monologue",
                        theme="Travel",
                        target_duration_sec=90,
                        llm_api_key="sk-secret",
                    )
                )

            job_payload = json.loads((config.jobs_dir / f"{created.assessment_id}.json").read_text(encoding="utf-8"))
            self.assertNotIn("llm_api_key", job_payload["request"])
            worker_request = process_factory.call_args.kwargs["args"][1]
            self.assertEqual(worker_request["llm_api_key"], "sk-secret")
            process.start.assert_called_once_with()

    def test_worker_persists_completed_payload_summary_and_runtime_env(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8770)
            assessment_id = "asmt_worker_success"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "queued", "phase": "queued", "progress": 0.0}),
                encoding="utf-8",
            )
            audio_path = config.app_data.uploads_dir / "sample.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"fake-audio")
            report_path = config.app_data.reports_dir / "report.json"
            saved_payload = {
                "report": {
                    "scores": {"final": 4.2, "band": "B2"},
                    "coaching": {"next_focus": "Use clearer connectors."},
                }
            }
            request_payload = {
                "whisper": "small",
                "provider": "openrouter",
                "llm_model": "demo-model",
                "target_cefr": "B2",
                "theme": "travel",
                "task_family": "free_monologue",
                "speaker_id": "speaker",
                "target_duration_sec": 90,
                "expected_language": "it",
                "language_profile_key": "it",
                "feedback_language": "en",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "dry_run": True,
                "log_dir": str(config.app_data.reports_dir),
                "label": "trial",
                "notes": "notes",
                "llm_api_key": "sk-worker",
                "openrouter_http_referer": "https://example.test/app",
                "openrouter_app_title": "Vostavo Desktop",
            }

            def fake_execute(run_request, status_callback):
                self.assertEqual(run_request.audio, audio_path)
                self.assertEqual(run_request.whisper_model, "small")
                self.assertEqual(run_request.provider, "openrouter")
                self.assertTrue(run_request.dry_run)
                status_callback("transcribing")
                return AssessmentRunResult(
                    meta={},
                    output={"report": {}},
                    stdout_json="{}",
                    report=saved_payload["report"],
                    report_path=report_path,
                    saved_payload=saved_payload,
                )

            with mock.patch.dict("app_backend.jobs.os.environ", {}, clear=True), mock.patch(
                "app_backend.jobs.execute_assessment_run",
                side_effect=fake_execute,
            ):
                _job_worker(str(job_file), request_payload, str(audio_path))

                self.assertEqual(os.environ["LLM_API_KEY"], "sk-worker")
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "sk-worker")
                self.assertEqual(os.environ["OPENROUTER_HTTP_REFERER"], "https://example.test/app")
                self.assertEqual(os.environ["OPENROUTER_APP_TITLE"], "Vostavo Desktop")

            payload = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], JobStatus.COMPLETED.value)
            self.assertEqual(payload["phase"], "done")
            self.assertEqual(payload["progress"], 1.0)
            self.assertEqual(payload["payload"], saved_payload)
            self.assertEqual(payload["summary"]["score_overall"], 4.2)
            self.assertEqual(payload["summary"]["band"], "B2")
            self.assertEqual(payload["summary"]["next_focus"], "Use clearer connectors.")
            self.assertEqual(payload["report_path"], str(report_path.resolve()))

    def test_worker_persists_local_provider_failure_code(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8771)
            assessment_id = "asmt_worker_failure"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "queued", "phase": "queued", "progress": 0.0}),
                encoding="utf-8",
            )
            audio_path = config.app_data.uploads_dir / "sample.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"fake-audio")
            request_payload = {
                "whisper": "small",
                "provider": "ollama",
                "llm_model": "demo-model",
                "theme": "travel",
                "task_family": "free_monologue",
                "speaker_id": "speaker",
                "target_duration_sec": 90,
                "log_dir": str(config.app_data.reports_dir),
            }

            with mock.patch(
                "app_backend.jobs.execute_assessment_run",
                side_effect=RuntimeError("connection refused by local provider"),
            ):
                _job_worker(str(job_file), request_payload, str(audio_path))

            payload = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], JobStatus.FAILED.value)
            self.assertEqual(payload["phase"], "failed")
            self.assertEqual(payload["progress"], 1.0)
            self.assertEqual(payload["error"]["code"], ErrorCode.LOCAL_PROVIDER_NOT_RUNNING.value)
            self.assertIn("connection refused", payload["error"]["detail"])

    def test_get_status_marks_unfinished_job_failed_when_worker_exited(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8772)
            manager = JobManager(config)
            assessment_id = "asmt_dead_worker"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "running", "phase": "transcribing", "progress": 0.0}),
                encoding="utf-8",
            )
            process = mock.Mock()
            process.is_alive.return_value = False
            process.exitcode = 9
            manager._processes[assessment_id] = process

            status = manager.get_status(assessment_id)

        self.assertEqual(status.status, JobStatus.FAILED)
        self.assertEqual(status.phase, "failed")
        self.assertEqual(status.progress, 1.0)
        self.assertEqual(status.error.code, ErrorCode.RUNTIME)
        self.assertIn("exited unexpectedly with code 9", status.error.detail)
        process.join.assert_called_once_with(timeout=0.1)
        self.assertNotIn(assessment_id, manager._processes)

    def test_cancel_marks_queued_job_cancelled_without_worker_process(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8773)
            manager = JobManager(config)
            assessment_id = "asmt_cancel_queued"
            job_file = config.jobs_dir / f"{assessment_id}.json"
            job_file.write_text(
                json.dumps({"assessment_id": assessment_id, "status": "queued", "phase": "queued", "progress": 0.0}),
                encoding="utf-8",
            )

            status = manager.cancel(assessment_id)

        self.assertEqual(status.status, JobStatus.CANCELLED)
        self.assertEqual(status.phase, "cancelled")
        self.assertEqual(status.progress, 1.0)
        self.assertEqual(status.error.code, ErrorCode.CANCELLATION)

    def test_shutdown_logs_worker_termination_failures(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8768)
            manager = JobManager(config)
            process = mock.Mock()
            process.pid = 12345
            process.is_alive.return_value = True
            manager._processes["asmt_stuck"] = process

            with mock.patch("app_backend.jobs.os.kill", side_effect=OSError("denied")), self.assertLogs(
                "app_backend.jobs",
                level="WARNING",
            ) as logs:
                manager.shutdown()

        self.assertIn("Could not terminate assessment worker", "\n".join(logs.output))
        process.join.assert_called_once_with(timeout=0.5)
        self.assertNotIn("asmt_stuck", manager._processes)


if __name__ == "__main__":
    unittest.main()
