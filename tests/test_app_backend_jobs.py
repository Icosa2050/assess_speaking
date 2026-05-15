import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest import mock

from app_backend.config import build_backend_runtime_config
from app_backend.contracts import AssessmentCreateRequest
from app_backend.jobs import JobManager, prunable_job_metadata_files


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
