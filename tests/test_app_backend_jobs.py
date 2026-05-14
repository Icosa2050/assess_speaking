import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest import mock

from app_backend.config import build_backend_runtime_config
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


if __name__ == "__main__":
    unittest.main()
