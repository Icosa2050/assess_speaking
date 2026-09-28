from datetime import UTC, datetime, timedelta
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_backend.config import (
    BACKEND_JOBS_DIRNAME,
    BACKEND_LOG_FILENAME,
    BACKEND_STATE_FILENAME,
    DEFAULT_JOB_METADATA_RETENTION_DAYS,
    allocate_local_port,
    build_backend_runtime_config,
    clear_backend_state,
    read_backend_state,
    resolve_backend_log_file,
    resolve_backend_state_file,
    resolve_jobs_dir,
    resolve_support_bundle_dir,
    write_backend_state,
)
from app_backend.contracts import CleanupTarget
from app_backend.jobs import prunable_job_metadata_files, recover_incomplete_job_metadata
from app_backend.maintenance import execute_cleanup
from app_core.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR


class BackendConfigTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_resolve_helpers_follow_app_data_root(self):
        with tempfile.TemporaryDirectory() as app_dir, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: app_dir},
            clear=False,
        ):
            root = Path(app_dir).resolve()
            self.assertEqual(resolve_backend_state_file("reports"), root / BACKEND_STATE_FILENAME)
            self.assertEqual(resolve_jobs_dir("reports"), root / BACKEND_JOBS_DIRNAME)
            self.assertEqual(resolve_backend_log_file("reports"), root / "logs" / BACKEND_LOG_FILENAME)
            self.assertEqual(resolve_support_bundle_dir("reports"), root / "tmp" / "support-bundles")

    def test_allocate_local_port_returns_bindable_port(self):
        port = allocate_local_port()
        self.assertGreater(port, 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))

    def test_build_write_read_and_clear_backend_state(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                host="0.0.0.0",
                port=8764,
            )

            self.assertEqual(config.base_url, "http://0.0.0.0:8764")
            self.assertEqual(config.state_file, Path(app_dir).resolve() / BACKEND_STATE_FILENAME)
            self.assertEqual(config.log_file, Path(app_dir).resolve() / "logs" / BACKEND_LOG_FILENAME)
            self.assertTrue(config.jobs_dir.exists())

            payload = write_backend_state(config, pid=321)

            self.assertEqual(payload["base_url"], config.base_url)
            self.assertEqual(payload["jobs_dir"], str(config.jobs_dir))
            self.assertEqual(payload["log_file"], str(config.log_file))
            self.assertEqual(read_backend_state(app_data_dir=app_dir, cache_dir=cache_dir), payload)

            clear_backend_state(app_data_dir=app_dir, cache_dir=cache_dir)
            self.assertIsNone(read_backend_state(app_data_dir=app_dir, cache_dir=cache_dir))

            # Missing state files should be ignored during cleanup.
            clear_backend_state(app_data_dir=app_dir, cache_dir=cache_dir)

    def test_read_backend_state_returns_none_for_invalid_json_and_non_dict_payloads(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ,
            {
                APP_DATA_HOME_ENV_VAR: app_dir,
                APP_CACHE_HOME_ENV_VAR: cache_dir,
            },
            clear=False,
        ):
            state_file = Path(app_dir).resolve() / BACKEND_STATE_FILENAME
            state_file.parent.mkdir(parents=True, exist_ok=True)

            state_file.write_text("{invalid", encoding="utf-8")
            self.assertIsNone(read_backend_state(app_data_dir=app_dir, cache_dir=cache_dir))

            state_file.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
            self.assertIsNone(read_backend_state(app_data_dir=app_dir, cache_dir=cache_dir))

    def test_build_backend_runtime_config_migrates_legacy_jobs_dir(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            root = Path(app_dir).resolve()
            legacy_jobs_dir = root / "reports" / BACKEND_JOBS_DIRNAME
            legacy_jobs_dir.mkdir(parents=True, exist_ok=True)
            (legacy_jobs_dir / "asmt-old.json").write_text('{"status":"completed"}', encoding="utf-8")

            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )

            self.assertTrue((config.jobs_dir / "asmt-old.json").exists())
            self.assertFalse(legacy_jobs_dir.exists())

    def test_build_backend_runtime_config_logs_legacy_jobs_migration_failures(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            root = Path(app_dir).resolve()
            legacy_jobs_dir = root / "reports" / BACKEND_JOBS_DIRNAME
            legacy_jobs_dir.mkdir(parents=True, exist_ok=True)
            (legacy_jobs_dir / "asmt-old.json").write_text('{"status":"completed"}', encoding="utf-8")

            with mock.patch("app_backend.config.shutil.move", side_effect=OSError("locked")), self.assertLogs(
                "app_backend.config",
                level="WARNING",
            ) as logs:
                config = build_backend_runtime_config(
                    app_data_dir=app_dir,
                    cache_dir=cache_dir,
                    port=8764,
                )

            self.assertTrue(config.jobs_dir.exists())
            self.assertIn("Could not migrate legacy backend job metadata", "\n".join(logs.output))
            self.assertTrue((legacy_jobs_dir / "asmt-old.json").exists())

    def test_clear_backend_state_logs_unlink_failures(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            state_file = Path(app_dir).resolve() / BACKEND_STATE_FILENAME
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text("{}", encoding="utf-8")

            with mock.patch.object(Path, "unlink", side_effect=OSError("locked")), self.assertLogs(
                "app_backend.config",
                level="WARNING",
            ) as logs:
                clear_backend_state(app_data_dir=app_dir, cache_dir=cache_dir)

            self.assertIn("Could not clear backend state file", "\n".join(logs.output))

    def test_recover_incomplete_job_metadata_marks_running_and_queued_as_failed(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )
            self._write_json(config.jobs_dir / "queued.json", {"status": "queued", "phase": "queued", "progress": 0.0})
            self._write_json(config.jobs_dir / "running.json", {"status": "running", "phase": "transcribing", "progress": 0.4})
            self._write_json(config.jobs_dir / "completed.json", {"status": "completed", "phase": "done", "progress": 1.0})

            recovered = recover_incomplete_job_metadata(config.jobs_dir)

            self.assertEqual(recovered, 2)
            queued = json.loads((config.jobs_dir / "queued.json").read_text(encoding="utf-8"))
            running = json.loads((config.jobs_dir / "running.json").read_text(encoding="utf-8"))
            completed = json.loads((config.jobs_dir / "completed.json").read_text(encoding="utf-8"))
            self.assertEqual(queued["status"], "failed")
            self.assertEqual(running["status"], "failed")
            self.assertEqual(completed["status"], "completed")
            self.assertIn("completed_at", queued)
            self.assertIn("Backend restarted", queued["error"]["detail"])

    def test_prunable_job_metadata_files_keeps_recent_and_incomplete_jobs(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )
            now = datetime(2026, 4, 22, 10, 0, tzinfo=UTC)
            old_completed = config.jobs_dir / "old-completed.json"
            old_failed = config.jobs_dir / "old-failed.json"
            recent_cancelled = config.jobs_dir / "recent-cancelled.json"
            running = config.jobs_dir / "running.json"
            self._write_json(
                old_completed,
                {"status": "completed", "completed_at": (now - timedelta(days=DEFAULT_JOB_METADATA_RETENTION_DAYS + 1)).isoformat()},
            )
            self._write_json(
                old_failed,
                {"status": "failed", "completed_at": (now - timedelta(days=DEFAULT_JOB_METADATA_RETENTION_DAYS + 5)).isoformat()},
            )
            self._write_json(
                recent_cancelled,
                {"status": "cancelled", "completed_at": (now - timedelta(days=2)).isoformat()},
            )
            self._write_json(
                running,
                {"status": "running", "created_at": (now - timedelta(days=90)).isoformat()},
            )

            candidates = prunable_job_metadata_files(config.jobs_dir, now=now)

            self.assertEqual(candidates, sorted([old_completed, old_failed]))

    def test_execute_cleanup_all_safe_preserves_fresh_support_bundles_active_log_and_user_content(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )
            now = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

            transient = config.app_data.temp_dir / "scratch" / "transient.tmp"
            transient.parent.mkdir(parents=True, exist_ok=True)
            transient.write_text("temp", encoding="utf-8")

            bundle_dir = config.app_data.temp_dir / "support-bundles"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            expired_bundle = bundle_dir / "expired.zip"
            fresh_bundle = bundle_dir / "fresh.zip"
            expired_bundle.write_text("expired", encoding="utf-8")
            fresh_bundle.write_text("fresh", encoding="utf-8")
            old_mtime = (now - timedelta(hours=25)).timestamp()
            fresh_mtime = (now - timedelta(hours=1)).timestamp()
            os.utime(expired_bundle, (old_mtime, old_mtime))
            os.utime(fresh_bundle, (fresh_mtime, fresh_mtime))

            active_log = config.log_file
            rotated_log = config.app_data.logs_dir / "backend.log.1"
            custom_log = config.app_data.logs_dir / "unexpected.log"
            active_log.parent.mkdir(parents=True, exist_ok=True)
            active_log.write_text("active", encoding="utf-8")
            rotated_log.write_text("rotated", encoding="utf-8")
            custom_log.write_text("custom", encoding="utf-8")

            stale_job = config.jobs_dir / "stale.json"
            recent_job = config.jobs_dir / "recent.json"
            queued_job = config.jobs_dir / "queued.json"
            self._write_json(
                stale_job,
                {"status": "completed", "completed_at": (now - timedelta(days=31)).isoformat()},
            )
            self._write_json(
                recent_job,
                {"status": "failed", "completed_at": (now - timedelta(days=2)).isoformat()},
            )
            self._write_json(
                queued_job,
                {"status": "queued", "created_at": (now - timedelta(days=60)).isoformat()},
            )

            report_file = config.app_data.reports_dir / "session.json"
            recording_file = config.app_data.recordings_dir / "sample.wav"
            upload_file = config.app_data.uploads_dir / "sample.wav"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            recording_file.parent.mkdir(parents=True, exist_ok=True)
            upload_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text("report", encoding="utf-8")
            recording_file.write_bytes(b"recording")
            upload_file.write_bytes(b"upload")

            tmp_dry_run = execute_cleanup(config, CleanupTarget.TMP, dry_run=True, now=now)

            self.assertEqual(tmp_dry_run.deleted_file_count, 1)
            self.assertTrue(transient.exists())

            all_safe = execute_cleanup(config, CleanupTarget.ALL_SAFE, dry_run=False, now=now)

            self.assertEqual(all_safe.target, CleanupTarget.ALL_SAFE)
            self.assertFalse(transient.exists())
            self.assertFalse(expired_bundle.exists())
            self.assertTrue(fresh_bundle.exists())
            self.assertFalse(stale_job.exists())
            self.assertTrue(recent_job.exists())
            self.assertTrue(queued_job.exists())
            self.assertTrue(active_log.exists())
            self.assertFalse(rotated_log.exists())
            self.assertFalse(custom_log.exists())
            self.assertTrue(report_file.exists())
            self.assertTrue(recording_file.exists())
            self.assertTrue(upload_file.exists())

    def test_execute_cleanup_counts_only_successful_non_dry_run_deletes(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(app_data_dir=app_dir, cache_dir=cache_dir, port=8765)
            deleted = config.app_data.temp_dir / "deleted.tmp"
            locked = config.app_data.temp_dir / "locked.tmp"
            vanished = config.app_data.temp_dir / "vanished.tmp"
            deleted.parent.mkdir(parents=True, exist_ok=True)
            deleted.write_text("gone", encoding="utf-8")
            locked.write_text("stay", encoding="utf-8")

            original_unlink = Path.unlink

            def flaky_unlink(path):
                if path == locked:
                    raise OSError("locked")
                return original_unlink(path)

            with mock.patch(
                "app_backend.maintenance.cleanup_candidates",
                return_value=[deleted, locked, vanished],
            ), mock.patch.object(Path, "unlink", autospec=True, side_effect=flaky_unlink):
                result = execute_cleanup(config, CleanupTarget.TMP, dry_run=False)

        self.assertEqual(result.deleted_file_count, 1)
        self.assertEqual(result.freed_bytes, 4)


if __name__ == "__main__":
    unittest.main()
