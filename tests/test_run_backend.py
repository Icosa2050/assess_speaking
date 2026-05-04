from datetime import UTC, datetime, timedelta
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from app_backend.config import build_backend_runtime_config
from scripts import run_backend


class RunBackendLoggingTests(unittest.TestCase):
    def test_configure_backend_logging_writes_to_rotating_log_file(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
                config = build_backend_runtime_config(
                    app_data_dir=app_dir,
                    cache_dir=cache_dir,
                    port=8764,
                )
                run_backend._configure_backend_logging(config.log_file)
                sys.stdout.write("backend hello\n")
                sys.stdout.flush()
                logging.getLogger("uvicorn.error").warning("uvicorn warning")
                sys.stderr.write("backend error\n")
                sys.stderr.flush()
                contents = config.log_file.read_text(encoding="utf-8")

            self.assertIn("backend hello", contents)
            self.assertIn("uvicorn warning", contents)
            self.assertIn("backend error", contents)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_run_startup_cleanup_prunes_safe_targets_and_preserves_active_log(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )
            now = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
            transient = config.app_data.temp_dir / "transient.tmp"
            transient.parent.mkdir(parents=True, exist_ok=True)
            transient.write_text("temp", encoding="utf-8")

            bundle_dir = config.app_data.temp_dir / "support-bundles"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            expired_bundle = bundle_dir / "expired.zip"
            fresh_bundle = bundle_dir / "fresh.zip"
            expired_bundle.write_text("expired", encoding="utf-8")
            fresh_bundle.write_text("fresh", encoding="utf-8")
            old_mtime = time.time() - (72 * 60 * 60)
            fresh_mtime = time.time()
            os.utime(expired_bundle, (old_mtime, old_mtime))
            os.utime(fresh_bundle, (fresh_mtime, fresh_mtime))

            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            config.log_file.write_text("active", encoding="utf-8")
            rotated_log = config.app_data.logs_dir / "backend.log.1"
            rotated_log.write_text("rotated", encoding="utf-8")

            stale_job = config.jobs_dir / "stale.json"
            stale_job.write_text(
                '{"status":"completed","completed_at":"2026-03-01T00:00:00+00:00"}',
                encoding="utf-8",
            )

            with mock.patch("scripts.run_backend.execute_cleanup", wraps=run_backend.execute_cleanup) as mock_cleanup:
                run_backend._run_startup_cleanup(config)

            self.assertFalse(transient.exists())
            self.assertFalse(expired_bundle.exists())
            self.assertTrue(fresh_bundle.exists())
            self.assertFalse(stale_job.exists())
            self.assertTrue(config.log_file.exists())
            self.assertFalse(rotated_log.exists())
            mock_cleanup.assert_called_once()

    def test_main_runs_startup_cleanup_before_logging_and_server(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            config = build_backend_runtime_config(
                app_data_dir=app_dir,
                cache_dir=cache_dir,
                port=8764,
            )
            call_order: list[str] = []

            def _cleanup(_config) -> None:
                call_order.append("cleanup")

            def _configure(_log_file: Path) -> None:
                call_order.append("logging")

            def _run(*_args, **_kwargs) -> None:
                call_order.append("uvicorn")

            with mock.patch("scripts.run_backend.build_backend_runtime_config", return_value=config) as mock_build_config, mock.patch(
                "scripts.run_backend._run_startup_cleanup",
                side_effect=_cleanup,
            ) as mock_startup_cleanup, mock.patch(
                "scripts.run_backend._configure_backend_logging",
                side_effect=_configure,
            ) as mock_configure_logging, mock.patch(
                "scripts.run_backend.uvicorn.run",
                side_effect=_run,
            ) as mock_uvicorn_run:
                result = run_backend.main(
                    [
                        "--app-data-dir",
                        app_dir,
                        "--cache-dir",
                        cache_dir,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8764",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(call_order, ["cleanup", "logging", "uvicorn"])
        mock_build_config.assert_called_once_with(
            log_dir=None,
            app_data_dir=app_dir,
            cache_dir=cache_dir,
            port=8764,
            host="127.0.0.1",
        )
        mock_startup_cleanup.assert_called_once_with(config)
        mock_configure_logging.assert_called_once_with(config.log_file)
        mock_uvicorn_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
