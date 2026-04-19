import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import httpx

from app_backend.config import BACKEND_STATE_FILENAME, BackendRuntimeConfig
from app_backend.lifecycle import (
    _backend_command,
    _safe_terminate_pid,
    ensure_local_backend,
    get_backend_state,
    is_backend_healthy,
    PROJECT_ROOT,
)
from app_shell.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR, AppDataPaths


def _build_paths(root: Path) -> AppDataPaths:
    reports_dir = root / "reports"
    cache_root = root / "cache"
    return AppDataPaths(
        root=root,
        reports_dir=reports_dir,
        jobs_dir=root / "jobs",
        logs_dir=root / "logs",
        recordings_dir=reports_dir / "recordings",
        uploads_dir=reports_dir / "uploads",
        temp_dir=root / "tmp",
        cache_root=cache_root,
        whisper_cache_dir=cache_root / "whisper",
    )


def _build_config(root: Path, *, host: str = "127.0.0.1", port: int = 8123) -> BackendRuntimeConfig:
    app_data = _build_paths(root)
    return BackendRuntimeConfig(
        host=host,
        port=port,
        app_data=app_data,
        state_file=root / BACKEND_STATE_FILENAME,
        jobs_dir=app_data.jobs_dir,
        log_file=app_data.logs_dir / "backend.log",
    )


class BackendLifecycleTests(unittest.TestCase):
    def test_backend_command_includes_optional_overrides(self):
        command = _backend_command(
            log_dir="reports/live",
            app_data_dir="/tmp/app-data",
            cache_dir="/tmp/cache-data",
            host="0.0.0.0",
            port=9001,
        )

        self.assertEqual(command[:6], [str(Path(os.sys.executable)), str(PROJECT_ROOT / "scripts" / "run_backend.py"), "--host", "0.0.0.0", "--port", "9001"])
        self.assertIn("--log-dir", command)
        self.assertIn("reports/live", command)
        self.assertIn("--app-data-dir", command)
        self.assertIn("/tmp/app-data", command)
        self.assertIn("--cache-dir", command)
        self.assertIn("/tmp/cache-data", command)
        self.assertEqual(command[-1], "/tmp/cache-data")

    def test_backend_command_skips_blank_overrides(self):
        command = _backend_command(log_dir=" ", app_data_dir=None, cache_dir="", port=9002)

        self.assertNotIn("--log-dir", command)
        self.assertNotIn("--app-data-dir", command)
        self.assertNotIn("--cache-dir", command)

    def test_is_backend_healthy_checks_status_and_handles_http_errors(self):
        with mock.patch("app_backend.lifecycle.httpx.get", return_value=mock.Mock(status_code=200)):
            self.assertTrue(is_backend_healthy("http://127.0.0.1:8123/"))

        with mock.patch("app_backend.lifecycle.httpx.get", return_value=mock.Mock(status_code=503)):
            self.assertFalse(is_backend_healthy("http://127.0.0.1:8123"))

        with mock.patch("app_backend.lifecycle.httpx.get", side_effect=httpx.ConnectError("offline")):
            self.assertFalse(is_backend_healthy("http://127.0.0.1:8123"))

    def test_safe_terminate_pid_ignores_invalid_pids_and_os_errors(self):
        with mock.patch("app_backend.lifecycle.os.kill") as mock_kill:
            _safe_terminate_pid(0)
        mock_kill.assert_not_called()

        with mock.patch("app_backend.lifecycle.os.kill") as mock_kill:
            _safe_terminate_pid(123)
        mock_kill.assert_called_once()

        with mock.patch("app_backend.lifecycle.os.kill", side_effect=OSError):
            _safe_terminate_pid(456)

    def test_get_backend_state_returns_none_for_invalid_state(self):
        with mock.patch("app_backend.lifecycle.read_backend_state", return_value=[]):
            self.assertIsNone(get_backend_state())

    def test_get_backend_state_clears_stale_backend_processes(self):
        stale_state = {"base_url": "http://127.0.0.1:8123", "pid": 99}
        with mock.patch("app_backend.lifecycle.read_backend_state", return_value=stale_state), mock.patch(
            "app_backend.lifecycle.is_backend_healthy",
            return_value=False,
        ), mock.patch("app_backend.lifecycle._safe_terminate_pid") as mock_terminate, mock.patch(
            "app_backend.lifecycle.clear_backend_state"
        ) as mock_clear:
            self.assertIsNone(get_backend_state(log_dir="reports", app_data_dir="/tmp/app", cache_dir="/tmp/cache"))

        mock_terminate.assert_called_once_with(99)
        mock_clear.assert_called_once_with("reports", app_data_dir="/tmp/app", cache_dir="/tmp/cache")

    def test_get_backend_state_returns_healthy_state(self):
        healthy_state = {"base_url": "http://127.0.0.1:8123", "pid": 99}
        with mock.patch("app_backend.lifecycle.read_backend_state", return_value=healthy_state), mock.patch(
            "app_backend.lifecycle.is_backend_healthy",
            return_value=True,
        ), mock.patch("app_backend.lifecycle._safe_terminate_pid") as mock_terminate, mock.patch(
            "app_backend.lifecycle.clear_backend_state"
        ) as mock_clear:
            result = get_backend_state()

        self.assertEqual(result, healthy_state)
        mock_terminate.assert_not_called()
        mock_clear.assert_not_called()

    def test_ensure_local_backend_reuses_existing_state(self):
        existing = {"base_url": "http://127.0.0.1:8123", "pid": 77}
        with mock.patch("app_backend.lifecycle.get_backend_state", return_value=existing), mock.patch(
            "app_backend.lifecycle.build_backend_runtime_config"
        ) as mock_build:
            result = ensure_local_backend()

        self.assertEqual(result, existing)
        mock_build.assert_not_called()

    def test_ensure_local_backend_starts_process_and_waits_for_health(self):
        with tempfile.TemporaryDirectory() as root_dir, mock.patch.dict(os.environ, {"EXISTING": "1"}, clear=True):
            config = _build_config(Path(root_dir).resolve())
            ready_state = {"base_url": config.base_url, "pid": 42}

            with mock.patch("app_backend.lifecycle.get_backend_state", return_value=None), mock.patch(
                "app_backend.lifecycle.build_backend_runtime_config",
                return_value=config,
            ), mock.patch(
                "app_backend.lifecycle._backend_command",
                return_value=["python", "scripts/run_backend.py"],
            ), mock.patch(
                "app_backend.lifecycle.subprocess.Popen"
            ) as mock_popen, mock.patch(
                "app_backend.lifecycle.read_backend_state",
                side_effect=[None, ready_state],
            ), mock.patch(
                "app_backend.lifecycle.is_backend_healthy",
                return_value=True,
            ), mock.patch(
                "app_backend.lifecycle.time.time",
                side_effect=[100.0, 100.1, 100.2],
            ), mock.patch("app_backend.lifecycle.time.sleep") as mock_sleep:
                result = ensure_local_backend(
                    log_dir="reports/live",
                    app_data_dir=str(config.app_data.root),
                    cache_dir=str(config.app_data.cache_root),
                    startup_timeout_sec=15.0,
                )

        self.assertEqual(result, ready_state)
        mock_popen.assert_called_once()
        self.assertEqual(mock_popen.call_args.kwargs["cwd"], str(PROJECT_ROOT))
        env = mock_popen.call_args.kwargs["env"]
        self.assertEqual(env["EXISTING"], "1")
        self.assertEqual(env[APP_DATA_HOME_ENV_VAR], str(config.app_data.root))
        self.assertEqual(env[APP_CACHE_HOME_ENV_VAR], str(config.app_data.cache_root))
        mock_sleep.assert_called_once_with(0.2)

    def test_ensure_local_backend_raises_after_timeout(self):
        with tempfile.TemporaryDirectory() as root_dir:
            config = _build_config(Path(root_dir).resolve(), port=9003)
            with mock.patch("app_backend.lifecycle.get_backend_state", return_value=None), mock.patch(
                "app_backend.lifecycle.build_backend_runtime_config",
                return_value=config,
            ), mock.patch(
                "app_backend.lifecycle._backend_command",
                return_value=["python", "scripts/run_backend.py"],
            ), mock.patch(
                "app_backend.lifecycle.subprocess.Popen"
            ) as mock_popen, mock.patch(
                "app_backend.lifecycle.read_backend_state",
                return_value={"base_url": config.base_url},
            ), mock.patch(
                "app_backend.lifecycle.is_backend_healthy",
                return_value=False,
            ), mock.patch(
                "app_backend.lifecycle.time.time",
                side_effect=[100.0, 100.1, 100.4],
            ), mock.patch("app_backend.lifecycle.time.sleep") as mock_sleep:
                with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                    ensure_local_backend(startup_timeout_sec=0.15)

        mock_popen.assert_called_once()
        mock_sleep.assert_called_once_with(0.2)


if __name__ == "__main__":
    unittest.main()
