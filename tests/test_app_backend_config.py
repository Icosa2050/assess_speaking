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
    allocate_local_port,
    build_backend_runtime_config,
    clear_backend_state,
    read_backend_state,
    resolve_backend_log_file,
    resolve_backend_state_file,
    resolve_jobs_dir,
    write_backend_state,
)
from app_shell.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR


class BackendConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
