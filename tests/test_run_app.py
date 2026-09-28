import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_core.app_data import AppDataPaths
from app_core import bootstrap as bootstrap_module
from scripts import run_app

ROOT = Path(__file__).resolve().parents[1]


class RunAppTests(unittest.TestCase):
    def test_launcher_payload_uses_overrides(self):
        with mock.patch.dict(os.environ, {}, clear=False), tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            args = run_app._parse_args(
                [
                    "--app-data-dir",
                    app_dir,
                    "--cache-dir",
                    cache_dir,
                    "--log-dir",
                    "reports",
                    "--dry-run",
                ]
            )
            with mock.patch("scripts.run_app.shutil.which", return_value="/opt/homebrew/bin/ffmpeg"), mock.patch(
                "scripts.run_app.get_backend_state",
                return_value={"base_url": "http://127.0.0.1:9000"},
            ):
                payload = run_app._launcher_payload(args)

        self.assertEqual(payload["app_data_root"], str(Path(app_dir).resolve()))
        self.assertEqual(payload["cache_root"], str(Path(cache_dir).resolve()))
        self.assertEqual(payload["reports_dir"], str((Path(app_dir).resolve() / "reports")))
        self.assertEqual(payload["jobs_dir"], str((Path(app_dir).resolve() / "jobs")))
        self.assertEqual(payload["logs_dir"], str((Path(app_dir).resolve() / "logs")))
        self.assertTrue(payload["ffmpeg_available"])
        self.assertEqual(payload["backend_base_url"], "http://127.0.0.1:9000")
        self.assertFalse(payload["repo_local_override_active"])
        self.assertEqual(payload["repo_local_override_targets"], [])
        self.assertEqual(payload["deployment_mode"], "local")
        self.assertEqual(payload["launch_mode"], "repo")
        self.assertTrue(payload["packaging_safe"])
        self.assertEqual(payload["auth_mode"], "guest")
        self.assertNotIn("command", payload)
        self.assertNotIn("entrypoint", payload)

    def test_main_prints_json_for_dry_run(self):
        with mock.patch.dict(os.environ, {}, clear=False), tempfile.TemporaryDirectory() as app_dir, mock.patch(
            "scripts.run_app.shutil.which",
            return_value="/opt/homebrew/bin/ffmpeg",
        ), mock.patch(
            "scripts.run_app.get_backend_state",
            return_value={"base_url": "http://127.0.0.1:9000"},
        ), mock.patch("sys.stdout.write") as mock_write:
            exit_code = run_app.main(["--app-data-dir", app_dir, "--dry-run"])

        written = "".join(call.args[0] for call in mock_write.call_args_list)
        payload = json.loads(written)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["app_data_root"], str(Path(app_dir).resolve()))
        self.assertNotIn("command", payload)

    def test_main_prints_desktop_bootstrap_env(self):
        with mock.patch(
            "scripts.run_app.build_bootstrap_env",
            return_value={
                "VOSTAVO_DESKTOP_API_BASE_URL": "http://127.0.0.1:9000",
                "VOSTAVO_DEPLOYMENT_MODE": "local",
                "VOSTAVO_LAUNCH_MODE": "repo",
                "VOSTAVO_DESKTOP_PACKAGING_SAFE": "true",
                "VOSTAVO_AUTH_MODE": "guest",
            },
        ), mock.patch("sys.stdout.write") as mock_write:
            exit_code = run_app.main(["--desktop-bootstrap"])

        written = "".join(call.args[0] for call in mock_write.call_args_list)
        self.assertEqual(exit_code, 0)
        self.assertIn("VOSTAVO_DESKTOP_API_BASE_URL=http://127.0.0.1:9000", written)
        self.assertIn("VOSTAVO_LAUNCH_MODE=repo", written)
        self.assertIn("VOSTAVO_DESKTOP_PACKAGING_SAFE=true", written)

    def test_main_starts_backend_without_streamlit_command(self):
        with tempfile.TemporaryDirectory() as app_dir, mock.patch(
            "scripts.run_app.ensure_local_backend",
            return_value={"base_url": "http://127.0.0.1:9000"},
        ), mock.patch("sys.stdout.write") as mock_write:
            exit_code = run_app.main(["--app-data-dir", app_dir])

        written = "".join(call.args[0] for call in mock_write.call_args_list)
        payload = json.loads(written)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["backend_base_url"], "http://127.0.0.1:9000")
        self.assertNotIn("command", payload)

    def test_parse_args_rejects_legacy_streamlit_option(self):
        with self.assertRaises(SystemExit):
            run_app._parse_args(["--legacy-streamlit"])

    def test_run_app_dry_run_works_outside_repo_root(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as outside_cwd:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_app.py"),
                    "--app-data-dir",
                    app_dir,
                    "--dry-run",
                ],
                cwd=outside_cwd,
                check=True,
                capture_output=True,
                text=True,
                env={**dict(os.environ), "VOSTAVO_HOME": app_dir, "VOSTAVO_CACHE_HOME": str(Path(app_dir) / "cache")},
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["app_data_root"], str(Path(app_dir).resolve()))
        self.assertEqual(payload["reports_dir"], str(Path(app_dir).resolve() / "reports"))
        self.assertEqual(payload["jobs_dir"], str(Path(app_dir).resolve() / "jobs"))
        self.assertEqual(payload["logs_dir"], str(Path(app_dir).resolve() / "logs"))
        self.assertNotIn("entrypoint", payload)
        self.assertNotIn("command", payload)
        self.assertIn("backend_base_url", payload)
        self.assertFalse(payload["repo_local_override_active"])

    def test_launcher_payload_sets_legacy_compatibility_envs(self):
        with mock.patch.dict(os.environ, {}, clear=False), tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
            args = run_app._parse_args(["--app-data-dir", app_dir, "--cache-dir", cache_dir, "--dry-run"])
            with mock.patch("scripts.run_app.get_backend_state", return_value={"base_url": "http://127.0.0.1:9000"}):
                run_app._launcher_payload(args)

            self.assertEqual(Path(os.environ["VOSTAVO_HOME"]).resolve(), Path(app_dir).resolve())
            self.assertEqual(Path(os.environ["VOSTAVO_CACHE_HOME"]).resolve(), Path(cache_dir).resolve())
            self.assertEqual(Path(os.environ["SPEAKING_STUDIO_HOME"]).resolve(), Path(app_dir).resolve())
            self.assertEqual(Path(os.environ["SPEAKING_STUDIO_CACHE_HOME"]).resolve(), Path(cache_dir).resolve())

    def test_bootstrap_rejects_repo_local_default_root(self):
        repo_root = ROOT.resolve()
        fake_paths = AppDataPaths(
            root=repo_root,
            reports_dir=repo_root / "reports",
            jobs_dir=repo_root / "jobs",
            logs_dir=repo_root / "logs",
            recordings_dir=repo_root / "reports" / "recordings",
            uploads_dir=repo_root / "reports" / "uploads",
            temp_dir=repo_root / "tmp",
            cache_root=Path("/tmp/vostavo-cache"),
            whisper_cache_dir=Path("/tmp/vostavo-cache/whisper"),
        )
        with mock.patch("app_core.bootstrap.build_app_data_paths", return_value=fake_paths), mock.patch(
            "app_core.bootstrap.ensure_app_data_dirs"
        ) as ensure_dirs, mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "inside the repository checkout"):
                bootstrap_module.bootstrap_app_environment()
        ensure_dirs.assert_not_called()

    def test_bootstrap_allows_repo_local_root_when_explicitly_overridden(self):
        repo_root = ROOT.resolve()
        fake_paths = AppDataPaths(
            root=repo_root,
            reports_dir=repo_root / "reports",
            jobs_dir=repo_root / "jobs",
            logs_dir=repo_root / "logs",
            recordings_dir=repo_root / "reports" / "recordings",
            uploads_dir=repo_root / "reports" / "uploads",
            temp_dir=repo_root / "tmp",
            cache_root=Path("/tmp/vostavo-cache"),
            whisper_cache_dir=Path("/tmp/vostavo-cache/whisper"),
        )
        with mock.patch("app_core.bootstrap.build_app_data_paths", return_value=fake_paths), mock.patch(
            "app_core.bootstrap.ensure_app_data_dirs",
            side_effect=lambda paths: paths,
        ), mock.patch.dict(os.environ, {}, clear=True):
            resolved = bootstrap_module.bootstrap_app_environment(app_data_dir=repo_root)
        self.assertEqual(resolved.root, repo_root)

    def test_launcher_payload_reports_repo_local_override_targets(self):
        repo_root = ROOT.resolve()
        args = run_app._parse_args(["--app-data-dir", str(repo_root), "--cache-dir", str(repo_root / ".cache"), "--dry-run"])
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch(
            "scripts.run_app.shutil.which",
            return_value="/opt/homebrew/bin/ffmpeg",
        ), mock.patch(
            "scripts.run_app.get_backend_state",
            return_value={"base_url": "http://127.0.0.1:9000"},
        ):
            payload = run_app._launcher_payload(args)

        self.assertTrue(payload["repo_local_override_active"])
        self.assertEqual(payload["repo_local_override_targets"], ["app_data", "cache"])
        self.assertFalse(payload["packaging_safe"])

    def test_launcher_payload_supports_packaged_launch_mode_override(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ,
            {
                "VOSTAVO_LAUNCH_MODE": "packaged",
                "VOSTAVO_DEPLOYMENT_MODE": "local",
                "VOSTAVO_AUTH_MODE": "guest",
            },
            clear=False,
        ):
            args = run_app._parse_args(["--app-data-dir", app_dir, "--cache-dir", cache_dir, "--dry-run"])
            with mock.patch("scripts.run_app.get_backend_state", return_value={"base_url": "http://127.0.0.1:9000"}):
                payload = run_app._launcher_payload(args)

        self.assertEqual(payload["launch_mode"], "packaged")
        self.assertEqual(payload["deployment_mode"], "local")
        self.assertEqual(payload["auth_mode"], "guest")
        self.assertTrue(payload["packaging_safe"])


if __name__ == "__main__":
    unittest.main()
