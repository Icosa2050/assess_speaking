import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_shell.app_data import AppDataPaths
from app_shell import bootstrap as bootstrap_module
from scripts import run_app

ROOT = Path(__file__).resolve().parents[1]


class RunAppTests(unittest.TestCase):
    def test_launcher_payload_uses_overrides(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
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
        self.assertEqual(payload["command"][-1], str(run_app.STREAMLIT_ENTRYPOINT))

    def test_main_prints_json_for_dry_run(self):
        with tempfile.TemporaryDirectory() as app_dir, mock.patch(
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
        self.assertIn("-m", payload["command"])

    def test_main_launches_streamlit_command(self):
        with mock.patch(
            "scripts.run_app.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ) as mock_run, mock.patch(
            "scripts.run_app.ensure_local_backend",
            return_value={"base_url": "http://127.0.0.1:9000"},
        ):
            exit_code = run_app.main([])

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:3], [command[0], "-m", "streamlit"])

    def test_launcher_payload_forwards_streamlit_args(self):
        args = run_app._parse_args(["--dry-run", "--server.port=9999", "--server.address=127.0.0.1"])
        with mock.patch(
            "scripts.run_app.get_backend_state",
            return_value={"base_url": "http://127.0.0.1:9000"},
        ):
            payload = run_app._launcher_payload(args)

        self.assertEqual(payload["command"][-2:], ["--server.port=9999", "--server.address=127.0.0.1"])

    def test_parse_args_strips_separator_from_streamlit_args(self):
        args = run_app._parse_args(["--dry-run", "--", "--server.port=9999"])

        self.assertEqual(args.streamlit_args, ["--server.port=9999"])

    def test_run_app_dry_run_works_outside_repo_root(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as outside_cwd:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_app.py"),
                    "--app-data-dir",
                    app_dir,
                    "--dry-run",
                    "--server.port=9998",
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
        self.assertEqual(payload["entrypoint"], str((ROOT / "streamlit_app.py").resolve()))
        self.assertEqual(payload["command"][4], str((ROOT / "streamlit_app.py").resolve()))
        self.assertIn("--server.port=9998", payload["command"])
        self.assertIn("backend_base_url", payload)
        self.assertFalse(payload["repo_local_override_active"])

    def test_launcher_payload_sets_legacy_compatibility_envs(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
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
        with mock.patch("app_shell.bootstrap.build_app_data_paths", return_value=fake_paths), mock.patch(
            "app_shell.bootstrap.ensure_app_data_dirs"
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
        with mock.patch("app_shell.bootstrap.build_app_data_paths", return_value=fake_paths), mock.patch(
            "app_shell.bootstrap.ensure_app_data_dirs",
            side_effect=lambda paths: paths,
        ), mock.patch.dict(os.environ, {}, clear=True):
            resolved = bootstrap_module.bootstrap_app_environment(app_data_dir=repo_root)
        self.assertEqual(resolved.root, repo_root)

    def test_launcher_payload_reports_repo_local_override_targets(self):
        repo_root = ROOT.resolve()
        args = run_app._parse_args(["--app-data-dir", str(repo_root), "--cache-dir", str(repo_root / ".cache"), "--dry-run"])
        with mock.patch("scripts.run_app.shutil.which", return_value="/opt/homebrew/bin/ffmpeg"), mock.patch(
            "scripts.run_app.get_backend_state",
            return_value={"base_url": "http://127.0.0.1:9000"},
        ):
            payload = run_app._launcher_payload(args)

        self.assertTrue(payload["repo_local_override_active"])
        self.assertEqual(payload["repo_local_override_targets"], ["app_data", "cache"])


if __name__ == "__main__":
    unittest.main()
