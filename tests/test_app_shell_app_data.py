import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_shell import app_data
from app_shell.app_data import (
    APP_AUTHOR,
    APP_CACHE_HOME_ENV_VAR,
    APP_DATA_HOME_ENV_VAR,
    LEGACY_APP_CACHE_HOME_ENV_VAR,
    LEGACY_APP_DATA_HOME_ENV_VAR,
    ROOT_MARKER_FILENAME,
    build_app_data_paths,
    ensure_app_data_dirs,
    resolve_app_data_root,
    resolve_cache_root,
    resolve_reports_dir,
)


class AppDataTests(unittest.TestCase):
    def test_resolve_app_data_root_uses_environment_override(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: tmpdir},
            clear=False,
        ):
            self.assertEqual(resolve_app_data_root(), Path(tmpdir).resolve())

    def test_resolve_reports_dir_keeps_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            absolute_reports = Path(tmpdir) / "reports"
            self.assertEqual(resolve_reports_dir(absolute_reports), absolute_reports.resolve())

    def test_resolve_reports_dir_scopes_relative_path_to_app_home(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: tmpdir},
            clear=False,
        ):
            resolved = resolve_reports_dir("reports")
            self.assertEqual(resolved, Path(tmpdir).resolve() / "reports")

    def test_build_app_data_paths_uses_custom_cache_root(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ,
            {
                APP_DATA_HOME_ENV_VAR: home_dir,
                APP_CACHE_HOME_ENV_VAR: cache_dir,
            },
            clear=False,
        ):
            paths = build_app_data_paths()

        self.assertEqual(paths.root, Path(home_dir).resolve())
        self.assertEqual(paths.cache_root, Path(cache_dir).resolve())
        self.assertEqual(paths.jobs_dir, Path(home_dir).resolve() / "jobs")
        self.assertEqual(paths.logs_dir, Path(home_dir).resolve() / "logs")
        self.assertEqual(paths.whisper_cache_dir, Path(cache_dir).resolve() / "whisper")

    def test_ensure_app_data_dirs_creates_expected_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: tmpdir},
            clear=False,
        ):
            paths = ensure_app_data_dirs(build_app_data_paths())
            self.assertTrue(paths.root.is_dir())
            self.assertTrue(paths.reports_dir.is_dir())
            self.assertTrue(paths.jobs_dir.is_dir())
            self.assertTrue(paths.logs_dir.is_dir())
            self.assertTrue(paths.recordings_dir.is_dir())
            self.assertTrue(paths.uploads_dir.is_dir())
            self.assertTrue(paths.temp_dir.is_dir())
            self.assertTrue(paths.cache_root.is_dir())
            self.assertTrue(paths.whisper_cache_dir.is_dir())

    def test_resolve_cache_root_uses_environment_override(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {APP_CACHE_HOME_ENV_VAR: tmpdir},
            clear=False,
        ):
            self.assertEqual(resolve_cache_root(), Path(tmpdir).resolve())

    def test_resolve_app_data_root_falls_back_to_legacy_environment_override(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {LEGACY_APP_DATA_HOME_ENV_VAR: tmpdir},
            clear=True,
        ):
            self.assertEqual(resolve_app_data_root(), Path(tmpdir).resolve())

    def test_resolve_cache_root_uses_custom_app_data_home_when_cache_home_is_absent(self):
        with tempfile.TemporaryDirectory() as data_home, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: data_home},
            clear=True,
        ):
            self.assertEqual(resolve_cache_root(), Path(data_home).resolve() / "cache")

    def test_legacy_windows_cache_home_does_not_append_cache_segment(self):
        with mock.patch("app_shell.app_data.os.name", "nt"), mock.patch.dict(
            os.environ,
            {"LOCALAPPDATA": "C:/Users/test/AppData/Local"},
            clear=True,
        ):
            self.assertEqual(
                app_data._legacy_platform_cache_home(),
                Path("C:/Users/test/AppData/Local"),
            )

    def test_resolve_app_data_root_prefers_initialized_legacy_root_over_empty_new_root(self):
        with tempfile.TemporaryDirectory() as base_dir:
            base_path = Path(base_dir)
            legacy_root = base_path / "Speaking Studio"
            legacy_root.mkdir()
            (legacy_root / "reports").mkdir()
            new_root = base_path / "Vostavo"
            new_root.mkdir()
            with mock.patch("app_shell.app_data._default_app_data_root", return_value=new_root.resolve()), mock.patch(
                "app_shell.app_data._legacy_default_app_data_root",
                return_value=legacy_root.resolve(),
            ), mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_app_data_root(), legacy_root.resolve())

    def test_temp_dir_is_root_scoped_not_reports_scoped(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: tmpdir},
            clear=False,
        ):
            paths = build_app_data_paths()
        self.assertEqual(paths.temp_dir, Path(tmpdir).resolve() / "tmp")

    def test_ensure_app_data_dirs_writes_root_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {APP_DATA_HOME_ENV_VAR: tmpdir},
            clear=False,
        ):
            paths = ensure_app_data_dirs(build_app_data_paths())
            self.assertTrue((paths.root / ROOT_MARKER_FILENAME).exists())

    def test_default_app_data_root_uses_platformdirs_identity(self):
        with mock.patch("app_shell.app_data._platform_dirs") as mock_dirs, mock.patch(
            "app_shell.app_data._legacy_default_app_data_root",
            return_value=Path("/tmp/legacy-speaking-studio").resolve(),
        ), mock.patch.dict(os.environ, {}, clear=True):
            mock_dirs.return_value.user_data_path = Path("/tmp/frommherz_it/Vostavo")
            self.assertEqual(resolve_app_data_root(), Path("/tmp/frommherz_it/Vostavo").resolve())
            self.assertEqual(APP_AUTHOR, "frommherz_it")

    def test_linux_default_roots_follow_xdg_style_locations(self):
        with mock.patch("app_shell.app_data._platform_dirs") as mock_dirs, mock.patch(
            "app_shell.app_data._legacy_default_app_data_root",
            return_value=Path("/tmp/legacy-speaking-studio").resolve(),
        ), mock.patch(
            "app_shell.app_data._legacy_default_cache_root",
            return_value=Path("/tmp/legacy-speaking-studio-cache").resolve(),
        ), mock.patch.dict(os.environ, {}, clear=True):
            mock_dirs.return_value.user_data_path = Path("/home/tester/.local/share/Vostavo")
            mock_dirs.return_value.user_cache_path = Path("/home/tester/.cache/Vostavo")

            self.assertEqual(resolve_app_data_root(), Path("/home/tester/.local/share/Vostavo").resolve())
            self.assertEqual(resolve_cache_root(), Path("/home/tester/.cache/Vostavo").resolve())


if __name__ == "__main__":
    unittest.main()
