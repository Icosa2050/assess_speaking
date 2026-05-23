import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_core.app_data import AppDataPaths
from app_core.diagnostics import collect_startup_diagnostics
from app_core.runtime_resolver import RuntimeConfig
from app_core.state import AppPreferences, AppState, ProviderConnection


def _local_state(*, provider_kind: str = "ollama", base_url: str = "http://localhost:11434/v1") -> AppState:
    return AppState(
        prefs=AppPreferences(
            provider=provider_kind,
            model="llama3" if provider_kind == "ollama" else "model-1",
            llm_base_url=base_url,
            connections=[
                ProviderConnection(
                    connection_id="primary",
                    provider_kind=provider_kind,
                    label="Primary",
                    base_url=base_url,
                    default_model="llama3" if provider_kind == "ollama" else "model-1",
                    is_default=True,
                    is_local=provider_kind in {"ollama", "lmstudio"},
                    provider_metadata={"deployment": "local"} if provider_kind == "ollama" else {},
                )
            ],
            active_connection_id="primary",
            setup_complete=True,
        )
    )


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


class StartupDiagnosticsTests(unittest.TestCase):
    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch("app_core.diagnostics.llm_health_check")
    def test_collect_startup_diagnostics_reports_local_runtime_health(
        self,
        mock_health_check,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        mock_health_check.return_value = {"endpoint": "http://localhost:11434/api/tags"}

        with tempfile.TemporaryDirectory() as tmpdir:
            state = _local_state()
            state.prefs.log_dir = tmpdir
            diagnostics = collect_startup_diagnostics(state)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["app_data"].status, "ok")
        self.assertEqual(by_key["ffmpeg"].status, "ok")
        self.assertEqual(by_key["whisper"].status, "ok")
        self.assertEqual(by_key["runtime"].status, "ok")
        self.assertEqual(by_key["runtime_local_health"].status, "ok")
        self.assertEqual(by_key["microphone"].status, "info")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value=None)
    def test_collect_startup_diagnostics_reports_missing_ffmpeg(self, _mock_which, mock_model_availability):
        mock_model_availability.return_value = {"cached": False, "cached_path": None}
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AppState(prefs=AppPreferences(log_dir=tmpdir))
            diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["ffmpeg"].status, "error")
        self.assertEqual(by_key["whisper"].status, "warning")
        self.assertEqual(by_key["runtime"].status, "warning")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch(
        "app_core.diagnostics.resolve_connection_runtime",
        return_value=RuntimeConfig(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key="",
        ),
    )
    @mock.patch(
        "app_core.diagnostics.build_client_snapshot",
        return_value={
            "has_active_connection": True,
            "provider_requires_auth": True,
            "has_saved_secret": False,
            "credentials_missing": True,
            "secure_storage_persistent": True,
            "credential_state": "missing",
        },
    )
    def test_collect_startup_diagnostics_reports_missing_saved_credentials_for_remote_provider(
        self,
        _mock_client_snapshot,
        _mock_runtime,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        state = AppState(
            prefs=AppPreferences(
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                llm_base_url="https://openrouter.ai/api/v1",
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="openrouter",
                        label="Primary",
                        base_url="https://openrouter.ai/api/v1",
                        default_model="google/gemini-3.1-pro-preview",
                        is_default=True,
                        auth_mode="bearer",
                    )
                ],
                active_connection_id="primary",
                setup_complete=True,
            )
        )
        diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["runtime"].status, "ok")
        self.assertEqual(by_key["runtime_api_key"].status, "error")
        self.assertEqual(by_key["runtime_api_key"].detail_args["credential_state"], "missing")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch("app_core.diagnostics.llm_health_check", side_effect=RuntimeError("connection refused"))
    def test_collect_startup_diagnostics_reports_unreachable_local_provider(
        self,
        _mock_health_check,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _local_state()
            state.prefs.log_dir = tmpdir
            diagnostics = collect_startup_diagnostics(state)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["runtime_local_health"].status, "error")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch("app_core.diagnostics.tempfile.NamedTemporaryFile", side_effect=OSError("permission denied"))
    def test_collect_startup_diagnostics_reports_unwritable_app_data_dir(
        self,
        _mock_tempfile,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _build_paths(Path(tmpdir).resolve())
            for directory in (
                paths.root,
                paths.reports_dir,
                paths.jobs_dir,
                paths.logs_dir,
                paths.temp_dir,
                paths.cache_root,
                paths.whisper_cache_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            state = AppState(prefs=AppPreferences(log_dir=str(paths.reports_dir)))
            with mock.patch("app_core.diagnostics.bootstrap_app_environment", return_value=paths):
                diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["app_data"].status, "error")
        self.assertEqual(by_key["app_data"].detail_args["path"], str(paths.temp_dir))

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    def test_collect_startup_diagnostics_warns_for_repo_local_app_data_override(self, _mock_which, mock_model_availability):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir).resolve()
            fake_paths = AppDataPaths(
                root=repo_root,
                reports_dir=temp_root / "reports",
                jobs_dir=temp_root / "jobs",
                logs_dir=temp_root / "logs",
                recordings_dir=temp_root / "reports" / "recordings",
                uploads_dir=temp_root / "reports" / "uploads",
                temp_dir=temp_root / "tmp",
                cache_root=temp_root / "cache",
                whisper_cache_dir=temp_root / "cache" / "whisper",
            )
            for directory in (
                fake_paths.reports_dir,
                fake_paths.jobs_dir,
                fake_paths.logs_dir,
                fake_paths.temp_dir,
                fake_paths.cache_root,
                fake_paths.whisper_cache_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            with mock.patch("app_core.diagnostics.bootstrap_app_environment", return_value=fake_paths):
                state = AppState(prefs=AppPreferences(log_dir=str(fake_paths.reports_dir)))
                diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["app_data_location"].status, "warning")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch(
        "app_core.diagnostics.resolve_connection_runtime",
        return_value=RuntimeConfig(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key="saved-key",
        ),
    )
    @mock.patch(
        "app_core.diagnostics.build_client_snapshot",
        return_value={
            "has_active_connection": True,
            "provider_requires_auth": True,
            "has_saved_secret": True,
            "credentials_missing": False,
            "secure_storage_persistent": True,
            "credential_state": "saved",
        },
    )
    def test_collect_startup_diagnostics_reports_saved_credentials_state(
        self,
        _mock_client_snapshot,
        _mock_runtime,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        state = AppState(
            prefs=AppPreferences(
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                llm_base_url="https://openrouter.ai/api/v1",
                connections=[
                    ProviderConnection(
                        connection_id="primary",
                        provider_kind="openrouter",
                        label="Primary",
                        base_url="https://openrouter.ai/api/v1",
                        default_model="google/gemini-3.1-pro-preview",
                        is_default=True,
                        auth_mode="bearer",
                    )
                ],
                active_connection_id="primary",
                setup_complete=True,
            )
        )

        diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["runtime_api_key"].status, "ok")
        self.assertEqual(by_key["runtime_api_key"].detail_args["credential_state"], "saved")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    def test_collect_startup_diagnostics_adds_maintenance_warnings(self, _mock_which, mock_model_availability):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _build_paths(Path(tmpdir).resolve())
            for directory in (
                paths.root,
                paths.reports_dir,
                paths.jobs_dir,
                paths.logs_dir,
                paths.temp_dir,
                paths.cache_root,
                paths.whisper_cache_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            stale_tmp = paths.temp_dir / "stale.tmp"
            stale_tmp.write_text("temp", encoding="utf-8")
            old_time = (Path(stale_tmp).stat().st_mtime - (48 * 60 * 60))
            os.utime(stale_tmp, (old_time, old_time))

            (paths.jobs_dir / "stale.json").write_text(
                '{"status":"completed","completed_at":"2026-03-01T00:00:00+00:00"}',
                encoding="utf-8",
            )
            (paths.logs_dir / "backend.log").write_text("active", encoding="utf-8")
            (paths.logs_dir / "unexpected.log").write_text("overflow", encoding="utf-8")

            with mock.patch("app_core.diagnostics.bootstrap_app_environment", return_value=paths):
                diagnostics = collect_startup_diagnostics(
                    AppState(prefs=AppPreferences(log_dir=str(paths.reports_dir))),
                    include_runtime_health=False,
                )

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["app_data"].detail_args["path"], str(paths.temp_dir))
        self.assertEqual(by_key["maintenance_tmp"].status, "warning")
        self.assertEqual(by_key["maintenance_jobs"].status, "warning")
        self.assertEqual(by_key["maintenance_logs"].status, "warning")
        self.assertEqual(by_key["maintenance_tmp"].detail_args["target_page"], "/settings")
        self.assertEqual(by_key["maintenance_jobs"].detail_args["target_page"], "/settings")
        self.assertEqual(by_key["maintenance_logs"].detail_args["target_page"], "/settings")

    @mock.patch("app_core.diagnostics.describe_model_availability")
    @mock.patch("app_core.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    def test_collect_startup_diagnostics_skips_maintenance_warnings_when_clean(self, _mock_which, mock_model_availability):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _build_paths(Path(tmpdir).resolve())
            for directory in (
                paths.root,
                paths.reports_dir,
                paths.jobs_dir,
                paths.logs_dir,
                paths.temp_dir,
                paths.cache_root,
                paths.whisper_cache_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            support_dir = paths.temp_dir / "support-bundles"
            support_dir.mkdir(parents=True, exist_ok=True)
            (support_dir / "fresh.zip").write_text("bundle", encoding="utf-8")
            (paths.logs_dir / "backend.log").write_text("active", encoding="utf-8")

            with mock.patch("app_core.diagnostics.bootstrap_app_environment", return_value=paths):
                diagnostics = collect_startup_diagnostics(
                    AppState(prefs=AppPreferences(log_dir=str(paths.reports_dir))),
                    include_runtime_health=False,
                )

        by_key = {item.key: item for item in diagnostics}
        self.assertNotIn("maintenance_tmp", by_key)
        self.assertNotIn("maintenance_jobs", by_key)
        self.assertNotIn("maintenance_logs", by_key)


if __name__ == "__main__":
    unittest.main()
