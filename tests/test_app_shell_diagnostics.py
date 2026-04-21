import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_shell.diagnostics import collect_startup_diagnostics
from app_shell.state import AppPreferences, AppShellState, ProviderConnection


def _local_state(*, provider_kind: str = "ollama", base_url: str = "http://localhost:11434/v1") -> AppShellState:
    return AppShellState(
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


class StartupDiagnosticsTests(unittest.TestCase):
    @mock.patch("app_shell.diagnostics.describe_model_availability")
    @mock.patch("app_shell.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch("app_shell.diagnostics.llm_health_check")
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

    @mock.patch("app_shell.diagnostics.describe_model_availability")
    @mock.patch("app_shell.diagnostics.shutil.which", return_value=None)
    def test_collect_startup_diagnostics_reports_missing_ffmpeg(self, _mock_which, mock_model_availability):
        mock_model_availability.return_value = {"cached": False, "cached_path": None}
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AppShellState(prefs=AppPreferences(log_dir=tmpdir))
            diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["ffmpeg"].status, "error")
        self.assertEqual(by_key["whisper"].status, "warning")
        self.assertEqual(by_key["runtime"].status, "warning")

    @mock.patch("app_shell.diagnostics.describe_model_availability")
    @mock.patch("app_shell.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    def test_collect_startup_diagnostics_reports_missing_api_key_for_remote_provider(
        self,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        with mock.patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "", "LLM_API_KEY": ""},
            clear=False,
        ):
            state = AppShellState(
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

    @mock.patch("app_shell.diagnostics.describe_model_availability")
    @mock.patch("app_shell.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch("app_shell.diagnostics.llm_health_check", side_effect=RuntimeError("connection refused"))
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

    @mock.patch("app_shell.diagnostics.describe_model_availability")
    @mock.patch("app_shell.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    @mock.patch("app_shell.diagnostics.tempfile.NamedTemporaryFile", side_effect=OSError("permission denied"))
    def test_collect_startup_diagnostics_reports_unwritable_app_data_dir(
        self,
        _mock_tempfile,
        _mock_which,
        mock_model_availability,
    ):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AppShellState(prefs=AppPreferences(log_dir=tmpdir))
            diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["app_data"].status, "error")

    @mock.patch("app_shell.diagnostics.describe_model_availability")
    @mock.patch("app_shell.diagnostics.shutil.which", return_value="/opt/homebrew/bin/ffmpeg")
    def test_collect_startup_diagnostics_warns_for_repo_local_app_data_override(self, _mock_which, mock_model_availability):
        mock_model_availability.return_value = {"cached": True, "cached_path": "/tmp/whisper-cache"}
        repo_root = Path(__file__).resolve().parents[1]

        fake_paths = mock.Mock(
            root=repo_root,
            reports_dir=repo_root / "reports",
            temp_dir=repo_root / "reports" / "tmp",
            whisper_cache_dir=repo_root / "cache" / "whisper",
        )
        fake_paths.temp_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch("app_shell.diagnostics.bootstrap_app_environment", return_value=fake_paths):
            state = AppShellState(prefs=AppPreferences(log_dir=str(repo_root / "reports")))
            diagnostics = collect_startup_diagnostics(state, include_runtime_health=False)

        by_key = {item.key: item for item in diagnostics}
        self.assertEqual(by_key["app_data_location"].status, "warning")


if __name__ == "__main__":
    unittest.main()
