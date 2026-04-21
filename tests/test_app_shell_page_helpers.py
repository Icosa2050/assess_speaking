import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_shell.page_helpers as page_helpers
from app_shell.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR
from app_shell.page_helpers import (
    BRAND_ICON_PATH,
    BRAND_LOGO_PATH,
    describe_whisper_download_event,
    format_byte_count,
    render_brand_logo,
    render_brand_mark,
    resolve_page_title_locale,
)
from app_shell.state import AppShellState
from streamlit.errors import StreamlitAPIException


class PageHelpersTests(unittest.TestCase):
    def test_brand_asset_paths_exist(self):
        self.assertTrue(BRAND_ICON_PATH.exists())
        self.assertTrue(BRAND_LOGO_PATH.exists())

    def test_render_brand_mark_uses_sidebar_image(self):
        with patch.object(page_helpers.st.sidebar, "image") as sidebar_image:
            render_brand_mark(width=72)
        sidebar_image.assert_called_once_with(str(BRAND_ICON_PATH), width=72)

    def test_render_brand_logo_uses_streamlit_image(self):
        with patch.object(page_helpers.st, "image") as image:
            render_brand_logo(width=320)
        image.assert_called_once_with(str(BRAND_LOGO_PATH), width=320)

    def test_format_byte_count_uses_human_readable_units(self):
        self.assertEqual(format_byte_count(512), "512 B")
        self.assertEqual(format_byte_count(1536), "1.5 KB")
        self.assertEqual(format_byte_count(5 * 1024 * 1024), "5.0 MB")

    def test_describe_whisper_download_event_reports_progress(self):
        status = describe_whisper_download_event(
            {
                "stage": "downloading",
                "current_file": "/tmp/cache/model.bin",
                "downloaded_bytes": 512,
                "total_bytes": 1024,
                "completed_files": 1,
                "total_files": 4,
            }
        )

        self.assertEqual(status["headline"], "Downloading model.bin...")
        self.assertEqual(status["detail"], "512 B / 1.0 KB · 1/4 files ready")
        self.assertEqual(status["progress_percent"], 50)

    def test_resolve_page_title_locale_reads_local_prefs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_dir.joinpath("workspace_prefs.json").write_text(
                json.dumps({"ui_locale": "de"}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_page_title_locale(log_dir), "de")

    def test_resolve_page_title_locale_bootstraps_app_environment(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
            with patch.dict(
                os.environ,
                {
                    APP_DATA_HOME_ENV_VAR: data_dir,
                    APP_CACHE_HOME_ENV_VAR: cache_dir,
                    "HF_HOME": "",
                    "WHISPER_CACHE_DIR": "",
                    "XDG_CACHE_HOME": "",
                },
                clear=False,
            ):
                resolve_page_title_locale()
                self.assertEqual(os.environ.get("HF_HOME"), str(Path(cache_dir).resolve() / "huggingface"))
                self.assertEqual(os.environ.get("WHISPER_CACHE_DIR"), str(Path(cache_dir).resolve() / "whisper"))

    def test_resolve_page_title_locale_follows_nested_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir) / "root"
            nested_dir = Path(tmpdir) / "nested"
            root_dir.mkdir()
            nested_dir.mkdir()
            root_dir.joinpath("workspace_prefs.json").write_text(
                json.dumps({"ui_locale": "en", "log_dir": str(nested_dir)}),
                encoding="utf-8",
            )
            nested_dir.joinpath("workspace_prefs.json").write_text(
                json.dumps({"ui_locale": "it"}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_page_title_locale(root_dir), "it")

    def test_resolve_page_title_locale_falls_back_for_invalid_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_dir.joinpath("workspace_prefs.json").write_text(
                json.dumps({"ui_locale": "fr"}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_page_title_locale(log_dir), "en")

    def test_go_to_sets_next_page_when_bootstrap_is_skipped(self):
        state = AppShellState()
        state.nav.current_page = "setup"
        session_state: dict[str, str] = {}
        with patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": "1"}, clear=False):
            with patch.object(page_helpers, "get_app_state", return_value=state), \
                    patch.object(page_helpers, "set_return_to") as set_return_to, \
                    patch.object(page_helpers.st, "session_state", session_state), \
                    patch.object(page_helpers.st, "stop", side_effect=RuntimeError("stop")):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    page_helpers.go_to("pages/02_Speak.py")
        self.assertEqual(session_state["_next_page"], "pages/02_Speak.py")
        set_return_to.assert_called_once_with("setup")

    def test_go_to_retries_with_page_alias_after_switch_page_error(self):
        state = AppShellState()
        switch_error = StreamlitAPIException("missing page")
        with patch.dict(os.environ, {"APP_SHELL_SKIP_BOOTSTRAP": ""}, clear=False):
            with patch.object(page_helpers, "get_app_state", return_value=state), \
                    patch.object(page_helpers, "set_return_to") as set_return_to, \
                    patch.object(page_helpers.st, "switch_page", side_effect=[switch_error, None]) as switch_page:
                page_helpers.go_to("pages/03_Review.py", return_to="home")
        self.assertEqual(
            [call.args[0] for call in switch_page.call_args_list],
            ["pages/03_Review.py", "03_Review.py"],
        )
        set_return_to.assert_called_once_with("home")

    def test_render_guard_stops_without_navigation_when_button_not_clicked(self):
        with patch.object(page_helpers.st, "warning") as warning, \
                patch.object(page_helpers.st, "button", return_value=False), \
                patch.object(page_helpers, "go_to") as go_to, \
                patch.object(page_helpers.st, "stop", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                page_helpers.render_guard("speak.guard_missing_setup", "speak.go_setup", "pages/01_Session_Setup.py")
        warning.assert_called_once()
        go_to.assert_not_called()

    def test_render_guard_navigates_when_button_is_clicked(self):
        with patch.object(page_helpers.st, "warning"), \
                patch.object(page_helpers.st, "button", return_value=True), \
                patch.object(page_helpers, "go_to") as go_to, \
                patch.object(page_helpers.st, "stop", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                page_helpers.render_guard("review.guard_missing_review", "review.go_speak", "pages/02_Speak.py")
        go_to.assert_called_once_with("pages/02_Speak.py")


if __name__ == "__main__":
    unittest.main()
