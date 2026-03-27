import json
import re
from pathlib import Path

import pytest
from playwright.sync_api import expect


APP_SHELL_E2E_LOCALES = ("en", "it")


def _seed_app_shell_runtime(project_root: Path, *, ui_locale: str) -> None:
    # The app shell now enforces a connection-first runtime flow. This fixture
    # seeds a default local connection so the happy-path E2E can exercise
    # Session Setup -> Speak -> Review instead of being redirected to Runtime Setup.
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dashboard_prefs.json").write_text(
        json.dumps(
            {
                "ui_locale": ui_locale,
                "log_dir": str(reports_dir.resolve()),
                "active_connection_id": "e2e-ollama",
                "setup_complete": True,
                "connections": [
                    {
                        "connection_id": "e2e-ollama",
                        "provider_kind": "ollama",
                        "label": "E2E Ollama",
                        "base_url": "http://localhost:11434/v1",
                        "default_model": "llama3",
                        "auth_mode": "none",
                        "secret_ref": "",
                        "is_default": True,
                        "is_local": True,
                        "provider_metadata": {"deployment": "local"},
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("ui_locale", APP_SHELL_E2E_LOCALES)
def test_app_shell_upload_reaches_review(page, app_shell_server, app_shell_text, project_root: Path, samples_dir: Path, ui_locale: str):
    _seed_app_shell_runtime(project_root, ui_locale=ui_locale)
    page.set_viewport_size({"width": 1440, "height": 960})
    page.goto(f"{app_shell_server}/", wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "home.title"))).to_be_visible(timeout=30000)
    start_button = page.get_by_role("button", name=app_shell_text(ui_locale, "home.start_new"))
    expect(start_button).to_be_visible(timeout=30000)
    start_button.click()

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "setup.title"))).to_be_visible(timeout=30000)
    page.get_by_label(app_shell_text(ui_locale, "setup.speaker_id")).fill(f"playwright-shell-user-{ui_locale}")
    page.get_by_role("button", name=app_shell_text(ui_locale, "setup.continue")).click()

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "speak.title"))).to_be_visible(timeout=30000)
    page.locator('label[data-baseweb="radio"]').filter(
        has_text=re.compile(re.escape(app_shell_text(ui_locale, "speak.input_method_upload")))
    ).click()
    page.locator('input[type="file"]').set_input_files(str(samples_dir / "demo.m4a"))
    expect(page.get_by_text(re.compile(re.escape(app_shell_text(ui_locale, "speak.status_ready"))))).to_be_visible(timeout=30000)

    page.get_by_role("button", name=app_shell_text(ui_locale, "speak.submit")).click()
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "review.title"))).to_be_visible(timeout=120000)
    expect(page.get_by_text(re.compile(rf"^{re.escape(app_shell_text(ui_locale, 'review.score'))}$"))).to_be_visible(timeout=30000)


@pytest.mark.parametrize("ui_locale", APP_SHELL_E2E_LOCALES)
def test_app_shell_runtime_setup_screen_is_localized(page, app_shell_server, app_shell_locale, app_shell_text, ui_locale: str):
    app_shell_locale(ui_locale)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"{app_shell_server}/", wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")

    runtime_setup_button = page.get_by_role("button", name=app_shell_text(ui_locale, "home.runtime_setup_button"))
    expect(runtime_setup_button).to_be_visible(timeout=30000)
    runtime_setup_button.click()

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "runtime_setup.title"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "runtime_setup.section_whisper"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "runtime_setup.section_provider"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "runtime_setup.section_connection"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "runtime_setup.section_actions"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("button", name=app_shell_text(ui_locale, "runtime_setup.back_home"))).to_be_visible(timeout=30000)
