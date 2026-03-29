import csv
import json
import re
import time
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


def _wait_for_saved_runs(
    reports_dir: Path,
    *,
    speaker_id: str,
    expected_labels: tuple[str, ...],
    timeout: float = 90.0,
) -> dict[str, dict[str, str]]:
    history_path = reports_dir / "history.csv"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            matching = {
                str(row.get("label") or ""): row
                for row in rows
                if row.get("speaker_id") == speaker_id and str(row.get("label") or "") in expected_labels
            }
            if all(label in matching for label in expected_labels):
                return matching
        time.sleep(0.5)
    raise AssertionError(
        f"history.csv did not contain all labels {expected_labels!r} for speaker {speaker_id!r} within {timeout:.0f}s"
    )


def _start_app_shell_session(page, app_shell_server: str, app_shell_text, ui_locale: str, *, speaker_id: str) -> None:
    page.set_viewport_size({"width": 1440, "height": 960})
    page.goto(f"{app_shell_server}/", wait_until="domcontentloaded")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "home.title"))).to_be_visible(timeout=30000)
    page.get_by_role("button", name=app_shell_text(ui_locale, "home.start_new")).click()

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "setup.title"))).to_be_visible(timeout=30000)
    page.get_by_label(app_shell_text(ui_locale, "setup.speaker_id")).fill(speaker_id)
    page.get_by_role("button", name=app_shell_text(ui_locale, "setup.continue")).click()
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "speak.title"))).to_be_visible(timeout=30000)


def _set_upload_mode(page, app_shell_text, ui_locale: str) -> None:
    page.locator('label[data-baseweb="radio"]').filter(
        has_text=re.compile(re.escape(app_shell_text(ui_locale, "speak.input_method_upload")))
    ).click()


def _attach_audio_for_review(page, app_shell_text, ui_locale: str, *, sample_path: Path) -> None:
    _set_upload_mode(page, app_shell_text, ui_locale)
    page.locator('input[type="file"]').set_input_files(str(sample_path))
    expect(page.get_by_text(re.compile(re.escape(app_shell_text(ui_locale, "speak.status_ready"))))).to_be_visible(timeout=30000)


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
def test_app_shell_complete_journey_reaches_history_in_english_and_italian(
    page,
    app_shell_server,
    app_shell_text,
    project_root: Path,
    reports_dir: Path,
    samples_dir: Path,
    ui_locale: str,
):
    _seed_app_shell_runtime(project_root, ui_locale=ui_locale)
    speaker_id = f"playwright-shell-history-{ui_locale}"
    first_label = f"{ui_locale}-history-first"
    second_label = f"{ui_locale}-history-second"
    first_notes = f"First {ui_locale} review pass."
    second_notes = f"Second {ui_locale} review pass."

    _start_app_shell_session(page, app_shell_server, app_shell_text, ui_locale, speaker_id=speaker_id)
    _attach_audio_for_review(page, app_shell_text, ui_locale, sample_path=samples_dir / "demo.m4a")
    page.get_by_role("textbox", name=app_shell_text(ui_locale, "speak.label")).fill(first_label)
    page.get_by_role("textbox", name=app_shell_text(ui_locale, "speak.notes")).fill(first_notes)
    page.get_by_role("button", name=app_shell_text(ui_locale, "speak.submit")).click()

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "review.title"))).to_be_visible(timeout=120000)
    expect(page.get_by_role("button", name=app_shell_text(ui_locale, "review.try_again"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("button", name=app_shell_text(ui_locale, "review.view_history"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("textbox", name=app_shell_text(ui_locale, "review.notes_title"))).to_have_value(first_notes)
    first_transcript = page.get_by_role("textbox", name=app_shell_text(ui_locale, "review.transcript_title")).input_value().strip()
    assert first_transcript

    saved_runs = _wait_for_saved_runs(
        reports_dir,
        speaker_id=speaker_id,
        expected_labels=(first_label,),
    )
    assert Path(saved_runs[first_label]["report_path"]).exists()

    page.get_by_role("button", name=app_shell_text(ui_locale, "review.try_again")).click()
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "speak.title"))).to_be_visible(timeout=30000)

    _attach_audio_for_review(page, app_shell_text, ui_locale, sample_path=samples_dir / "demo.m4a")
    page.get_by_role("textbox", name=app_shell_text(ui_locale, "speak.label")).fill(second_label)
    page.get_by_role("textbox", name=app_shell_text(ui_locale, "speak.notes")).fill(second_notes)
    page.get_by_role("button", name=app_shell_text(ui_locale, "speak.submit")).click()

    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "review.title"))).to_be_visible(timeout=120000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "review.progress_title"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("textbox", name=app_shell_text(ui_locale, "review.notes_title"))).to_have_value(second_notes)

    saved_runs = _wait_for_saved_runs(
        reports_dir,
        speaker_id=speaker_id,
        expected_labels=(first_label, second_label),
    )
    assert Path(saved_runs[second_label]["report_path"]).exists()

    page.get_by_role("button", name=app_shell_text(ui_locale, "review.view_history")).click()
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "history.title"))).to_be_visible(timeout=30000)
    scope_pattern = re.compile(
        "|".join(
            re.escape(text)
            for text in (
                app_shell_text(ui_locale, "history.scope_current_speaker").format(count=2, speaker=speaker_id),
                app_shell_text(ui_locale, "history.scope_current_speaker_language").format(
                    count=2,
                    speaker=speaker_id,
                    language="IT",
                ),
            )
        )
    )
    expect(page.get_by_text(scope_pattern)).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "history.attempts_title"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=app_shell_text(ui_locale, "history.details_title"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("combobox", name=app_shell_text(ui_locale, "history.details_select"))).to_be_visible(timeout=30000)
    expect(page.get_by_role("textbox", name=app_shell_text(ui_locale, "review.notes_title"))).to_have_value(second_notes)
    history_transcript = page.get_by_role("textbox", name=app_shell_text(ui_locale, "review.transcript_title")).input_value().strip()
    assert history_transcript


@pytest.mark.parametrize("ui_locale", APP_SHELL_E2E_LOCALES)
def test_app_shell_remove_recording_returns_to_idle_in_english_and_italian(
    page,
    app_shell_server,
    app_shell_text,
    project_root: Path,
    samples_dir: Path,
    ui_locale: str,
):
    _seed_app_shell_runtime(project_root, ui_locale=ui_locale)
    speaker_id = f"playwright-shell-remove-{ui_locale}"

    _start_app_shell_session(page, app_shell_server, app_shell_text, ui_locale, speaker_id=speaker_id)
    _attach_audio_for_review(page, app_shell_text, ui_locale, sample_path=samples_dir / "demo.m4a")

    remove_button = page.get_by_role("button", name=app_shell_text(ui_locale, "speak.remove_recording"))
    expect(remove_button).to_be_visible(timeout=30000)
    remove_button.click()

    expect(page.get_by_text(re.compile(re.escape(app_shell_text(ui_locale, "speak.status_idle"))))).to_be_visible(timeout=30000)
    expect(page.get_by_role("button", name=app_shell_text(ui_locale, "speak.submit"))).to_be_disabled(timeout=30000)


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
