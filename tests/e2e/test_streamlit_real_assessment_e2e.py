import csv
import os
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

DEFAULT_REAL_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio" / "test1.m4a"


def localized_pattern(*labels: str, exact: bool = True):
    joined = "|".join(re.escape(label) for label in labels)
    if exact:
        return re.compile(rf"^({joined})$")
    return re.compile(joined)


def _require_real_audio_path() -> Path:
    if os.getenv("RUN_STREAMLIT_REAL_E2E") != "1":
        pytest.skip("Set RUN_STREAMLIT_REAL_E2E=1 to run the real Streamlit E2E assessment.")
    audio_path = os.getenv("ASSESS_SPEAKING_REAL_AUDIO_PATH")
    path = Path(audio_path).expanduser() if audio_path else DEFAULT_REAL_AUDIO_PATH
    if not path.exists():
        pytest.skip(f"Real audio file not found: {path}")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for the real Streamlit E2E assessment.")
    return path


def wait_for_labeled_history(reports_dir: Path, label: str, timeout: float = 180.0) -> dict:
    history_path = reports_dir / "history.csv"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            for row in reversed(rows):
                if row.get("label") == label:
                    return row
        time.sleep(1.0)
    raise AssertionError(f"history.csv did not contain label {label!r} within timeout")


def activate_manual_upload_mode(page) -> None:
    page.get_by_text("Stattdessen eine vorhandene Aufnahme nutzen", exact=True).click()
    page.get_by_role("button", name="Alternative aktivieren").click()
    expect(page.get_by_text("Du nutzt gerade eine vorhandene Aufnahme", exact=False)).to_be_visible(timeout=10000)
    expect(page.locator('[aria-label="Audio-Datei hinzufügen"] input[type="file"]')).to_be_attached(timeout=10000)


def test_real_upload_returns_feedback(page, reports_dir, streamlit_real_server):
    audio_path = _require_real_audio_path()
    run_label = "playwright-real-audio"
    page.goto(f"{streamlit_real_server}/")
    page.wait_for_load_state("networkidle")

    page.get_by_label("Speaker ID").fill("playwright-real-user")
    page.get_by_role("textbox", name=localized_pattern("Thema", "Theme")).fill("tema libero")
    page.get_by_role(
        "spinbutton",
        name=localized_pattern("Zielsprechdauer (Sekunden)", "Target speaking duration (seconds)"),
    ).fill("60")
    activate_manual_upload_mode(page)
    page.locator('[aria-label="Audio-Datei hinzufügen"] input[type="file"]').set_input_files(str(audio_path))
    page.get_by_label("Label").fill(run_label)

    page.get_by_text("Technik und Notizen", exact=True).click()
    page.get_by_role("textbox", name="Whisper-Modell (lokal)", exact=True).fill(
        os.getenv("ASSESS_SPEAKING_REAL_WHISPER_MODEL", "tiny")
    )

    run_button = page.get_by_role("button", name="Datei auswerten")
    expect(run_button).to_be_enabled(timeout=15000)
    run_button.click()

    expect(
        page.get_by_text(localized_pattern("Deine Rückmeldung", "Your feedback", exact=False))
    ).to_be_visible(timeout=180000)
    expect(
        page.get_by_text(
            localized_pattern(
                "Darauf solltest du als Nächstes achten",
                "What to focus on next",
                exact=False,
            )
        )
    ).to_be_visible(timeout=180000)

    row = wait_for_labeled_history(reports_dir, run_label, timeout=180.0)
    assert row["speaker_id"] == "playwright-real-user"
    assert row["task_family"] == "travel_narrative"
    assert row["theme"] == "tema libero"
    assert row["report_path"]
