import csv
import os
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

DEFAULT_WEAKER_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio" / "test1.m4a"
DEFAULT_BETTER_AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio" / "test2.m4a"


def localized_pattern(*labels: str, exact: bool = True):
    joined = "|".join(re.escape(label) for label in labels)
    if exact:
        return re.compile(rf"^({joined})$")
    return re.compile(joined)


def _require_real_audio_pair() -> tuple[Path, Path]:
    if os.getenv("RUN_APP_SHELL_REAL_E2E") != "1" and os.getenv("RUN_STREAMLIT_REAL_E2E") != "1":
        pytest.skip("Set RUN_APP_SHELL_REAL_E2E=1 (or RUN_STREAMLIT_REAL_E2E=1) to run the real app-shell E2E test.")

    weaker_path = Path(
        os.getenv("ASSESS_SPEAKING_REAL_AUDIO_PATH") or DEFAULT_WEAKER_AUDIO_PATH
    ).expanduser()
    better_path = Path(
        os.getenv("ASSESS_SPEAKING_REAL_BETTER_AUDIO_PATH") or DEFAULT_BETTER_AUDIO_PATH
    ).expanduser()
    if not weaker_path.exists():
        pytest.skip(f"Weaker real audio file not found: {weaker_path}")
    if not better_path.exists():
        pytest.skip(f"Better real audio file not found: {better_path}")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for the real app-shell E2E test.")
    return weaker_path, better_path


def wait_for_label(reports_dir: Path, label: str, timeout: float = 180.0) -> dict[str, str]:
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


def submit_uploaded_attempt(page, audio_path: Path, *, label: str, notes: str) -> None:
    page.locator('label[data-baseweb="radio"]').filter(has_text=localized_pattern("Upload", "Hochladen", "Carica", exact=False)).click()
    page.locator('input[type="file"]').set_input_files(str(audio_path))
    expect(
        page.get_by_text(
            localized_pattern(
                "A recording is attached and ready for assessment",
                "Eine Aufnahme ist vorhanden und bereit fuer die Auswertung",
                exact=False,
            )
        )
    ).to_be_visible(timeout=15000)
    page.get_by_role("textbox", name=localized_pattern("Label", exact=False)).fill(label)
    page.get_by_role("textbox", name=localized_pattern("Notes", "Notizen", "Note", exact=False)).fill(notes)
    page.get_by_role(
        "button",
        name=localized_pattern("Submit for review", "Zur Auswertung senden", "Invia alla revisione"),
    ).click()


def _parse_score(row: dict[str, str]) -> float:
    value = row.get("final_score") or ""
    return float(value)


def _parse_band(row: dict[str, str]) -> int:
    value = row.get("band") or ""
    return int(value)


def test_app_shell_real_upload_review_history_progression(page, app_shell_real_server, reports_dir: Path):
    weaker_audio, better_audio = _require_real_audio_pair()
    speaker_id = "playwright-real-shell-user"
    first_label = "playwright-real-shell-bad"
    second_label = "playwright-real-shell-better"
    first_notes = "First take: weaker sample."
    second_notes = "Second take: stronger sample."

    page.set_viewport_size({"width": 1440, "height": 960})
    page.goto(f"{app_shell_real_server}/", wait_until="domcontentloaded")

    expect(page.get_by_role("heading", name=localized_pattern("Speaking Studio", exact=False))).to_be_visible()
    page.get_by_role("button", name=localized_pattern("Start new session", "Neue Session starten", "Nuova sessione")).click()

    expect(
        page.get_by_role(
            "heading",
            name=localized_pattern(
                "Prepare one speaking session",
                "Eine Sprechsession vorbereiten",
                "Prepara una sessione orale",
            ),
        )
    ).to_be_visible()
    page.get_by_label(localized_pattern("Speaker ID", exact=False)).fill(speaker_id)
    page.get_by_role("combobox", name=localized_pattern("Theme", "Thema", "Tema", exact=False)).click()
    page.get_by_role("option", name=localized_pattern("Custom theme", "Eigenes Thema", "Tema personalizzato")).click()
    page.get_by_role("textbox", name=localized_pattern("Custom theme text", exact=False)).fill("tema libero")
    page.get_by_role(
        "button",
        name=localized_pattern("Continue to speaking", "Weiter zum Sprechen", "Continua alla registrazione"),
    ).click()

    expect(
        page.get_by_role(
            "heading",
            name=localized_pattern("Record one response", "Eine Antwort aufnehmen", "Registra una risposta"),
        )
    ).to_be_visible()

    submit_uploaded_attempt(page, weaker_audio, label=first_label, notes=first_notes)
    expect(
        page.get_by_role(
            "heading",
            name=localized_pattern("Review one attempt", "Einen Versuch auswerten", "Rivedi un tentativo"),
        )
    ).to_be_visible(timeout=180000)
    expect(page.get_by_text(localized_pattern("Overall score", "Gesamtwert", "Punteggio complessivo"), exact=False)).to_be_visible()
    expect(page.get_by_role("heading", name=localized_pattern("Coach summary", exact=False))).to_be_visible()

    first_notes_view = page.get_by_role("textbox", name=localized_pattern("Your notes", "Deine Notizen", "Le tue note"))
    expect(first_notes_view).to_have_value(first_notes)
    first_transcript_view = page.get_by_role("textbox", name=localized_pattern("Transcript", "Transkript", "Trascrizione"))
    first_transcript = first_transcript_view.input_value().strip()
    assert first_transcript
    assert page.get_by_text(localized_pattern("Progress delta", exact=False)).count() == 0

    first_row = wait_for_label(reports_dir, first_label, timeout=180.0)
    assert first_row["speaker_id"] == speaker_id
    assert first_row["theme"] == "tema libero"
    assert first_row["report_path"]
    assert Path(first_row["report_path"]).exists()

    page.get_by_role("button", name=localized_pattern("Try again", "Noch einmal", "Riprova")).click()
    expect(
        page.get_by_role(
            "heading",
            name=localized_pattern("Record one response", "Eine Antwort aufnehmen", "Registra una risposta"),
        )
    ).to_be_visible()

    submit_uploaded_attempt(page, better_audio, label=second_label, notes=second_notes)
    expect(
        page.get_by_role(
            "heading",
            name=localized_pattern("Review one attempt", "Einen Versuch auswerten", "Rivedi un tentativo"),
        )
    ).to_be_visible(timeout=180000)
    expect(page.get_by_role("heading", name=localized_pattern("Progress delta", exact=False))).to_be_visible(timeout=30000)

    second_notes_view = page.get_by_role("textbox", name=localized_pattern("Your notes", "Deine Notizen", "Le tue note"))
    expect(second_notes_view).to_have_value(second_notes)
    second_transcript_view = page.get_by_role("textbox", name=localized_pattern("Transcript", "Transkript", "Trascrizione"))
    second_transcript = second_transcript_view.input_value().strip()
    assert second_transcript

    second_row = wait_for_label(reports_dir, second_label, timeout=180.0)
    assert second_row["speaker_id"] == speaker_id
    assert second_row["theme"] == "tema libero"
    assert second_row["report_path"]
    assert Path(second_row["report_path"]).exists()
    assert second_row["task_family"] == first_row["task_family"]
    assert _parse_score(second_row) > _parse_score(first_row)
    assert _parse_band(second_row) >= _parse_band(first_row)

    page.get_by_role("button", name=localized_pattern("Open history", "Verlauf oeffnen", "Apri lo storico")).click()
    expect(
        page.get_by_role(
            "heading",
            name=localized_pattern("History and trends", "Verlauf und Trends", "Storico e andamento"),
        )
    ).to_be_visible(timeout=30000)
    expect(
        page.get_by_text(
            re.compile(rf"Showing 2 saved runs for speaker {re.escape(speaker_id)}(?: in [A-Z]+)?\."),
            exact=False,
        )
    ).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name=localized_pattern("Saved attempts", exact=False))).to_be_visible()
    expect(page.get_by_role("heading", name=localized_pattern("Attempt details", exact=False))).to_be_visible()

    history_notes_view = page.get_by_role("textbox", name=localized_pattern("Your notes", "Deine Notizen", "Le tue note"))
    expect(history_notes_view).to_have_value(second_notes)
    history_transcript_view = page.get_by_role("textbox", name=localized_pattern("Transcript", "Transkript", "Trascrizione"))
    expect(history_transcript_view).to_have_value(second_transcript)
