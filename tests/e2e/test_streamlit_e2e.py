import csv
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect


def localized_pattern(*labels: str, exact: bool = True):
    joined = "|".join(re.escape(label) for label in labels)
    if exact:
        return re.compile(rf"^({joined})$")
    return re.compile(joined)


def expand_summary(page, *labels: str) -> None:
    page.locator("summary").filter(has_text=localized_pattern(*labels, exact=False)).first.click()


@pytest.fixture
def samples_dir(project_root: Path) -> Path:
    return project_root / "samples"


@pytest.fixture
def reports_dir(project_root: Path) -> Path:
    return project_root / "reports"


def wait_for_history(reports_dir: Path, timeout: float = 90.0) -> Path:
    history = reports_dir / "history.csv"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if history.exists():
            return history
        time.sleep(0.5)
    raise AssertionError("history.csv was not created within timeout")


def read_history(reports_dir: Path):
    history_path = wait_for_history(reports_dir)
    with history_path.open() as fh:
        return list(csv.DictReader(fh))


def latest_matching_row(rows, **criteria):
    for row in reversed(rows):
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise AssertionError(f"No history row matched {criteria!r}")


def open_prompt_trainer(page) -> None:
    prompt_tab = page.get_by_role("tab", name="Prompt-Trainer")
    if prompt_tab.count() == 0 or not prompt_tab.first.is_visible():
        expand_summary(page, "Weitere Werkzeuge und Verlauf", "More tools and progress")
        expect(prompt_tab).to_be_visible(timeout=10000)
    prompt_tab.click()


def activate_manual_upload_mode(page) -> None:
    page.get_by_text("Stattdessen eine vorhandene Aufnahme nutzen", exact=True).click()
    page.get_by_role("button", name="Alternative aktivieren").click()
    expect(page.get_by_text("Du nutzt gerade eine vorhandene Aufnahme", exact=False)).to_be_visible(timeout=10000)
    expect(page.locator('[aria-label="Audio-Datei hinzufügen"] input[type="file"]')).to_be_attached(timeout=10000)


def test_01_basic_upload_creates_history(page, base_url, samples_dir, reports_dir, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    expect(page.get_by_text("Speaking Studio", exact=False)).to_be_visible()
    expect(page.get_by_text("Dein Sprechauftrag", exact=False)).to_be_visible()
    expect(page.get_by_role("button", name="Neue Aufgabenfassung")).to_be_visible()
    expect(page.get_by_role("textbox", name="Speaker ID")).to_be_visible()
    theme_input = page.get_by_role("textbox", name=localized_pattern("Thema", "Theme"))
    expect(theme_input).to_be_visible()
    expect(page.get_by_text("Primärer Weg: direkt sprechen", exact=False)).to_be_visible()

    page.get_by_role("textbox", name="Speaker ID").fill("playwright-user")
    theme_input.fill("Il mio ultimo viaggio all'estero")
    activate_manual_upload_mode(page)
    page.locator('[aria-label="Audio-Datei hinzufügen"] input[type="file"]').set_input_files(
        str(samples_dir / "demo.m4a")
    )
    page.get_by_text("Technik und Notizen", exact=True).click()
    page.get_by_label("Label").fill("playwright-basic")
    run_button = page.get_by_role("button", name="Datei auswerten")
    expect(run_button).to_be_enabled(timeout=10000)
    run_button.click()
    expect(
        page.get_by_text(localized_pattern("Deine Rückmeldung", "Your feedback", exact=False))
    ).to_be_visible(timeout=60000)

    rows = read_history(reports_dir)
    assert rows
    row = latest_matching_row(rows, label="playwright-basic", speaker_id="playwright-user")
    assert row["theme"] == "Il mio ultimo viaggio all'estero"


def test_02_prompt_file_upload_generates_baseline(page, base_url, samples_dir, reports_dir, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    open_prompt_trainer(page)

    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B2 – Lavoro da casa (B2)').click()

    page.get_by_role("button", name="Übung starten").click()
    open_prompt_trainer(page)
    expect(
        page.get_by_text(localized_pattern("Verbleibende Zeit", "Zeitlimit überschritten", exact=False))
    ).to_be_visible(timeout=10000)

    play_button = page.get_by_role("button", name=re.compile(r"^Prompt abspielen"))
    play_button.click()

    page.get_by_text("Stattdessen eine fertige Antwort hochladen", exact=True).click()
    page.locator('[aria-label="Antwortdatei hochladen (wav/mp3/m4a)"] input[type="file"]').set_input_files(
        str(samples_dir / "demo.m4a")
    )

    open_prompt_trainer(page)
    expect(page.get_by_text("Letztes Prompt-Ergebnis", exact=True)).to_be_visible(timeout=60000)
    expect(
        page.get_by_text(localized_pattern("Deine Rückmeldung", "Your feedback", exact=False))
    ).to_be_visible(timeout=60000)

    rows = read_history(reports_dir)
    assert rows
    row = latest_matching_row(rows, speaker_id="playwright-user")
    assert row["label"].startswith("prompt:b2_remote_work")


def test_03_switching_prompts_shows_warning(page, base_url, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    open_prompt_trainer(page)
    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B1 – Racconto di viaggio (B1)').click()

    page.get_by_role("button", name="Übung starten").click()
    open_prompt_trainer(page)
    expect(page.get_by_text("Verbleibende Zeit", exact=False)).to_be_visible(timeout=10000)

    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B2 – Lavoro da casa (B2)').click()

    expect(page.get_by_text("Es läuft gerade ein Versuch", exact=False)).to_be_visible()


def test_04_prompt_timeout_blocks_submission(page, base_url, samples_dir, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    open_prompt_trainer(page)
    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B1 – Timer breve (test) (B1)').click()

    page.get_by_role("button", name="Übung starten").click()
    open_prompt_trainer(page)
    expect(page.get_by_text("Verbleibende Zeit", exact=False)).to_be_visible(timeout=10000)

    page.wait_for_timeout(4000)
    page.get_by_text("Stattdessen eine fertige Antwort hochladen", exact=True).click()
    page.locator('[aria-label="Antwortdatei hochladen (wav/mp3/m4a)"] input[type="file"]').set_input_files(
        str(samples_dir / "demo.m4a")
    )
    open_prompt_trainer(page)
    expect(page.get_by_text("Zeitlimit überschritten – starte die Übung neu", exact=False)).to_be_visible()
    page.get_by_text("Stattdessen eine fertige Antwort hochladen", exact=True).click()
    expect(page.get_by_text("Upload ist nach Ablauf des Zeitlimits gesperrt", exact=False)).to_be_visible()
