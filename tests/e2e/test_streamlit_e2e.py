import csv
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect


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


def test_01_basic_upload_creates_history(page, base_url, samples_dir, reports_dir, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    expect(page.get_by_text("Assess Speaking", exact=False)).to_be_visible()

    page.locator('input[type="file"]').first.set_input_files(str(samples_dir / "demo.m4a"))
    page.get_by_label("Label").fill("playwright-basic")
    page.get_by_role("button", name="Bewertung starten").click()

    rows = read_history(reports_dir)
    assert rows
    assert rows[-1]["label"] == "playwright-basic"


def test_02_prompt_file_upload_generates_baseline(page, base_url, samples_dir, reports_dir, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("text=Prompt-Trainer", timeout=60000)
    page.get_by_text('Prompt-Trainer', exact=True).click()

    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B2 – Lavoro da casa (B2)').click()

    page.get_by_role("button", name="Versuch starten").click()
    expect(page.get_by_text("Verbleibende Zeit", exact=False)).to_be_visible()

    play_button = page.get_by_role("button", name=re.compile(r"^Prompt abspielen"))
    play_button.click()

    uploaders = page.locator('input[type="file"]')
    uploaders.nth(1).set_input_files(str(samples_dir / "demo.m4a"))

    expect(page.get_by_text("Bewertung abgeschlossen", exact=False)).to_be_visible(timeout=60000)
    expect(page.get_by_text("Baseline B2", exact=False)).to_be_visible()

    rows = read_history(reports_dir)
    assert rows
    assert rows[-1]["label"].startswith("prompt:b2_remote_work")


def test_03_switching_prompts_shows_warning(page, base_url, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("text=Prompt-Trainer", timeout=60000)
    page.get_by_text('Prompt-Trainer', exact=True).click()
    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B1 – Racconto di viaggio (B1)').click()

    page.get_by_role("button", name="Versuch starten").click()
    expect(page.get_by_text("Verbleibende Zeit", exact=False)).to_be_visible()

    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B2 – Lavoro da casa (B2)').click()

    expect(page.get_by_text("Es läuft gerade ein Versuch", exact=False)).to_be_visible()


def test_04_prompt_timeout_blocks_submission(page, base_url, samples_dir, streamlit_server):
    page.goto(f"{base_url}/")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector("text=Prompt-Trainer", timeout=60000)
    page.get_by_text('Prompt-Trainer', exact=True).click()
    page.get_by_text('B1 – Racconto di viaggio (B1)', exact=True).click()
    page.get_by_role('option', name='B1 – Timer breve (test) (B1)').click()

    page.get_by_role("button", name="Versuch starten").click()
    expect(page.get_by_text("Verbleibende Zeit", exact=False)).to_be_visible()

    page.wait_for_timeout(4000)

    uploaders = page.locator('input[type="file"]')
    uploaders.nth(1).set_input_files(str(samples_dir / "demo.m4a"))

    expect(page.get_by_text("Zeitlimit überschritten", exact=False)).to_be_visible()
