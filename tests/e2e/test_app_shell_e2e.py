import re
from pathlib import Path

from playwright.sync_api import expect


def test_app_shell_upload_reaches_review(page, app_shell_server, samples_dir: Path):
    page.set_viewport_size({"width": 1440, "height": 960})
    page.goto(f"{app_shell_server}/", wait_until="domcontentloaded")

    expect(page.get_by_role("heading", name="Speaking Studio")).to_be_visible()
    page.get_by_role("button", name=re.compile(r"^(Start new session|Neue Session starten)$")).click()

    expect(page.get_by_role("heading", name=re.compile(r"^(Prepare one speaking session|Eine Sprechsession vorbereiten)$"))).to_be_visible()
    page.get_by_label("Speaker ID").fill("playwright-shell-user")
    page.get_by_role("button", name=re.compile(r"^(Continue to speaking|Weiter zum Sprechen)$")).click()

    expect(page.get_by_role("heading", name=re.compile(r"^(Record one response|Eine Antwort aufnehmen)$"))).to_be_visible()
    page.locator('label[data-baseweb="radio"]').filter(has_text=re.compile(r"(Upload|Hochladen)")).click()
    page.locator('input[type="file"]').set_input_files(str(samples_dir / "demo.m4a"))
    expect(page.get_by_text(re.compile(r"(A recording is attached and ready for assessment|Eine Aufnahme ist vorhanden und bereit fuer die Auswertung)"))).to_be_visible()

    page.get_by_role("button", name=re.compile(r"^(Submit for review|Zur Auswertung senden)$")).click()
    expect(page.get_by_role("heading", name=re.compile(r"^(Review one attempt|Einen Versuch auswerten)$"))).to_be_visible(timeout=120000)
    expect(page.get_by_text(re.compile(r"^(Overall score|Gesamtwert)$"))).to_be_visible()
