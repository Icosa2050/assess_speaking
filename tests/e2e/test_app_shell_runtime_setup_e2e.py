import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from app_shell.services import discover_runtime_models


def _pick_free_port(preferred: int = 8510) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, *, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit port {host}:{port} did not become reachable within {timeout:.0f}s: {last_error}")


def _require_local_runtime_models(provider_choice: str, base_url: str) -> list[str]:
    if str(os.getenv("RUN_APP_SHELL_LOCAL_RUNTIME_E2E") or "") != "1":
        pytest.skip("Set RUN_APP_SHELL_LOCAL_RUNTIME_E2E=1 to run the local runtime setup E2E tests.")

    try:
        detection = discover_runtime_models(
            provider=provider_choice,
            provider_choice=provider_choice,
            base_url=base_url,
            timeout_sec=5.0,
        )
    except Exception as exc:  # pragma: no cover - live environment dependent
        pytest.skip(f"{provider_choice} is not reachable for local runtime setup E2E: {exc}")

    models = list(detection.get("models") or [])
    if not models:
        pytest.skip(f"{provider_choice} returned no local models for runtime setup E2E.")
    return models


@pytest.fixture(scope="session")
def app_shell_clean_server(project_root: Path):
    runtime_root = Path(tempfile.mkdtemp(prefix="app-shell-runtime-setup-e2e-"))
    port = _pick_free_port()
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(project_root))
    env.setdefault("ASSESS_SPEAKING_DRY_RUN", "1")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(project_root / "streamlit_app.py"),
            "--server.headless=true",
            f"--server.port={port}",
            "--server.address=127.0.0.1",
        ],
        cwd=runtime_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port("127.0.0.1", port, timeout=60.0)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _open_runtime_setup(page: Page, base_url: str) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"{base_url}/", wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name=re.compile(r"^Speaking Studio$", re.I))).to_be_visible(timeout=30000)
    page.get_by_role("button", name=re.compile(r"^(Open runtime setup|Runtime-Setup oeffnen)$")).click()
    expect(page.get_by_role("heading", name=re.compile(r"^Runtime setup$", re.I))).to_be_visible(timeout=30000)


def _choose_selectbox_option(page: Page, label: str, option: str) -> None:
    page.get_by_role("combobox", name=label).click()
    page.get_by_role("option", name=option, exact=True).click()


def _locator_must_be_in_viewport(locator) -> None:
    expect(locator).to_be_visible(timeout=30000)
    box = locator.bounding_box()
    assert box is not None
    assert box["y"] >= 0
    assert box["y"] < 844


def test_runtime_setup_ollama_local_detects_models_and_sanitizes_small_screen_feedback(page, app_shell_clean_server):
    ollama_models = _require_local_runtime_models("ollama_local", "http://localhost:11434")

    _open_runtime_setup(page, app_shell_clean_server)
    _choose_selectbox_option(page, "Provider", "Ollama local")

    base_url_input = page.get_by_role("textbox", name="Base URL")
    model_input = page.get_by_role("textbox", name="Model")
    base_url_input.fill("http://localhost:11434/api")
    model_input.fill("")
    page.get_by_role("button", name="Detect local models", exact=True).click()

    expect(page.get_by_text(re.compile(r"Detected \d+ local model\(s\) via http://localhost:11434/api/tags\."))).to_be_visible(timeout=30000)
    expect(base_url_input).to_have_value("http://localhost:11434", timeout=30000)
    expect(model_input).to_have_value(ollama_models[0], timeout=30000)
    expect(page.get_by_role("combobox", name="Detected local models")).to_be_visible(timeout=30000)

    page.get_by_role("button", name="Test connection", exact=True).click()
    section_heading = page.get_by_role("heading", name=re.compile(r"^Section D.*Test connection$"))
    _locator_must_be_in_viewport(section_heading)
    _locator_must_be_in_viewport(
        page.get_by_text(re.compile(r"^Health check passed at http://localhost:11434/api/tags\.", re.I))
    )


def test_runtime_setup_lmstudio_local_shows_small_screen_failure_summary_and_detail(page, app_shell_clean_server):
    lmstudio_models = _require_local_runtime_models("lmstudio_local", "http://localhost:1234/v1")

    _open_runtime_setup(page, app_shell_clean_server)
    _choose_selectbox_option(page, "Provider", "LM Studio local")

    model_input = page.get_by_role("textbox", name="Model")
    model_input.fill("")
    page.get_by_role("button", name="Detect local models", exact=True).click()

    expect(page.get_by_text(re.compile(r"Detected \d+ local model\(s\) via http://localhost:1234/v1/models\."))).to_be_visible(timeout=30000)
    expect(model_input).to_have_value(lmstudio_models[0], timeout=30000)

    model_input.fill("nonexistingmodelshouldgiverror")
    model_input.press("Tab")
    expect(model_input).to_have_value("nonexistingmodelshouldgiverror", timeout=30000)
    page.get_by_role("button", name="Test connection", exact=True).click()

    expect(page.get_by_text("Runtime setup is already complete.")).to_have_count(0)

    section_heading = page.get_by_role("heading", name=re.compile(r"^Section D.*Test connection$"))
    _locator_must_be_in_viewport(section_heading)
    _locator_must_be_in_viewport(page.get_by_text(re.compile(r"Connection test failed: HTTP \d{3}:", re.I)))
