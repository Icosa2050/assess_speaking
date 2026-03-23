import os
import socket
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

DEFAULT_STREAMLIT_PORT = 8502


def _pick_free_port(preferred: int = DEFAULT_STREAMLIT_PORT) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, *, timeout: float) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit port {host}:{port} did not become reachable within {timeout:.0f}s: {last_err}")


STREAMLIT_PORT = _pick_free_port(DEFAULT_STREAMLIT_PORT)
REAL_STREAMLIT_PORT = _pick_free_port(STREAMLIT_PORT + 1)
APP_SHELL_PORT = _pick_free_port(REAL_STREAMLIT_PORT + 1)
APP_SHELL_REAL_PORT = _pick_free_port(APP_SHELL_PORT + 1)
BASE_URL = f"http://127.0.0.1:{STREAMLIT_PORT}"
REAL_BASE_URL = f"http://127.0.0.1:{REAL_STREAMLIT_PORT}"
APP_SHELL_BASE_URL = f"http://127.0.0.1:{APP_SHELL_PORT}"
APP_SHELL_REAL_BASE_URL = f"http://127.0.0.1:{APP_SHELL_REAL_PORT}"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def reports_dir(project_root: Path) -> Path:
    return project_root / "reports"


@pytest.fixture(scope="session")
def samples_dir(project_root: Path) -> Path:
    return project_root / "samples"


def _faster_whisper_cache_exists(model_name: str, env: dict[str, str]) -> bool:
    hub_cache = Path(
        env.get("HF_HUB_CACHE")
        or env.get("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    return (hub_cache / f"models--Systran--faster-whisper-{model_name}").exists()


@pytest.fixture(scope="session", autouse=True)
def prepare_reports(project_root: Path):
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir, ignore_errors=True)
    yield


def _start_streamlit_server(project_root: Path, *, dry_run: bool, port: int, entrypoint: Path):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(project_root))
    if dry_run:
        env.setdefault("ASSESS_SPEAKING_DRY_RUN", "1")
    else:
        env.pop("ASSESS_SPEAKING_DRY_RUN", None)
        chosen_model = env.get("ASSESS_SPEAKING_REAL_WHISPER_MODEL", "tiny")
        if "HF_HUB_OFFLINE" not in env and _faster_whisper_cache_exists(chosen_model, env):
            env["HF_HUB_OFFLINE"] = "1"
    streamlit_cmd = [sys.executable,
                     "-m",
                     "streamlit",
                     "run",
                     str(entrypoint),
                     "--server.headless=true",
                     f"--server.port={port}",
                     "--server.address=127.0.0.1"]

    proc = subprocess.Popen(
        streamlit_cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise RuntimeError(f"Streamlit failed to start.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        try:
            _wait_for_port("127.0.0.1", port, timeout=2)
            break
        except RuntimeError:
            time.sleep(0.5)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        stdout, stderr = proc.communicate(timeout=1)
        raise RuntimeError(f"Streamlit health check failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    return proc


@pytest.fixture(scope="session")
def streamlit_server(project_root: Path):
    proc = _start_streamlit_server(
        project_root,
        dry_run=True,
        port=STREAMLIT_PORT,
        entrypoint=project_root / "scripts" / "interactive_dashboard.py",
    )

    yield BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def streamlit_real_server(project_root: Path):
    proc = _start_streamlit_server(
        project_root,
        dry_run=False,
        port=REAL_STREAMLIT_PORT,
        entrypoint=project_root / "scripts" / "interactive_dashboard.py",
    )

    yield REAL_BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def app_shell_server(project_root: Path):
    proc = _start_streamlit_server(
        project_root,
        dry_run=True,
        port=APP_SHELL_PORT,
        entrypoint=project_root / "streamlit_app.py",
    )

    yield APP_SHELL_BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def app_shell_real_server(project_root: Path):
    proc = _start_streamlit_server(
        project_root,
        dry_run=False,
        port=APP_SHELL_REAL_PORT,
        entrypoint=project_root / "streamlit_app.py",
    )

    yield APP_SHELL_REAL_BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
