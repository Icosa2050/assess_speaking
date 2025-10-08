import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

STREAMLIT_PORT = 8502
BASE_URL = f"http://127.0.0.1:{STREAMLIT_PORT}"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def prepare_reports(project_root: Path):
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    yield


@pytest.fixture(scope="session", autouse=True)
def streamlit_server(project_root: Path):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(project_root))
    streamlit_cmd = [sys.executable,
                     "-m",
                     "streamlit",
                     "run",
                     str(project_root / "scripts" / "interactive_dashboard.py"),
                     "--server.headless=true",
                     f"--server.port={STREAMLIT_PORT}",
                     "--server.address=127.0.0.1"]

    proc = subprocess.Popen(
        streamlit_cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    health_url = f"{BASE_URL}/_stcore/health"
    deadline = time.time() + 60
    last_err = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise RuntimeError(f"Streamlit failed to start.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        try:
            resp = requests.get(health_url, timeout=1)
            if resp.status_code == 200:
                break
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(0.5)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError(f"Streamlit health check failed: {last_err}")

    yield BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL
