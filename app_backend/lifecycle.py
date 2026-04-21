from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import httpx

from app_backend.config import build_backend_runtime_config, clear_backend_state, read_backend_state
from app_shell.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR
from app_shell.bootstrap import PROJECT_ROOT

HEALTH_ENDPOINT = "/v1/health"


def _backend_command(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int,
) -> list[str]:
    command = [
        str(Path(sys.executable)),
        str(PROJECT_ROOT / "scripts" / "run_backend.py"),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if log_dir is not None and str(log_dir).strip():
        command.extend(["--log-dir", str(log_dir)])
    if app_data_dir is not None and str(app_data_dir).strip():
        command.extend(["--app-data-dir", str(app_data_dir)])
    if cache_dir is not None and str(cache_dir).strip():
        command.extend(["--cache-dir", str(cache_dir)])
    return command


def is_backend_healthy(base_url: str, *, timeout_sec: float = 1.0) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}{HEALTH_ENDPOINT}", timeout=timeout_sec)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _safe_terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def get_backend_state(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    state = read_backend_state(
        log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    )
    if not isinstance(state, dict):
        return None
    base_url = str(state.get("base_url") or "").strip()
    if base_url and is_backend_healthy(base_url):
        return state
    pid = int(state.get("pid") or 0)
    _safe_terminate_pid(pid)
    clear_backend_state(log_dir, app_data_dir=app_data_dir, cache_dir=cache_dir)
    return None


def ensure_local_backend(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    startup_timeout_sec: float = 15.0,
) -> dict[str, Any]:
    existing = get_backend_state(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    )
    if existing is not None:
        return existing

    config = build_backend_runtime_config(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    )
    command = _backend_command(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
        host=config.host,
        port=config.port,
    )
    env = os.environ.copy()
    if app_data_dir is not None and str(app_data_dir).strip():
        env[APP_DATA_HOME_ENV_VAR] = str(app_data_dir)
    if cache_dir is not None and str(cache_dir).strip():
        env[APP_CACHE_HOME_ENV_VAR] = str(cache_dir)
    subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + startup_timeout_sec
    while time.time() < deadline:
        state = read_backend_state(
            log_dir,
            app_data_dir=app_data_dir,
            cache_dir=cache_dir,
        )
        if isinstance(state, dict):
            base_url = str(state.get("base_url") or "").strip()
            if base_url and is_backend_healthy(base_url):
                return state
        time.sleep(0.2)
    raise RuntimeError("The local backend did not become ready in time.")
