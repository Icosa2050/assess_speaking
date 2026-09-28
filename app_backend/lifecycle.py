from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import httpx

from app_backend.config import build_backend_runtime_config, clear_backend_state, read_backend_state
from app_core.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR
from app_core.bootstrap import (
    AUTH_MODE_ENV_VAR,
    DEPLOYMENT_MODE_ENV_VAR,
    LAUNCH_MODE_ENV_VAR,
    PROJECT_ROOT,
    bootstrap_app_environment,
    build_runtime_metadata,
)

HEALTH_ENDPOINT = "/v1/health"
DESKTOP_API_BASE_URL_ENV_VAR = "VOSTAVO_DESKTOP_API_BASE_URL"
DESKTOP_PACKAGING_SAFE_ENV_VAR = "VOSTAVO_DESKTOP_PACKAGING_SAFE"
BACKEND_START_ATTEMPTS = 3

logger = logging.getLogger(__name__)


def _backend_python_executable() -> Path:
    venv_python = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else PROJECT_ROOT / ".venv" / "bin" / "python"
    )
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _backend_command(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int,
) -> list[str]:
    command = [
        str(_backend_python_executable()),
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


def _pid_matches_backend_command(pid: int, expected_command_marker: str) -> bool:
    if not expected_command_marker:
        return True
    if os.name != "posix":
        return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and expected_command_marker in result.stdout


def _safe_terminate_pid(pid: int, *, expected_command_marker: str = "") -> None:
    if pid <= 0:
        return
    if not _pid_matches_backend_command(pid, expected_command_marker):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        logger.warning("Could not terminate stale backend process %s.", pid, exc_info=True)
        return


def _terminate_spawned_backend(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning(
                "Could not force-stop spawned backend process %s.",
                getattr(process, "pid", "<unknown>"),
                exc_info=True,
            )
            return


def _spawn_backend_process(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise RuntimeError(f"Could not start the local backend: {exc}") from exc


def _runtime_metadata_payload(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = bootstrap_app_environment(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    )
    return build_runtime_metadata(paths).as_dict()


def _merge_runtime_metadata(
    state: dict[str, Any],
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    merged = dict(state)
    merged.update(
        _runtime_metadata_payload(
            log_dir=log_dir,
            app_data_dir=app_data_dir,
            cache_dir=cache_dir,
        )
    )
    return merged


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
        return _merge_runtime_metadata(
            state,
            log_dir=log_dir,
            app_data_dir=app_data_dir,
            cache_dir=cache_dir,
        )
    pid = int(state.get("pid") or 0)
    _safe_terminate_pid(pid, expected_command_marker="scripts/run_backend.py")
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

    env = os.environ.copy()
    if app_data_dir is not None and str(app_data_dir).strip():
        env[APP_DATA_HOME_ENV_VAR] = str(app_data_dir)
    if cache_dir is not None and str(cache_dir).strip():
        env[APP_CACHE_HOME_ENV_VAR] = str(cache_dir)

    for attempt in range(1, BACKEND_START_ATTEMPTS + 1):
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
        process = _spawn_backend_process(command, env)
        deadline = time.time() + startup_timeout_sec
        exit_code: int | None = None
        while time.time() < deadline:
            state = read_backend_state(
                log_dir,
                app_data_dir=app_data_dir,
                cache_dir=cache_dir,
            )
            if isinstance(state, dict):
                base_url = str(state.get("base_url") or "").strip()
                if base_url and is_backend_healthy(base_url):
                    return _merge_runtime_metadata(
                        state,
                        log_dir=log_dir,
                        app_data_dir=app_data_dir,
                        cache_dir=cache_dir,
                    )
            poll_result = process.poll()
            if isinstance(poll_result, int):
                exit_code = poll_result
                logger.warning(
                    "The local backend exited before it became ready (attempt %s/%s, exit code %s).",
                    attempt,
                    BACKEND_START_ATTEMPTS,
                    exit_code,
                )
                break
            time.sleep(0.2)
        if exit_code is not None:
            if attempt < BACKEND_START_ATTEMPTS:
                continue
            raise RuntimeError(f"The local backend exited before it became ready (exit code {exit_code}).")
        _terminate_spawned_backend(process)
        raise RuntimeError("The local backend did not become ready in time.")

    raise RuntimeError("The local backend did not become ready in time.")


def build_desktop_bootstrap_env(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    startup_timeout_sec: float = 15.0,
) -> dict[str, str]:
    state = ensure_local_backend(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
        startup_timeout_sec=startup_timeout_sec,
    )
    return {
        DESKTOP_API_BASE_URL_ENV_VAR: str(state.get("base_url") or ""),
        DEPLOYMENT_MODE_ENV_VAR: str(state.get("deployment_mode") or "local"),
        LAUNCH_MODE_ENV_VAR: str(state.get("launch_mode") or "repo"),
        DESKTOP_PACKAGING_SAFE_ENV_VAR: "true" if bool(state.get("packaging_safe")) else "false",
        AUTH_MODE_ENV_VAR: str(state.get("auth_mode") or "guest"),
    }
