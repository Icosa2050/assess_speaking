from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import socket
from pathlib import Path
import shutil
from typing import Any

from app_core.app_data import AppDataPaths, build_app_data_paths
from app_core.bootstrap import bootstrap_app_environment

BACKEND_STATE_FILENAME = "backend_state.json"
BACKEND_LOG_FILENAME = "backend.log"
BACKEND_JOBS_DIRNAME = "jobs"
SUPPORT_BUNDLE_DIRNAME = "support-bundles"
BACKEND_LOG_MAX_BYTES = 1_000_000
BACKEND_LOG_BACKUP_COUNT = 3
DEFAULT_SUPPORT_BUNDLE_RETENTION_HOURS = 24
DEFAULT_JOB_METADATA_RETENTION_DAYS = 30
DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_STARTUP_TIMEOUT_SEC = 15.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendRuntimeConfig:
    host: str
    port: int
    app_data: AppDataPaths
    state_file: Path
    jobs_dir: Path
    log_file: Path
    startup_timeout_sec: float = DEFAULT_BACKEND_STARTUP_TIMEOUT_SEC

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def resolve_backend_state_file(log_dir: str | Path | None = None) -> Path:
    return build_app_data_paths(log_dir).root / BACKEND_STATE_FILENAME


def resolve_jobs_dir(log_dir: str | Path | None = None) -> Path:
    return build_app_data_paths(log_dir).jobs_dir


def resolve_backend_log_file(log_dir: str | Path | None = None) -> Path:
    return build_app_data_paths(log_dir).logs_dir / BACKEND_LOG_FILENAME


def resolve_support_bundle_dir(log_dir: str | Path | None = None) -> Path:
    return build_app_data_paths(log_dir).temp_dir / SUPPORT_BUNDLE_DIRNAME


def _migrate_legacy_jobs_dir(app_data: AppDataPaths) -> None:
    legacy_jobs_dir = app_data.reports_dir / BACKEND_JOBS_DIRNAME
    jobs_dir = app_data.jobs_dir
    if not legacy_jobs_dir.exists() or not legacy_jobs_dir.is_dir():
        return
    try:
        has_new_jobs = any(jobs_dir.iterdir())
    except OSError:
        has_new_jobs = False
    if has_new_jobs:
        return
    jobs_dir.mkdir(parents=True, exist_ok=True)
    try:
        for child in legacy_jobs_dir.iterdir():
            target = jobs_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        legacy_jobs_dir.rmdir()
    except OSError:
        logger.warning(
            "Could not migrate legacy backend job metadata from %s to %s.",
            legacy_jobs_dir,
            jobs_dir,
            exc_info=True,
        )
        return


def allocate_local_port(host: str = DEFAULT_BACKEND_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def build_backend_runtime_config(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    port: int | None = None,
    host: str = DEFAULT_BACKEND_HOST,
) -> BackendRuntimeConfig:
    app_data = bootstrap_app_environment(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    )
    _migrate_legacy_jobs_dir(app_data)
    resolved_port = int(port) if port is not None else allocate_local_port(host)
    jobs_dir = app_data.jobs_dir
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return BackendRuntimeConfig(
        host=host,
        port=resolved_port,
        app_data=app_data,
        state_file=app_data.root / BACKEND_STATE_FILENAME,
        jobs_dir=jobs_dir,
        log_file=app_data.logs_dir / BACKEND_LOG_FILENAME,
    )


def write_backend_state(config: BackendRuntimeConfig, *, pid: int) -> dict[str, Any]:
    payload = {
        "pid": int(pid),
        "host": config.host,
        "port": int(config.port),
        "base_url": config.base_url,
        "app_data_root": str(config.app_data.root),
        "reports_dir": str(config.app_data.reports_dir),
        "jobs_dir": str(config.jobs_dir),
        "log_file": str(config.log_file),
    }
    config.state_file.parent.mkdir(parents=True, exist_ok=True)
    config.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_backend_state(
    log_dir: str | Path | None = None,
    *,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    state_file = bootstrap_app_environment(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    ).root / BACKEND_STATE_FILENAME
    if not state_file.exists():
        return None
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def clear_backend_state(
    log_dir: str | Path | None = None,
    *,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    state_file = bootstrap_app_environment(
        log_dir=log_dir,
        app_data_dir=app_data_dir,
        cache_dir=cache_dir,
    ).root / BACKEND_STATE_FILENAME
    try:
        state_file.unlink()
    except OSError:
        logger.warning("Could not clear backend state file %s.", state_file, exc_info=True)
        return
