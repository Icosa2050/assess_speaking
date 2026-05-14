from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app_backend.config import (
    BACKEND_LOG_BACKUP_COUNT,
    BACKEND_LOG_FILENAME,
    BACKEND_LOG_MAX_BYTES,
    DEFAULT_JOB_METADATA_RETENTION_DAYS,
    SUPPORT_BUNDLE_DIRNAME,
)
from app_backend.jobs import prunable_job_metadata_files
from assessment_runtime.asr import describe_model_availability
from assessment_runtime.llm_client import health_check as llm_health_check
from app_shell.bootstrap import PROJECT_ROOT, bootstrap_app_environment, is_within_project_checkout
from app_shell.services import build_client_snapshot
from app_shell.runtime_providers import requires_api_key
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime

DiagnosticStatus = Literal["ok", "warning", "error", "info"]
REDACTED_DETAIL_VALUE = "[redacted]"
_SECRET_DETAIL_ARG_TOKENS = ("api_key", "authorization", "password", "secret", "token")
MAINTENANCE_SETTINGS_PAGE = "pages/06_Settings.py"
MAINTENANCE_ACTION_LABEL_KEY = "diagnostics.maintenance_open_settings"
TMP_WARNING_AGE_HOURS = 24
TMP_WARNING_FILE_COUNT = 10
TMP_WARNING_SIZE_BYTES = 50 * 1024 * 1024
JOB_WARNING_FILE_COUNT = 100
LOG_WARNING_FOOTPRINT_BYTES = BACKEND_LOG_MAX_BYTES * (BACKEND_LOG_BACKUP_COUNT + 1)


@dataclass(frozen=True)
class StartupDiagnostic:
    key: str
    status: DiagnosticStatus
    title_key: str
    detail_key: str
    detail_args: dict[str, Any] = field(default_factory=dict)


def _app_data_writable_diagnostic(paths) -> StartupDiagnostic:
    try:
        with tempfile.NamedTemporaryFile(dir=paths.temp_dir, delete=True):
            pass
    except OSError as exc:
        return StartupDiagnostic(
            key="app_data",
            status="error",
            title_key="diagnostics.app_data_title",
            detail_key="diagnostics.app_data_error_detail",
            detail_args={"path": str(paths.temp_dir), "detail": str(exc)},
        )
    return StartupDiagnostic(
        key="app_data",
        status="ok",
        title_key="diagnostics.app_data_title",
        detail_key="diagnostics.app_data_ok_detail",
        detail_args={"path": str(paths.temp_dir)},
    )


def _ffmpeg_diagnostic() -> StartupDiagnostic:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return StartupDiagnostic(
            key="ffmpeg",
            status="ok",
            title_key="diagnostics.ffmpeg_title",
            detail_key="diagnostics.ffmpeg_ok_detail",
            detail_args={"path": ffmpeg_path},
        )
    return StartupDiagnostic(
        key="ffmpeg",
        status="error",
        title_key="diagnostics.ffmpeg_title",
        detail_key="diagnostics.ffmpeg_error_detail",
    )


def _repo_local_override_diagnostic(paths) -> StartupDiagnostic | None:
    if not is_within_project_checkout(paths.root):
        return None
    return StartupDiagnostic(
        key="app_data_location",
        status="warning",
        title_key="diagnostics.app_data_location_title",
        detail_key="diagnostics.app_data_location_warning_detail",
        detail_args={"path": str(paths.root), "project_root": str(PROJECT_ROOT)},
    )


def _whisper_diagnostic(model_size: str) -> StartupDiagnostic:
    availability = describe_model_availability(model_size)
    if availability["cached"]:
        return StartupDiagnostic(
            key="whisper",
            status="ok",
            title_key="diagnostics.whisper_title",
            detail_key="diagnostics.whisper_ok_detail",
            detail_args={
                "model": model_size,
                "path": str(availability.get("cached_path") or ""),
            },
        )
    return StartupDiagnostic(
        key="whisper",
        status="warning",
        title_key="diagnostics.whisper_title",
        detail_key="diagnostics.whisper_warning_detail",
        detail_args={"model": model_size},
    )


def _runtime_diagnostics(state, *, include_runtime_health: bool) -> list[StartupDiagnostic]:
    connection = active_connection(state.prefs)
    if connection is None:
        return [
            StartupDiagnostic(
                key="runtime",
                status="warning",
                title_key="diagnostics.runtime_title",
                detail_key="diagnostics.runtime_warning_detail",
            )
        ]

    client_snapshot = build_client_snapshot(state)
    runtime = resolve_connection_runtime(connection)
    diagnostics = [
        StartupDiagnostic(
            key="runtime",
            status="ok",
            title_key="diagnostics.runtime_title",
            detail_key="diagnostics.runtime_ok_detail",
            detail_args={
                "provider": runtime.provider,
                "model": runtime.model,
                "credential_state": client_snapshot["credential_state"],
            },
        )
    ]
    if client_snapshot["provider_requires_auth"] and client_snapshot["credentials_missing"]:
        diagnostics.append(
            StartupDiagnostic(
                key="runtime_api_key",
                status="error",
                title_key="diagnostics.runtime_api_key_title",
                detail_key="diagnostics.runtime_api_key_error_detail",
                detail_args={
                    "provider": runtime.provider,
                    "credential_state": client_snapshot["credential_state"],
                    "secure_storage_persistent": client_snapshot["secure_storage_persistent"],
                },
            )
        )
    elif client_snapshot["provider_requires_auth"]:
        diagnostics.append(
            StartupDiagnostic(
                key="runtime_api_key",
                status="ok",
                title_key="diagnostics.runtime_api_key_title",
                detail_key="diagnostics.runtime_api_key_ok_detail",
                detail_args={
                    "provider": runtime.provider,
                    "credential_state": client_snapshot["credential_state"],
                },
            )
        )

    if include_runtime_health and connection.provider_kind in {"ollama", "lmstudio"}:
        try:
            result = llm_health_check(
                provider=runtime.provider,
                base_url=runtime.base_url,
                api_key=runtime.api_key,
                timeout_sec=1.5,
                openrouter_http_referer=runtime.extra_headers.get("HTTP-Referer", ""),
                openrouter_app_title=runtime.extra_headers.get("X-Title", ""),
            )
            diagnostics.append(
                StartupDiagnostic(
                    key="runtime_local_health",
                    status="ok",
                    title_key="diagnostics.runtime_local_health_title",
                    detail_key="diagnostics.runtime_local_health_ok_detail",
                    detail_args={
                        "provider": runtime.provider,
                        "endpoint": str(result.get("endpoint") or runtime.base_url),
                    },
                )
            )
        except Exception as exc:  # quality: allow[broad-except] provider health errors should stay learner-facing
            diagnostics.append(
                StartupDiagnostic(
                    key="runtime_local_health",
                    status="error",
                    title_key="diagnostics.runtime_local_health_title",
                    detail_key="diagnostics.runtime_local_health_error_detail",
                    detail_args={"provider": runtime.provider, "detail": str(exc)},
                )
            )
    return diagnostics


def _microphone_diagnostic() -> StartupDiagnostic:
    return StartupDiagnostic(
        key="microphone",
        status="info",
        title_key="diagnostics.microphone_title",
        detail_key="diagnostics.microphone_info_detail",
    )


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def _bundle_root(paths) -> Path:
    return paths.temp_dir / SUPPORT_BUNDLE_DIRNAME


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _maintenance_detail_args(**extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "target_page": MAINTENANCE_SETTINGS_PAGE,
        "action_label_key": MAINTENANCE_ACTION_LABEL_KEY,
    }


def _maintenance_tmp_diagnostic(paths) -> StartupDiagnostic | None:
    bundle_root = _bundle_root(paths)
    temp_files = [path for path in _iter_files(paths.temp_dir) if not _is_within(path, bundle_root)]
    if not temp_files:
        return None
    total_size = sum(path.stat().st_size for path in temp_files)
    oldest_mtime = min(path.stat().st_mtime for path in temp_files)
    oldest_age_hours = int(max(0, (datetime.now(UTC).timestamp() - oldest_mtime) // 3600))
    is_stale = oldest_age_hours >= TMP_WARNING_AGE_HOURS
    is_oversized = total_size >= TMP_WARNING_SIZE_BYTES or len(temp_files) >= TMP_WARNING_FILE_COUNT
    if not (is_stale or is_oversized):
        return None
    return StartupDiagnostic(
        key="maintenance_tmp",
        status="warning",
        title_key="diagnostics.maintenance_tmp_title",
        detail_key="diagnostics.maintenance_tmp_warning_detail",
        detail_args=_maintenance_detail_args(
            file_count=len(temp_files),
            size_bytes=total_size,
            oldest_age_hours=oldest_age_hours,
        ),
    )


def _maintenance_jobs_diagnostic(paths) -> StartupDiagnostic | None:
    job_files = sorted(paths.jobs_dir.glob("*.json")) if paths.jobs_dir.exists() else []
    stale_jobs = prunable_job_metadata_files(paths.jobs_dir, retention_days=DEFAULT_JOB_METADATA_RETENTION_DAYS)
    if not stale_jobs and len(job_files) < JOB_WARNING_FILE_COUNT:
        return None
    return StartupDiagnostic(
        key="maintenance_jobs",
        status="warning",
        title_key="diagnostics.maintenance_jobs_title",
        detail_key="diagnostics.maintenance_jobs_warning_detail",
        detail_args=_maintenance_detail_args(
            file_count=len(job_files),
            stale_count=len(stale_jobs),
            retention_days=DEFAULT_JOB_METADATA_RETENTION_DAYS,
        ),
    )


def _maintenance_logs_diagnostic(paths) -> StartupDiagnostic | None:
    log_files = _iter_files(paths.logs_dir)
    if not log_files:
        return None
    rotated_logs = [path for path in log_files if path.name.startswith(f"{BACKEND_LOG_FILENAME}.")]
    unexpected_logs = [
        path for path in log_files if path.name != BACKEND_LOG_FILENAME and not path.name.startswith(f"{BACKEND_LOG_FILENAME}.")
    ]
    total_size = sum(path.stat().st_size for path in log_files)
    exceeds_footprint = (
        total_size > LOG_WARNING_FOOTPRINT_BYTES
        or len(rotated_logs) > BACKEND_LOG_BACKUP_COUNT
        or bool(unexpected_logs)
    )
    if not exceeds_footprint:
        return None
    return StartupDiagnostic(
        key="maintenance_logs",
        status="warning",
        title_key="diagnostics.maintenance_logs_title",
        detail_key="diagnostics.maintenance_logs_warning_detail",
        detail_args=_maintenance_detail_args(
            file_count=len(log_files),
            size_bytes=total_size,
            expected_size_bytes=LOG_WARNING_FOOTPRINT_BYTES,
            unexpected_count=len(unexpected_logs),
        ),
    )


def _maintenance_diagnostics(paths) -> list[StartupDiagnostic]:
    diagnostics = [
        _maintenance_tmp_diagnostic(paths),
        _maintenance_jobs_diagnostic(paths),
        _maintenance_logs_diagnostic(paths),
    ]
    return [item for item in diagnostics if item is not None]


def _sanitize_support_detail_args(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            lowered = str(key).strip().lower()
            if lowered == "secret_ref":
                continue
            if any(token in lowered for token in _SECRET_DETAIL_ARG_TOKENS):
                sanitized[str(key)] = "" if child in (None, "") else REDACTED_DETAIL_VALUE
                continue
            sanitized[str(key)] = _sanitize_support_detail_args(child)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_support_detail_args(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_support_detail_args(item) for item in value]
    return value


def serialize_startup_diagnostics(items: list[StartupDiagnostic]) -> list[dict[str, Any]]:
    return [
        {
            "key": item.key,
            "status": item.status,
            "title_key": item.title_key,
            "detail_key": item.detail_key,
            "detail_args": _sanitize_support_detail_args(item.detail_args),
        }
        for item in items
    ]


def collect_startup_diagnostics(state, *, include_runtime_health: bool = True) -> list[StartupDiagnostic]:
    paths = bootstrap_app_environment(
        log_dir=getattr(state.prefs, "log_dir", ""),
        whisper_cache_dir=getattr(state.prefs, "whisper_cache_dir", ""),
    )
    diagnostics = [
        _app_data_writable_diagnostic(paths),
        _ffmpeg_diagnostic(),
        _whisper_diagnostic(str(getattr(state.prefs, "whisper_model", "") or "small")),
    ]
    repo_local_override = _repo_local_override_diagnostic(paths)
    if repo_local_override is not None:
        diagnostics.append(repo_local_override)
    diagnostics.extend(_maintenance_diagnostics(paths))
    diagnostics.extend(_runtime_diagnostics(state, include_runtime_health=include_runtime_health))
    diagnostics.append(_microphone_diagnostic())
    return diagnostics


def collect_support_bundle_diagnostics(state, *, include_runtime_health: bool = False) -> list[dict[str, Any]]:
    return serialize_startup_diagnostics(
        collect_startup_diagnostics(state, include_runtime_health=include_runtime_health)
    )
