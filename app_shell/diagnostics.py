from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from assessment_runtime.asr import describe_model_availability
from assessment_runtime.llm_client import health_check as llm_health_check
from app_shell.bootstrap import PROJECT_ROOT, bootstrap_app_environment, is_within_project_checkout
from app_shell.runtime_providers import requires_api_key
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime

DiagnosticStatus = Literal["ok", "warning", "error", "info"]


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
            detail_args={"path": str(paths.reports_dir), "detail": str(exc)},
        )
    return StartupDiagnostic(
        key="app_data",
        status="ok",
        title_key="diagnostics.app_data_title",
        detail_key="diagnostics.app_data_ok_detail",
        detail_args={"path": str(paths.reports_dir)},
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

    runtime = resolve_connection_runtime(connection)
    diagnostics = [
        StartupDiagnostic(
            key="runtime",
            status="ok",
            title_key="diagnostics.runtime_title",
            detail_key="diagnostics.runtime_ok_detail",
            detail_args={"provider": runtime.provider, "model": runtime.model},
        )
    ]
    if requires_api_key(runtime.provider) and not runtime.api_key:
        diagnostics.append(
            StartupDiagnostic(
                key="runtime_api_key",
                status="error",
                title_key="diagnostics.runtime_api_key_title",
                detail_key="diagnostics.runtime_api_key_error_detail",
                detail_args={"provider": runtime.provider},
            )
        )
    elif requires_api_key(runtime.provider):
        diagnostics.append(
            StartupDiagnostic(
                key="runtime_api_key",
                status="ok",
                title_key="diagnostics.runtime_api_key_title",
                detail_key="diagnostics.runtime_api_key_ok_detail",
                detail_args={"provider": runtime.provider},
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
    diagnostics.extend(_runtime_diagnostics(state, include_runtime_health=include_runtime_health))
    diagnostics.append(_microphone_diagnostic())
    return diagnostics
