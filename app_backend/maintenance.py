from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app_backend.config import (
    BACKEND_LOG_FILENAME,
    DEFAULT_JOB_METADATA_RETENTION_DAYS,
    DEFAULT_SUPPORT_BUNDLE_RETENTION_HOURS,
    BackendRuntimeConfig,
    SUPPORT_BUNDLE_DIRNAME,
)
from app_backend.contracts import CleanupTarget, MaintenanceCleanupResponse
from app_backend.jobs import prunable_job_metadata_files


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def support_bundle_dir(runtime_config: BackendRuntimeConfig) -> Path:
    return runtime_config.app_data.temp_dir / SUPPORT_BUNDLE_DIRNAME


def _normalize_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def temporary_cleanup_candidates(runtime_config: BackendRuntimeConfig) -> list[Path]:
    bundle_root = support_bundle_dir(runtime_config)
    return [
        path
        for path in _iter_files(runtime_config.app_data.temp_dir)
        if not _is_within(path, bundle_root)
    ]


def expired_support_bundle_candidates(
    runtime_config: BackendRuntimeConfig,
    *,
    now: datetime | None = None,
    retention_hours: int = DEFAULT_SUPPORT_BUNDLE_RETENTION_HOURS,
) -> list[Path]:
    bundle_root = support_bundle_dir(runtime_config)
    cutoff = _normalize_now(now) - timedelta(hours=max(int(retention_hours), 0))
    candidates: list[Path] = []
    for path in _iter_files(bundle_root):
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        if modified_at <= cutoff:
            candidates.append(path)
    return sorted(candidates)


def prunable_log_files(runtime_config: BackendRuntimeConfig) -> list[Path]:
    active_log = runtime_config.log_file.resolve()
    candidates: list[Path] = []
    for path in _iter_files(runtime_config.app_data.logs_dir):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved == active_log:
            continue
        candidates.append(path)
    return sorted(candidates)


def cleanup_candidates(
    runtime_config: BackendRuntimeConfig,
    target: CleanupTarget,
    *,
    now: datetime | None = None,
) -> list[Path]:
    if target == CleanupTarget.TMP:
        return temporary_cleanup_candidates(runtime_config)
    if target == CleanupTarget.JOBS:
        return prunable_job_metadata_files(
            runtime_config.jobs_dir,
            retention_days=DEFAULT_JOB_METADATA_RETENTION_DAYS,
            now=now,
        )
    if target == CleanupTarget.LOGS:
        return prunable_log_files(runtime_config)
    if target == CleanupTarget.ALL_SAFE:
        combined = (
            temporary_cleanup_candidates(runtime_config)
            + expired_support_bundle_candidates(runtime_config, now=now)
            + prunable_job_metadata_files(
                runtime_config.jobs_dir,
                retention_days=DEFAULT_JOB_METADATA_RETENTION_DAYS,
                now=now,
            )
            + prunable_log_files(runtime_config)
        )
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in combined:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return sorted(unique)
    return []


def execute_cleanup(
    runtime_config: BackendRuntimeConfig,
    target: CleanupTarget,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> MaintenanceCleanupResponse:
    candidates = cleanup_candidates(runtime_config, target, now=now)
    freed_bytes = sum(path.stat().st_size for path in candidates if path.exists())
    if not dry_run:
        for path in candidates:
            try:
                path.unlink()
            except OSError:
                continue
    return MaintenanceCleanupResponse(
        target=target,
        dry_run=dry_run,
        deleted_file_count=len(candidates),
        freed_bytes=freed_bytes,
        warnings=[],
    )


def startup_cleanup_targets() -> tuple[CleanupTarget, ...]:
    return (CleanupTarget.ALL_SAFE,)


def is_active_backend_log(path: Path) -> bool:
    return path.name == BACKEND_LOG_FILENAME
