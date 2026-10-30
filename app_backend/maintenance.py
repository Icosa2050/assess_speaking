from __future__ import annotations

from datetime import UTC, datetime, timedelta
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import re

from app_backend.config import (
    BACKEND_LOG_FILENAME,
    DEFAULT_JOB_METADATA_RETENTION_DAYS,
    DEFAULT_SUPPORT_BUNDLE_RETENTION_HOURS,
    BackendRuntimeConfig,
    SUPPORT_BUNDLE_DIRNAME,
)
from app_backend.contracts import CleanupTarget, MaintenanceCleanupResponse
from app_backend.jobs import prunable_job_metadata_files, unreferenced_job_stage_directories
from app_core.cloud_policy import locked_file

logger = logging.getLogger(__name__)
JOB_STAGING_RETENTION_HOURS = 24
# Encrypted desktop receipts cannot be inspected by the helper. Their take
# expires after seven days, so an upload younger than eight days is protected.
ORPHAN_UPLOAD_RETENTION_DAYS = 8


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
            logger.warning("Could not inspect support bundle candidate %s.", path, exc_info=True)
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


def orphaned_job_staging_candidates(runtime_config: BackendRuntimeConfig, *, now: datetime | None = None) -> list[Path]:
    # A terminated worker can leave the private file used for atomic status
    # replacement. Keep recent files because a live writer does not hold the
    # retention lock and might still be staging its replacement.
    cutoff = _normalize_now(now) - timedelta(hours=JOB_STAGING_RETENTION_HOURS)
    candidates: list[Path] = []
    for path in runtime_config.jobs_dir.glob(".*.json.*.tmp"):
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            logger.warning("Could not inspect job staging candidate %s.", path, exc_info=True)
            continue
        if path.is_file() and modified_at <= cutoff:
            candidates.append(path)
    return sorted(candidates)


def orphan_upload_candidates(runtime_config: BackendRuntimeConfig, *, now: datetime | None = None) -> list[Path]:
    """Keep uploads while any retained job or journal entry can still name them."""
    root = runtime_config.app_data.uploads_dir
    cutoff = _normalize_now(now) - timedelta(days=ORPHAN_UPLOAD_RETENTION_DAYS)
    references: set[str] = set()
    for job_file in runtime_config.jobs_dir.glob("*.json"):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
            if not isinstance(job, dict):
                return []
        except (OSError, ValueError):
            return []
        request = job.get("request") or {}
        if not isinstance(request, dict):
            return []
        references.add(str(request.get("audio_id") or ""))
        references.add(str(job.get("audio_path") or ""))
    # Reports and restored journal rows can outlive job metadata. A text match
    # is conservative and also recognizes older report schemas.
    journal_ids: set[str] = set()
    audio_id_pattern = re.compile(r"aud_[0-9a-f]{32}")
    for path in runtime_config.app_data.reports_dir.rglob("*"):
        if path.is_relative_to(root):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        try:
            with path.open("r", encoding="utf-8") as source:
                tail = ""
                while chunk := source.read(64 * 1024):
                    journal_ids.update(audio_id_pattern.findall(tail + chunk))
                    tail = chunk[-64:]
        except (OSError, UnicodeError):
            return []
    result: list[Path] = []
    for sidecar in root.glob("aud_*.json"):
        if sidecar.is_symlink():
            continue
        try:
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(str(saved.get("created_at") or "").replace("Z", "+00:00"))
            audio_id = str(saved.get("audio_id") or "")
            media = Path(str(saved.get("stored_path") or ""))
            if (created.tzinfo is None or created.astimezone(UTC) > cutoff or
                audio_id != sidecar.stem or media.is_symlink() or (media.exists() and not media.is_file()) or
                media.parent.resolve() != root.resolve() or not media.name.startswith(audio_id + ".")):
                continue
        except (OSError, ValueError, TypeError, AttributeError):
            # Old or malformed sidecars have no trustworthy age or identity.
            continue
        if audio_id in references or str(media.resolve()) in references or audio_id in journal_ids:
            continue
        if media.is_file():
            result.append(media)
        result.append(sidecar)
    for media in root.glob("aud_*.*"):
        if media.suffix == ".json" or media.is_symlink() or not media.is_file() or not re.fullmatch(r"aud_[0-9a-f]{32}", media.stem):
            continue
        if (root / f"{media.stem}.json").exists():
            continue
        try:
            modified = datetime.fromtimestamp(media.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        if modified <= cutoff and media.stem not in references and str(media.resolve()) not in references and media.stem not in journal_ids:
            result.append(media)
    return sorted(result)


def cleanup_candidates(
    runtime_config: BackendRuntimeConfig,
    target: CleanupTarget,
    *,
    now: datetime | None = None,
) -> list[Path]:
    if target == CleanupTarget.TMP:
        return temporary_cleanup_candidates(runtime_config)
    if target == CleanupTarget.JOBS:
        return sorted(
            prunable_job_metadata_files(runtime_config.jobs_dir, retention_days=DEFAULT_JOB_METADATA_RETENTION_DAYS, now=now)
            + orphaned_job_staging_candidates(runtime_config, now=now)
            + orphan_upload_candidates(runtime_config, now=now)
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
            + orphaned_job_staging_candidates(runtime_config, now=now)
            + orphan_upload_candidates(runtime_config, now=now)
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


def _delete_candidates(candidates: list[Path], *, dry_run: bool) -> tuple[int, int, list[str]]:
    deleted_file_count = 0
    freed_bytes = 0
    warnings: list[str] = []
    for path in candidates:
        try:
            size_bytes = (path.lstat() if path.is_symlink() else path.stat()).st_size
        except OSError:
            logger.warning("Could not inspect cleanup candidate %s.", path, exc_info=True)
            warnings.append(f"Could not inspect cleanup candidate: {path}")
            continue
        if dry_run:
            deleted_file_count += 1
            freed_bytes += size_bytes
            continue
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not delete cleanup candidate %s.", path, exc_info=True)
            warnings.append(f"Could not delete cleanup candidate: {path}")
            continue
        deleted_file_count += 1
        freed_bytes += size_bytes
    return deleted_file_count, freed_bytes, warnings


def execute_cleanup(
    runtime_config: BackendRuntimeConfig,
    target: CleanupTarget,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> MaintenanceCleanupResponse:
    prune_jobs = target in {CleanupTarget.JOBS, CleanupTarget.ALL_SAFE}
    # A resume publishes another reference to a shared cache. Serialize that
    # publication with retention, including dry-run inspection.
    guard = locked_file(runtime_config.jobs_dir / "retention.lock") if prune_jobs else nullcontext()
    with guard, (locked_file(runtime_config.app_data.uploads_dir / ".upload-requests.lock") if prune_jobs else nullcontext()):
        candidates = cleanup_candidates(runtime_config, target, now=now)
        deleted_file_count, freed_bytes, warnings = _delete_candidates(candidates, dry_run=dry_run)
        if prune_jobs:
            # Re-read after successful deletion: failed metadata deletes must
            # still protect their checkpoints. Preview excludes planned deletes.
            roots = unreferenced_job_stage_directories(runtime_config.jobs_dir, excluding=candidates if dry_run else None)
            for root in roots:
                entries = list(root.rglob("*"))
                files = [path for path in entries if path.is_file() or path.is_symlink()]
                count, size, failures = _delete_candidates(files, dry_run=dry_run)
                deleted_file_count += count
                freed_bytes += size
                warnings.extend(failures)
                if not dry_run:
                    directories = [path for path in entries if path.is_dir() and not path.is_symlink()]
                    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True) + [root]:
                        try:
                            directory.rmdir()
                        except OSError:
                            logger.warning("Could not remove checkpoint directory %s.", directory, exc_info=True)
                            warnings.append(f"Could not remove checkpoint directory: {directory}")
        return MaintenanceCleanupResponse(
            target=target,
            dry_run=dry_run,
            deleted_file_count=deleted_file_count,
            freed_bytes=freed_bytes,
            warnings=warnings,
        )


def startup_cleanup_targets() -> tuple[CleanupTarget, ...]:
    return (CleanupTarget.ALL_SAFE,)


def is_active_backend_log(path: Path) -> bool:
    return path.name == BACKEND_LOG_FILENAME
