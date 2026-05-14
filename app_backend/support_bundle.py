from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
import platform
from pathlib import Path
import re
from typing import Any
from uuid import uuid4
import zipfile

from app_backend.config import BackendRuntimeConfig
from app_backend.contracts import (
    MaintenanceStorageResponse,
    StorageAreaSummary,
    SupportBundleCreateRequest,
    SupportBundleCreateResponse,
)
from app_shell.bootstrap import build_runtime_metadata

SUPPORT_BUNDLE_RETENTION_HOURS = 24
RECENT_JOB_LIMIT = 20
APP_VERSION = "0.1.0"
REDACTED_VALUE = "[redacted]"
BUNDLE_ID_PATTERN = re.compile(r"^bundle_[A-Za-z0-9]+$")
logger = logging.getLogger(__name__)
_SECRET_VALUE_FIELD_TOKENS = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
)
_SECRET_REF_FIELD = "secret_ref"
_TEXT_REDACTION_PATTERNS = (
    (
        re.compile(r'(?im)("?(?:llm_api_key|openrouter_api_key|ollama_api_key|api_key|authorization|password|token)"?\s*[:=]\s*")([^"\n]*)(")'),
        lambda match: f'{match.group(1)}{REDACTED_VALUE}{match.group(3)}',
    ),
    (
        re.compile(r'(?im)("?secret_ref"?\s*[:=]\s*")([^"\n]*)(")'),
        lambda match: "",
    ),
)


@dataclass
class RedactionStats:
    removed_secret_refs: int = 0
    redacted_secret_values: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "removed_secret_refs": self.removed_secret_refs,
            "redacted_secret_values": self.redacted_secret_values,
        }


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if not path.is_symlink() and path.is_file()]


def _storage_area_summary(path: Path) -> StorageAreaSummary:
    files = _iter_files(path)
    return StorageAreaSummary(
        path=str(path),
        size_bytes=sum(file.stat().st_size for file in files),
        file_count=len(files),
    )


def build_storage_summary(runtime_config: BackendRuntimeConfig) -> MaintenanceStorageResponse:
    app_data = runtime_config.app_data
    areas = {
        "jobs": _storage_area_summary(app_data.jobs_dir),
        "logs": _storage_area_summary(app_data.logs_dir),
        "reports": _storage_area_summary(app_data.reports_dir),
        "recordings": _storage_area_summary(app_data.recordings_dir),
        "uploads": _storage_area_summary(app_data.uploads_dir),
        "tmp": _storage_area_summary(app_data.temp_dir),
        "cache": _storage_area_summary(app_data.cache_root),
    }
    return MaintenanceStorageResponse(
        app_data_root=str(app_data.root),
        cache_root=str(app_data.cache_root),
        areas=areas,
    )


def support_bundle_dir(runtime_config: BackendRuntimeConfig) -> Path:
    bundle_dir = runtime_config.app_data.temp_dir / "support-bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def support_bundle_path(runtime_config: BackendRuntimeConfig, bundle_id: str) -> Path:
    candidate = str(bundle_id or "").strip()
    if not BUNDLE_ID_PATTERN.fullmatch(candidate):
        raise ValueError("Invalid support bundle id.")
    bundle_root = support_bundle_dir(runtime_config).resolve()
    bundle_path = (bundle_root / f"{candidate}.zip").resolve()
    bundle_path.relative_to(bundle_root)
    return bundle_path


def _looks_like_secret_value_field(key: str) -> bool:
    lowered = key.strip().lower()
    if lowered == _SECRET_REF_FIELD:
        return False
    return any(token in lowered for token in _SECRET_VALUE_FIELD_TOKENS)


def _sanitize_text(content: str, stats: RedactionStats) -> str:
    redacted = content
    for pattern, replacer in _TEXT_REDACTION_PATTERNS:
        matches = list(pattern.finditer(redacted))
        if not matches:
            continue
        for match in matches:
            key = match.group(1).lower()
            if "secret_ref" in key:
                stats.removed_secret_refs += 1
            else:
                stats.redacted_secret_values += 1
        redacted = pattern.sub(replacer, redacted)
    return redacted


def _sanitize_for_bundle(value: Any, stats: RedactionStats) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            text_key = str(key)
            if text_key.strip().lower() == _SECRET_REF_FIELD:
                stats.removed_secret_refs += 1
                continue
            if _looks_like_secret_value_field(text_key):
                if child in (None, ""):
                    sanitized[text_key] = ""
                else:
                    sanitized[text_key] = REDACTED_VALUE
                    stats.redacted_secret_values += 1
                continue
            sanitized[text_key] = _sanitize_for_bundle(child, stats)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_bundle(item, stats) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_bundle(item, stats) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, (dict, list)) else None


def _write_json(archive: zipfile.ZipFile, name: str, payload: Any) -> None:
    archive.writestr(name, json.dumps(payload, ensure_ascii=False, indent=2))


def _backend_state_payload(runtime_config: BackendRuntimeConfig) -> dict[str, Any]:
    payload = _read_json(runtime_config.state_file)
    if isinstance(payload, dict):
        return payload
    return {
        "host": runtime_config.host,
        "port": runtime_config.port,
        "base_url": runtime_config.base_url,
        "app_data_root": str(runtime_config.app_data.root),
        "reports_dir": str(runtime_config.app_data.reports_dir),
        "jobs_dir": str(runtime_config.jobs_dir),
        "log_file": str(runtime_config.log_file),
    }


def _backend_diagnostics_payload(runtime_config: BackendRuntimeConfig) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for key, path in (
        ("state_file", runtime_config.state_file),
        ("jobs_dir", runtime_config.jobs_dir),
        ("log_file", runtime_config.log_file),
        ("temp_dir", runtime_config.app_data.temp_dir),
    ):
        diagnostics.append(
            {
                "key": key,
                "status": "ok" if path.exists() else "warning",
                "path": str(path),
                "exists": path.exists(),
            }
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "items": diagnostics,
    }


def _recent_job_files(runtime_config: BackendRuntimeConfig) -> list[Path]:
    files = [path for path in runtime_config.jobs_dir.glob("*.json") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:RECENT_JOB_LIMIT]


def _add_json_file_to_archive(
    archive: zipfile.ZipFile,
    *,
    entry_name: str,
    payload: Any,
    stats: RedactionStats,
) -> None:
    _write_json(archive, entry_name, _sanitize_for_bundle(payload, stats))


def _add_recent_jobs(archive: zipfile.ZipFile, runtime_config: BackendRuntimeConfig, stats: RedactionStats) -> None:
    for job_file in _recent_job_files(runtime_config):
        payload = _read_json(job_file)
        if payload is None:
            continue
        entry_name = f"jobs/{job_file.name}"
        _add_json_file_to_archive(archive, entry_name=entry_name, payload=payload, stats=stats)


def _add_text_file_to_archive(
    archive: zipfile.ZipFile,
    *,
    entry_name: str,
    source: Path,
    stats: RedactionStats,
) -> None:
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping support bundle text file %s: %s", source, exc)
        return
    archive.writestr(entry_name, _sanitize_text(content, stats))


def _relative_entry(root: Path, path: Path, *, folder_name: str) -> str:
    relative_path = path.relative_to(root).as_posix()
    return f"{folder_name}/{relative_path}"


def _add_optional_tree(
    archive: zipfile.ZipFile,
    *,
    root: Path,
    folder_name: str,
    stats: RedactionStats,
) -> None:
    for path in _iter_files(root):
        entry_name = _relative_entry(root, path, folder_name=folder_name)
        if path.suffix.lower() in {".json", ".log", ".txt", ".csv"}:
            payload = _read_json(path)
            if payload is not None:
                _add_json_file_to_archive(archive, entry_name=entry_name, payload=payload, stats=stats)
            else:
                _add_text_file_to_archive(archive, entry_name=entry_name, source=path, stats=stats)
            continue
        try:
            archive.write(path, arcname=entry_name)
        except OSError as exc:
            logger.warning("Skipping support bundle file %s: %s", path, exc)


def create_support_bundle(
    runtime_config: BackendRuntimeConfig,
    request: SupportBundleCreateRequest,
) -> SupportBundleCreateResponse:
    bundle_id = f"bundle_{uuid4().hex[:12]}"
    bundle_path = support_bundle_path(runtime_config, bundle_id)
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(hours=SUPPORT_BUNDLE_RETENTION_HOURS)
    stats = RedactionStats()
    runtime_metadata = build_runtime_metadata(runtime_config.app_data).as_dict()

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_json(
            archive,
            "runtime_metadata.json",
            {
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                },
                "app": {
                    "name": "Vostavo Local Backend",
                    "version": APP_VERSION,
                },
                "runtime_metadata": runtime_metadata,
            },
        )
        _add_json_file_to_archive(
            archive,
            entry_name="backend_state.json",
            payload=_backend_state_payload(runtime_config),
            stats=stats,
        )
        _write_json(archive, "backend_diagnostics.json", _backend_diagnostics_payload(runtime_config))
        _add_json_file_to_archive(
            archive,
            entry_name="shell/client_snapshot.json",
            payload=request.client_snapshot,
            stats=stats,
        )
        _add_json_file_to_archive(
            archive,
            entry_name="shell/client_diagnostics.json",
            payload=request.client_diagnostics,
            stats=stats,
        )
        archive.writestr(
            "storage_summary.json",
            build_storage_summary(runtime_config).model_dump_json(indent=2),
        )
        if runtime_config.log_file.exists():
            _add_text_file_to_archive(
                archive,
                entry_name="logs/backend.log",
                source=runtime_config.log_file,
                stats=stats,
            )
        _add_recent_jobs(archive, runtime_config, stats)
        if request.include_reports:
            _add_optional_tree(
                archive,
                root=runtime_config.app_data.reports_dir,
                folder_name="reports",
                stats=stats,
            )
        if request.include_recordings:
            _add_optional_tree(
                archive,
                root=runtime_config.app_data.recordings_dir,
                folder_name="recordings",
                stats=stats,
            )
        if request.include_uploads:
            _add_optional_tree(
                archive,
                root=runtime_config.app_data.uploads_dir,
                folder_name="uploads",
                stats=stats,
            )
        _write_json(
            archive,
            "manifest.json",
            {
                "bundle_id": bundle_id,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "include_reports": request.include_reports,
                "include_recordings": request.include_recordings,
                "include_uploads": request.include_uploads,
                "runtime_metadata": runtime_metadata,
                "redaction": {
                    **stats.as_dict(),
                    "secret_ref_policy": "full_redaction",
                },
            },
        )

    return SupportBundleCreateResponse(
        bundle_id=bundle_id,
        filename=bundle_path.name,
        size_bytes=bundle_path.stat().st_size,
        expires_at=expires_at,
    )
