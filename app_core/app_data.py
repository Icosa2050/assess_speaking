from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

APP_DATA_HOME_ENV_VAR = "VOSTAVO_HOME"
APP_CACHE_HOME_ENV_VAR = "VOSTAVO_CACHE_HOME"
LEGACY_APP_DATA_HOME_ENV_VAR = "SPEAKING_STUDIO_HOME"
LEGACY_APP_CACHE_HOME_ENV_VAR = "SPEAKING_STUDIO_CACHE_HOME"
APP_AUTHOR = "frommherz_it"
APP_DATA_DIRNAME = "Vostavo"
LEGACY_APP_DATA_DIRNAME = "Speaking Studio"
ROOT_MARKER_FILENAME = ".vostavo-root"
DEFAULT_REPORTS_DIRNAME = "reports"
DEFAULT_JOBS_DIRNAME = "jobs"
DEFAULT_LOGS_DIRNAME = "logs"
DEFAULT_RECORDINGS_DIRNAME = "recordings"
DEFAULT_UPLOADS_DIRNAME = "uploads"
DEFAULT_TEMP_DIRNAME = "tmp"
DEFAULT_WHISPER_CACHE_DIRNAME = "whisper"


@dataclass(frozen=True)
class AppDataPaths:
    root: Path
    reports_dir: Path
    jobs_dir: Path
    logs_dir: Path
    recordings_dir: Path
    uploads_dir: Path
    temp_dir: Path
    cache_root: Path
    whisper_cache_dir: Path


def _platform_dirs() -> PlatformDirs:
    return PlatformDirs(appname=APP_DATA_DIRNAME, appauthor=APP_AUTHOR, roaming=False, opinion=True, ensure_exists=False)


def _legacy_platform_data_home() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))


def _legacy_platform_cache_home() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))


def _default_app_data_root() -> Path:
    return Path(_platform_dirs().user_data_path).expanduser().resolve()


def _default_cache_root() -> Path:
    return Path(_platform_dirs().user_cache_path).expanduser().resolve()


def _legacy_default_app_data_root() -> Path:
    return (_legacy_platform_data_home() / LEGACY_APP_DATA_DIRNAME).expanduser().resolve()


def _legacy_default_cache_root() -> Path:
    return (_legacy_platform_cache_home() / LEGACY_APP_DATA_DIRNAME).expanduser().resolve()


def _env_override(*names: str) -> Path | None:
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return Path(value).expanduser().resolve()
    return None


def _root_marker(path: Path) -> Path:
    return path / ROOT_MARKER_FILENAME


def _is_initialized_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if _root_marker(path).exists():
        return True
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False
    except OSError:
        return False


def _resolve_default_root(*, new_root: Path, legacy_root: Path) -> Path:
    if _root_marker(new_root).exists():
        return new_root
    if _root_marker(legacy_root).exists():
        return legacy_root

    legacy_initialized = _is_initialized_root(legacy_root)
    new_initialized = _is_initialized_root(new_root)

    if legacy_initialized:
        return legacy_root
    if new_initialized:
        return new_root
    return new_root


def resolve_app_data_root(root: str | Path | None = None) -> Path:
    if root is not None and str(root).strip():
        return Path(root).expanduser().resolve()
    env_override = _env_override(APP_DATA_HOME_ENV_VAR, LEGACY_APP_DATA_HOME_ENV_VAR)
    if env_override is not None:
        return env_override
    return _resolve_default_root(new_root=_default_app_data_root(), legacy_root=_legacy_default_app_data_root())


def resolve_cache_root(root: str | Path | None = None) -> Path:
    if root is not None and str(root).strip():
        return Path(root).expanduser().resolve()
    env_override = _env_override(APP_CACHE_HOME_ENV_VAR)
    if env_override is not None:
        return env_override
    if _env_override(APP_DATA_HOME_ENV_VAR, LEGACY_APP_DATA_HOME_ENV_VAR) is not None:
        return resolve_app_data_root() / "cache"
    env_override = _env_override(LEGACY_APP_CACHE_HOME_ENV_VAR)
    if env_override is not None:
        return env_override
    return _resolve_default_root(new_root=_default_cache_root(), legacy_root=_legacy_default_cache_root())


def resolve_reports_dir(log_dir: str | Path | None = None) -> Path:
    if log_dir is None or not str(log_dir).strip():
        return (resolve_app_data_root() / DEFAULT_REPORTS_DIRNAME).resolve()
    candidate = Path(log_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (resolve_app_data_root() / candidate).resolve()


def build_app_data_paths(log_dir: str | Path | None = None) -> AppDataPaths:
    reports_dir = resolve_reports_dir(log_dir)
    root = resolve_app_data_root()
    cache_root = resolve_cache_root()
    return AppDataPaths(
        root=root,
        reports_dir=reports_dir,
        jobs_dir=root / DEFAULT_JOBS_DIRNAME,
        logs_dir=root / DEFAULT_LOGS_DIRNAME,
        recordings_dir=reports_dir / DEFAULT_RECORDINGS_DIRNAME,
        uploads_dir=reports_dir / DEFAULT_UPLOADS_DIRNAME,
        temp_dir=root / DEFAULT_TEMP_DIRNAME,
        cache_root=cache_root,
        whisper_cache_dir=cache_root / DEFAULT_WHISPER_CACHE_DIRNAME,
    )


def ensure_app_data_dirs(paths: AppDataPaths) -> AppDataPaths:
    for directory in (
        paths.root,
        paths.reports_dir,
        paths.jobs_dir,
        paths.logs_dir,
        paths.recordings_dir,
        paths.uploads_dir,
        paths.temp_dir,
        paths.cache_root,
        paths.whisper_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _root_marker(paths.root).touch(exist_ok=True)
    return paths
