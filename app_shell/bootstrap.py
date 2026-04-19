from __future__ import annotations

import os
import sys
from pathlib import Path

from app_shell.app_data import (
    APP_CACHE_HOME_ENV_VAR,
    APP_DATA_HOME_ENV_VAR,
    LEGACY_APP_CACHE_HOME_ENV_VAR,
    LEGACY_APP_DATA_HOME_ENV_VAR,
    AppDataPaths,
    build_app_data_paths,
    ensure_app_data_dirs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSESS_SCRIPT = PROJECT_ROOT / "assess_speaking.py"
STREAMLIT_ENTRYPOINT = PROJECT_ROOT / "streamlit_app.py"
BACKEND_ENTRYPOINT = PROJECT_ROOT / "scripts" / "run_backend.py"


def _set_default_env(name: str, value: str) -> None:
    if not str(os.environ.get(name) or "").strip():
        os.environ[name] = value


def _set_compat_envs(*, app_data_root: Path | None = None, cache_root: Path | None = None, default_only: bool = False) -> None:
    pairs: list[tuple[str, Path]] = []
    if app_data_root is not None:
        pairs.extend(
            [
                (APP_DATA_HOME_ENV_VAR, app_data_root),
                (LEGACY_APP_DATA_HOME_ENV_VAR, app_data_root),
            ]
        )
    if cache_root is not None:
        pairs.extend(
            [
                (APP_CACHE_HOME_ENV_VAR, cache_root),
                (LEGACY_APP_CACHE_HOME_ENV_VAR, cache_root),
            ]
        )
    for name, value in pairs:
        if default_only:
            _set_default_env(name, str(value))
        else:
            os.environ[name] = str(value)


def _has_explicit_override(root: str | Path | None, *env_names: str) -> bool:
    if root is not None and str(root).strip():
        return True
    return any(str(os.environ.get(name) or "").strip() for name in env_names)


def is_within_project_checkout(path: Path) -> bool:
    candidate = path.expanduser().resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _guard_default_root_outside_project(path: Path, *, explicit_override: bool, label: str) -> None:
    if explicit_override or not is_within_project_checkout(path):
        return
    raise RuntimeError(
        f"Resolved {label} root '{path}' inside the repository checkout without an explicit override. "
        f"Use --app-data-dir/--cache-dir or the Vostavo environment variables when you intentionally want repo-local state."
    )


def bootstrap_app_environment(
    *,
    log_dir: str | Path | None = None,
    app_data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    whisper_cache_dir: str | Path | None = None,
) -> AppDataPaths:
    explicit_app_data_override = _has_explicit_override(
        app_data_dir,
        APP_DATA_HOME_ENV_VAR,
        LEGACY_APP_DATA_HOME_ENV_VAR,
    )
    explicit_cache_override = _has_explicit_override(
        cache_dir,
        APP_CACHE_HOME_ENV_VAR,
        LEGACY_APP_CACHE_HOME_ENV_VAR,
    )
    if app_data_dir is not None and str(app_data_dir).strip():
        _set_compat_envs(app_data_root=Path(app_data_dir).expanduser().resolve())
    if cache_dir is not None and str(cache_dir).strip():
        _set_compat_envs(cache_root=Path(cache_dir).expanduser().resolve())
    paths = build_app_data_paths(log_dir)
    _guard_default_root_outside_project(paths.root, explicit_override=explicit_app_data_override, label="app-data")
    _guard_default_root_outside_project(paths.cache_root, explicit_override=explicit_cache_override, label="cache")
    paths = ensure_app_data_dirs(paths)
    resolved_whisper_cache_dir = (
        Path(whisper_cache_dir).expanduser().resolve()
        if whisper_cache_dir is not None and str(whisper_cache_dir).strip()
        else paths.whisper_cache_dir
    )
    resolved_whisper_cache_dir.mkdir(parents=True, exist_ok=True)

    _set_compat_envs(app_data_root=paths.root, cache_root=paths.cache_root, default_only=True)
    _set_default_env("HF_HOME", str(paths.cache_root / "huggingface"))
    _set_default_env("XDG_CACHE_HOME", str(paths.cache_root))
    _set_default_env("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    _set_default_env("STREAMLIT_SERVER_HEADLESS", "true")
    _set_default_env("WHISPER_CACHE_DIR", str(resolved_whisper_cache_dir))

    return AppDataPaths(
        root=paths.root,
        reports_dir=paths.reports_dir,
        jobs_dir=paths.jobs_dir,
        logs_dir=paths.logs_dir,
        recordings_dir=paths.recordings_dir,
        uploads_dir=paths.uploads_dir,
        temp_dir=paths.temp_dir,
        cache_root=paths.cache_root,
        whisper_cache_dir=resolved_whisper_cache_dir,
    )


def build_streamlit_launch_command(
    *,
    python_executable: str | Path | None = None,
    entrypoint: str | Path | None = None,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    command = [
        str(Path(python_executable) if python_executable is not None else Path(sys.executable)),
        "-m",
        "streamlit",
        "run",
        str(Path(entrypoint) if entrypoint is not None else STREAMLIT_ENTRYPOINT),
    ]
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command
