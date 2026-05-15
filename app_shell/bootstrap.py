from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import sys
from pathlib import Path
from typing import Literal, TypeVar, cast

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
DEPLOYMENT_MODE_ENV_VAR = "VOSTAVO_DEPLOYMENT_MODE"
LAUNCH_MODE_ENV_VAR = "VOSTAVO_LAUNCH_MODE"
AUTH_MODE_ENV_VAR = "VOSTAVO_AUTH_MODE"
WHISPER_CACHE_DIR_ENV_VAR = "WHISPER_CACHE_DIR"

DeploymentMode = Literal["local", "hosted"]
LaunchMode = Literal["repo", "packaged"]
RuntimeAuthMode = Literal["guest", "optional", "required"]
TEnvChoice = TypeVar("TEnvChoice", bound=str)


@dataclass(frozen=True)
class RuntimeMetadata:
    deployment_mode: DeploymentMode
    launch_mode: LaunchMode
    packaging_safe: bool
    auth_mode: RuntimeAuthMode

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


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


def _resolve_env_choice(env_name: str, *, allowed: tuple[TEnvChoice, ...], default: TEnvChoice) -> TEnvChoice:
    raw = str(os.environ.get(env_name) or "").strip().lower()
    return cast(TEnvChoice, raw) if raw in allowed else default


def _resolved_writable_roots(paths: AppDataPaths) -> tuple[Path, ...]:
    return (
        paths.root,
        paths.cache_root,
        paths.reports_dir,
        paths.jobs_dir,
        paths.logs_dir,
        paths.recordings_dir,
        paths.uploads_dir,
        paths.temp_dir,
        paths.whisper_cache_dir,
    )


def build_runtime_metadata(paths: AppDataPaths) -> RuntimeMetadata:
    deployment_mode = _resolve_env_choice(
        DEPLOYMENT_MODE_ENV_VAR,
        allowed=("local", "hosted"),
        default="local",
    )
    launch_mode = _resolve_env_choice(
        LAUNCH_MODE_ENV_VAR,
        allowed=("repo", "packaged"),
        default="repo",
    )
    auth_mode = _resolve_env_choice(
        AUTH_MODE_ENV_VAR,
        allowed=("guest", "optional", "required"),
        default="guest",
    )
    packaging_safe = not any(is_within_project_checkout(path) for path in _resolved_writable_roots(paths))
    return RuntimeMetadata(
        deployment_mode=deployment_mode,
        launch_mode=launch_mode,
        packaging_safe=packaging_safe,
        auth_mode=auth_mode,
    )


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
    explicit_whisper_cache_override = _has_explicit_override(whisper_cache_dir, WHISPER_CACHE_DIR_ENV_VAR)
    if app_data_dir is not None and str(app_data_dir).strip():
        _set_compat_envs(app_data_root=Path(app_data_dir).expanduser().resolve())
    if cache_dir is not None and str(cache_dir).strip():
        _set_compat_envs(cache_root=Path(cache_dir).expanduser().resolve())
    paths = build_app_data_paths(log_dir)
    _guard_default_root_outside_project(paths.root, explicit_override=explicit_app_data_override, label="app-data")
    _guard_default_root_outside_project(paths.cache_root, explicit_override=explicit_cache_override, label="cache")
    paths = ensure_app_data_dirs(paths)
    whisper_cache_choice = str(whisper_cache_dir or os.environ.get(WHISPER_CACHE_DIR_ENV_VAR) or "").strip()
    resolved_whisper_cache_dir = (
        Path(whisper_cache_choice).expanduser().resolve()
        if whisper_cache_choice
        else paths.whisper_cache_dir
    )
    _guard_default_root_outside_project(
        resolved_whisper_cache_dir,
        explicit_override=explicit_whisper_cache_override,
        label="whisper cache",
    )
    resolved_whisper_cache_dir.mkdir(parents=True, exist_ok=True)

    _set_compat_envs(app_data_root=paths.root, cache_root=paths.cache_root, default_only=True)
    _set_default_env("HF_HOME", str(paths.cache_root / "huggingface"))
    _set_default_env("XDG_CACHE_HOME", str(paths.cache_root))
    _set_default_env("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    _set_default_env("STREAMLIT_SERVER_HEADLESS", "true")
    if explicit_whisper_cache_override:
        os.environ[WHISPER_CACHE_DIR_ENV_VAR] = str(resolved_whisper_cache_dir)
    else:
        _set_default_env(WHISPER_CACHE_DIR_ENV_VAR, str(resolved_whisper_cache_dir))

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
