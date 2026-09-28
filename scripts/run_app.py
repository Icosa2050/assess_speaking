#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_core.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR
from app_backend.lifecycle import ensure_local_backend, get_backend_state
from app_core.bootstrap import (
    bootstrap_app_environment,
    build_runtime_metadata,
    is_within_project_checkout,
)
from scripts.bootstrap_backend import build_bootstrap_env, format_bootstrap_env


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the Vostavo local backend.")
    parser.add_argument("--app-data-dir", default="", help="Override the local app-data root.")
    parser.add_argument("--cache-dir", default="", help="Override the local cache root.")
    parser.add_argument("--log-dir", default="", help="Override the reports/history directory.")
    parser.add_argument("--check", action="store_true", help="Print launcher diagnostics as JSON and exit.")
    parser.add_argument(
        "--desktop-bootstrap",
        action="store_true",
        help="Start or reuse the local backend and print desktop bridge env vars for the Tauri shell.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print backend launcher diagnostics as JSON and exit.")
    return parser.parse_args(argv)


def _repo_local_override_targets(*, app_data_root: Path, cache_root: Path) -> list[str]:
    targets: list[str] = []
    if is_within_project_checkout(app_data_root):
        targets.append("app_data")
    if is_within_project_checkout(cache_root):
        targets.append("cache")
    return targets


def _launcher_payload(args: argparse.Namespace) -> dict[str, object]:
    if args.app_data_dir:
        os.environ[APP_DATA_HOME_ENV_VAR] = args.app_data_dir
    if args.cache_dir:
        os.environ[APP_CACHE_HOME_ENV_VAR] = args.cache_dir
    paths = bootstrap_app_environment(
        log_dir=args.log_dir or None,
        app_data_dir=args.app_data_dir or None,
        cache_dir=args.cache_dir or None,
    )
    runtime_metadata = build_runtime_metadata(paths).as_dict()
    backend_state = (
        get_backend_state(
            log_dir=args.log_dir or None,
            app_data_dir=args.app_data_dir or None,
            cache_dir=args.cache_dir or None,
        )
        if args.check or args.dry_run
        else ensure_local_backend(
            log_dir=args.log_dir or None,
            app_data_dir=args.app_data_dir or None,
            cache_dir=args.cache_dir or None,
        )
    ) or {}
    repo_local_override_targets = _repo_local_override_targets(
        app_data_root=paths.root,
        cache_root=paths.cache_root,
    )
    payload: dict[str, object] = {
        "app_data_root": str(paths.root),
        "reports_dir": str(paths.reports_dir),
        "jobs_dir": str(paths.jobs_dir),
        "logs_dir": str(paths.logs_dir),
        "cache_root": str(paths.cache_root),
        "whisper_cache_dir": str(paths.whisper_cache_dir),
        "repo_local_override_active": bool(repo_local_override_targets),
        "repo_local_override_targets": repo_local_override_targets,
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
        "backend_base_url": str(backend_state.get("base_url") or ""),
        **runtime_metadata,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.desktop_bootstrap:
        print(format_bootstrap_env(build_bootstrap_env(args)))
        return 0
    payload = _launcher_payload(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
