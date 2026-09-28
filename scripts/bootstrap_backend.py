#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_backend.lifecycle import build_desktop_bootstrap_env


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start or reuse the local backend and print desktop bridge env vars.")
    parser.add_argument("--app-data-dir", default="", help="Override the local app-data root.")
    parser.add_argument("--cache-dir", default="", help="Override the local cache root.")
    parser.add_argument("--log-dir", default="", help="Override the reports/history directory.")
    return parser.parse_args(argv)


def build_bootstrap_env(args: argparse.Namespace) -> dict[str, str]:
    return build_desktop_bootstrap_env(
        log_dir=args.log_dir or None,
        app_data_dir=args.app_data_dir or None,
        cache_dir=args.cache_dir or None,
    )


def format_bootstrap_env(bootstrap_env: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in bootstrap_env.items())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(format_bootstrap_env(build_bootstrap_env(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
