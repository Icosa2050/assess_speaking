#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_backend.app import create_app
from app_backend.config import (
    BACKEND_LOG_BACKUP_COUNT,
    BACKEND_LOG_MAX_BYTES,
    build_backend_runtime_config,
)
from app_backend.contracts import CleanupTarget
from app_backend.maintenance import execute_cleanup
from app_core.app_data import APP_CACHE_HOME_ENV_VAR, APP_DATA_HOME_ENV_VAR


class _LoggerWriter:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buffer += str(message)
        written = len(message)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line)
        return written

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.strip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False


def _configure_backend_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=BACKEND_LOG_MAX_BYTES,
        backupCount=BACKEND_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    for logger_name, level in (
        ("uvicorn", logging.INFO),
        ("uvicorn.error", logging.INFO),
        ("uvicorn.access", logging.INFO),
        ("vostavo.stdout", logging.INFO),
        ("vostavo.stderr", logging.ERROR),
    ):
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.setLevel(level)
        logger.propagate = False

    sys.stdout = _LoggerWriter(logging.getLogger("vostavo.stdout"), logging.INFO)
    sys.stderr = _LoggerWriter(logging.getLogger("vostavo.stderr"), logging.ERROR)
    atexit.register(sys.stdout.flush)
    atexit.register(sys.stderr.flush)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Vostavo local backend.")
    parser.add_argument("--app-data-dir", default="", help="Override the local app-data root.")
    parser.add_argument("--cache-dir", default="", help="Override the local cache root.")
    parser.add_argument("--log-dir", default="", help="Override the reports/history directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to.")
    return parser.parse_args(argv)


def _run_startup_cleanup(config) -> None:
    execute_cleanup(config, CleanupTarget.ALL_SAFE, dry_run=False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.app_data_dir:
        os.environ[APP_DATA_HOME_ENV_VAR] = args.app_data_dir
    if args.cache_dir:
        os.environ[APP_CACHE_HOME_ENV_VAR] = args.cache_dir
    config = build_backend_runtime_config(
        log_dir=args.log_dir or None,
        app_data_dir=args.app_data_dir or None,
        cache_dir=args.cache_dir or None,
        port=args.port,
        host=args.host,
    )
    _run_startup_cleanup(config)
    _configure_backend_logging(config.log_file)
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="warning", log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
