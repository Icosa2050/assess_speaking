#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

import json
import socket
import threading
import signal
import atexit
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _utility_mode() -> None:
    if "--validate-audio" in sys.argv:
        from assessment_runtime.media import validate_duration
        parser = argparse.ArgumentParser()
        parser.add_argument("--validate-audio", type=Path, required=True)
        parser.add_argument("--max-seconds", type=float, required=True)
        parser.add_argument("--validation-result", type=Path, required=True)
        args = parser.parse_args()
        try:
            validate_duration(args.validate_audio, args.max_seconds)
            result = {"ok": True}
        except (ValueError, OSError, ImportError) as exc:
            result = {"ok": False, "detail": str(exc)}
        args.validation_result.write_text(json.dumps(result), encoding="utf-8")
        raise SystemExit(0 if result["ok"] else 2)
    if "--self-test" in sys.argv:
        import importlib
        import certifi
        import onnxruntime
        import faster_whisper
        from assessment_runtime.media import pcm_blocks
        for module in ("app_backend.app", "assessment_runtime.responses_client", "assessment_runtime.asr", "keyring.backends.macOS", "uvicorn.protocols.http.h11_impl"):
            importlib.import_module(module)
        sample = PROJECT_ROOT / "samples/cefr/en/B1/travel_story.wav"
        vad = Path(faster_whisper.__file__).parent / "assets/silero_vad_v6.onnx"
        if not vad.exists():
            vad = next((Path(faster_whisper.__file__).parent / "assets").glob("*.onnx"))
        onnxruntime.InferenceSession(str(vad), providers=["CPUExecutionProvider"])
        assert Path(certifi.where()).is_file()
        result = json.dumps({"ok": True, "frozen": bool(getattr(sys, "frozen", False)), "sample_bytes": sum(map(len, pcm_blocks(sample))), "vad": True, "certificates": True})
        # PyInstaller windowed mode sets sys.stdout=None; preserve redirected fd.
        with os.fdopen(os.dup(1), "w") as output:
            output.write(result + "\n")
        raise SystemExit(0)


if __name__ == "__main__":
    _utility_mode()

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
    parser.add_argument("--port", type=int, required=True, help="Port to bind to; 0 allocates atomically.")
    parser.add_argument("--desktop-owned", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    return parser.parse_args(argv)


def _run_startup_cleanup(config) -> None:
    execute_cleanup(config, CleanupTarget.ALL_SAFE, dry_run=False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.app_data_dir:
        os.environ[APP_DATA_HOME_ENV_VAR] = args.app_data_dir
    if args.cache_dir:
        os.environ[APP_CACHE_HOME_ENV_VAR] = args.cache_dir
    if getattr(sys, "frozen", False) and not args.desktop_owned:
        raise RuntimeError("Packaged backend requires its desktop owner.")
    owner_input = None
    if args.desktop_owned:
        owner_input = os.fdopen(os.dup(0), "r")
        token = owner_input.readline().strip()
        media_token = owner_input.readline().strip()
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in (token, media_token)):
            raise RuntimeError("Desktop session token is required.")
        os.environ["VOSTAVO_SESSION_TOKEN"] = token
        os.environ["VOSTAVO_MEDIA_TOKEN"] = media_token
        os.environ["VOSTAVO_LAUNCH_MODE"] = "packaged" if getattr(sys, "frozen", False) else "repo"
        os.environ["VOSTAVO_DEPLOYMENT_MODE"] = "local"
        os.environ["VOSTAVO_AUTH_MODE"] = "guest"
    if getattr(sys, "frozen", False):
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bound_socket.bind((args.host, args.port))
    bound_socket.listen(128)
    config = build_backend_runtime_config(
        log_dir=args.log_dir or None,
        app_data_dir=args.app_data_dir or None,
        cache_dir=args.cache_dir or None,
        port=bound_socket.getsockname()[1],
        host=args.host,
    )
    _run_startup_cleanup(config)
    _configure_backend_logging(config.log_file)
    server = uvicorn.Server(uvicorn.Config(create_app(config), host=config.host, port=config.port, log_level="warning", log_config=None, access_log=False))
    # Uvicorn replays SIGTERM to the original handler after graceful shutdown.
    # The default handler would kill Python before our readiness cleanup runs.
    def request_shutdown(_signal=None, _frame=None):
        server.should_exit = True
    signal.signal(signal.SIGTERM, request_shutdown)
    if args.ready_file:
        pending = args.ready_file.with_suffix(".pending")
        pending.write_text(json.dumps({"port": config.port, "pid": os.getpid()}), encoding="utf-8")
        pending.replace(args.ready_file)
    if owner_input is not None:
        def watch_owner():
            owner_input.read()
            request_shutdown()
        threading.Thread(target=watch_owner, daemon=True).start()
    try:
        server.run(sockets=[bound_socket])
    finally:
        bound_socket.close()
        if args.ready_file:
            args.ready_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
