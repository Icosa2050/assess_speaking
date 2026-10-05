"""Bounded multipart ingestion for the local practice app."""
from __future__ import annotations

import shutil
import tempfile
import subprocess
import signal
import sys
import json
from pathlib import Path

from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DISK_RESERVE_BYTES = 512 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
MAX_AUDIO_SECONDS = 20 * 60


def ensure_copy_space(destination: Path, size: int) -> None:
    if shutil.disk_usage(destination).free < DISK_RESERVE_BYTES + size:
        raise UploadRejected("Disk space ran low. Free some space and retry.", 507, "storage_error")


def validate_audio_duration(path: Path) -> None:
    """Bound decoding even for compressed files with absent/untrusted duration."""
    # Run bundled native decoding in a killable subprocess, with no PCM expansion
    # in memory or on disk. The result file contains only a bounded status object.
    with tempfile.NamedTemporaryFile(suffix=".json") as result_file:
        previous_handler = signal.getsignal(signal.SIGTERM)
        def cancel_validation(signum, _frame):
            raise SystemExit(128 + signum)
        decoder = None
        try:
            signal.signal(signal.SIGTERM, cancel_validation)
            command = [sys.executable]
            if not getattr(sys, "frozen", False):
                command.append(str(Path(__file__).resolve().parents[1] / "scripts/run_backend.py"))
            command.extend(["--validate-audio", str(path), "--max-seconds", str(MAX_AUDIO_SECONDS), "--validation-result", result_file.name])
            decoder = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            decoder.wait(timeout=90)
            try:
                result = json.loads(Path(result_file.name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ValueError("This file could not be decoded as audio. Choose a playable recording and retry.") from None
            if not result.get("ok"):
                raise ValueError(result.get("detail") or "This file could not be decoded as audio.")
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Audio decoding took too long. Export a shorter MP3 or WAV and retry.") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("The bundled audio decoder is unavailable. Reinstall Vostavo and retry.") from exc
        finally:
            try:
                if decoder is not None:
                    if decoder.poll() is None:
                        decoder.kill()
                    decoder.wait()
            finally:
                signal.signal(signal.SIGTERM, previous_handler)


class UploadRejected(MultiPartException):
    def __init__(self, message: str, status: int = 413, code: str = "validation_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def upload_limits(destination: Path) -> dict[str, int]:
    # Allow room for both the multipart spool and its durable copy, even when
    # temp and app data share a volume. Reserve space for decoding and reports.
    free = min(shutil.disk_usage(destination).free, shutil.disk_usage(tempfile.gettempdir()).free)
    available = max(0, (free - DISK_RESERVE_BYTES) // 2 - MULTIPART_OVERHEAD_BYTES)
    return {"max_bytes": MAX_UPLOAD_BYTES, "available_bytes": min(MAX_UPLOAD_BYTES, available),
            "disk_reserve_bytes": DISK_RESERVE_BYTES}


class BoundedMultipartParser(MultiPartParser):
    async def parse(self):
        try:
            return await super().parse()
        except BaseException:  # quality: allow[broad-except] close temporary spools on disconnect/cancellation, then re-raise
            # Starlette 0.46 closes spools only for MultiPartException. Also
            # release them on disconnect, cancellation, and disk write errors.
            for file in self._files_to_close_on_error:
                file.close()
            raise


async def parse_upload(request: Request, destination: Path):
    limits = upload_limits(destination)
    available = limits["available_bytes"]
    if available <= 0:
        raise UploadRejected("Not enough free disk space. Free some space and retry; your recording is still selected.", 507, "storage_error")
    body_limit = available + MULTIPART_OVERHEAD_BYTES
    length = request.headers.get("content-length")
    if length is not None:
        try:
            too_large = int(length) > body_limit or int(length) < 0
        except ValueError:
            raise UploadRejected("Invalid upload length.", 400) from None
        if too_large:
            raise UploadRejected("Audio upload exceeds the available limit. Choose a smaller recording.")
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise UploadRejected("Send one audio file as multipart/form-data.", 415)

    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > body_limit:
                raise UploadRejected("Audio upload exceeds the available limit. Choose a smaller recording.")
            if min(shutil.disk_usage(destination).free, shutil.disk_usage(tempfile.gettempdir()).free) < DISK_RESERVE_BYTES:
                raise UploadRejected("Disk space ran low. Free some space and retry.", 507, "storage_error")
            yield chunk

    parser = BoundedMultipartParser(request.headers, bounded_stream(), max_files=1, max_fields=0)
    form = await parser.parse()
    file = form.get("file")
    if file is None or not hasattr(file, "file") or len(form.multi_items()) != 1:
        await form.close()
        raise UploadRejected("Choose one audio file.", 400)
    if not file.size or file.size > available:
        await form.close()
        raise UploadRejected("Uploaded audio file is empty." if not file.size else "Audio file exceeds the available limit.", 400 if not file.size else 413)
    return form, file
