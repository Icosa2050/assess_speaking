from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import threading
from typing import Any, BinaryIO
from uuid import uuid4

from app_backend.config import BackendRuntimeConfig, DEFAULT_JOB_METADATA_RETENTION_DAYS
from app_backend.contracts import (
    AssessmentCreateRequest,
    AssessmentCreateResponse,
    AssessmentStatusResponse,
    AssessmentSummary,
    ErrorCode,
    ErrorResponse,
    JobStatus,
    UploadResponse,
)
from assessment_runtime.runner import AssessmentRunRequest, execute_assessment_run
from app_backend.uploads import COPY_CHUNK_BYTES, MAX_UPLOAD_BYTES, UploadRejected, ensure_copy_space, upload_limits, validate_audio_duration

INCOMPLETE_JOB_STATUSES = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
TERMINAL_JOB_STATUSES = {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
TERMINAL_JOB_STATUSES_NORMALIZED = {status.lower() for status in TERMINAL_JOB_STATUSES}

logger = logging.getLogger(__name__)


class AssessmentBusyError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("status"), str):
        payload["status"] = payload["status"].strip().lower()
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _job_file(jobs_dir: Path, assessment_id: str) -> Path:
    return jobs_dir / f"{assessment_id}.json"


def _sanitize_request_metadata(request_payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request_payload.items() if key != "llm_api_key"}


def _upload_meta_file(uploads_dir: Path, audio_id: str) -> Path:
    return uploads_dir / f"{audio_id}.json"


def _coerce_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _job_retention_anchor(payload: dict[str, Any], path: Path) -> datetime | None:
    timestamp = _coerce_timestamp(payload.get("completed_at")) or _coerce_timestamp(payload.get("created_at"))
    if timestamp is not None:
        return timestamp
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None


def recover_incomplete_job_metadata(jobs_dir: Path) -> int:
    recovered = 0
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for job_file in jobs_dir.glob("*.json"):
        payload = _read_json(job_file)
        if payload.get("status") not in INCOMPLETE_JOB_STATUSES:
            continue
        payload["status"] = JobStatus.FAILED.value
        payload["phase"] = "failed"
        payload["progress"] = 1.0
        payload["completed_at"] = _now_iso()
        payload["error"] = ErrorResponse(
            code=ErrorCode.RUNTIME,
            detail="Backend restarted before the assessment finished.",
        ).model_dump()
        _write_json(job_file, payload)
        recovered += 1
    return recovered


def prunable_job_metadata_files(
    jobs_dir: Path,
    *,
    retention_days: int = DEFAULT_JOB_METADATA_RETENTION_DAYS,
    now: datetime | None = None,
) -> list[Path]:
    reference_now = now.astimezone(UTC) if now is not None and now.tzinfo is not None else (now.replace(tzinfo=UTC) if now is not None else datetime.now(UTC))
    cutoff = reference_now - timedelta(days=max(int(retention_days), 0))
    candidates: list[Path] = []
    if not jobs_dir.exists():
        return candidates
    for job_file in jobs_dir.glob("*.json"):
        payload = _read_json(job_file)
        if str(payload.get("status") or "").strip().lower() not in TERMINAL_JOB_STATUSES_NORMALIZED:
            continue
        anchor = _job_retention_anchor(payload, job_file)
        if anchor is not None and anchor <= cutoff:
            candidates.append(job_file)
    return sorted(candidates)


def _error_code_from_detail(detail: str, *, provider: str = "") -> ErrorCode:
    lowered = str(detail or "").lower()
    if "audio" in lowered and any(value in lowered for value in ("decoder", "decode", "conversion", "chunking")):
        return ErrorCode.MEDIA_DECODE_ERROR
    if "whisper" in lowered and ("missing" in lowered or "cache" in lowered or "download" in lowered):
        return ErrorCode.MISSING_WHISPER_MODEL
    if provider in {"ollama", "lmstudio"}:
        if any(token in lowered for token in ("connection refused", "not reachable", "not running", "failed to connect")):
            return ErrorCode.LOCAL_PROVIDER_NOT_RUNNING
        if any(token in lowered for token in ("command not found", "no such file", "not installed")):
            return ErrorCode.LOCAL_PROVIDER_NOT_INSTALLED
    if any(token in lowered for token in ("missing", "invalid", "unsupported", "enter a model name")):
        return ErrorCode.VALIDATION
    if any(token in lowered for token in ("history.csv schema", "permission denied", "read-only", "storage")):
        return ErrorCode.STORAGE
    return ErrorCode.RUNTIME


def _build_summary(payload: dict[str, Any] | None) -> AssessmentSummary | None:
    if not isinstance(payload, dict):
        return None
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    scores = report.get("scores") if isinstance(report.get("scores"), dict) else {}
    coaching = report.get("coaching") if isinstance(report.get("coaching"), dict) else {}
    return AssessmentSummary(
        score_overall=scores.get("final"),
        band=str(scores.get("band") or ""),
        next_focus=str(coaching.get("next_focus") or ""),
    )


def _update_job_file(
    path: Path,
    *,
    status: str | None = None,
    phase: str | None = None,
    progress: float | None = None,
    error: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    current = _read_json(path)
    if status is not None:
        current["status"] = status
    if phase is not None:
        current["phase"] = phase
    if progress is not None:
        current["progress"] = progress
    if error is not None or "error" in extra:
        current["error"] = error
    current.update(extra)
    _write_json(path, current)
    return current


def _job_worker(job_file: str, request_payload: dict[str, Any], audio_path: str, credentials=None) -> None:
    path = Path(job_file)
    _update_job_file(
        path,
        status=JobStatus.RUNNING.value,
        phase="analyzing_audio",
        progress=0.0,
        started_at=_now_iso(),
        worker_pid=os.getpid(),
        error=None,
    )

    def _status_callback(phase: str) -> None:
        _update_job_file(
            path,
            status=JobStatus.RUNNING.value,
            phase=phase,
            progress=0.0,
            error=None,
        )

    def current_access_token():
        from assessment_runtime.llm_client import LLMClientError
        try:
            credentials.send("access-token")
        except (EOFError, OSError):
            raise LLMClientError("ChatGPT session connection ended. Reconnect your account.") from None
        if not credentials.poll(60):
            raise LLMClientError("ChatGPT session renewal timed out. Retry the assessment.")
        try:
            result = credentials.recv()
        except (EOFError, OSError):
            raise LLMClientError("ChatGPT session connection ended. Reconnect your account.") from None
        if not result.get("token"):
            raise LLMClientError(result.get("error") or "Reconnect your ChatGPT account.")
        return result["token"]

    try:
        request_api_key = str(request_payload.get("llm_api_key") or "").strip()
        if request_api_key and request_payload.get("provider") not in {"chatgpt", "xai", "groq"}:
            os.environ["LLM_API_KEY"] = request_api_key
            if request_payload.get("provider") == "openrouter":
                os.environ["OPENROUTER_API_KEY"] = request_api_key
            if request_payload.get("provider") == "ollama":
                os.environ["OLLAMA_API_KEY"] = request_api_key
        if request_payload.get("openrouter_http_referer"):
            os.environ["OPENROUTER_HTTP_REFERER"] = str(request_payload["openrouter_http_referer"])
        if request_payload.get("openrouter_app_title"):
            os.environ["OPENROUTER_APP_TITLE"] = str(request_payload["openrouter_app_title"])
        validate_audio_duration(Path(audio_path))
        result = execute_assessment_run(
            AssessmentRunRequest(
                audio=Path(audio_path),
                whisper_model=request_payload["whisper"],
                llm_model=request_payload.get("llm_model"),
                provider=request_payload.get("provider"),
                target_cefr=request_payload.get("target_cefr"),
                prompt_id=request_payload.get("prompt_id", ""),
                prompt_text=request_payload.get("prompt_text", ""),
                retry_of_session_id=request_payload.get("retry_of_session_id", ""),
                theme=request_payload["theme"],
                task_family=request_payload["task_family"],
                speaker_id=request_payload["speaker_id"],
                target_duration_sec=float(request_payload["target_duration_sec"]),
                expected_language=request_payload.get("expected_language"),
                language_profile_key=request_payload.get("language_profile_key"),
                feedback_language=request_payload.get("feedback_language"),
                llm_base_url=request_payload.get("llm_base_url"),
                llm_api_key=current_access_token if credentials is not None else (request_payload.get("llm_api_key") or None),
                dry_run=bool(request_payload.get("dry_run", False)),
                log_dir=Path(request_payload["log_dir"]),
                label=request_payload.get("label", ""),
                notes=request_payload.get("notes", ""),
                no_log=False,
            ),
            status_callback=_status_callback,
        )
        summary = _build_summary(result.saved_payload or result.output)
        _update_job_file(
            path,
            status=JobStatus.COMPLETED.value,
            phase="done",
            progress=1.0,
            completed_at=_now_iso(),
            error=None,
            report_path=str(result.report_path.resolve()) if result.report_path is not None else None,
            payload=result.saved_payload or result.output,
            summary=summary.model_dump() if summary else None,
        )
    except Exception as exc:  # quality: allow[broad-except] worker boundary must persist a failed job
        detail = str(exc)
        _update_job_file(
            path,
            status=JobStatus.FAILED.value,
            phase="failed",
            progress=1.0,
            completed_at=_now_iso(),
            error=ErrorResponse(
                code=_error_code_from_detail(detail, provider=str(request_payload.get("provider") or "")),
                detail=detail,
            ).model_dump(),
        )


class JobManager:
    def __init__(self, config: BackendRuntimeConfig) -> None:
        self._config = config
        self._ctx = multiprocessing.get_context("spawn")
        self._processes: dict[str, multiprocessing.Process] = {}
        self._submit_lock = threading.Lock()
        self._recover_existing_jobs()

    @property
    def uploads_dir(self) -> Path:
        return self._config.app_data.uploads_dir

    def _recover_existing_jobs(self) -> None:
        recover_incomplete_job_metadata(self._config.jobs_dir)

    def register_upload(self, *, data: bytes, filename: str) -> UploadResponse:
        return self.register_upload_stream(source=io.BytesIO(data), filename=filename)

    def register_upload_stream(self, *, source: BinaryIO, filename: str) -> UploadResponse:
        audio_id = f"aud_{uuid4().hex}"
        suffix = Path(filename or "audio.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".webm", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".flac", ".aiff", ".aif"}:
            suffix = ".audio"
        stored_path = self.uploads_dir / f"{audio_id}{suffix}"
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = _upload_meta_file(self.uploads_dir, audio_id)
        hasher = hashlib.sha1()
        total = 0
        # The multipart spool already exists. Reserve only its durable copy now.
        limit = MAX_UPLOAD_BYTES
        try:
            with stored_path.open("xb") as target:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    total += len(chunk)
                    if total > limit:
                        raise UploadRejected("Audio exceeds the available upload limit.")
                    ensure_copy_space(self.uploads_dir, len(chunk))
                    target.write(chunk)
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            _write_json(metadata_path, {"audio_id": audio_id, "stored_path": str(stored_path.resolve()),
                                      "sha1": digest, "original_name": filename})
        except BaseException:  # quality: allow[broad-except] remove partial files on cancellation too, then re-raise
            stored_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
        return UploadResponse(
            audio_id=audio_id,
            stored_path=str(stored_path.resolve()),
            sha1=digest,
            original_name=filename,
        )

    def _resolve_audio_path(self, audio_id: str) -> Path:
        payload = _read_json(_upload_meta_file(self.uploads_dir, audio_id))
        candidate = Path(str(payload.get("stored_path") or "")).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Uploaded audio {audio_id} is not available anymore.")
        return candidate

    def submit(self, request: AssessmentCreateRequest, *, credential_ref: str = "") -> AssessmentCreateResponse:
        with self._submit_lock:
            if any(process.is_alive() for process in list(self._processes.values())):
                raise AssessmentBusyError("Another attempt is still being analysed. Wait for it to finish or cancel it, then retry.")
            if upload_limits(self.uploads_dir)["available_bytes"] <= 0:
                raise OSError("Not enough free disk space to analyse audio. Free some space and retry.")
            return self._submit(request, credential_ref=credential_ref)

    def _submit(self, request: AssessmentCreateRequest, *, credential_ref: str = "") -> AssessmentCreateResponse:
        assessment_id = f"asmt_{uuid4().hex}"
        audio_path = self._resolve_audio_path(request.audio_id)
        worker_request = {
            **request.model_dump(),
            "log_dir": str(self._config.app_data.reports_dir),
        }
        payload = {
            "assessment_id": assessment_id,
            "status": JobStatus.QUEUED.value,
            "phase": "queued",
            "progress": 0.0,
            "created_at": _now_iso(),
            "report_path": None,
            "summary": None,
            "payload": None,
            "error": None,
            "request": _sanitize_request_metadata(worker_request),
            "audio_path": str(audio_path.resolve()),
        }
        job_file = _job_file(self._config.jobs_dir, assessment_id)
        _write_json(job_file, payload)
        parent = child = None
        if request.provider == "chatgpt":
            if not credential_ref:
                raise ValueError("A saved ChatGPT account is required.")
            parent, child = self._ctx.Pipe()
            worker_request["llm_api_key"] = ""
        args = (str(job_file), worker_request, str(audio_path.resolve()))
        process = self._ctx.Process(target=_job_worker, args=(*args, child) if child is not None else args)
        try:
            process.start()
        except BaseException:  # quality: allow[broad-except] close IPC handles on failed process creation, then re-raise
            if parent is not None:
                parent.close()
                child.close()
            raise
        self._processes[assessment_id] = process
        if child is not None:
            child.close()
            threading.Thread(target=self._serve_credentials, args=(process, parent, credential_ref), daemon=True).start()
        return AssessmentCreateResponse(assessment_id=assessment_id, status=JobStatus.QUEUED)

    @staticmethod
    def _serve_credentials(process, channel, credential_ref):
        from app_core.chatgpt_auth import access_token, ChatGPTAuthError
        try:
            while process.is_alive():
                if not channel.poll(0.5):
                    continue
                if channel.recv() != "access-token":
                    break
                try:
                    result = {"token": access_token(credential_ref)}
                except ChatGPTAuthError as exc:
                    result = {"error": str(exc)}
                except (ValueError, TypeError, KeyError):
                    result = {"error": "Reconnect your ChatGPT account."}
                channel.send(result)
        except (EOFError, BrokenPipeError, OSError):
            pass  # Worker cancellation closes the pipe; no credentials are logged.
        finally:
            channel.close()

    def has_active_provider(self, provider: str) -> bool:
        return any(process.is_alive() and _read_json(_job_file(self._config.jobs_dir, assessment_id)).get('request', {}).get('provider') == provider
                   for assessment_id, process in list(self._processes.items()))

    def cancel_provider(self, provider: str) -> None:
        with self._submit_lock:
            for assessment_id in list(self._processes):
                payload = _read_json(_job_file(self._config.jobs_dir, assessment_id))
                if payload.get("request", {}).get("provider") == provider:
                    self.cancel(assessment_id)

    def _reconcile_process(self, assessment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._processes.get(assessment_id)
        if process is None:
            return payload
        if process.is_alive():
            return payload
        process.join(timeout=0.1)
        if payload.get("status") in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
            payload["status"] = JobStatus.FAILED.value
            payload["phase"] = "failed"
            payload["progress"] = 1.0
            payload["completed_at"] = _now_iso()
            payload["error"] = ErrorResponse(
                code=ErrorCode.RUNTIME,
                detail=f"Assessment worker exited unexpectedly with code {process.exitcode}.",
            ).model_dump()
            _write_json(_job_file(self._config.jobs_dir, assessment_id), payload)
        self._processes.pop(assessment_id, None)
        return payload

    def get_status(self, assessment_id: str) -> AssessmentStatusResponse | None:
        job_file = _job_file(self._config.jobs_dir, assessment_id)
        if not job_file.exists():
            return None
        payload = self._reconcile_process(assessment_id, _read_json(job_file))
        return AssessmentStatusResponse(
            assessment_id=assessment_id,
            status=JobStatus(str(payload.get("status") or JobStatus.FAILED.value)),
            phase=str(payload.get("phase") or ""),
            progress=float(payload.get("progress") or 0.0),
            error=(ErrorResponse(**payload["error"]) if isinstance(payload.get("error"), dict) else None),
            report_path=str(payload.get("report_path") or "") or None,
            summary=(AssessmentSummary(**payload["summary"]) if isinstance(payload.get("summary"), dict) else None),
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else None,
        )

    def cancel(self, assessment_id: str) -> AssessmentStatusResponse | None:
        status = self.get_status(assessment_id)
        if status is None:
            return None
        if status.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return status
        process = self._processes.get(assessment_id)
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        payload = _read_json(_job_file(self._config.jobs_dir, assessment_id))
        if str(payload.get("status") or "").strip().lower() in TERMINAL_JOB_STATUSES_NORMALIZED:
            self._processes.pop(assessment_id, None)
            return self.get_status(assessment_id)
        payload.update(
            {
                "status": JobStatus.CANCELLED.value,
                "phase": "cancelled",
                "progress": 1.0,
                "completed_at": _now_iso(),
                "error": ErrorResponse(
                    code=ErrorCode.CANCELLATION,
                    detail="Assessment cancelled by the user.",
                ).model_dump(),
            }
        )
        _write_json(_job_file(self._config.jobs_dir, assessment_id), payload)
        self._processes.pop(assessment_id, None)
        return self.get_status(assessment_id)

    def shutdown(self) -> None:
        for assessment_id, process in list(self._processes.items()):
            if process.is_alive():
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except OSError:
                    logger.warning(
                        "Could not terminate assessment worker %s for %s.",
                        process.pid,
                        assessment_id,
                        exc_info=True,
                    )
                process.join(timeout=0.5)
            self._processes.pop(assessment_id, None)
