from __future__ import annotations

import csv
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from app_backend.config import BackendRuntimeConfig, build_backend_runtime_config, clear_backend_state, write_backend_state
from app_backend.contracts import (
    AssessmentCreateRequest,
    AssessmentCreateResponse,
    AssessmentStatusResponse,
    DiagnosticsResponse,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    HistoryDetailResponse,
    HistoryResponse,
    RuntimeResponse,
    SamplesResponse,
    SampleItem,
    UploadResponse,
)
from app_backend.jobs import JobManager
from app_shell.diagnostics import collect_startup_diagnostics
from app_shell.runtime_providers import requires_api_key
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime
from app_shell.services import history_rows, hydrate_state_from_storage, load_report_payload
from app_shell.state import build_default_state
from app_shell.bootstrap import PROJECT_ROOT


def _load_persisted_state():
    state = build_default_state()
    return hydrate_state_from_storage(state)


def _http_error(status_code: int, code: ErrorCode, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=ErrorResponse(code=code, detail=detail).model_dump())


def _find_history_payload(session_id: str, reports_dir: Path) -> dict[str, Any] | None:
    history_path = reports_dir / "history.csv"
    if not history_path.exists():
        return None
    with history_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("session_id") or "").strip() != session_id:
                continue
            payload = load_report_payload(row.get("report_path") or "")
            if payload is not None:
                return payload
    return None


def _sample_items() -> list[SampleItem]:
    root = PROJECT_ROOT / "samples" / "cefr"
    items: list[SampleItem] = []
    if not root.exists():
        return items
    for language_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for cefr_dir in sorted(path for path in language_dir.iterdir() if path.is_dir()):
            for audio_file in sorted(path for path in cefr_dir.iterdir() if path.is_file()):
                items.append(
                    SampleItem(
                        sample_id=f"{language_dir.name}_{cefr_dir.name}_{audio_file.stem}",
                        language=language_dir.name,
                        cefr=cefr_dir.name,
                        title=audio_file.stem.replace("_", " "),
                        path=str(audio_file.resolve()),
                    )
                )
    return items


def create_app(config: BackendRuntimeConfig | None = None) -> FastAPI:
    runtime_config = config or build_backend_runtime_config()
    started_at = datetime.now(UTC)
    job_manager = JobManager(runtime_config)
    
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        write_backend_state(runtime_config, pid=os.getpid())
        try:
            yield
        finally:
            app.state.job_manager.shutdown()
            clear_backend_state(
                runtime_config.app_data.reports_dir,
                app_data_dir=runtime_config.app_data.root,
                cache_dir=runtime_config.app_data.cache_root,
            )

    app = FastAPI(title="Vostavo Local Backend", version="0.1.0", lifespan=_lifespan)
    app.state.runtime_config = runtime_config
    app.state.started_at = started_at
    app.state.job_manager = job_manager

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        uptime = (datetime.now(UTC) - app.state.started_at).total_seconds()
        return HealthResponse(
            app_data_root=str(runtime_config.app_data.root),
            uptime_sec=uptime,
        )

    @app.get("/v1/diagnostics", response_model=DiagnosticsResponse)
    def diagnostics() -> DiagnosticsResponse:
        state = _load_persisted_state()
        items = collect_startup_diagnostics(state)
        return DiagnosticsResponse(
            items=[
                {
                    "key": item.key,
                    "status": item.status,
                    "title_key": item.title_key,
                    "detail_key": item.detail_key,
                    "detail_args": item.detail_args,
                }
                for item in items
            ]
        )

    @app.get("/v1/runtime", response_model=RuntimeResponse)
    def runtime() -> RuntimeResponse:
        state = _load_persisted_state()
        connection = active_connection(state.prefs)
        if connection is None:
            return RuntimeResponse(configured=False)
        runtime_state = resolve_connection_runtime(connection)
        return RuntimeResponse(
            configured=True,
            provider=runtime_state.provider,
            model=runtime_state.model,
            base_url=runtime_state.base_url,
            requires_api_key=requires_api_key(runtime_state.provider),
            has_api_key=bool(runtime_state.api_key),
        )

    @app.post("/v1/uploads", response_model=UploadResponse)
    async def upload_audio(file: UploadFile = File(...)) -> UploadResponse:
        data = await file.read()
        if not data:
            raise _http_error(400, ErrorCode.VALIDATION, "Uploaded audio file is empty.")
        return app.state.job_manager.register_upload(data=data, filename=file.filename or "audio.wav")

    @app.post("/v1/assessments", response_model=AssessmentCreateResponse)
    def create_assessment(request: AssessmentCreateRequest) -> AssessmentCreateResponse:
        try:
            return app.state.job_manager.submit(request)
        except FileNotFoundError as exc:
            raise _http_error(404, ErrorCode.VALIDATION, str(exc)) from exc
        except OSError as exc:
            raise _http_error(500, ErrorCode.STORAGE, str(exc)) from exc

    @app.get("/v1/assessments/{assessment_id}", response_model=AssessmentStatusResponse)
    def assessment_status(assessment_id: str) -> AssessmentStatusResponse:
        status = app.state.job_manager.get_status(assessment_id)
        if status is None:
            raise _http_error(404, ErrorCode.VALIDATION, f"Assessment {assessment_id} does not exist.")
        return status

    @app.post("/v1/assessments/{assessment_id}/cancel", response_model=AssessmentStatusResponse)
    def cancel_assessment(assessment_id: str) -> AssessmentStatusResponse:
        status = app.state.job_manager.cancel(assessment_id)
        if status is None:
            raise _http_error(404, ErrorCode.VALIDATION, f"Assessment {assessment_id} does not exist.")
        return status

    @app.get("/v1/history", response_model=HistoryResponse)
    def history() -> HistoryResponse:
        return HistoryResponse(items=history_rows(runtime_config.app_data.reports_dir))

    @app.get("/v1/history/{session_id}", response_model=HistoryDetailResponse)
    def history_detail(session_id: str) -> HistoryDetailResponse:
        payload = _find_history_payload(session_id, runtime_config.app_data.reports_dir)
        if payload is None:
            raise _http_error(404, ErrorCode.VALIDATION, f"History entry {session_id} does not exist.")
        return HistoryDetailResponse(payload=payload)

    @app.get("/v1/samples", response_model=SamplesResponse)
    def samples() -> SamplesResponse:
        return SamplesResponse(items=_sample_items())

    return app
