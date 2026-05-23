from __future__ import annotations

import csv
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app_backend.config import BackendRuntimeConfig, build_backend_runtime_config, clear_backend_state, write_backend_state
from app_backend.contracts import (
    AssessmentCreateRequest,
    AssessmentCreateResponse,
    AssessmentStatusResponse,
    CANONICAL_PRODUCT_API_ROUTES,
    ConnectionSecretState,
    DiagnosticsResponse,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    HistoryDetailResponse,
    HistoryResponse,
    LOCAL_RUNTIME_MANAGEMENT_ROUTES,
    LOCAL_SUPPORT_API_ROUTES,
    LocalApiContractResponse,
    MaintenanceCleanupRequest,
    MaintenanceCleanupResponse,
    MaintenanceStorageResponse,
    RuntimeConnectionTestRequest,
    RuntimeConnectionTestResponse,
    RuntimeResponse,
    RuntimeSettingsConnection,
    RuntimeSettingsResponse,
    RuntimeSettingsSaveRequest,
    SamplesResponse,
    SampleItem,
    SupportBundleCreateRequest,
    SupportBundleCreateResponse,
    UploadResponse,
    WhisperModelStatusResponse,
)
from app_backend.jobs import JobManager
from app_backend.maintenance import execute_cleanup
from app_backend.support_bundle import (
    build_storage_summary,
    create_support_bundle as create_support_bundle_archive,
    support_bundle_path,
)
from app_core.diagnostics import collect_startup_diagnostics
from app_core.runtime_providers import (
    default_base_url,
    default_connection_label,
    default_setup_base_url,
    normalize_provider,
    normalize_setup_provider_choice,
    provider_kind_from_choice,
    requires_api_key,
)
from app_core.runtime_resolver import active_connection, resolve_connection_runtime
from app_core.services import (
    build_provider_connection,
    delete_provider_connection,
    download_whisper_model,
    history_rows,
    hydrate_state_from_storage,
    load_report_payload,
    provider_choice_for_connection,
    save_provider_connection,
    sanitize_setup_base_url,
    set_default_provider_connection,
    test_runtime_connection,
    whisper_model_status,
)
from app_core.secret_store import delete_secret, get_secret
from app_core.state import (
    build_default_state,
    normalize_openrouter_app_title,
    normalize_openrouter_http_referer,
)
from app_core.bootstrap import PROJECT_ROOT, bootstrap_app_environment

CONTRACT_TAG = "contract"
PRODUCT_API_TAG = "product-api"
LOCAL_RUNTIME_TAG = "local-runtime"
LOCAL_SUPPORT_TAG = "local-support"
LOCAL_GUEST_ORIGIN_REGEX = r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://tauri\.localhost|tauri://localhost)$"

OPENAPI_TAGS = [
    {
        "name": CONTRACT_TAG,
        "description": "Phase-2 local desktop contract metadata for the shared frontend and launcher.",
    },
    {
        "name": PRODUCT_API_TAG,
        "description": "Canonical shared product endpoints that stay stable between local desktop and future hosted work.",
    },
    {
        "name": LOCAL_RUNTIME_TAG,
        "description": "Local-desktop runtime management endpoints for saved connections and whisper tooling.",
    },
    {
        "name": LOCAL_SUPPORT_TAG,
        "description": "Local-desktop support extensions that do not redefine the canonical product API.",
    },
]


def _load_persisted_state(runtime_config: BackendRuntimeConfig):
    bootstrap_app_environment(
        log_dir=runtime_config.app_data.reports_dir,
        app_data_dir=runtime_config.app_data.root,
        cache_dir=runtime_config.app_data.cache_root,
        whisper_cache_dir=runtime_config.app_data.whisper_cache_dir,
    )
    state = build_default_state()
    state.prefs.log_dir = str(runtime_config.app_data.reports_dir)
    state.prefs.whisper_cache_dir = str(runtime_config.app_data.whisper_cache_dir)
    return hydrate_state_from_storage(state)


def _http_error(status_code: int, code: ErrorCode, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=ErrorResponse(code=code, detail=detail).model_dump())


def _serialize_diagnostics(state, *, include_runtime_health: bool = False) -> DiagnosticsResponse:
    items = collect_startup_diagnostics(state, include_runtime_health=include_runtime_health)
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


def _saved_connection_api_key(connection) -> str:
    secret_ref = str(getattr(connection, "secret_ref", "") or "").strip()
    if not secret_ref:
        return ""
    return str(get_secret(secret_ref) or "").strip()


def _connection_secret_state(connection) -> ConnectionSecretState:
    if _saved_connection_api_key(connection):
        return ConnectionSecretState.PRESENT
    if str(connection.auth_mode or "").strip().lower() == "bearer" or requires_api_key(connection.provider_kind):
        return ConnectionSecretState.MISSING
    return ConnectionSecretState.ABSENT


def _serialize_connection(connection) -> RuntimeSettingsConnection:
    provider_choice = provider_choice_for_connection(connection)
    metadata = dict(connection.provider_metadata or {})
    if normalize_provider(connection.provider_kind) == "openrouter":
        metadata["http_referer"] = normalize_openrouter_http_referer(metadata.get("http_referer"))
        metadata["app_title"] = normalize_openrouter_app_title(metadata.get("app_title"))
    saved_api_key = _saved_connection_api_key(connection)
    return RuntimeSettingsConnection(
        connection_id=connection.connection_id,
        provider_key=normalize_provider(connection.provider_kind),
        provider_choice=provider_choice,
        provider_label=default_connection_label(provider_choice),
        label=connection.label,
        model=connection.default_model,
        base_url=connection.base_url,
        is_default=bool(connection.is_default),
        is_local=bool(connection.is_local),
        requires_api_key=requires_api_key(connection.provider_kind),
        has_api_key=bool(saved_api_key),
        secret_state=_connection_secret_state(connection),
        last_test_status=connection.last_test_status,
        last_tested_at=connection.last_tested_at,
        openrouter_http_referer=str(metadata.get("http_referer") or ""),
        openrouter_app_title=str(metadata.get("app_title") or ""),
        provider_metadata=metadata,
    )


def _build_runtime_settings_response(state) -> RuntimeSettingsResponse:
    return RuntimeSettingsResponse(
        ui_locale=str(state.prefs.ui_locale or ""),
        whisper_model=str(state.prefs.whisper_model or ""),
        active_connection_id=str(state.prefs.active_connection_id or ""),
        connections=[_serialize_connection(connection) for connection in list(state.prefs.connections or [])],
    )


def _runtime_settings_provider_identity(provider_choice: str, base_url: str) -> tuple[str, str]:
    normalized_choice = normalize_setup_provider_choice(provider_choice)
    provider_kind = provider_kind_from_choice(normalized_choice)
    sanitized_base_url = sanitize_setup_base_url(normalized_choice, base_url)
    resolved_base_url = str(
        sanitized_base_url or default_setup_base_url(normalized_choice) or default_base_url(provider_kind)
    ).strip()
    return normalized_choice, resolved_base_url.rstrip("/")


def _find_runtime_settings_connection(state, connection_id: str):
    resolved_connection_id = str(connection_id or "").strip()
    if not resolved_connection_id:
        return None
    return next(
        (
            item
            for item in list(state.prefs.connections or [])
            if item.connection_id == resolved_connection_id
        ),
        None,
    )


def _runtime_settings_draft_matches_existing(state, draft, existing_connection) -> bool:
    provider_choice = str(
        draft.provider_choice or provider_choice_for_connection(existing_connection, state.prefs.provider)
    ).strip()
    existing_identity = _runtime_settings_provider_identity(
        provider_choice_for_connection(existing_connection, state.prefs.provider),
        existing_connection.base_url,
    )
    draft_identity = _runtime_settings_provider_identity(provider_choice, str(draft.base_url or ""))
    return existing_identity == draft_identity


def _runtime_settings_test_connection_secret(state, draft) -> str:
    api_key = str(draft.api_key or "").strip()
    if api_key:
        return api_key
    existing_connection = _find_runtime_settings_connection(
        state,
        str(draft.connection_id or "").strip(),
    )
    if existing_connection is None:
        return ""
    if not _runtime_settings_draft_matches_existing(state, draft, existing_connection):
        return ""
    return _saved_connection_api_key(existing_connection)


def _save_runtime_settings(state, request: RuntimeSettingsSaveRequest):
    state.prefs.ui_locale = str(request.ui_locale or state.prefs.ui_locale or "").strip() or state.prefs.ui_locale
    state.prefs.whisper_model = str(request.whisper_model or state.prefs.whisper_model or "").strip() or state.prefs.whisper_model

    existing_connection = _find_runtime_settings_connection(
        state,
        str(request.connection.connection_id or "").strip(),
    )
    draft = request.connection
    provider_choice = str(draft.provider_choice or provider_choice_for_connection(existing_connection, state.prefs.provider)).strip()
    if not provider_choice:
        raise ValueError("Choose a provider before saving the runtime settings.")

    api_key = str(draft.api_key or "").strip()
    preserve_existing_secret = False
    if not api_key and existing_connection is not None and not request.clear_saved_secret:
        existing_identity = _runtime_settings_provider_identity(
            provider_choice_for_connection(existing_connection, state.prefs.provider),
            existing_connection.base_url,
        )
        draft_identity = _runtime_settings_provider_identity(provider_choice, str(draft.base_url or ""))
        preserve_existing_secret = existing_identity == draft_identity

    existing_secret_ref = str(getattr(existing_connection, "secret_ref", "") or "").strip()
    should_drop_existing_secret = (
        bool(existing_secret_ref)
        and existing_connection is not None
        and (request.clear_saved_secret or (not api_key and not preserve_existing_secret))
    )
    if should_drop_existing_secret:
        delete_secret(existing_secret_ref)
    elif preserve_existing_secret and existing_connection is not None:
        api_key = _saved_connection_api_key(existing_connection)

    connection = build_provider_connection(
        provider_choice=provider_choice,
        label=str(draft.label or "").strip(),
        model=str(draft.model or "").strip(),
        base_url=str(draft.base_url or "").strip(),
        api_key=api_key,
        openrouter_http_referer=str(draft.openrouter_http_referer or "").strip(),
        openrouter_app_title=str(draft.openrouter_app_title or "").strip(),
        existing_connection=existing_connection,
    )
    save_provider_connection(
        state,
        connection,
        api_key=api_key,
        persist_draft=False,
    )
    return connection


def _assessment_request_with_saved_runtime_secret(
    state,
    request: AssessmentCreateRequest,
) -> AssessmentCreateRequest:
    if str(request.llm_api_key or "").strip():
        return request
    connection = active_connection(state.prefs)
    if connection is None:
        return request
    runtime_state = resolve_connection_runtime(connection)
    if normalize_provider(request.provider) != runtime_state.provider:
        return request
    request_base_url = str(request.llm_base_url or "").strip().rstrip("/")
    runtime_base = str(runtime_state.base_url or "").strip().rstrip("/")
    if request_base_url and runtime_base and request_base_url != runtime_base:
        return request

    updates: dict[str, Any] = {}
    if runtime_state.api_key:
        updates["llm_api_key"] = runtime_state.api_key
    if not str(request.llm_base_url or "").strip() and runtime_state.base_url:
        updates["llm_base_url"] = runtime_state.base_url
    if runtime_state.provider == "openrouter":
        if not str(request.openrouter_http_referer or "").strip():
            updates["openrouter_http_referer"] = runtime_state.extra_headers.get("HTTP-Referer", "")
        if not str(request.openrouter_app_title or "").strip():
            updates["openrouter_app_title"] = runtime_state.extra_headers.get("X-Title", "")
    return request.model_copy(update=updates) if updates else request


def _find_history_payload(session_id: str, reports_dir: Path) -> dict[str, Any] | None:
    history_path = reports_dir / "history.csv"
    if not history_path.exists():
        return None
    try:
        with history_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("session_id") or "").strip() != session_id:
                    continue
                payload = load_report_payload(row.get("report_path") or "")
                if payload is not None:
                    return payload
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
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

    app = FastAPI(
        title="Vostavo Local Backend",
        version="0.1.0",
        lifespan=_lifespan,
        openapi_tags=OPENAPI_TAGS,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCAL_GUEST_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime_config = runtime_config
    app.state.started_at = started_at
    app.state.job_manager = job_manager

    @app.get("/v1/contract", response_model=LocalApiContractResponse, tags=[CONTRACT_TAG])
    def contract() -> LocalApiContractResponse:
        return LocalApiContractResponse(
            product_routes=list(CANONICAL_PRODUCT_API_ROUTES),
            local_support_routes=list(LOCAL_SUPPORT_API_ROUTES),
            local_runtime_routes=list(LOCAL_RUNTIME_MANAGEMENT_ROUTES),
        )

    @app.get("/v1/health", response_model=HealthResponse, tags=[PRODUCT_API_TAG])
    def health() -> HealthResponse:
        uptime = (datetime.now(UTC) - app.state.started_at).total_seconds()
        return HealthResponse(
            app_data_root=str(runtime_config.app_data.root),
            uptime_sec=uptime,
        )

    @app.get("/v1/diagnostics", response_model=DiagnosticsResponse, tags=[PRODUCT_API_TAG])
    def diagnostics() -> DiagnosticsResponse:
        state = _load_persisted_state(runtime_config)
        return _serialize_diagnostics(state)

    @app.get("/v1/runtime", response_model=RuntimeResponse, tags=[PRODUCT_API_TAG])
    def runtime() -> RuntimeResponse:
        state = _load_persisted_state(runtime_config)
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

    @app.get("/v1/runtime/settings", response_model=RuntimeSettingsResponse, tags=[LOCAL_RUNTIME_TAG])
    def runtime_settings() -> RuntimeSettingsResponse:
        state = _load_persisted_state(runtime_config)
        return _build_runtime_settings_response(state)

    @app.put("/v1/runtime/settings", response_model=RuntimeSettingsResponse, tags=[LOCAL_RUNTIME_TAG])
    def save_runtime_settings(request: RuntimeSettingsSaveRequest) -> RuntimeSettingsResponse:
        state = _load_persisted_state(runtime_config)
        try:
            _save_runtime_settings(state, request)
        except ValueError as exc:
            raise _http_error(400, ErrorCode.VALIDATION, str(exc)) from exc
        return _build_runtime_settings_response(state)

    @app.post("/v1/runtime/settings/test-connection", response_model=RuntimeConnectionTestResponse, tags=[LOCAL_RUNTIME_TAG])
    def runtime_settings_test_connection(request: RuntimeConnectionTestRequest) -> RuntimeConnectionTestResponse:
        state = _load_persisted_state(runtime_config)
        draft = request.connection
        try:
            result = test_runtime_connection(
                provider=str(draft.provider_choice or "").strip(),
                provider_choice=str(draft.provider_choice or "").strip(),
                model=str(draft.model or "").strip(),
                base_url=str(draft.base_url or "").strip(),
                api_key=_runtime_settings_test_connection_secret(state, draft),
                openrouter_http_referer=str(draft.openrouter_http_referer or "").strip(),
                openrouter_app_title=str(draft.openrouter_app_title or "").strip(),
            )
        except ValueError as exc:
            raise _http_error(400, ErrorCode.VALIDATION, str(exc)) from exc
        except OSError as exc:
            raise _http_error(500, ErrorCode.RUNTIME, str(exc)) from exc
        tested_at = str((result.get("test_payload") or {}).get("tested_at") or "")
        content_preview = str((result.get("test_payload") or {}).get("content_preview") or "")
        return RuntimeConnectionTestResponse(
            provider=str(result.get("provider") or ""),
            base_url=str(result.get("base_url") or ""),
            service_base_url=str(result.get("service_base_url") or ""),
            health_endpoint=str(result.get("health_endpoint") or ""),
            discovered_models=[str(item) for item in list(result.get("models") or []) if str(item).strip()],
            tested_at=tested_at,
            content_preview=content_preview,
        )

    @app.post(
        "/v1/runtime/settings/connections/{connection_id}/default",
        response_model=RuntimeSettingsResponse,
        tags=[LOCAL_RUNTIME_TAG],
    )
    def runtime_settings_set_default(connection_id: str) -> RuntimeSettingsResponse:
        state = _load_persisted_state(runtime_config)
        if not set_default_provider_connection(state, connection_id, persist_draft=False):
            raise _http_error(404, ErrorCode.VALIDATION, f"Runtime connection {connection_id} does not exist.")
        return _build_runtime_settings_response(state)

    @app.delete(
        "/v1/runtime/settings/connections/{connection_id}",
        response_model=RuntimeSettingsResponse,
        tags=[LOCAL_RUNTIME_TAG],
    )
    def runtime_settings_delete(connection_id: str) -> RuntimeSettingsResponse:
        state = _load_persisted_state(runtime_config)
        if not delete_provider_connection(state, connection_id, persist_draft=False):
            raise _http_error(404, ErrorCode.VALIDATION, f"Runtime connection {connection_id} does not exist.")
        return _build_runtime_settings_response(state)

    @app.get("/v1/runtime/whisper-models/{model_size}", response_model=WhisperModelStatusResponse, tags=[LOCAL_RUNTIME_TAG])
    def runtime_whisper_model_status(model_size: str) -> WhisperModelStatusResponse:
        status = whisper_model_status(str(model_size or "").strip())
        return WhisperModelStatusResponse(
            model=str(status.get("model") or model_size),
            repo_id=str(status.get("repo_id") or ""),
            cached=bool(status.get("cached")),
            cached_path=str(status.get("cached_path") or ""),
            recommended=bool(status.get("recommended")),
            recommendation_reason=str(status.get("recommendation_reason") or ""),
        )

    @app.post(
        "/v1/runtime/whisper-models/{model_size}/download",
        response_model=WhisperModelStatusResponse,
        tags=[LOCAL_RUNTIME_TAG],
    )
    def runtime_whisper_model_download(model_size: str) -> WhisperModelStatusResponse:
        try:
            status = download_whisper_model(str(model_size or "").strip())
        except OSError as exc:
            raise _http_error(500, ErrorCode.RUNTIME, str(exc)) from exc
        return WhisperModelStatusResponse(
            model=str(status.get("model") or model_size),
            repo_id=str(status.get("repo_id") or ""),
            cached=bool(status.get("cached")),
            cached_path=str(status.get("cached_path") or ""),
            recommended=bool(status.get("recommended")),
            recommendation_reason=str(status.get("recommendation_reason") or ""),
        )

    @app.get("/v1/maintenance/storage", response_model=MaintenanceStorageResponse, tags=[LOCAL_SUPPORT_TAG])
    def maintenance_storage() -> MaintenanceStorageResponse:
        return build_storage_summary(runtime_config)

    @app.post("/v1/maintenance/cleanup", response_model=MaintenanceCleanupResponse, tags=[LOCAL_SUPPORT_TAG])
    def maintenance_cleanup(request: MaintenanceCleanupRequest) -> MaintenanceCleanupResponse:
        return execute_cleanup(
            runtime_config,
            request.target,
            dry_run=request.dry_run,
        )

    @app.post("/v1/support-bundles", response_model=SupportBundleCreateResponse, tags=[LOCAL_SUPPORT_TAG])
    def create_support_bundle(request: SupportBundleCreateRequest) -> SupportBundleCreateResponse:
        try:
            if request.include_runtime_health:
                state = _load_persisted_state(runtime_config)
                diagnostics = _serialize_diagnostics(state, include_runtime_health=True)
                request = request.model_copy(
                    update={
                        "client_diagnostics": [
                            *list(request.client_diagnostics or []),
                            *[item.model_dump(mode="json") for item in diagnostics.items],
                        ]
                    }
                )
            return create_support_bundle_archive(runtime_config, request)
        except OSError as exc:
            raise _http_error(500, ErrorCode.STORAGE, str(exc)) from exc

    @app.get("/v1/support-bundles/{bundle_id}", tags=[LOCAL_SUPPORT_TAG])
    def download_support_bundle(bundle_id: str) -> FileResponse:
        try:
            bundle_path = support_bundle_path(runtime_config, bundle_id)
        except ValueError as exc:
            raise _http_error(400, ErrorCode.VALIDATION, str(exc)) from exc
        if not bundle_path.exists():
            raise _http_error(404, ErrorCode.VALIDATION, f"Support bundle {bundle_id} does not exist.")
        return FileResponse(bundle_path, media_type="application/zip", filename=bundle_path.name)

    @app.post("/v1/uploads", response_model=UploadResponse, tags=[PRODUCT_API_TAG])
    async def upload_audio(file: UploadFile = File(...)) -> UploadResponse:
        data = await file.read()
        if not data:
            raise _http_error(400, ErrorCode.VALIDATION, "Uploaded audio file is empty.")
        return app.state.job_manager.register_upload(data=data, filename=file.filename or "audio.wav")

    @app.post("/v1/assessments", response_model=AssessmentCreateResponse, tags=[PRODUCT_API_TAG])
    def create_assessment(request: AssessmentCreateRequest) -> AssessmentCreateResponse:
        try:
            state = _load_persisted_state(runtime_config)
            return app.state.job_manager.submit(_assessment_request_with_saved_runtime_secret(state, request))
        except FileNotFoundError as exc:
            raise _http_error(404, ErrorCode.VALIDATION, str(exc)) from exc
        except OSError as exc:
            raise _http_error(500, ErrorCode.STORAGE, str(exc)) from exc

    @app.get("/v1/assessments/{assessment_id}", response_model=AssessmentStatusResponse, tags=[PRODUCT_API_TAG])
    def assessment_status(assessment_id: str) -> AssessmentStatusResponse:
        status = app.state.job_manager.get_status(assessment_id)
        if status is None:
            raise _http_error(404, ErrorCode.VALIDATION, f"Assessment {assessment_id} does not exist.")
        return status

    @app.post(
        "/v1/assessments/{assessment_id}/cancel",
        response_model=AssessmentStatusResponse,
        tags=[PRODUCT_API_TAG],
    )
    def cancel_assessment(assessment_id: str) -> AssessmentStatusResponse:
        status = app.state.job_manager.cancel(assessment_id)
        if status is None:
            raise _http_error(404, ErrorCode.VALIDATION, f"Assessment {assessment_id} does not exist.")
        return status

    @app.get("/v1/history", response_model=HistoryResponse, tags=[PRODUCT_API_TAG])
    def history() -> HistoryResponse:
        return HistoryResponse(items=history_rows(runtime_config.app_data.reports_dir))

    @app.get("/v1/history/{session_id}", response_model=HistoryDetailResponse, tags=[PRODUCT_API_TAG])
    def history_detail(session_id: str) -> HistoryDetailResponse:
        payload = _find_history_payload(session_id, runtime_config.app_data.reports_dir)
        if payload is None:
            raise _http_error(404, ErrorCode.VALIDATION, f"History entry {session_id} does not exist.")
        return HistoryDetailResponse(payload=payload)

    @app.get("/v1/samples", response_model=SamplesResponse, tags=[PRODUCT_API_TAG])
    def samples() -> SamplesResponse:
        return SamplesResponse(items=_sample_items())

    return app
