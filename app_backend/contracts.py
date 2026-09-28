from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    BACKEND_UNAVAILABLE = "backend_unavailable"
    LOCAL_PROVIDER_NOT_INSTALLED = "local_provider_not_installed"
    LOCAL_PROVIDER_NOT_RUNNING = "local_provider_not_running"
    MISSING_FFMPEG = "missing_ffmpeg"
    MISSING_WHISPER_MODEL = "missing_whisper_model"
    VALIDATION = "validation_error"
    CONFIG = "configuration_error"
    RUNTIME = "runtime_error"
    STORAGE = "storage_error"
    CANCELLATION = "cancellation_error"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CleanupTarget(str, Enum):
    TMP = "tmp"
    JOBS = "jobs"
    LOGS = "logs"
    ALL_SAFE = "all_safe"


class ConnectionSecretState(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"
    MISSING = "missing"


PHASE_2_LOCAL_API_VERSION = "phase2-local-v1"

CANONICAL_PRODUCT_API_ROUTES: tuple[str, ...] = (
    "/v1/health",
    "/v1/diagnostics",
    "/v1/runtime",
    "/v1/uploads",
    "/v1/assessments",
    "/v1/assessments/{assessment_id}",
    "/v1/assessments/{assessment_id}/cancel",
    "/v1/history",
    "/v1/history/{session_id}",
    "/v1/samples",
)

LOCAL_SUPPORT_API_ROUTES: tuple[str, ...] = (
    "/v1/maintenance/storage",
    "/v1/maintenance/cleanup",
    "/v1/support-bundles",
    "/v1/support-bundles/{bundle_id}",
)

LOCAL_RUNTIME_MANAGEMENT_ROUTES: tuple[str, ...] = (
    "/v1/runtime/settings",
    "/v1/runtime/settings/test-connection",
    "/v1/runtime/settings/connections/{connection_id}/default",
    "/v1/runtime/settings/connections/{connection_id}",
    "/v1/runtime/whisper-models/{model_size}",
    "/v1/runtime/whisper-models/{model_size}/download",
)


class HealthResponse(BaseModel):
    status: str = "ready"
    version: str = "0.1.0"
    app_data_root: str
    uptime_sec: float


class DiagnosticItem(BaseModel):
    key: str
    status: str
    title_key: str
    detail_key: str
    detail_args: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsResponse(BaseModel):
    items: list[DiagnosticItem]


class RuntimeResponse(BaseModel):
    configured: bool
    provider: str = ""
    model: str = ""
    base_url: str = ""
    requires_api_key: bool = False
    has_api_key: bool = False


class RuntimeSettingsConnection(BaseModel):
    connection_id: str
    provider_key: str
    provider_choice: str
    provider_label: str
    label: str
    model: str
    base_url: str
    is_default: bool = False
    is_local: bool = False
    requires_api_key: bool = False
    has_api_key: bool = False
    secret_state: ConnectionSecretState = ConnectionSecretState.ABSENT
    last_test_status: str = ""
    last_tested_at: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_title: str = ""
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeSettingsResponse(BaseModel):
    ui_locale: str
    whisper_model: str
    active_connection_id: str = ""
    connections: list[RuntimeSettingsConnection] = Field(default_factory=list)


class RuntimeConnectionDraft(BaseModel):
    connection_id: str = ""
    provider_choice: str = ""
    label: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_title: str = ""


class RuntimeSettingsSaveRequest(BaseModel):
    ui_locale: str = ""
    whisper_model: str = ""
    clear_saved_secret: bool = False
    connection: RuntimeConnectionDraft


class RuntimeConnectionTestRequest(BaseModel):
    connection: RuntimeConnectionDraft


class RuntimeConnectionTestResponse(BaseModel):
    provider: str
    base_url: str
    service_base_url: str = ""
    health_endpoint: str = ""
    discovered_models: list[str] = Field(default_factory=list)
    tested_at: str = ""
    content_preview: str = ""


class WhisperModelStatusResponse(BaseModel):
    model: str
    repo_id: str
    cached: bool = False
    cached_path: str = ""
    recommended: bool = False
    recommendation_reason: str = ""


class LocalApiContractResponse(BaseModel):
    contract_version: Literal["phase2-local-v1"] = PHASE_2_LOCAL_API_VERSION
    session_mode: Literal["local_guest"] = "local_guest"
    auth_mode: Literal["none"] = "none"
    tenancy_mode: Literal["none"] = "none"
    storage_mode: Literal["local_app_data"] = "local_app_data"
    product_routes: list[str] = Field(default_factory=lambda: list(CANONICAL_PRODUCT_API_ROUTES))
    local_support_routes: list[str] = Field(default_factory=lambda: list(LOCAL_SUPPORT_API_ROUTES))
    local_runtime_routes: list[str] = Field(default_factory=lambda: list(LOCAL_RUNTIME_MANAGEMENT_ROUTES))


class ErrorResponse(BaseModel):
    code: ErrorCode
    detail: str


class StorageAreaSummary(BaseModel):
    path: str
    size_bytes: int
    file_count: int


class MaintenanceStorageResponse(BaseModel):
    app_data_root: str
    cache_root: str
    areas: dict[str, StorageAreaSummary]


class MaintenanceCleanupRequest(BaseModel):
    target: CleanupTarget
    dry_run: bool = False


class MaintenanceCleanupResponse(BaseModel):
    target: CleanupTarget
    dry_run: bool
    deleted_file_count: int
    freed_bytes: int
    warnings: list[str] = Field(default_factory=list)


class SupportBundleCreateRequest(BaseModel):
    include_reports: bool = False
    include_recordings: bool = False
    include_uploads: bool = False
    include_runtime_health: bool = False
    client_snapshot: dict[str, Any] = Field(default_factory=dict)
    client_diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class SupportBundleCreateResponse(BaseModel):
    bundle_id: str
    filename: str
    size_bytes: int
    expires_at: datetime


class UploadResponse(BaseModel):
    audio_id: str
    stored_path: str
    sha1: str
    original_name: str = ""


class AssessmentCreateRequest(BaseModel):
    audio_id: str
    whisper: str
    provider: str
    llm_model: str
    expected_language: str
    feedback_language: str
    speaker_id: str
    task_family: str
    theme: str
    target_duration_sec: int
    target_cefr: str | None = None
    language_profile_key: str | None = None
    label: str = ""
    notes: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    openrouter_http_referer: str = ""
    openrouter_app_title: str = ""
    dry_run: bool = False


class AssessmentCreateResponse(BaseModel):
    assessment_id: str
    status: JobStatus


class AssessmentSummary(BaseModel):
    score_overall: float | None = None
    band: str = ""
    next_focus: str = ""


class AssessmentStatusResponse(BaseModel):
    assessment_id: str
    status: JobStatus
    phase: str
    progress: float
    error: ErrorResponse | None = None
    report_path: str | None = None
    summary: AssessmentSummary | None = None
    payload: dict[str, Any] | None = None


class HistoryRow(BaseModel):
    timestamp: str = ""
    session_id: str = ""
    speaker_id: str = ""
    learning_language: str = ""
    theme: str = ""
    task_family: str = ""
    overall: Any = ""
    wpm: Any = ""
    report_path: str = ""
    requires_human_review: Any = ""
    duration_pass: Any = ""
    topic_pass: Any = ""
    language_pass: Any = ""
    min_words_pass: Any = ""
    top_priorities: Any = []
    grammar_error_categories: Any = []
    coherence_issue_categories: Any = []
    final_score: Any = ""
    band: Any = ""


class HistoryResponse(BaseModel):
    items: list[HistoryRow]


class HistoryDetailResponse(BaseModel):
    payload: dict[str, Any]


class SampleItem(BaseModel):
    sample_id: str
    language: str
    cefr: str
    title: str
    path: str


class SamplesResponse(BaseModel):
    items: list[SampleItem]
