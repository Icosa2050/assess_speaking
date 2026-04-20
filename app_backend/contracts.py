from __future__ import annotations

from enum import Enum
from typing import Any

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


class ErrorResponse(BaseModel):
    code: ErrorCode
    detail: str


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
