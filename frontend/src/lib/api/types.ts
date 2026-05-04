export type JsonRecord = Record<string, unknown>;

export type ErrorCode =
  | "backend_unavailable"
  | "local_provider_not_installed"
  | "local_provider_not_running"
  | "missing_ffmpeg"
  | "missing_whisper_model"
  | "validation_error"
  | "configuration_error"
  | "runtime_error"
  | "storage_error"
  | "cancellation_error";

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type CleanupTarget = "tmp" | "jobs" | "logs" | "all_safe";
export type ConnectionSecretState = "absent" | "present" | "missing";

export type FrozenApiRoute =
  | "/v1/health"
  | "/v1/diagnostics"
  | "/v1/runtime"
  | "/v1/runtime/settings"
  | "/v1/runtime/settings/test-connection"
  | `/v1/runtime/settings/connections/${string}`
  | `/v1/runtime/settings/connections/${string}/default`
  | `/v1/runtime/whisper-models/${string}`
  | `/v1/runtime/whisper-models/${string}/download`
  | "/v1/maintenance/storage"
  | "/v1/maintenance/cleanup"
  | "/v1/support-bundles"
  | `/v1/support-bundles/${string}`
  | "/v1/uploads"
  | "/v1/assessments"
  | `/v1/assessments/${string}`
  | `/v1/assessments/${string}/cancel`
  | "/v1/history"
  | `/v1/history/${string}`
  | "/v1/samples";

export interface HealthResponse {
  status: string;
  version: string;
  app_data_root: string;
  uptime_sec: number;
}

export interface DiagnosticItem {
  key: string;
  status: string;
  title_key: string;
  detail_key: string;
  detail_args: JsonRecord;
}

export interface DiagnosticsResponse {
  items: DiagnosticItem[];
}

export interface RuntimeResponse {
  configured: boolean;
  provider: string;
  model: string;
  base_url: string;
  requires_api_key: boolean;
  has_api_key: boolean;
}

export interface RuntimeSettingsConnection {
  connection_id: string;
  provider_key: string;
  provider_choice: string;
  provider_label: string;
  label: string;
  model: string;
  base_url: string;
  is_default: boolean;
  is_local: boolean;
  requires_api_key: boolean;
  has_api_key: boolean;
  secret_state: ConnectionSecretState;
  last_test_status: string;
  last_tested_at: string;
  openrouter_http_referer: string;
  openrouter_app_title: string;
  provider_metadata: JsonRecord;
}

export interface RuntimeSettingsResponse {
  ui_locale: string;
  whisper_model: string;
  active_connection_id: string;
  connections: RuntimeSettingsConnection[];
}

export interface RuntimeConnectionDraft {
  connection_id?: string;
  provider_choice: string;
  label: string;
  model: string;
  base_url: string;
  api_key?: string;
  openrouter_http_referer?: string;
  openrouter_app_title?: string;
}

export interface RuntimeSettingsSaveRequest {
  ui_locale: string;
  whisper_model: string;
  clear_saved_secret?: boolean;
  connection: RuntimeConnectionDraft;
}

export interface RuntimeConnectionTestRequest {
  connection: RuntimeConnectionDraft;
}

export interface RuntimeConnectionTestResponse {
  provider: string;
  base_url: string;
  service_base_url: string;
  health_endpoint: string;
  discovered_models: string[];
  tested_at: string;
  content_preview: string;
}

export interface WhisperModelStatusResponse {
  model: string;
  repo_id: string;
  cached: boolean;
  cached_path: string;
  recommended: boolean;
  recommendation_reason: string;
}

export interface ErrorResponse {
  code: ErrorCode;
  detail: string;
}

export interface StorageAreaSummary {
  path: string;
  size_bytes: number;
  file_count: number;
}

export interface MaintenanceStorageResponse {
  app_data_root: string;
  cache_root: string;
  areas: Record<string, StorageAreaSummary>;
}

export interface MaintenanceCleanupRequest {
  target: CleanupTarget;
  dry_run?: boolean;
}

export interface MaintenanceCleanupResponse {
  target: CleanupTarget;
  dry_run: boolean;
  deleted_file_count: number;
  freed_bytes: number;
  warnings: string[];
}

export interface SupportBundleCreateRequest {
  include_reports?: boolean;
  include_recordings?: boolean;
  include_uploads?: boolean;
  include_runtime_health?: boolean;
  client_snapshot?: JsonRecord;
  client_diagnostics?: JsonRecord[];
}

export interface SupportBundleCreateResponse {
  bundle_id: string;
  filename: string;
  size_bytes: number;
  expires_at: string;
}

export interface UploadResponse {
  audio_id: string;
  stored_path: string;
  sha1: string;
  original_name: string;
}

export interface AssessmentCreateRequest {
  audio_id: string;
  whisper: string;
  provider: string;
  llm_model: string;
  expected_language: string;
  feedback_language: string;
  speaker_id: string;
  task_family: string;
  theme: string;
  target_duration_sec: number;
  target_cefr?: string | null;
  language_profile_key?: string | null;
  label?: string;
  notes?: string;
  llm_base_url?: string;
  llm_api_key?: string;
  openrouter_http_referer?: string;
  openrouter_app_title?: string;
  dry_run?: boolean;
}

export interface AssessmentCreateResponse {
  assessment_id: string;
  status: JobStatus;
}

export interface AssessmentSummary {
  score_overall?: number | null;
  band?: string;
  next_focus?: string;
}

export interface AssessmentStatusResponse {
  assessment_id: string;
  status: JobStatus;
  phase: string;
  progress: number;
  error?: ErrorResponse | null;
  report_path?: string | null;
  summary?: AssessmentSummary | null;
  payload?: JsonRecord | null;
}

export interface HistoryRow {
  timestamp: string;
  session_id: string;
  speaker_id: string;
  learning_language: string;
  theme: string;
  task_family: string;
  overall: unknown;
  wpm: unknown;
  report_path: string;
  requires_human_review: unknown;
  duration_pass: unknown;
  topic_pass: unknown;
  language_pass: unknown;
  min_words_pass: unknown;
  top_priorities: string[];
  grammar_error_categories: string[];
  coherence_issue_categories: string[];
  final_score: unknown;
  band: unknown;
}

export interface HistoryResponse {
  items: HistoryRow[];
}

export interface HistoryDetailResponse {
  payload: JsonRecord;
}

export interface SampleItem {
  sample_id: string;
  language: string;
  cefr: string;
  title: string;
  path: string;
}

export interface SamplesResponse {
  items: SampleItem[];
}
