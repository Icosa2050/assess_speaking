import { buildApiUrl, resolveLocalDesktopApiBaseUrl } from "@/lib/runtime/environment";

import type {
  AssessmentCreateRequest,
  AssessmentCreateResponse,
  AssessmentStatusResponse,
  DiagnosticsResponse,
  ErrorCode,
  ErrorResponse,
  FrozenApiRoute,
  HealthResponse,
  HistoryDetailResponse,
  HistoryResponse,
  JsonRecord,
  MaintenanceCleanupRequest,
  MaintenanceCleanupResponse,
  MaintenanceStorageResponse,
  RuntimeConnectionTestRequest,
  RuntimeConnectionTestResponse,
  RuntimeResponse,
  RuntimeSettingsSaveRequest,
  RuntimeSettingsResponse,
  SamplesResponse,
  SupportBundleCreateRequest,
  SupportBundleCreateResponse,
  UploadResponse,
  WhisperModelStatusResponse,
} from "./types";

type ClientOptions = {
  baseUrl?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
};

type JsonRequestOptions = ClientOptions & {
  headers?: HeadersInit;
};

const DEFAULT_TIMEOUT_MS = 10_000;
const LONG_RUNNING_TIMEOUT_MS = 30_000;

const isErrorCode = (value: string): value is ErrorCode =>
  [
    "backend_unavailable",
    "local_provider_not_installed",
    "local_provider_not_running",
    "missing_ffmpeg",
    "missing_whisper_model",
    "validation_error",
    "configuration_error",
    "runtime_error",
    "storage_error",
    "cancellation_error",
  ].includes(value);

const mergeSignals = (signal: AbortSignal | undefined, timeoutMs: number): AbortSignal => {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }

  controller.signal.addEventListener(
    "abort",
    () => {
      window.clearTimeout(timeoutId);
    },
    { once: true },
  );

  return controller.signal;
};

const parseErrorResponse = async (response: Response): Promise<ErrorResponse> => {
  let fallbackDetail = `${response.status} ${response.statusText}`.trim();
  let code: ErrorCode = "backend_unavailable";

  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (payload && typeof payload.detail === "object" && payload.detail !== null) {
      const detailPayload = payload.detail as { code?: unknown; detail?: unknown };
      if (typeof detailPayload.detail === "string" && detailPayload.detail.trim()) {
        fallbackDetail = detailPayload.detail;
      }
      if (typeof detailPayload.code === "string" && isErrorCode(detailPayload.code)) {
        code = detailPayload.code;
      }
    }
  } catch {
    // Keep the HTTP status fallback.
  }

  return {
    code,
    detail: fallbackDetail,
  };
};

export class ApiClientError extends Error {
  readonly code: ErrorCode;
  readonly detail: string;
  readonly responseStatus: number;

  constructor(responseStatus: number, error: ErrorResponse) {
    super(error.detail);
    this.name = "ApiClientError";
    this.code = error.code;
    this.detail = error.detail;
    this.responseStatus = responseStatus;
  }
}

const requestJson = async <TResponse>(
  route: FrozenApiRoute,
  init: RequestInit,
  options: JsonRequestOptions = {},
): Promise<TResponse> => {
  const signal = mergeSignals(options.signal, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const response = await fetch(buildApiUrl(route, options.baseUrl), {
    ...init,
    headers: {
      Accept: "application/json",
      ...options.headers,
    },
    signal,
  });

  if (!response.ok) {
    throw new ApiClientError(response.status, await parseErrorResponse(response));
  }

  return (await response.json()) as TResponse;
};

const requestBlob = async (
  route: FrozenApiRoute,
  init: RequestInit,
  options: ClientOptions = {},
): Promise<Blob> => {
  const signal = mergeSignals(options.signal, options.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS);
  const response = await fetch(buildApiUrl(route, options.baseUrl), {
    ...init,
    signal,
  });

  if (!response.ok) {
    throw new ApiClientError(response.status, await parseErrorResponse(response));
  }

  return response.blob();
};

export const createApiClient = (baseUrl = resolveLocalDesktopApiBaseUrl()) => ({
  getHealth: (options?: ClientOptions) =>
    requestJson<HealthResponse>("/v1/health", { method: "GET" }, { ...options, baseUrl }),

  getDiagnostics: (options?: ClientOptions) =>
    requestJson<DiagnosticsResponse>("/v1/diagnostics", { method: "GET" }, { ...options, baseUrl }),

  getRuntime: (options?: ClientOptions) =>
    requestJson<RuntimeResponse>("/v1/runtime", { method: "GET" }, { ...options, baseUrl }),

  getRuntimeSettings: (options?: ClientOptions) =>
    requestJson<RuntimeSettingsResponse>(
      "/v1/runtime/settings",
      { method: "GET" },
      { ...options, baseUrl },
    ),

  putRuntimeSettings: (
    request: RuntimeSettingsSaveRequest,
    options?: ClientOptions,
  ) =>
    requestJson<RuntimeSettingsResponse>(
      "/v1/runtime/settings",
      {
        method: "PUT",
        body: JSON.stringify(request),
      },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
        headers: {
          "Content-Type": "application/json",
        },
      },
    ),

  postRuntimeSettingsTestConnection: (
    request: RuntimeConnectionTestRequest,
    options?: ClientOptions,
  ) =>
    requestJson<RuntimeConnectionTestResponse>(
      "/v1/runtime/settings/test-connection",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
        headers: {
          "Content-Type": "application/json",
        },
      },
    ),

  postRuntimeSettingsSetDefault: (connectionId: string, options?: ClientOptions) =>
    requestJson<RuntimeSettingsResponse>(
      `/v1/runtime/settings/connections/${connectionId}/default`,
      { method: "POST" },
      { ...options, baseUrl, timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS },
    ),

  deleteRuntimeSettingsConnection: (connectionId: string, options?: ClientOptions) =>
    requestJson<RuntimeSettingsResponse>(
      `/v1/runtime/settings/connections/${connectionId}`,
      { method: "DELETE" },
      { ...options, baseUrl, timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS },
    ),

  getWhisperModelStatus: (modelSize: string, options?: ClientOptions) =>
    requestJson<WhisperModelStatusResponse>(
      `/v1/runtime/whisper-models/${modelSize}`,
      { method: "GET" },
      { ...options, baseUrl },
    ),

  postWhisperModelDownload: (modelSize: string, options?: ClientOptions) =>
    requestJson<WhisperModelStatusResponse>(
      `/v1/runtime/whisper-models/${modelSize}/download`,
      { method: "POST" },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
      },
    ),

  uploadAudio: (file: File, options?: ClientOptions & { filename?: string }) => {
    const formData = new FormData();
    formData.append("file", file, options?.filename || file.name || "audio.wav");

    return requestJson<UploadResponse>(
      "/v1/uploads",
      {
        method: "POST",
        body: formData,
      },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
      },
    );
  },

  createAssessment: (request: AssessmentCreateRequest, options?: ClientOptions) =>
    requestJson<AssessmentCreateResponse>(
      "/v1/assessments",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
      {
        ...options,
        baseUrl,
        headers: {
          "Content-Type": "application/json",
        },
      },
    ),

  getAssessmentStatus: (assessmentId: string, options?: ClientOptions) =>
    requestJson<AssessmentStatusResponse>(
      `/v1/assessments/${assessmentId}`,
      { method: "GET" },
      { ...options, baseUrl },
    ),

  cancelAssessment: (assessmentId: string, options?: ClientOptions) =>
    requestJson<AssessmentStatusResponse>(
      `/v1/assessments/${assessmentId}/cancel`,
      { method: "POST" },
      { ...options, baseUrl },
    ),

  getHistory: (options?: ClientOptions) =>
    requestJson<HistoryResponse>("/v1/history", { method: "GET" }, { ...options, baseUrl }),

  getHistoryDetail: (sessionId: string, options?: ClientOptions) =>
    requestJson<HistoryDetailResponse>(
      `/v1/history/${sessionId}`,
      { method: "GET" },
      { ...options, baseUrl },
    ),

  getSamples: (options?: ClientOptions) =>
    requestJson<SamplesResponse>("/v1/samples", { method: "GET" }, { ...options, baseUrl }),

  getMaintenanceStorage: (options?: ClientOptions) =>
    requestJson<MaintenanceStorageResponse>(
      "/v1/maintenance/storage",
      { method: "GET" },
      { ...options, baseUrl },
    ),

  postMaintenanceCleanup: (request: MaintenanceCleanupRequest, options?: ClientOptions) =>
    requestJson<MaintenanceCleanupResponse>(
      "/v1/maintenance/cleanup",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
        headers: {
          "Content-Type": "application/json",
        },
      },
    ),

  createSupportBundle: (request: SupportBundleCreateRequest, options?: ClientOptions) =>
    requestJson<SupportBundleCreateResponse>(
      "/v1/support-bundles",
      {
        method: "POST",
        body: JSON.stringify(request),
      },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
        headers: {
          "Content-Type": "application/json",
        },
      },
    ),

  downloadSupportBundle: (bundleId: string, options?: ClientOptions) =>
    requestBlob(
      `/v1/support-bundles/${bundleId}`,
      { method: "GET" },
      {
        ...options,
        baseUrl,
        timeoutMs: options?.timeoutMs ?? LONG_RUNNING_TIMEOUT_MS,
      },
    ),
});

export type ApiClient = ReturnType<typeof createApiClient>;

export const apiClient = createApiClient();

export const toHistoryPayload = (response: HistoryDetailResponse): JsonRecord => response.payload;
