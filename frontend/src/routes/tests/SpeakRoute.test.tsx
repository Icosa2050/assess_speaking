import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppFrame } from "@/App";
import { renderWithProviders } from "@/test/renderWithProviders";
import { queryKeys } from "@/lib/query/queryClient";

vi.mock("@/lib/api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    code = "runtime_error";
    detail: string;
    responseStatus = 500;

    constructor(responseStatus: number, error: { detail: string }) {
      super(error.detail);
      this.detail = error.detail;
      this.responseStatus = responseStatus;
    }
  },
  apiClient: {
    getSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getResumeSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    cancelAssessment: vi.fn(),
    createAssessment: vi.fn(),
    getAssessmentStatus: vi.fn(),
    getRuntime: vi.fn(),
    getDiagnostics: vi.fn(),
    getRuntimeSettings: vi.fn(),
    getWhisperModelStatus: vi.fn(),
    uploadAudio: vi.fn(),
    getUploadLimits: vi.fn(),
  },
}));

import { apiClient, ApiClientError } from "@/lib/api/client";

const mockedCancelAssessment = vi.mocked(apiClient.cancelAssessment);
const mockedCreateAssessment = vi.mocked(apiClient.createAssessment);
const mockedGetAssessmentStatus = vi.mocked(apiClient.getAssessmentStatus);
const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetRuntimeSettings = vi.mocked(apiClient.getRuntimeSettings);
const mockedUploadAudio = vi.mocked(apiClient.uploadAudio);

type FakeRecorderEvent = {
  data: Blob;
};

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = [];
  static isTypeSupported = vi.fn((mimeType: string) => mimeType === "audio/webm;codecs=opus");

  mimeType: string;
  ondataavailable: ((event: FakeRecorderEvent) => void) | null = null;
  onerror: ((event: { error?: Error }) => void) | null = null;
  onstop: (() => void) | null = null;
  state: "inactive" | "recording" = "inactive";

  constructor(
    readonly stream: MediaStream,
    options: MediaRecorderOptions = {},
  ) {
    this.mimeType = options.mimeType || "audio/webm";
    FakeMediaRecorder.instances.push(this);
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    this.ondataavailable?.({
      data: new Blob(["browser audio"], { type: this.mimeType }),
    });
    this.onstop?.();
  }
}

const validDraftState = {
  microphoneSetupPassed: true,
  draft: {
    speakerId: "bern",
    learningLanguage: "en",
    learningLanguageLabel: "English",
    cefrLevel: "B2" as const,
    themeId: "work-home-b2",
    themeLabel: "The pros and cons of working from home",
    taskFamily: "opinion_monologue" as const,
    durationSec: 120 as const,
    promptId: "work-home-b2",
    promptText: "Give your opinion on working from home.",
  },
  preferences: {
    activeConnectionId: "conn-primary",
    setupComplete: true,
  },
};

const uploadFile = new File(["audio"], "attempt.wav", {
  type: "audio/wav",
});

const expectDecorativeIcon = (element: HTMLElement) => {
  expect(element.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument();
};

describe("Speak route", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows an editable, clearable retry goal before recording", async () => {
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Add a clear ending", retryGoalEligible: true } } });
    const brief = await screen.findByTestId("speak-retry-brief");
    expect(brief).toHaveTextContent("Add a clear ending");
    expect(screen.queryByTestId("speak-retry-announcement")).not.toBeInTheDocument();
    await waitFor(() => expect(within(brief).getByRole("heading")).toHaveFocus());
    expect(screen.getByTestId("speak-retry-goal-text")).toHaveTextContent("Add a clear ending");
    expect(screen.getByTestId("speak.record_start")).toBeInTheDocument();
    fireEvent.change(screen.getByTestId("speak-retry-goal"), { target: { value: "Use one transition" } });
    expect(store.getState().draft.coachingGoal).toBe("Use one transition");
    fireEvent.click(within(brief).getByRole("button", { name: "Clear goal" }));
    expect(store.getState().draft.coachingGoal).toBe("");
  });
  it("does not refocus the brief on queued, running, or failed transitions", async () => {
    mockedGetAssessmentStatus.mockImplementation(() => new Promise(() => undefined));
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Add a clear ending", retryGoalEligible: true } } });
    const heading = within(await screen.findByTestId("speak-retry-brief")).getByRole("heading");
    await waitFor(() => expect(heading).toHaveFocus());
    const historyLink = screen.getByTestId("home.open_history");
    historyLink.focus();
    for (const status of ["queued", "running", "failed"]) {
      act(() => store.getState().setRecordingJob({ status }));
      expect(historyLink).toHaveFocus();
      expect(heading).not.toHaveFocus();
    }
  });
  it("redirects a completed take before a retained goal can reappear", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: {
      ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Retained goal", retryGoalEligible: true },
      recording: { job: { assessmentId: "asmt", status: "completed", phase: "", progress: 1,
        error: "", reportPath: "completed" }, assessmentState: "completed" },
      review: { reportId: "completed", payload: {} },
    } });
    await screen.findByTestId("review-action-try-again");
    expect(screen.queryByTestId("speak-retry-brief")).not.toBeInTheDocument();
  });
  it("shows a History fallback when assessment completes without a report", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: {
      ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Retained goal", retryGoalEligible: true },
      recording: { assessmentState: "completed" },
    } });
    const status = await screen.findByTestId("speak-completed-no-report");
    expect(status).toHaveTextContent("Assessment completed.");
    expect(within(status).getByRole("link", { name: "Open history" })).toHaveAttribute("href", "/history");
    expect(screen.queryByTestId("speak-retry-brief")).not.toBeInTheDocument();
  });
  it("withholds the goal editor from a non-assessable parent", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "invalid", coachingGoal: "Stale goal", retryGoalEligible: false } } });
    expect(await screen.findByTestId("speak.session_summary")).toBeVisible();
    expect(screen.queryByTestId("speak-retry-brief")).not.toBeInTheDocument();
  });
  it("asks who owns a restored retry before showing its goal", async () => {
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, retryRestorePending: true, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Add a clear ending", retryGoalEligible: true } } });
    expect(await screen.findByTestId("speak-retry-resume")).toHaveTextContent("bern");
    expect(screen.queryByTestId("speak-retry-brief")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Resume this retry" }));
    expect(store.getState().retryRestorePending).toBe(false);
    expect(await screen.findByTestId("speak-retry-brief")).toHaveTextContent("Add a clear ending");
  });
  it("keeps recording unavailable while checking runtime setup for a retry", async () => {
    mockedGetRuntime.mockImplementationOnce(() => new Promise(() => undefined));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Add a clear ending", retryGoalEligible: true } } });
    expect(await screen.findByTestId("speak-retry-runtime-check")).toHaveTextContent("Add a clear ending");
    expect(screen.queryByTestId("speak.record_start")).not.toBeInTheDocument();
  });
  it("locks the brief during an actual recording and unlocks it afterward", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, draft: { ...validDraftState.draft,
        retryOfSessionId: "first", coachingGoal: "Add a clear ending", retryGoalEligible: true } } });
    const brief = await screen.findByTestId("speak-retry-brief");
    const goalField = screen.getByTestId("speak-retry-goal");
    const clearGoal = within(brief).getByRole("button", { name: "Clear goal" });
    expect(goalField).toBeEnabled();
    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    await waitFor(() => expect(goalField).toBeDisabled());
    expect(clearGoal).toBeDisabled();
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));
    await waitFor(() => expect(goalField).toBeEnabled());
    expect(clearGoal).toBeEnabled();
  });
  it("keeps recording blocked without calibration while upload remains usable", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, microphoneSetupPassed: false } });
    expect(await screen.findByTestId("speak.record_start")).toBeDisabled();
    expect(screen.getByTestId("speak.microphone_setup")).toBeEnabled();
    expect(getUserMedia).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.input_mode_upload"));
    expect(screen.getByTestId("speak.upload_input")).toBeEnabled();
  });

  it("redirects a direct Speak link when saved setup points to a missing runtime", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false, provider: "", model: "", base_url: "",
      requires_api_key: false, has_api_key: false,
    });
    const { queryClient } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.getByText("Finish setup to enter the recording step")).toBeVisible();
    expect(screen.getByText(/needs attention before you can continue/)).toBeVisible();
    expect(screen.queryByTestId("speak.record_start")).not.toBeInTheDocument();

    act(() => queryClient.setQueryData(["runtime"], {
      configured: true, provider: "ollama_local", model: "llama3.2:3b", base_url: "http://localhost:11434",
      requires_api_key: false, has_api_key: false,
    }));
    fireEvent.click(await screen.findByRole("button", { name: "Return to the recording step" }));
    expect(await screen.findByTestId("speak.recording_panel", {}, { timeout: 5_000 })).toBeVisible();
  });

  it("allows local speech when settings fetch fails without cached settings", async () => {
    mockedGetRuntimeSettings.mockRejectedValue(new Error("settings offline"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    expect(await screen.findByTestId("speak.recording_panel", {}, { timeout: 5_000 })).toBeVisible();
    expect(screen.queryByTestId("runtime_setup.screen")).not.toBeInTheDocument();
  });

  it("redirects a direct Speak link when a selected cloud tutor lacks its key", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "openrouter", model: "vendor/model", base_url: "https://example.test",
      requires_api_key: true, has_api_key: false,
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.queryByTestId("speak.record_start")).not.toBeInTheDocument();
  });
  const stopTrack = vi.fn();
  const getUserMedia = vi.fn();

  beforeEach(() => {
    window.localStorage.removeItem("assess-speaking.first-take-recorded");
    vi.mocked(apiClient.getUploadLimits).mockResolvedValue({ max_bytes: 104857600, available_bytes: 104857600 });
    vi.clearAllMocks();
    vi.mocked(apiClient.getSharingRoute).mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null });
    stopTrack.mockClear();
    FakeMediaRecorder.instances = [];
    FakeMediaRecorder.isTypeSupported.mockClear();
    getUserMedia.mockResolvedValue({
      getTracks: () => [{ stop: stopTrack }],
    } as unknown as MediaStream);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia,
      },
    });
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    ] });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "small",
      active_connection_id: "conn-primary",
      connections: [],
    });
    mockedUploadAudio.mockResolvedValue({
      audio_id: "audio-1",
      stored_path: "/tmp/audio-1.wav",
      sha1: "sha1",
      original_name: "attempt.wav",
    });
    mockedCreateAssessment.mockResolvedValue({
      assessment_id: "asmt-1",
      status: "queued",
    });
    mockedCancelAssessment.mockResolvedValue({
      assessment_id: "asmt-1",
      status: "cancelled",
      phase: "cancelled",
      progress: 0,
      error: null,
      report_path: null,
      payload: null,
      summary: null,
    });
    mockedGetAssessmentStatus.mockResolvedValue({
      assessment_id: "asmt-1",
      status: "running",
      phase: "transcribing",
      progress: 0.35,
      error: null,
      report_path: null,
      payload: null,
      summary: null,
    });
  });

  it("shows one current Practice step and explains unavailable Review in AppFrame", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    const nav = screen.getByRole("navigation", { name: "Main navigation" });
    expect(await screen.findByTestId("speak.input_mode_upload")).toBeVisible();
    expect(within(nav).getByRole("link", { name: "Session Setup" })).toHaveAttribute("href", "/session-setup");
    expect(within(nav).getByRole("link", { name: "Speak" })).toHaveAttribute("aria-current", "page");
    const review = within(nav).getByRole("link", { name: "Review" });
    expect(review).toHaveAttribute("aria-disabled", "true");
    expect(review).toHaveAttribute("href", "/review");
    expect(review).toHaveAttribute("tabindex", "0");
    expect(review).toHaveAccessibleDescription("Submit a speaking attempt before opening the review screen.");
    expect(nav.querySelectorAll('[aria-current="page"]')).toHaveLength(1);
  });

  it("enables the Review navigation link when a report exists", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: { ...validDraftState, review: { reportId: "report-1" } },
    });

    const nav = screen.getByRole("navigation", { name: "Main navigation" });
    expect(await screen.findByTestId("speak.input_mode_upload")).toBeVisible();
    const review = within(nav).getByRole("link", { name: "Review" });
    expect(review).toHaveAttribute("href", "/review");
    expect(review).not.toHaveAttribute("aria-disabled");
    expect(review).not.toHaveAccessibleDescription();
    expect(nav.querySelectorAll('[aria-current="page"]')).toHaveLength(1);
  });

  it("stops before uploading if the displayed sharing route changes", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    const displayed = await apiClient.getSharingRoute();
    vi.mocked(apiClient.getSharingRoute).mockResolvedValue({ ...displayed, fingerprint: "b".repeat(64), analysis: { ...displayed.analysis, host: "other.example", local: false } });
    fireEvent.click(screen.getByTestId("speak.submit"));
    await screen.findByText(/Sharing destinations changed/i);
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "attempt.wav");
  });

  it("reuses a saved upload after a busy assessment response", async () => {
    mockedCreateAssessment.mockRejectedValueOnce(new ApiClientError(409, { code: "runtime_error", detail: "Another attempt is still being analysed." }));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [new File(["audio"], "retry.wav", { type: "audio/wav" })] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    expect(screen.getByTestId("speak.media_state")).toHaveTextContent("Audio uploaded and saved. Start the assessment to continue.");
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "retry.wav");
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
  });

  it("retries an upload with the same request ID after its response is lost", async () => {
    mockedUploadAudio.mockRejectedValueOnce(new DOMException("Upload connection lost.", "NetworkError"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Upload connection lost"));
    expect(screen.getByTestId("speak.media_state")).toHaveTextContent("Upload confirmation was lost");
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedUploadAudio).toHaveBeenCalledTimes(2));
    const first = mockedUploadAudio.mock.calls[0][1]?.requestId;
    expect(first).toBeTruthy();
    expect(mockedUploadAudio.mock.calls[1][1]?.requestId).toBe(first);
  });

  it("keeps upload percentages out of the live status and describes an aborted upload accurately", async () => {
    mockedUploadAudio.mockImplementationOnce((_file, options) => new Promise((_resolve, reject) => {
      options?.signal?.addEventListener("abort", () => reject(new DOMException("Upload cancelled", "AbortError")));
      options?.onProgress?.(12);
      options?.onProgress?.(38);
    }));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    const panel = await screen.findByTestId("speak.status_panel");
    await waitFor(() => expect(panel).toHaveTextContent("38%"));
    expect(within(panel).queryByRole("status")).toBeNull();
    fireEvent.click(screen.getByTestId("speak.cancel_assessment"));
    await waitFor(() => expect(panel).toHaveTextContent("Upload cancelled"));
    expect(screen.getByTestId("speak.media_state")).not.toHaveTextContent("Upload confirmation was lost");
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
  });

  it("shows a failure returned by Cancel without calling it a cancellation", async () => {
    mockedCancelAssessment.mockResolvedValueOnce({ assessment_id: "asmt-1", status: "failed", phase: "failed", last_phase: "scoring_rubric", progress: 1,
      error: { code: "runtime_error", detail: "Scoring failed" }, report_path: null, payload: null, summary: null });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await screen.findByTestId("speak.cancel_assessment");
    fireEvent.click(screen.getByTestId("speak.cancel_assessment"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Assessment failed: Scoring failed"));
    expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Stopped during Scoring rubric.");
    expect(screen.getByTestId("speak.status_panel")).not.toHaveTextContent("Assessment failed: The assessment was cancelled");
  });

  it("keeps a cancelled assessment resumable while its provider request finishes", async () => {
    mockedCreateAssessment.mockResolvedValueOnce({ assessment_id: "asmt-1", status: "queued" })
      .mockRejectedValueOnce(new ApiClientError(409, { code: "runtime_error", detail: "A prior provider request is still finishing." }))
      .mockResolvedValueOnce({ assessment_id: "asmt-2", status: "queued" });
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    fireEvent.click(await screen.findByTestId("speak.cancel_assessment"));
    fireEvent.click(await screen.findByTestId("speak.retry_assessment"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Retry is not ready yet"));
    expect(store.getState().recording.job.assessmentId).toBe("asmt-1");
    expect(screen.getByTestId("speak.retry_assessment")).toBeVisible();
    fireEvent.click(screen.getByTestId("speak.retry_assessment"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(3));
    expect(mockedCreateAssessment.mock.calls[2][0]).toMatchObject({ resume_assessment_id: "asmt-1", audio_id: "audio-1" });
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
  });

  it("recovers a lost accepted response with the same submission ID", async () => {
    mockedCreateAssessment.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    const first = mockedCreateAssessment.mock.calls[0][0];
    const recovered = mockedCreateAssessment.mock.calls[1][0];
    expect(first.request_id).toBeTruthy();
    expect(recovered).toEqual(first);
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Your assessment is running"));
  });

  it("explains a stall and retries only after cancellation with retained audio", async () => {
    mockedGetAssessmentStatus.mockResolvedValue({ assessment_id: "asmt-1", status: "running",
      phase: "scoring_rubric", progress: 0, created_at: new Date(Date.now() - 240_000).toISOString(),
      updated_at: new Date(Date.now() - 240_000).toISOString(),
      phase_started_at: new Date(Date.now() - 240_000).toISOString(),
      worker_seen_at: new Date(Date.now() - 240_000).toISOString(), quiet_warning_after_sec: 210 });
    mockedCreateAssessment.mockResolvedValueOnce({ assessment_id: "asmt-1", status: "queued" })
      .mockResolvedValueOnce({ assessment_id: "asmt-2", status: "queued" });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "true"));
    expect(screen.queryByTestId("speak.retry_assessment")).not.toBeInTheDocument();
    expect(screen.getByTestId("speak.media_state")).toHaveTextContent("Audio uploaded and retained");
    fireEvent.click(screen.getByTestId("speak.cancel_assessment"));
    const retry = await screen.findByTestId("speak.retry_assessment");
    fireEvent.click(retry);
    fireEvent.click(retry);
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    expect(mockedCancelAssessment).toHaveBeenCalledTimes(1);
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
    expect(mockedCreateAssessment.mock.calls[1][0]).toMatchObject({
      audio_id: "audio-1", resume_assessment_id: "asmt-1",
    });
    expect(mockedCreateAssessment.mock.calls[1][0].request_id).not.toBe(mockedCreateAssessment.mock.calls[0][0].request_id);
  });

  it("keeps monitoring a stalled job when cancellation fails", async () => {
    mockedGetAssessmentStatus.mockResolvedValue({ assessment_id: "asmt-1", status: "running",
      phase: "scoring_rubric", progress: 0, created_at: new Date(Date.now() - 240_000).toISOString(),
      updated_at: new Date(Date.now() - 240_000).toISOString(),
      worker_seen_at: new Date(Date.now() - 240_000).toISOString(), quiet_warning_after_sec: 210 });
    mockedCancelAssessment.mockRejectedValueOnce(new ApiClientError(503, { code: "runtime_error", detail: "Connection lost" }));
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "true"));
    expect(screen.queryByTestId("speak.retry_assessment")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("speak.cancel_assessment"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Cancellation did not finish: Connection lost"));
    expect(store.getState().recording.job.assessmentId).toBe("asmt-1");
    expect(screen.getByTestId("speak.cancel_assessment")).toBeVisible();
    expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "true");
    expect(mockedCreateAssessment).toHaveBeenCalledTimes(1);
  });

  it("replaces a failed cancellation message when the worker later fails", async () => {
    const { queryClient } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    mockedCancelAssessment.mockRejectedValueOnce(new ApiClientError(503, { code: "runtime_error", detail: "Connection lost" }));
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    fireEvent.click(await screen.findByTestId("speak.cancel_assessment"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Cancellation did not finish"));
    act(() => { queryClient.setQueryData(queryKeys.assessment("asmt-1"), {
      assessment_id: "asmt-1", status: "failed", phase: "failed", last_phase: "scoring_rubric", progress: 1,
      error: { code: "runtime_error", detail: "Scoring failed" },
    }); });
    const panel = screen.getByTestId("speak.status_panel");
    await waitFor(() => expect(panel).toHaveTextContent("Scoring failed"));
    expect(panel).not.toHaveTextContent("Cancellation did not finish");
    expect(within(panel).getByRole("status")).toHaveTextContent("Scoring failed");
  });

  it("clears a polling error after an unchanged status response recovers", async () => {
    const { queryClient } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedGetAssessmentStatus).toHaveBeenCalled());
    const originalStatus = await mockedGetAssessmentStatus.mock.results[0].value;
    mockedGetAssessmentStatus.mockRejectedValue(new ApiClientError(503, { code: "runtime_error", detail: "Poll unavailable" }));
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.assessment("asmt-1") }); });
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Could not refresh progress: Poll unavailable"));
    mockedGetAssessmentStatus.mockResolvedValue(originalStatus);
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.assessment("asmt-1") }); });
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).not.toHaveTextContent("Could not refresh progress"));
  });

  it("rejects a file when current disk space is insufficient, retaining a backup link", async () => {
    vi.mocked(apiClient.getUploadLimits).mockResolvedValueOnce({ max_bytes: 104857600, available_bytes: 1 });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [new File(["audio"], "saved.wav", { type: "audio/wav" })] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("not enough free disk space"));
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.download_recording")).toBeVisible();
  });

  it("redirects to Session Setup when the learner draft is missing", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    expect(await screen.findByRole("heading", { name: "Set up today's speaking practice" })).toBeVisible();
  });

  it("keeps the speaking brief and recording action dominant with compact metadata", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByRole("heading", { name: "Speaking brief" })).toBeVisible();
    expect(screen.getByTestId("speak.session_summary")).toHaveTextContent(
      "English · B2 · 120 s · Speaker bern",
    );
    expect(screen.getByTestId("speak.status_rail")).toBeVisible();
    expect(screen.getByTestId("speak.status_rail_step_brief")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_record")).toHaveAttribute(
      "data-step-state",
      "active",
    );
    expect(screen.getByTestId("speak.status_rail_step_submit")).toHaveAttribute(
      "data-step-state",
      "upcoming",
    );
    expect(screen.getByTestId("speak.recording_panel")).toBeVisible();

    const statusPanel = screen.getByTestId("speak.status_panel");
    expect(await screen.findByTestId("speak.runtime_detail")).toHaveTextContent(
      "Runtime: ollama_local · llama3.2:3b · Whisper small",
    );
    expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
      "Record or upload one take. You can listen before sending it.",
    );
    expect(screen.queryByTestId("speak.recording_ready_checkpoint")).not.toBeInTheDocument();
    expect(within(statusPanel).getByTestId("speak.submit_disabled_help")).toHaveTextContent(
      "Record or upload audio before submitting for review.",
    );
    expect(within(statusPanel).queryByTestId("speak.optional_context")).not.toBeInTheDocument();
    expect(within(statusPanel).getByTestId("speak.label")).toBeVisible();
    expect(within(statusPanel).getByTestId("speak.notes")).toBeVisible();
    expect(screen.getByTestId("speak.submit")).toHaveAttribute(
      "aria-describedby",
      "speak-submit-help",
    );
    const recordModeButton = screen.getByRole("button", { name: "Record" });
    const uploadModeButton = screen.getByRole("button", { name: "Upload" });
    const startButton = screen.getByRole("button", { name: "Start recording" });
    expect(recordModeButton).toHaveAttribute("data-semantic-id", "speak.input_mode_record");
    expect(uploadModeButton).toHaveAttribute("data-semantic-id", "speak.input_mode_upload");
    expect(startButton).toHaveAttribute("data-semantic-id", "speak.record_start");
    expectDecorativeIcon(recordModeButton);
    expectDecorativeIcon(uploadModeButton);
    expectDecorativeIcon(startButton);
    expect(screen.queryByText("Target CEFR level")).not.toBeInTheDocument();
    expect(screen.queryByText("Target duration (seconds)")).not.toBeInTheDocument();
  });

  it("exposes a stable accessible recording visualizer while idle", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByRole("heading", { name: "Speaking brief" })).toBeVisible();

    const visualizer = screen.getByTestId("speak.recording_visualizer");
    expect(visualizer).toHaveAttribute("data-semantic-id", "speak.recording_visualizer");
    expect(visualizer).toHaveAttribute("data-recording-state", "idle");
    expect(screen.getByRole("img", { name: "No recording is attached yet." })).toBe(visualizer);
  });

  it("updates the recording visualizer when browser recording is active", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));

    const visualizer = await screen.findByRole("img", { name: "Recording... 0 s" });
    const stopButton = screen.getByRole("button", { name: "Stop recording" });
    expect(visualizer).toHaveAttribute("data-recording-state", "recording");
    expect(stopButton).toHaveAttribute("data-semantic-id", "speak.record_stop");
    expectDecorativeIcon(stopButton);
    expect(stopButton).toBeVisible();
  });

  it("keeps the active recorder mounted during a cloud-settings refresh", async () => {
    const cloudResponse = {
      settings: { asr_provider: "groq", asr_connection_id: "groq-1" },
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(cloudResponse), {
      status: 200, headers: { "Content-Type": "application/json" },
    })));
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "conn-primary", asr_provider: "groq",
      connections: [{
        connection_id: "groq-1", provider_key: "groq", provider_choice: "groq", provider_label: "Groq",
        label: "Speech", model: "whisper-large-v3", base_url: "https://api.groq.com", is_default: false,
        is_local: false, requires_api_key: true, has_api_key: true, secret_state: "present",
        last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
      }],
    });
    const { queryClient } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"], locale: "en", appState: validDraftState,
    });
    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    const panel = screen.getByTestId("speak.recording_panel");
    await screen.findByRole("button", { name: "Stop recording" });
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));

    act(() => { void queryClient.invalidateQueries({ queryKey: ["runtime", "cloud"] }); });
    await waitFor(() => expect(queryClient.getQueryState(["runtime", "cloud"])?.fetchStatus).toBe("fetching"));
    expect(screen.getByTestId("speak.recording_panel")).toBe(panel);
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeVisible();
    expect(FakeMediaRecorder.instances.at(-1)?.state).toBe("recording");
  });

  it("keeps the active recorder mounted when cached readiness refreshes fail", async () => {
    const { queryClient } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"], locale: "en", appState: validDraftState,
    });
    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    const panel = screen.getByTestId("speak.recording_panel");
    await screen.findByRole("button", { name: "Stop recording" });
    mockedGetDiagnostics.mockRejectedValue(new Error("offline"));
    mockedGetRuntime.mockRejectedValue(new Error("offline"));
    mockedGetRuntimeSettings.mockRejectedValue(new Error("offline"));

    act(() => {
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ["diagnostics"] }),
        queryClient.invalidateQueries({ queryKey: ["runtime"] }),
        queryClient.invalidateQueries({ queryKey: ["runtime", "settings"] }),
      ]);
    });
    await waitFor(() => {
      expect(queryClient.getQueryState(["diagnostics"])?.status).toBe("error");
      expect(queryClient.getQueryState(["runtime"])?.status).toBe("error");
      expect(queryClient.getQueryState(["runtime", "settings"])?.status).toBe("error");
    }, { timeout: 5_000 });
    expect(screen.getByTestId("speak.recording_panel")).toBe(panel);
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeVisible();
    expect(FakeMediaRecorder.instances.at(-1)?.state).toBe("recording");
  });

  it("clears the previous attachment when switching between record and upload modes", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(await screen.findByTestId("speak.first_take_milestone")).toHaveTextContent("First take recorded on this Mac");
    expect(window.localStorage.getItem("assess-speaking.first-take-recorded")).toBe("1");
    expect(statusPanel).toHaveTextContent("Record at least 30 seconds before requesting a review.");
    expect(statusPanel).not.toHaveTextContent("A recording is attached and ready for assessment.");
    expect(screen.getByRole("button", { name: "Remove recording" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Upload" }));

    await waitFor(() => {
      expect(within(statusPanel).getByText("No recording is attached yet.")).toBeVisible();
    });
    expect(screen.queryByRole("button", { name: "Remove recording" })).not.toBeInTheDocument();
    const uploadInput = screen.getByTestId("speak.upload_input");
    expect(uploadInput).toBeInTheDocument();
    expect(uploadInput).not.toBeVisible();
    expect(uploadInput.closest("label")?.className).toContain("uploadControl");
    expectDecorativeIcon(screen.getByText("Choose audio file"));
  });

  it("does not repeat the first-take banner for a later recording", async () => {
    const { rerender } = renderWithProviders(<></>, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    window.localStorage.setItem("assess-speaking.first-take-recorded", "1");
    rerender(<AppFrame />);

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    expect(await screen.findByTestId("speak.status_panel")).toHaveTextContent("Record at least 30 seconds");
    expect(screen.queryByTestId("speak.first_take_milestone")).not.toBeInTheDocument();
  });

  it("shows the milestone for the first take but not the next take in one session", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));
    expect(await screen.findByTestId("speak.first_take_milestone")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Remove recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));
    expect(await screen.findByTestId("speak.status_panel")).toHaveTextContent("Record at least 30 seconds");
    expect(screen.queryByTestId("speak.first_take_milestone")).not.toBeInTheDocument();
  });

  it("removes an uploaded recording and returns to the idle state", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), {
      target: { files: [uploadFile] },
    });

    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(
      await within(statusPanel).findByText("A recording is attached and ready for assessment."),
    ).toBeVisible();
    expect(screen.getByTestId("speak.status_rail_step_submit")).toHaveAttribute(
      "data-step-state",
      "active",
    );
    expect(screen.getByTestId("speak.recording_ready_checkpoint")).toHaveTextContent(
      "Saved take",
    );
    expect(screen.getByTestId("speak.recording_ready_checkpoint")).toHaveTextContent(
      "Listen back if you want to check the take before sending it.",
    );
    expect(within(statusPanel).getByTestId("speak.submit")).toBeEnabled();
    expect(within(statusPanel).getByTestId("speak.optional_context")).toBeInTheDocument();
    expect(within(statusPanel).queryByTestId("speak.assessment_progress")).not.toBeInTheDocument();
    expect(within(statusPanel).getByText("Add optional context")).toBeVisible();
    expect(within(statusPanel).getByTestId("speak.optional_context")).not.toHaveAttribute(
      "open",
      "",
    );
    expect(within(statusPanel).getByTestId("speak.label")).toBeInTheDocument();
    expect(within(statusPanel).getByTestId("speak.notes")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("speak.remove_recording"));

    await waitFor(() => {
      expect(within(statusPanel).getByText("No recording is attached yet.")).toBeVisible();
    });
    expect(screen.queryByTestId("speak.remove_recording")).not.toBeInTheDocument();
    expect(within(statusPanel).queryByTestId("speak.optional_context")).not.toBeInTheDocument();
    expect(screen.getByTestId("speak.submit")).toBeDisabled();
  });

  it("requires reattaching a local file after Speak unmounts", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), {
      target: { files: [uploadFile] },
    });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());

    fireEvent.click(screen.getByRole("link", { name: "How scores work" }));
    expect(await screen.findByRole("heading", { name: "Understand your feedback" })).toBeVisible();
    fireEvent.click(screen.getByRole("link", { name: "Speak" }));

    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(
      within(statusPanel).getByText(
        "A recording was selected earlier, but the audio file is no longer available.",
      ),
    ).toBeVisible();
    expect(within(statusPanel).getByTestId("speak.submit")).toBeDisabled();
    expect(screen.queryByTestId("speak.recording_ready_checkpoint")).not.toBeInTheDocument();

    fireEvent.click(within(statusPanel).getByTestId("speak.submit"));
    expect(mockedUploadAudio).not.toHaveBeenCalled();
  });

  it.each([29, 30, 31])("requires the encoder margin when recorded audio cannot be decoded (recorded: %s)", async (duration) => {
    const clock = vi.spyOn(Date, "now").mockReturnValue(100_000);
    try {
      const {store} = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
      fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
      await screen.findByRole("button", { name: "Stop recording" });
      clock.mockReturnValue(100_000 + duration * 1000);
      fireEvent.click(screen.getByRole("button", { name: "Stop recording" }));
      await waitFor(() => expect(store.getState().recording.status).toBe("ready"));
      const submit = screen.getByTestId("speak.submit");
      if (duration < 31) expect(submit).toBeDisabled();
      else await waitFor(() => expect(submit).toBeEnabled());
      expect(mockedCreateAssessment).not.toHaveBeenCalled();
    } finally {
      clock.mockRestore();
    }
  });

  it("keeps short browser audio for playback while blocking assessment", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));

    expect(getUserMedia).toHaveBeenCalledWith({ audio: { autoGainControl: false, echoCancellation: true, noiseSuppression: true } });
    expect(await screen.findByRole("button", { name: "Stop recording" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Stop recording" }));

    expect(stopTrack).toHaveBeenCalled();
    expect(FakeMediaRecorder.instances[0]?.mimeType).toBe("audio/webm;codecs=opus");
    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(statusPanel).toHaveTextContent("Record at least 30 seconds before requesting a review.");
    expect(statusPanel).not.toHaveTextContent("A recording is attached and ready for assessment.");
    expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
      "Record at least 30 seconds before requesting a review.",
    );
    expect(within(statusPanel).getByTestId("speak.submit")).toBeDisabled();
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.recording_ready_checkpoint")).toHaveTextContent(
      "Saved take",
    );
    expect(screen.getByRole("button", { name: "Remove recording" })).toBeVisible();
  });

  it("surfaces a stuck microphone permission request and leaves upload available", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });
    const startButton = await screen.findByRole("button", { name: "Start recording" });
    getUserMedia.mockReturnValueOnce(new Promise(() => undefined));

    vi.useFakeTimers();
    try {
      fireEvent.click(startButton);

      expect(screen.getByText("Asking for microphone access...")).toBeVisible();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(8_000);
      });

      expect(
        screen.getByText(
          "Microphone access is still pending. Check the permission prompt in your browser or upload an audio file.",
        ),
      ).toBeVisible();

      fireEvent.click(screen.getByRole("button", { name: "Upload" }));

      expect(screen.getByTestId("speak.upload_input")).toBeInTheDocument();
      expect(screen.getByTestId("speak.upload_input")).not.toBeVisible();
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears pending microphone status when switching to upload", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });
    const startButton = await screen.findByRole("button", { name: "Start recording" });
    getUserMedia.mockReturnValueOnce(new Promise(() => undefined));

    vi.useFakeTimers();
    try {
      fireEvent.click(startButton);

      expect(screen.getByText("Asking for microphone access...")).toBeVisible();

      fireEvent.click(screen.getByRole("button", { name: "Upload" }));

      expect(screen.queryByText("Asking for microphone access...")).not.toBeInTheDocument();
      expect(screen.getByTestId("speak.upload_input")).toBeInTheDocument();
      expect(screen.getByTestId("speak.upload_input")).not.toBeVisible();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(8_000);
      });

      expect(
        screen.queryByText(
          "Microphone access is still pending. Check the permission prompt in your browser or upload an audio file.",
        ),
      ).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("runs the cloud provider lifecycle and acknowledges cancel", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "OpenRouter",
      model: "google/gemini-3.1-pro-preview",
      base_url: "https://openrouter.ai/api/v1",
      requires_api_key: true,
      has_api_key: true,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "small",
      active_connection_id: "conn-openrouter",
      connections: [
        {
          connection_id: "conn-openrouter",
          provider_key: "openrouter",
          provider_choice: "openrouter",
          provider_label: "OpenRouter",
          label: "OpenRouter review",
          model: "google/gemini-3.1-pro-preview",
          base_url: "https://openrouter.ai/api/v1",
          is_default: true,
          is_local: false,
          requires_api_key: true,
          has_api_key: true,
          secret_state: "present",
          last_test_status: "",
          last_tested_at: "",
          openrouter_http_referer: "https://example.test/assess-speaking",
          openrouter_app_title: "Vostavo Review",
          provider_metadata: {},
        },
      ],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Upload" }));
    fireEvent.change(screen.getByLabelText("Or upload an audio file"), {
      target: { files: [uploadFile] },
    });
    fireEvent.change(screen.getByLabelText("Label (optional)"), {
      target: { value: "Morning run" },
    });
    fireEvent.change(screen.getByLabelText("Notes (optional)"), {
      target: { value: "Mention two concrete examples." },
    });

    const statusPanel = await screen.findByTestId("speak.status_panel");

    expect(screen.queryByText(/this saved connection is missing its API key/)).not.toBeInTheDocument();

    await waitFor(() => expect(screen.getByRole("button", { name: "Submit for review" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));

    expect(
      await within(statusPanel).findByText(
        "Your assessment is running via OpenRouter with model `google/gemini-3.1-pro-preview`.",
      ),
    ).toBeVisible();
    expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
      "Your review is being prepared.",
    );
    expect(screen.queryByTestId("speak.recording_ready_checkpoint")).not.toBeInTheDocument();
    expect(screen.getByTestId("speak.status_rail_step_review")).toHaveAttribute(
      "data-step-state",
      "active",
    );
    expect(within(statusPanel).getByText("Current step: Transcribing recording.")).toBeVisible();
    expect(await within(statusPanel).findByRole("list", { name: "Assessment phases" })).toBeVisible();
    expect(
      within(statusPanel).queryByText("This status refreshes automatically every few seconds."),
    ).not.toBeInTheDocument();
    expect(
      within(statusPanel).queryByText(
        "This can take several minutes. Keep this page open; the review will appear automatically when it is ready.",
      ),
    ).not.toBeInTheDocument();
    await waitFor(() => {
      expect(mockedCreateAssessment).toHaveBeenCalledWith(
        expect.objectContaining({
          openrouter_app_title: "Vostavo Review",
          openrouter_http_referer: "https://example.test/assess-speaking",
          whisper: "small",
        }),
      );
    });
    expect(within(statusPanel).getByText("Runtime: OpenRouter · google/gemini-3.1-pro-preview · Whisper small")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Cancel assessment" }));

    expect(
      await within(statusPanel).findByText(
        "The assessment was cancelled. You can adjust your notes or recording and try again.",
      ),
    ).toBeVisible();
    expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
      "Assessment cancelled. Adjust the take or submit again when ready.",
    );
    expect(screen.queryByTestId("speak.recording_ready_checkpoint")).not.toBeInTheDocument();
    expect(within(statusPanel).getByTestId("speak.optional_context")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Morning run")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Mention two concrete examples.")).toBeInTheDocument();
  });

  it("shows explicit failure handling and automatically hands completed work to Review", async () => {
    mockedCreateAssessment
      .mockResolvedValueOnce({
        assessment_id: "asmt-1",
        status: "queued",
      })
      .mockResolvedValueOnce({
        assessment_id: "asmt-2",
        status: "queued",
      });
    mockedGetAssessmentStatus
      .mockResolvedValueOnce({
        assessment_id: "asmt-1",
        status: "failed",
        phase: "failed",
        progress: 1,
        error: {
          code: "runtime_error",
          detail: "boom",
        },
        report_path: null,
        payload: null,
        summary: null,
      })
      .mockResolvedValueOnce({
        assessment_id: "asmt-2",
        status: "completed",
        phase: "finalizing_report",
        progress: 1,
        error: null,
        report_path: "/tmp/report.json",
        payload: {
          metrics: { duration_sec: 120 },
          report: {
            scores: { final: 3.5, band: 4, mode: "hybrid" },
            checks: { language_pass: true, topic_pass: true, content_validity_pass: true,
              duration_pass: true, min_words_pass: true },
            transcript: {
              text: "Transcript text",
            },
            coaching: {
              coach_summary: "Focus on connectors.",
            },
          },
        },
        summary: {
          score_overall: 3.5,
          band: "4",
          next_focus: "Focus on connectors.",
        },
      });

    const { rerender } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Upload" }));
    fireEvent.change(screen.getByLabelText("Or upload an audio file"), {
      target: { files: [uploadFile] },
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit for review" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));

    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(await within(statusPanel).findByText("Assessment failed: boom")).toBeVisible();
    expect(within(statusPanel).getByTestId("speak.handoff_hint")).toHaveTextContent(
      "Assessment stopped. Keep the take, adjust notes, or submit again.",
    );
    expect(screen.queryByTestId("speak.recording_ready_checkpoint")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Or upload an audio file"), {
      target: { files: [uploadFile] },
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "Submit for review" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));

    rerender(<AppFrame />);

    expect(await screen.findByTestId("review-summary")).toBeVisible();
    expect(screen.getByTestId("review-estimate-text")).toHaveTextContent("3.5/5");
  });
});
