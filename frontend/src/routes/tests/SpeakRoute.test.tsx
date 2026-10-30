import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppFrame } from "@/App";
import { renderWithProviders } from "@/test/renderWithProviders";
import { readyReadiness, readinessWith } from "@/test/readinessFixtures";
import { queryKeys } from "@/lib/query/queryClient";
import { draftAdapterFixture, savedDraftFixture } from "@/test/draftAdapterFixture";
import { desktopBridge } from "@/lib/runtime/desktopBridge";
import { persistRetryDraft, RETRY_DRAFT_STORAGE_KEY } from "@/lib/state/retryBriefStorage";

vi.mock("@/lib/api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    code = "runtime_error";
    detail: string;
    responseStatus = 500;

    constructor(responseStatus: number, error: { detail: string; code?: string }) {
      super(error.detail);
      this.detail = error.detail;
      this.code = error.code || "runtime_error";
      this.responseStatus = responseStatus;
    }
  },
  apiClient: {
    getSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getResumeSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    cancelAssessment: vi.fn(),
    createAssessment: vi.fn(),
    getAssessmentStatus: vi.fn(),
    getAssessmentByRequest: vi.fn(),
    getHistoryDetail: vi.fn(),
    getUploadByRequest: vi.fn(),
    getRuntime: vi.fn(),
    getDiagnostics: vi.fn(),
    getReadiness: vi.fn().mockResolvedValue({ ready: true, checks: [
      "speech_recognition", "ai_tutor", "media_decoder", "storage",
    ].map(key => ({ key, state: "available", blocking: true, reason: "available", detail_args: {} })) }),
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
const mockedGetReadiness = vi.mocked(apiClient.getReadiness);
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

  it("retains a saved take when changing mode and cancelling replacement", async () => {
    const saved = savedDraftFixture();
    const adapter = draftAdapterFixture([saved]);
    vi.stubGlobal("confirm", vi.fn(() => true));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac"));
    fireEvent.click(screen.getByRole("button", { name: "Record" }));
    expect(adapter.discard).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "saved.wav");
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    expect(adapter.discard).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac");
  });

  it("blocks Submit during a replacement recording while keeping Stop usable", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    vi.stubGlobal("confirm", vi.fn(() => true));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Record" }));
    fireEvent.click(screen.getByTestId("speak.record_start"));
    expect(await screen.findByTestId("speak.record_stop")).toBeEnabled();
    expect(screen.getByTestId("speak.submit")).toBeDisabled();
    fireEvent.click(screen.getByTestId("speak.submit"));
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(adapter.discard).not.toHaveBeenCalled();
  });

  it("blocks assessment Retry while a replacement recording is active", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    vi.stubGlobal("confirm", vi.fn(() => true));
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    act(() => store.getState().setRecordingJob({ assessmentId: "asmt-1", status: "failed", error: "retry" }));
    expect(await screen.findByTestId("speak.retry_assessment")).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Record" }));
    fireEvent.click(screen.getByTestId("speak.record_start"));
    await screen.findByTestId("speak.record_stop");
    expect(screen.getByTestId("speak.retry_assessment")).toBeDisabled();
    fireEvent.click(screen.getByTestId("speak.retry_assessment"));
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(adapter.discard).not.toHaveBeenCalled();
  });

  it("keeps a queued assessment live when its local audio is reattached", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await screen.findByTestId("speak.recording_status");
    act(() => store.getState().setRecordingAssessing({ assessmentId: "asmt-1", status: "queued" }));
    fireEvent.click(screen.getByRole("link", { name: "Practice" }));
    fireEvent.click(screen.getByRole("link", { name: "Speak" }));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveAttribute("data-assessment-state", "running"));
    expect(screen.getByTestId("speak.cancel_assessment")).toBeVisible();
    expect(screen.getByTestId("speak.submit")).toBeDisabled();
    expect(store.getState().recording.job.assessmentId).toBe("asmt-1");
  });

  it("removes the local duplicate after a completed report and does not offer it on remount", async () => {
    const desktopStore = vi.spyOn(desktopBridge, "hasDraftStore").mockReturnValue(true);
    try {
      const saved = savedDraftFixture();
      saved.setup = { ...saved.setup, retryOfSessionId: "prior-attempt", coachingGoal: "Finish clearly", retryGoalEligible: true };
      saved.status = "assessing";
      saved.submission = { uploadRequestId: "upload-one", audioId: "audio-1", assessmentRequestId: "request-one",
        assessmentId: "asmt-1", requestFingerprint: "a".repeat(64),
        frozenRequestBody: JSON.stringify({ audio_id: "audio-1", request_id: "request-one", sharing_fingerprint: "a".repeat(64) }),
        lastConfirmedPhase: "assessment" };
      vi.mocked(apiClient.getAssessmentByRequest).mockResolvedValue({ assessment_id: "asmt-1", status: "completed", report_id: "report-1" });
      vi.mocked(apiClient.getAssessmentByRequest).mockRejectedValueOnce(new ApiClientError(404, { code: "validation_error", detail: "absent" }));
      vi.mocked(apiClient.getHistoryDetail).mockResolvedValue({ payload: { report: { session_id: "report-1" } } } as Awaited<ReturnType<typeof apiClient.getHistoryDetail>>);
      const adapter = draftAdapterFixture([saved]);
      adapter.list.mockResolvedValueOnce({ drafts: [saved], unreadableIds: [] })
        .mockResolvedValueOnce({ drafts: [], unreadableIds: [] });
      const first = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
      fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
      await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac"));
      persistRetryDraft(saved.setup);
      expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).not.toBeNull();
      act(() => first.store.getState().setRecordingJob({ assessmentId: "asmt-1", status: "queued" }));
      await waitFor(() => expect(mockedGetAssessmentStatus).toHaveBeenCalled());
      act(() => first.queryClient.setQueryData(queryKeys.assessment("asmt-1"), {
        assessment_id: "asmt-1", status: "completed", phase: "completed", progress: 1,
        error: null, report_path: "/private/tmp/report-1.json", summary: null,
        payload: { report: { session_id: "report-1" } },
      }));
      await waitFor(() => expect(adapter.discard).toHaveBeenCalledWith(saved.draftId));
      expect(first.store.getState().review.reportId).toBe("report-1");
      await waitFor(() => expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull());
      first.unmount();
      renderWithProviders(<AppFrame />, { initialEntries: ["/"], locale: "en", appState: validDraftState, draftAdapter: adapter });
      await screen.findByTestId("home.resume");
      expect(screen.queryByTestId("draft.recovery")).not.toBeInTheDocument();
    } finally { desktopStore.mockRestore(); }
  });

  it("does not mark a second selected file saved while the first finish is pending", async () => {
    const adapter = draftAdapterFixture();
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 32 }); });
    let finish: (value: Awaited<ReturnType<typeof adapter.saveTake>>) => void = () => undefined;
    adapter.saveTake.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    const audio = (name: string) => {
      const bytes = new Uint8Array(44 + 8000 * 32 * 2);
      const view = new DataView(bytes.buffer);
      bytes.set(new TextEncoder().encode("RIFF"), 0); view.setUint32(4, bytes.length - 8, true);
      bytes.set(new TextEncoder().encode("WAVEfmt "), 8); view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, 8000, true);
      view.setUint32(28, 16000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
      bytes.set(new TextEncoder().encode("data"), 36); view.setUint32(40, bytes.length - 44, true);
      return new File([bytes], name, { type: "audio/wav" });
    };
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [audio("first.wav")] } });
    fireEvent.click(screen.getByRole("button", { name: "Keep this selected file as the local take" }));
    await waitFor(() => expect(adapter.saveTake).toHaveBeenCalledOnce());
    expect(screen.getByTestId("speak.recording_status")).not.toHaveTextContent("Saved on this Mac");
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [audio("second.wav")] } });
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "first.wav");
    const saved = adapter.saveTake.mock.calls[0][0];
    act(() => finish({ ...saved, revision: saved.revision + 1, status: "take_ready",
      take: { ...savedDraftFixture().take!, originalFilename: "first.wav" } }));
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac"));
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "first.wav");
  });

  it("keeps the local audio download usable after a draft save fails", async () => {
    const adapter = draftAdapterFixture();
    adapter.saveTake.mockRejectedValue(new Error("Disk unavailable"));
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 32 }); });
    vi.stubGlobal("confirm", vi.fn(() => false));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    fireEvent.click(screen.getByRole("button", { name: "Keep this selected file as the local take" }));
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Could not save locally"));
    const download = screen.getByTestId("speak.download_recording");
    expect(download).toHaveAttribute("download", "attempt.wav");
    let allowed = false;
    download.addEventListener("click", event => { allowed = !event.defaultPrevented; event.preventDefault(); }, { once: true });
    const click = new MouseEvent("click", { bubbles: true, cancelable: true });
    download.dispatchEvent(click);
    expect(allowed).toBe(true);
    expect(window.confirm).not.toHaveBeenCalled();
  });

  it("keeps a saved take after an invalid replacement upload", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    vi.stubGlobal("confirm", vi.fn(() => true));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "saved.wav"));
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [new File([], "empty.wav", { type: "audio/wav" })] } });
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "saved.wav");
    expect(adapter.discard).not.toHaveBeenCalled();
  });

  it("keeps and reattaches the saved take after removing a failed unsaved replacement", async () => {
    const saved = savedDraftFixture();
    const adapter = draftAdapterFixture([saved]);
    adapter.saveTake.mockRejectedValue(new Error("disk unavailable"));
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 32 }); });
    vi.stubGlobal("confirm", vi.fn(() => true));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "saved.wav"));
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    fireEvent.click(screen.getByRole("button", { name: "Keep this selected file as the local take" }));
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Could not save locally"));
    fireEvent.click(screen.getByTestId("speak.remove_recording"));
    await waitFor(() => expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "saved.wav"));
    expect(adapter.discard).not.toHaveBeenCalled();
    expect(adapter.loadTake).toHaveBeenCalledTimes(2);
  });

  it("recreates setup before recording after a discarded take's setup save fails", async () => {
    const saved = savedDraftFixture();
    saved.take!.inputMethod = "record";
    const adapter = draftAdapterFixture([saved]);
    adapter.saveSetup.mockRejectedValueOnce(new Error("Mac storage is unavailable"))
      .mockRejectedValueOnce(new Error("Mac storage is still unavailable"));
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 32 }); });
    vi.stubGlobal("confirm", vi.fn(() => true));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac"));
    fireEvent.click(screen.getByTestId("speak.remove_recording"));
    await waitFor(() => expect(adapter.discard).toHaveBeenCalledWith(saved.draftId));
    await waitFor(() => expect(adapter.saveSetup).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.queryByTestId("speak.download_recording")).not.toBeInTheDocument());
    expect(screen.queryByTestId("speak.remove_recording")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try saving this recording again" })).not.toBeInTheDocument();
    expect(adapter.saveTake).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.record_start"));
    await waitFor(() => expect(adapter.saveSetup).toHaveBeenCalledTimes(2));
    expect(getUserMedia).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.record_start"));
    await waitFor(() => expect(adapter.saveSetup).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledOnce());
    expect(adapter.saveSetup.mock.invocationCallOrder[2]).toBeLessThan(getUserMedia.mock.invocationCallOrder[0]);
    fireEvent.click(await screen.findByTestId("speak.record_stop"));
    await waitFor(() => expect(adapter.saveTake).toHaveBeenCalledOnce());
    const [createdSetup, recordedFile] = adapter.saveTake.mock.calls[0];
    expect(recordedFile.lastModified).toBeGreaterThanOrEqual(createdSetup.createdAt);
    expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac");
  });

  it("retries saving the same finalized recording after a transient failure", async () => {
    const adapter = draftAdapterFixture();
    adapter.saveTake.mockRejectedValueOnce(new Error("disk unavailable"));
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 32 }); });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByTestId("speak.record_start"));
    fireEvent.click(await screen.findByTestId("speak.record_stop"));
    const retry = await screen.findByRole("button", { name: "Try saving this recording again" });
    expect(screen.getByTestId("speak.recording_status")).not.toHaveTextContent("Saved on this Mac");
    fireEvent.click(retry);
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Saved on this Mac"));
    expect(adapter.saveTake).toHaveBeenCalledTimes(2);
    expect(adapter.saveTake.mock.calls[0][1]).toBe(adapter.saveTake.mock.calls[1][1]);
  });

  it("does not persist or label a take shorter than the review minimum", async () => {
    const adapter = draftAdapterFixture();
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 1 }); });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByTestId("speak.record_start"));
    fireEvent.click(await screen.findByTestId("speak.record_stop"));
    await waitFor(() => expect(screen.getByTestId("speak.recording_status")).toHaveTextContent("Record at least 30 seconds"));
    expect(adapter.saveTake).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.recording_status")).not.toHaveTextContent("Saved on this Mac");
  });

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
    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "not_configured"));
    mockedGetRuntime.mockResolvedValue({
      configured: false, provider: "", model: "", base_url: "",
      requires_api_key: false, has_api_key: false,
    });
    const { queryClient } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.getByText("Finish setup to enter the recording step")).toBeVisible();
    expect(screen.getByText(/needs attention before you can continue/)).toBeVisible();
    expect(screen.queryByTestId("speak.record_start")).not.toBeInTheDocument();

    act(() => queryClient.setQueryData(queryKeys.readiness, readyReadiness));
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
    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "credential_missing"));
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "openrouter", model: "vendor/model", base_url: "https://example.test",
      requires_api_key: true, has_api_key: false,
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.queryByTestId("speak.record_start")).not.toBeInTheDocument();
  });

  it.each([
    ["media_decoder", "media_decoder_missing", "The media decoder is unavailable."],
    ["storage", "storage_not_writable", "Practice storage is not writable."],
  ] as const)("guards Speak when %s blocks readiness", async (key, reason, detail) => {
    mockedGetReadiness.mockResolvedValue(readinessWith(key, reason, "unavailable"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.getByText(detail)).toBeVisible();
    expect(screen.queryByTestId("speak.record_start")).not.toBeInTheDocument();
  });
  const stopTrack = vi.fn();
  const getUserMedia = vi.fn();

  beforeEach(() => {
    window.localStorage.removeItem("assess-speaking.first-take-recorded");
    vi.mocked(apiClient.getUploadLimits).mockResolvedValue({ max_bytes: 104857600, available_bytes: 104857600 });
    vi.clearAllMocks();
    vi.mocked(apiClient.getAssessmentByRequest).mockRejectedValue(new ApiClientError(404, { code: "validation_error", detail: "absent" }));
    vi.mocked(apiClient.getUploadByRequest).mockRejectedValue(new ApiClientError(404, { code: "validation_error", detail: "absent" }));
    mockedGetReadiness.mockResolvedValue(readyReadiness);
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
    vi.mocked(apiClient.getUploadLimits).mockResolvedValueOnce({ max_bytes: 104857600, available_bytes: 0 });
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedUploadAudio).toHaveBeenCalledTimes(2));
    const first = mockedUploadAudio.mock.calls[0][1]?.requestId;
    expect(first).toBeTruthy();
    expect(mockedUploadAudio.mock.calls[1][1]?.requestId).toBe(first);
  });

  it.each([
    ["network replies are lost", () => new TypeError("Failed to fetch")],
    ["assessment replies time out", () => new DOMException("The assessment request timed out", "AbortError")],
  ])("keeps the prior job and request ID when %s", async (_scenario, makeError) => {
    mockedCreateAssessment.mockResolvedValueOnce({ assessment_id: "asmt-1", status: "queued" })
      .mockRejectedValueOnce(makeError())
      .mockResolvedValueOnce({ assessment_id: "asmt-2", status: "queued" });
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    fireEvent.click(await screen.findByTestId("speak.cancel_assessment"));
    fireEvent.click(await screen.findByTestId("speak.retry_assessment"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Could not confirm the retry"));
    expect(screen.getByTestId("speak.status_panel")).not.toHaveTextContent("Upload cancelled");
    expect(store.getState().recording.job.assessmentId).toBe("asmt-1");
    expect(screen.getByTestId("speak.retry_assessment")).toBeVisible();
    fireEvent.click(screen.getByTestId("speak.retry_assessment"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(3));
    expect(mockedCreateAssessment.mock.calls[2][0]).toEqual(mockedCreateAssessment.mock.calls[1][0]);
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
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
    expect(screen.getByTestId("speak.retry_assessment")).toBeVisible();
    expect(screen.queryByTestId("speak.submit")).not.toBeInTheDocument();
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
    await screen.findByTestId("speak.retry_assessment");
    expect(screen.queryByTestId("speak.submit")).not.toBeInTheDocument();
    fireEvent.click(await screen.findByTestId("speak.retry_assessment"));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Retry is not ready yet"));
    expect(store.getState().recording.job.assessmentId).toBe("asmt-1");
    expect(screen.getByTestId("speak.retry_assessment")).toBeVisible();
    fireEvent.click(screen.getByTestId("speak.retry_assessment"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(3));
    expect(mockedCreateAssessment.mock.calls[2][0]).toMatchObject({ resume_assessment_id: "asmt-1", audio_id: "audio-1" });
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
  });

  it("keeps Retry visible and disabled while a retained assessment is being resumed", async () => {
    let releaseRetry: () => void = () => { throw new Error("Retry was not requested"); };
    mockedCreateAssessment.mockResolvedValueOnce({ assessment_id: "asmt-1", status: "queued" })
      .mockImplementationOnce(() => new Promise<Awaited<ReturnType<typeof apiClient.createAssessment>>>(resolve => {
        releaseRetry = () => resolve({ assessment_id: "asmt-2", status: "queued" });
      }));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    fireEvent.click(await screen.findByTestId("speak.cancel_assessment"));
    fireEvent.click(await screen.findByTestId("speak.retry_assessment"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    expect(screen.getByTestId("speak.retry_assessment")).toBeDisabled();
    expect(screen.queryByTestId("speak.submit")).not.toBeInTheDocument();
    expect(screen.getByTestId("speak.retry_assessment")).toHaveAttribute("data-semantic-id", "speak.retry_assessment");
    act(() => releaseRetry());
    await waitFor(() => expect(screen.getByTestId("speak.cancel_assessment")).toBeVisible());
    expect(mockedCreateAssessment.mock.calls[1][0]).toMatchObject({ resume_assessment_id: "asmt-1" });
  });

  it("waits for a second explicit Submit after a lost response and reuses the submission ID", async () => {
    mockedCreateAssessment.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Could not confirm submission"));
    expect(store.getState().recording.assessmentState).toBe("uncertain");
    expect(mockedCreateAssessment).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    const first = mockedCreateAssessment.mock.calls[0][0];
    const recovered = mockedCreateAssessment.mock.calls[1][0];
    expect(first.request_id).toBeTruthy();
    expect(recovered).toEqual(first);
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent("Your assessment is running"));
  });

  it.each(["record", "upload"] as const)("keeps a saved %s take unchanged while submission is uncertain", async (method) => {
    const saved = savedDraftFixture();
    saved.take = { ...saved.take!, inputMethod: method };
    const adapter = draftAdapterFixture([saved]);
    mockedCreateAssessment.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter,
    });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(store.getState().recording.assessmentState).toBe("uncertain"));
    expect(screen.getByTestId("speak.remove_recording")).toBeDisabled();
    expect(screen.getByTestId("speak.input_mode_record")).toBeDisabled();
    expect(screen.getByTestId("speak.input_mode_upload")).toBeDisabled();
    fireEvent.click(screen.getByTestId("speak.remove_recording"));
    if (method === "record") {
      expect(screen.getByTestId("speak.record_start")).toBeDisabled();
      fireEvent.click(screen.getByTestId("speak.record_start"));
      expect(getUserMedia).not.toHaveBeenCalled();
    } else {
      const input = screen.getByTestId("speak.upload_input");
      expect(input).toBeDisabled();
      fireEvent.change(input, { target: { files: [new File(["new audio"], "new.wav", { type: "audio/wav" })] } });
    }
    expect(adapter.discard).not.toHaveBeenCalled();
    expect(adapter.saveTake).not.toHaveBeenCalled();
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "saved.wav");
    expect(screen.getByTestId("speak.submit")).toBeEnabled();
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    expect(mockedCreateAssessment.mock.calls[1][0]).toEqual(mockedCreateAssessment.mock.calls[0][0]);
    expect(mockedUploadAudio).toHaveBeenCalledTimes(1);
  });

  it("commits the upload and assessment receipts before their POSTs", async () => {
    const saved = savedDraftFixture();
    const adapter = draftAdapterFixture([saved]);
    mockedUploadAudio.mockImplementation(async () => {
      expect(adapter.markSubmission).toHaveBeenCalledWith(expect.objectContaining({ draftId: saved.draftId }), "upload_uncertain", expect.objectContaining({ uploadRequestId: expect.any(String) }));
      return { audio_id: "audio-1", stored_path: "/tmp/audio-1.wav", sha1: "sha1", original_name: "saved.wav" };
    });
    mockedCreateAssessment.mockImplementation(async () => {
      expect(adapter.markSubmission).toHaveBeenCalledWith(expect.objectContaining({ draftId: saved.draftId }), "assessment_uncertain", expect.objectContaining({ assessmentRequestId: expect.any(String) }));
      return { assessment_id: "asmt-1", status: "queued" };
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledOnce());
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledWith(expect.anything(), "assessing", expect.objectContaining({ assessmentId: "asmt-1" })));
  });

  it("never uploads when the local receipt cannot be committed", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    adapter.markSubmission.mockRejectedValueOnce(new Error("disk full"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledOnce());
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
  });

  it("never posts when a receipt write resolves without a saved draft", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    adapter.markSubmission.mockResolvedValueOnce(null as unknown as Awaited<ReturnType<typeof adapter.markSubmission>>);
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledOnce());
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
  });

  it("never submits an assessment when its receipt write fails", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    adapter.markSubmission.mockImplementation(async (draft, status, submission) => {
      if (status === "assessment_uncertain") throw new Error("disk unavailable");
      return { ...draft, status, submission, revision: draft.revision + 1 };
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledWith(expect.anything(), "assessment_uncertain", expect.anything()));
    expect(mockedUploadAudio).toHaveBeenCalledOnce();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
  });

  it("reconciles an uploaded file after its receipt write fails once", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    let failUploaded = true;
    adapter.markSubmission.mockImplementation(async (draft, status, submission) => {
      if (status === "uploaded" && failUploaded) {
        failUploaded = false;
        throw new Error("disk unavailable");
      }
      if (status === "assessment_uncertain" && draft.status !== "uploaded") {
        throw new Error("Invalid submission transition");
      }
      if (status === "assessment_uncertain" && !submission.audioId) {
        throw new Error("Incomplete submission receipt");
      }
      return { ...draft, status, submission, revision: draft.revision + 1 };
    });
    vi.mocked(apiClient.getUploadByRequest).mockResolvedValueOnce({ audio_id: "audio-1" });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledWith(expect.anything(), "uploaded", expect.anything()));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    expect(mockedUploadAudio).toHaveBeenCalledOnce();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledOnce());
    expect(mockedUploadAudio).toHaveBeenCalledOnce();
    expect(apiClient.getUploadByRequest).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ sha256: expect.any(String) }));
    const uploaded = adapter.markSubmission.mock.calls.filter(([, status]) => status === "uploaded");
    expect(uploaded).toHaveLength(2);
    expect(uploaded[1][2]).toMatchObject({ uploadRequestId: uploaded[0][2].uploadRequestId, audioId: "audio-1" });
  });

  it("keeps the take and uses a new ID after a confirmed sharing-route rejection", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    const firstFingerprint = "a".repeat(64);
    const nextFingerprint = "b".repeat(64);
    let routeFingerprint = firstFingerprint;
    vi.mocked(apiClient.getSharingRoute).mockImplementation(async () => ({ version: 1, available: true,
      fingerprint: routeFingerprint, audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" },
      analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }));
    mockedCreateAssessment.mockImplementationOnce(async () => {
      routeFingerprint = nextFingerprint;
      throw new ApiClientError(409, { code: "sharing_changed", detail: "Sharing route changed" });
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledWith(expect.anything(), "uploaded",
      expect.objectContaining({ assessmentRequestId: "", requestFingerprint: "", audioId: "audio-1" })));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    expect(mockedCreateAssessment).toHaveBeenCalledOnce();
    expect(mockedUploadAudio).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    expect(mockedCreateAssessment.mock.calls[1][0]).toMatchObject({ sharing_fingerprint: nextFingerprint, audio_id: "audio-1" });
    expect(mockedCreateAssessment.mock.calls[1][0].request_id).not.toBe(mockedCreateAssessment.mock.calls[0][0].request_id);
    expect(mockedUploadAudio).toHaveBeenCalledOnce();
  });

  it("submits a persisted route-rejected take after restart without reuploading", async () => {
    const saved = savedDraftFixture();
    saved.status = "uploaded";
    saved.submission = { uploadRequestId: "upload-one", audioId: "audio-1", assessmentRequestId: "", assessmentId: "",
      requestFingerprint: "", frozenRequestBody: "", lastConfirmedPhase: "upload" };
    const adapter = draftAdapterFixture([saved]);
    vi.mocked(apiClient.getSharingRoute).mockResolvedValue({ version: 1, available: true, fingerprint: "b".repeat(64),
      audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" },
      analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledOnce());
    expect(mockedCreateAssessment.mock.calls[0][0]).toMatchObject({ audio_id: "audio-1", sharing_fingerprint: "b".repeat(64) });
    expect(mockedUploadAudio).not.toHaveBeenCalled();
  });

  it("releases an absent frozen request after restart and a sharing-route change", async () => {
    const saved = savedDraftFixture();
    saved.status = "assessment_uncertain";
    saved.submission = { uploadRequestId: "upload-one", audioId: "audio-1", assessmentRequestId: "old-request",
      assessmentId: "", requestFingerprint: "a".repeat(64), lastConfirmedPhase: "upload",
      frozenRequestBody: JSON.stringify({ request_id: "old-request", audio_id: "audio-1", sharing_fingerprint: "a".repeat(64),
        speaker_id: saved.setup.speakerId, prompt_text: saved.setup.promptText, label: "Original label" }) };
    const adapter = draftAdapterFixture([saved]);
    vi.mocked(apiClient.getSharingRoute).mockResolvedValue({ version: 1, available: true, fingerprint: "b".repeat(64),
      audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" },
      analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null });
    mockedCreateAssessment.mockRejectedValueOnce(new ApiClientError(409, { code: "sharing_changed", detail: "Route changed" }));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledOnce());
    expect(mockedCreateAssessment.mock.calls[0][0]).toMatchObject({ request_id: "old-request",
      sharing_fingerprint: "a".repeat(64), label: "Original label" });
    await waitFor(() => expect(adapter.markSubmission).toHaveBeenCalledWith(expect.anything(), "uploaded",
      expect.objectContaining({ assessmentRequestId: "", audioId: "audio-1" })));
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
    expect(mockedCreateAssessment.mock.calls[1][0]).toMatchObject({ audio_id: "audio-1", sharing_fingerprint: "b".repeat(64) });
    expect(mockedCreateAssessment.mock.calls[1][0].request_id).not.toBe("old-request");
  });

  it("retains an uncertain assessment identity if a sharing conflict lookup finds a job", async () => {
    const adapter = draftAdapterFixture([savedDraftFixture()]);
    mockedCreateAssessment.mockRejectedValueOnce(new ApiClientError(409, { code: "sharing_changed", detail: "Request ID conflict" }));
    vi.mocked(apiClient.getAssessmentByRequest).mockResolvedValueOnce({ assessment_id: "asmt-existing", status: "queued", report_id: null });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(apiClient.getAssessmentByRequest).toHaveBeenCalledOnce());
    expect(adapter.markSubmission.mock.calls.filter(([, status]) => status === "uploaded")).toHaveLength(1);
    expect(adapter.markSubmission.mock.calls.find(([, status]) => status === "assessment_uncertain")?.[2].assessmentRequestId).toBe(
      mockedCreateAssessment.mock.calls[0][0].request_id,
    );
    expect(mockedCreateAssessment).toHaveBeenCalledOnce();
  });

  it("does not write a receipt or POST after the submitting draft is left mid-upload check", async () => {
    const saved = savedDraftFixture();
    const adapter = draftAdapterFixture([saved]);
    let releaseLimits: (value: { max_bytes: number; available_bytes: number }) => void = () => undefined;
    vi.mocked(apiClient.getUploadLimits).mockImplementation(() => new Promise(resolve => { releaseLimits = resolve; }));
    vi.stubGlobal("confirm", vi.fn(() => true));
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(apiClient.getUploadLimits).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole("link", { name: "Practice" }));
    fireEvent.click(await screen.findByTestId("home.start_new"));
    await act(async () => releaseLimits({ max_bytes: 104857600, available_bytes: 104857600 }));
    expect(adapter.markSubmission).not.toHaveBeenCalled();
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    expect(mockedCreateAssessment).not.toHaveBeenCalled();
  });

  it("leaves saving state after a pre-submit save error and succeeds on explicit retry", async () => {
    const adapter = draftAdapterFixture();
    adapter.saveTake.mockRejectedValueOnce(new Error("disk unavailable"));
    vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = async () => ({ duration: 40 }); });
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(adapter.saveTake).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    expect(mockedUploadAudio).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledOnce());
    expect(adapter.saveTake).toHaveBeenCalledTimes(2);
  });

  it("replays frozen label, notes, model and retry origin with the same ID after restart", async () => {
    const saved = savedDraftFixture();
    saved.status = "assessment_uncertain";
    saved.submission = { uploadRequestId: "upload-one", audioId: "audio-1", assessmentRequestId: "assessment-one",
      assessmentId: "", requestFingerprint: "a".repeat(64), lastConfirmedPhase: "upload",
      frozenRequestBody: JSON.stringify({ request_id: "assessment-one", sharing_fingerprint: "a".repeat(64),
        audio_id: "audio-1", whisper: "small", provider: "ollama_local", llm_model: "old-model",
        expected_language: "en", feedback_language: "de", speaker_id: "bern", task_family: "opinion_monologue",
        theme: "The pros and cons of working from home", target_duration_sec: 120,
        prompt_id: "work-home-b2", prompt_text: "Give your opinion on working from home.",
        retry_of_session_id: "", resume_assessment_id: "asmt-failed", label: "Original label", notes: "Original notes" }),
    };
    const adapter = draftAdapterFixture([saved]);
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState, draftAdapter: adapter });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledOnce());
    expect(mockedCreateAssessment.mock.calls[0][0]).toEqual(expect.objectContaining({ request_id: "assessment-one",
      resume_assessment_id: "asmt-failed", label: "Original label", notes: "Original notes", llm_model: "old-model", feedback_language: "de" }));
    expect(mockedUploadAudio).not.toHaveBeenCalled();
  });

  it("explains a stall and retries only after cancellation with retained audio", async () => {
    mockedGetAssessmentStatus.mockResolvedValue({ assessment_id: "asmt-1", status: "running",
      phase: "scoring_rubric", progress: 0, created_at: new Date(Date.now() - 240_000).toISOString(),
      updated_at: new Date(Date.now() - 240_000).toISOString(),
      phase_started_at: new Date(Date.now() - 240_000).toISOString(),
      worker_seen_at: new Date().toISOString(), quiet_warning_after_sec: 210 });
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
      worker_seen_at: new Date().toISOString(), quiet_warning_after_sec: 210 });
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

  it.each([
    ["network failure", new TypeError("Failed to fetch"), "Failed to fetch"],
    ["request timeout", new DOMException("The request timed out", "AbortError"), "The request timed out"],
  ])("reports a %s while retaining the active job and Cancel", async (_scenario, error, detail) => {
    const { queryClient, store } = renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedGetAssessmentStatus).toHaveBeenCalled());
    mockedGetAssessmentStatus.mockRejectedValue(error);
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.assessment("asmt-1") }); });
    await waitFor(() => expect(screen.getByTestId("speak.status_panel")).toHaveTextContent(`Could not refresh progress: ${detail}`));
    expect(store.getState().recording.job.assessmentId).toBe("asmt-1");
    expect(screen.getByTestId("speak.cancel_assessment")).toBeVisible();
    expect(screen.getByTestId("speak.assessment_progress")).toBeVisible();
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

    mockedGetReadiness.mockImplementation(() => new Promise(() => undefined));
    act(() => { void queryClient.invalidateQueries({ queryKey: queryKeys.readiness }); });
    await waitFor(() => expect(queryClient.getQueryState(queryKeys.readiness)?.fetchStatus).toBe("fetching"));
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
    mockedGetReadiness.mockRejectedValue(new Error("offline"));

    act(() => {
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.readiness }),
      ]);
    });
    await waitFor(() => {
      expect(queryClient.getQueryState(queryKeys.readiness)?.status).toBe("error");
    }, { timeout: 5_000 });
    expect(screen.getByTestId("speak.recording_panel")).toBe(panel);
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeVisible();
    expect(FakeMediaRecorder.instances.at(-1)?.state).toBe("recording");
  });

  it("keeps recording when a successful readiness refresh reports an unreachable tutor", async () => {
    const { queryClient } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"], locale: "en", appState: validDraftState,
    });
    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    const panel = screen.getByTestId("speak.recording_panel");
    await screen.findByRole("button", { name: "Stop recording" });
    expect(FakeMediaRecorder.instances.at(-1)?.state).toBe("recording");

    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "local_service_unreachable", "unavailable"));
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.readiness }); });

    expect(queryClient.getQueryState(queryKeys.readiness)?.status).toBe("success");
    expect(screen.getByTestId("speak.recording_panel")).toBe(panel);
    expect(screen.getByRole("button", { name: "Stop recording" })).toBeVisible();
    expect(FakeMediaRecorder.instances.at(-1)?.state).toBe("recording");
    expect(await screen.findByTestId("speak.readiness_changed")).toHaveTextContent("The local tutor service cannot be reached.");
    expect(screen.getByTestId("speak.readiness_changed")).toHaveTextContent("your take is preserved");
  });

  it("preserves an attached take and blocks submission until readiness recovers", async () => {
    const { queryClient } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"], locale: "en", appState: validDraftState,
    });
    fireEvent.click(await screen.findByTestId("speak.input_mode_upload"));
    fireEvent.change(screen.getByTestId("speak.upload_input"), { target: { files: [uploadFile] } });
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    const panel = screen.getByTestId("speak.recording_panel");

    const failedReadiness = readinessWith("ai_tutor", "local_service_unreachable", "unavailable");
    failedReadiness.checks[0] = { ...failedReadiness.checks[0], state: "needs_setup", reason: "model_not_cached", blocking: false };
    mockedGetReadiness.mockResolvedValue(failedReadiness);
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.readiness }); });

    expect(queryClient.getQueryState(queryKeys.readiness)?.status).toBe("success");
    expect(screen.getByTestId("speak.recording_panel")).toBe(panel);
    expect(screen.getByRole("button", { name: "Remove recording" })).toBeVisible();
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeDisabled());
    expect(screen.getByTestId("speak.readiness_changed")).toHaveTextContent("Start the local tutor app");
    expect(screen.getByTestId("speak.readiness_changed")).not.toHaveTextContent("The selected speech model is not cached.");

    mockedGetReadiness.mockResolvedValue(readyReadiness);
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.readiness }); });
    await waitFor(() => expect(screen.queryByTestId("speak.readiness_changed")).not.toBeInTheDocument());
    expect(screen.getByTestId("speak.recording_panel")).toBe(panel);
    expect(screen.getByTestId("speak.submit")).toBeEnabled();
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
      "Take ready",
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
      "Take ready",
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
