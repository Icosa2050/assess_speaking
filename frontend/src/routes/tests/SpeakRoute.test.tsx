import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppFrame } from "@/App";
import { renderWithProviders } from "@/test/renderWithProviders";

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
    getRuntimeSettings: vi.fn(),
    uploadAudio: vi.fn(),
    getUploadLimits: vi.fn(),
  },
}));

import { apiClient, ApiClientError } from "@/lib/api/client";

const mockedCancelAssessment = vi.mocked(apiClient.cancelAssessment);
const mockedCreateAssessment = vi.mocked(apiClient.createAssessment);
const mockedGetAssessmentStatus = vi.mocked(apiClient.getAssessmentStatus);
const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
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
  it("keeps recording blocked without calibration while upload remains usable", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en",
      appState: { ...validDraftState, microphoneSetupPassed: false } });
    expect(await screen.findByTestId("speak.record_start")).toBeDisabled();
    expect(screen.getByTestId("speak.microphone_setup")).toBeEnabled();
    expect(getUserMedia).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("speak.input_mode_upload"));
    expect(screen.getByTestId("speak.upload_input")).toBeEnabled();
  });
  const stopTrack = vi.fn();
  const getUserMedia = vi.fn();

  beforeEach(() => {
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
    expect(screen.getByTestId("speak.download_recording")).toHaveAttribute("download", "retry.wav");
    await waitFor(() => expect(screen.getByTestId("speak.submit")).toBeEnabled());
    fireEvent.click(screen.getByTestId("speak.submit"));
    await waitFor(() => expect(mockedCreateAssessment).toHaveBeenCalledTimes(2));
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

  it("clears the previous attachment when switching between record and upload modes", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
    fireEvent.click(await screen.findByRole("button", { name: "Stop recording" }));

    const statusPanel = await screen.findByTestId("speak.status_panel");
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
    expect(await screen.findByTestId("speak.submit")).toBeEnabled();

    fireEvent.click(screen.getByRole("link", { name: "Scoring Guide" }));
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

  it.each([29, 30, 31])("only enables a recorded review after 30 decoded seconds (recorded: %s)", async (duration) => {
    const clock = vi.spyOn(Date, "now").mockReturnValue(100_000);
    try {
      renderWithProviders(<AppFrame />, { initialEntries: ["/speak"], locale: "en", appState: validDraftState });
      fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));
      await screen.findByRole("button", { name: "Stop recording" });
      clock.mockReturnValue(100_000 + duration * 1000);
      fireEvent.click(screen.getByRole("button", { name: "Stop recording" }));
      const submit = screen.getByTestId("speak.submit");
      if (duration < 30) expect(submit).toBeDisabled();
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

  it("shows the provider warning, runs the queued or running lifecycle, and acknowledges cancel", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "OpenRouter",
      model: "google/gemini-3.1-pro-preview",
      base_url: "https://openrouter.ai/api/v1",
      requires_api_key: true,
      has_api_key: false,
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
          has_api_key: false,
          secret_state: "missing",
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

    expect(
      await screen.findByText(
        "OpenRouter is selected, but this saved connection is missing its API key. Open Runtime Setup or Settings to add one.",
      ),
    ).toBeVisible();

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
    expect(
      within(statusPanel).getByText(
        "This usually takes a short moment. You can leave this screen open while the review finishes.",
      ),
    ).toBeVisible();
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
          report: {
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
          band: "B2",
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
  });
});
