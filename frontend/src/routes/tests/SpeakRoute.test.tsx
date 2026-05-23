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
    cancelAssessment: vi.fn(),
    createAssessment: vi.fn(),
    getAssessmentStatus: vi.fn(),
    getRuntime: vi.fn(),
    getRuntimeSettings: vi.fn(),
    uploadAudio: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";

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

describe("Speak route", () => {
  const stopTrack = vi.fn();
  const getUserMedia = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
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

    expect(await screen.findByRole("heading", { name: "Prepare one speaking session" })).toBeVisible();
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
    expect(
      await within(statusPanel).findByText("A recording is attached and ready for assessment."),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Remove recording" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Upload" }));

    await waitFor(() => {
      expect(within(statusPanel).getByText("No recording is attached yet.")).toBeVisible();
    });
    expect(screen.queryByRole("button", { name: "Remove recording" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("Or upload an audio file")).toBeVisible();
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

    fireEvent.click(screen.getByTestId("speak.remove_recording"));

    await waitFor(() => {
      expect(within(statusPanel).getByText("No recording is attached yet.")).toBeVisible();
    });
    expect(screen.queryByTestId("speak.remove_recording")).not.toBeInTheDocument();
    expect(screen.getByTestId("speak.submit")).toBeDisabled();
  });

  it("records browser audio and attaches it for assessment", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/speak"],
      locale: "en",
      appState: validDraftState,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Start recording" }));

    expect(getUserMedia).toHaveBeenCalledWith({ audio: true });
    expect(await screen.findByRole("button", { name: "Stop recording" })).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Stop recording" }));

    expect(stopTrack).toHaveBeenCalled();
    expect(FakeMediaRecorder.instances[0]?.mimeType).toBe("audio/webm;codecs=opus");
    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(
      await within(statusPanel).findByText("A recording is attached and ready for assessment."),
    ).toBeVisible();
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
          "Microphone access did not finish. Check the browser permission prompt or use Upload.",
        ),
      ).toBeVisible();

      fireEvent.click(screen.getByRole("button", { name: "Upload" }));

      expect(screen.getByLabelText("Or upload an audio file")).toBeVisible();
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
      expect(screen.getByLabelText("Or upload an audio file")).toBeVisible();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(8_000);
      });

      expect(
        screen.queryByText(
          "Microphone access did not finish. Check the browser permission prompt or use Upload.",
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

    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));

    expect(
      await within(statusPanel).findByText(
        "Your assessment is running via OpenRouter with model `google/gemini-3.1-pro-preview`.",
      ),
    ).toBeVisible();
    expect(within(statusPanel).getByText("Current step: Transcribing recording.")).toBeVisible();
    expect(
      within(statusPanel).getByText(
        "This can take several minutes. Keep this page open; the review will appear automatically when it is ready.",
      ),
    ).toBeVisible();
    await waitFor(() => {
      expect(mockedCreateAssessment).toHaveBeenCalledWith(
        expect.objectContaining({
          openrouter_app_title: "Vostavo Review",
          openrouter_http_referer: "https://example.test/assess-speaking",
          whisper: "small",
        }),
      );
    });
    expect(within(statusPanel).getByText("Provider `OpenRouter` · model `google/gemini-3.1-pro-preview` · Whisper `small`")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Cancel assessment" }));

    expect(
      await within(statusPanel).findByText(
        "The assessment was cancelled. You can adjust your notes or recording and try again.",
      ),
    ).toBeVisible();
    expect(screen.getByDisplayValue("Morning run")).toBeVisible();
    expect(screen.getByDisplayValue("Mention two concrete examples.")).toBeVisible();
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
    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));

    const statusPanel = await screen.findByTestId("speak.status_panel");
    expect(await within(statusPanel).findByText("Assessment failed: boom")).toBeVisible();

    fireEvent.change(screen.getByLabelText("Or upload an audio file"), {
      target: { files: [uploadFile] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));

    rerender(<AppFrame />);

    expect(await screen.findByTestId("review-summary")).toBeVisible();
  });
});
