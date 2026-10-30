import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/renderWithProviders";
import { draftAdapterFixture, savedDraftFixture } from "@/test/draftAdapterFixture";
import { buildSessionDraft } from "@/lib/state/sessionDraft";

vi.mock("@/lib/api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    detail: string;

    constructor(detail: string) {
      super(detail);
      this.detail = detail;
    }
  },
  apiClient: {
    getSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getResumeSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getDiagnostics: vi.fn(),
    getReadiness: vi.fn().mockResolvedValue({ ready: true, checks: [
      "speech_recognition", "ai_tutor", "media_decoder", "storage",
    ].map(key => ({ key, state: "available", blocking: true, reason: "available", detail_args: {} })) }),
    getRuntime: vi.fn(),
    getRuntimeSettings: vi.fn().mockResolvedValue({ ui_locale: "en", whisper_model: "small", active_connection_id: "conn-local", connections: [] }),
    getWhisperModelStatus: vi.fn().mockResolvedValue({ available: true }),
    getSamples: vi.fn(),
  },
}));

import { AppFrame } from "@/App";
import { apiClient } from "@/lib/api/client";

const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetSamples = vi.mocked(apiClient.getSamples);

describe("Library and Guide routes", () => {
  it("opens Session Setup from a Library sample before a learner name exists on desktop", async () => {
    const adapter = draftAdapterFixture();
    renderWithProviders(<AppFrame />, { initialEntries: ["/library"], locale: "en", draftAdapter: adapter,
      appState: { draft: { speakerId: "" }, preferences: { activeConnectionId: "conn-local", setupComplete: true } } });
    fireEvent.click(await screen.findByTestId("library.sample_prepare"));
    expect(await screen.findByLabelText("Learner name or nickname")).toBeVisible();
    expect(adapter.saveSetup).not.toHaveBeenCalled();
    expect(adapter.replaceSetup).not.toHaveBeenCalled();
    expect(screen.queryByText("Could not use local draft storage. Keep this window open or try again.")).not.toBeInTheDocument();
  });

  it("replaces a saved task before preparing a different Library sample", async () => {
    const saved = savedDraftFixture("Alex");
    saved.setup = buildSessionDraft({ ...saved.setup, learningLanguage: "en", learningLanguageLabel: "English" });
    const adapter = draftAdapterFixture([saved]);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/library"], locale: "en", draftAdapter: adapter,
      appState: { draft: saved.setup, preferences: { activeConnectionId: "conn-local", setupComplete: true } } });
    fireEvent.click(await screen.findByRole("button", { name: "Resume" }));
    await screen.findByTestId("speak.recording_panel");
    fireEvent.click(screen.getByRole("link", { name: "Library" }));
    fireEvent.click(await screen.findByTestId("library.sample_prepare"));
    await waitFor(() => expect(adapter.replaceSetup).toHaveBeenCalledWith(expect.objectContaining({ themeId: "en_B1_travel_story" }), saved));
    expect(adapter.discard).not.toHaveBeenCalled();
    await waitFor(() => expect(store.getState().draft.themeId).toBe("en_B1_travel_story"));
    expect(store.getState().draft.sessionId).not.toBe(saved.setup.sessionId);
    expect(adapter.saveSetup).not.toHaveBeenCalled();
    expect(await screen.findByTestId("speak.recording_panel")).toBeVisible();
    expect(screen.queryByTestId("speak.download_recording")).not.toBeInTheDocument();
    confirm.mockRestore();
  });

  it("keeps the previous Library task and take when replacement storage fails", async () => {
    const saved = savedDraftFixture("Alex");
    saved.setup = buildSessionDraft({ ...saved.setup, learningLanguage: "en", learningLanguageLabel: "English" });
    const adapter = draftAdapterFixture([saved]);
    adapter.replaceSetup.mockRejectedValue(new Error("disk unavailable"));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/library"], locale: "en", draftAdapter: adapter,
      appState: { draft: saved.setup, preferences: { activeConnectionId: "conn-local", setupComplete: true } } });
    fireEvent.click(await screen.findByRole("button", { name: "Continue without resuming" }));
    fireEvent.click(await screen.findByTestId("library.sample_prepare"));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not use local draft storage");
    expect(store.getState().draft.themeId).toBe(saved.setup.themeId);
    expect(adapter.discard).not.toHaveBeenCalled();
    expect(screen.getByTestId("library.sample_prepare")).toBeVisible();
    confirm.mockRestore();
  });
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetDiagnostics.mockResolvedValue({ items: [] });
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetSamples.mockResolvedValue({
      items: [
        {
          sample_id: "en_B1_travel_story",
          language: "en",
          cefr: "B1",
          title: "TRAVEL STORY",
          path: "/samples/cefr/en/B1/travel_story.wav",
        },
        {
          sample_id: "it_C1_public_debate",
          language: "it",
          cefr: "C1",
          title: "public debate",
          path: "/samples/cefr/it/C1/public_debate.wav",
        },
      ],
    });
  });

  it("opens Library and Guide from Home without Streamlit navigation", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-local",
          setupComplete: true,
        },
      },
    });

    fireEvent.click(await screen.findByTestId("home.open_library"));
    expect(await screen.findByRole("heading", { name: "Pick your next exercise" })).toBeVisible();

    fireEvent.click(screen.getByRole("link", { name: "How scores work" }));
    expect(await screen.findByRole("heading", { name: "Understand your feedback" })).toBeVisible();
    expect(screen.getByText("Content validity")).toBeVisible();
  });

  it("shows localized library themes and prepares a shipped sample for Speak", async () => {
    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/library"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-local",
          setupComplete: true,
        },
        draft: {
          speakerId: "learner-1",
          learningLanguage: "en",
          learningLanguageLabel: "English",
          cefrLevel: "B1",
          themeId: "b1-my-last-trip-abroad",
          themeLabel: "My last trip abroad",
          promptText: "Talk about your last trip.",
        },
      },
    });

    expect(await screen.findByRole("heading", { name: "Pick your next exercise" })).toBeVisible();
    expect(screen.getByText("Choose a prompt or sample and move straight back into speaking practice.")).toBeVisible();
    expect(screen.getByText("My last trip abroad")).toBeVisible();
    expect(await screen.findByText("Travel Story")).toBeVisible();
    expect(screen.getByRole("button", { name: "Practice this prompt" })).toBeVisible();
    expect(screen.getByTestId("library.custom_theme_builder")).toHaveTextContent(
      "Optional: manage your own themes when the shipped prompts do not fit today's practice.",
    );

    fireEvent.click(screen.getByTestId("library.sample_prepare"));

    await waitFor(() => {
      expect(store.getState().draft.themeId).toBe("en_B1_travel_story");
      expect(store.getState().draft.themeLabel).toBe("Travel Story");
    });
  });

  it("only allows themes to be added for profiled practice languages", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/library"], locale: "en" });
    await screen.findByRole("heading", { name: "Pick your next exercise" });

    expect(screen.getByTestId("library.language_filter")).toBeInTheDocument();
    expect(screen.getByLabelText("Practice language")).toBeInTheDocument();
    expect(screen.getByLabelText("Learning language")).toBeInTheDocument();
    const languageSelects = screen.getAllByRole("combobox") as HTMLSelectElement[];
    expect(languageSelects.map((select) => Array.from(select.options).map((option) => option.value))).toEqual([
      ["en", "it"], ["en", "it"], ["B1", "B2", "C1"],
      ["travel_narrative", "personal_experience", "opinion_monologue", "picture_description", "free_monologue"],
    ]);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "A new English topic" } });
    fireEvent.click(screen.getByTestId("library.save_theme"));
    expect(await screen.findByText("A new English topic")).toBeVisible();
  });

  it("shows a loading state while shipped samples are loading", async () => {
    mockedGetSamples.mockImplementationOnce(() => new Promise(() => undefined));

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/library"],
      locale: "en",
    });

    expect(await screen.findByTestId("library-samples-loading")).toHaveTextContent(
      "Loading sample recordings...",
    );
  });

  it("shows a localized error when shipped samples cannot be loaded", async () => {
    mockedGetSamples.mockRejectedValueOnce(
      Object.assign(new Error("offline"), { responseStatus: 400 }),
    );

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/library"],
      locale: "en",
    });

    expect(await screen.findByTestId("library-samples-error")).toHaveTextContent(
      "The sample recordings could not be loaded.",
    );
  });

  it("renders the scoring guide sections from localized copy", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/guide"],
      locale: "en",
    });

    expect(await screen.findByRole("heading", { name: "Understand your feedback" })).toBeVisible();
    expect(screen.getByTestId("guide.practice_support_intro")).toHaveTextContent(
      "Use this guide to understand a score or a note in your feedback.",
    );
    expect(screen.getByRole("link", { name: "Quality checks" })).toHaveAttribute("href", "#guide-gates");
    expect(screen.getByText("Final score formula")).toBeVisible();
    expect(screen.getByText("Speaking pace (WPM)")).toBeVisible();
    expect(screen.getByText("Provisional CEFR estimate")).toBeVisible();
  });
});
