import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
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
    getDiagnostics: vi.fn(),
    getHistory: vi.fn(),
    getRuntime: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";

const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetHistory = vi.mocked(apiClient.getHistory);
const STORAGE_KEY = "assess-speaking.session-setup.theme-library";
const createLocalStorageMock = () => {
  const store = new Map<string, string>();

  return {
    clear: () => {
      store.clear();
    },
    getItem: (key: string) => store.get(key) ?? null,
    removeItem: (key: string) => {
      store.delete(key);
    },
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
  };
};

describe("Session Setup route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: createLocalStorageMock(),
    });
    mockedGetDiagnostics.mockResolvedValue({
      items: [
        {
          key: "whisper",
          status: "warning",
          title_key: "diagnostics.whisper_title",
          detail_key: "diagnostics.whisper_warning_detail",
          detail_args: {
            model: "small",
            path: "/tmp/whisper",
          },
        },
      ],
    });
    mockedGetHistory.mockResolvedValue({ items: [] });
  });

  it.each([
    { initial: "", edited: "maria", description: "an unsubmitted nickname" },
    { initial: "maria", edited: "alex", description: "an edited nickname" },
    { initial: "maria", edited: "", description: "an explicitly cleared nickname" },
  ])("keeps $description when leaving setup and returning", async ({ initial, edited }) => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: { draft: { speakerId: initial } },
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: edited },
    });
    fireEvent.click(screen.getByRole("link", { name: "Practice Home" }));
    await screen.findByRole("button", { name: "Start new session" });
    fireEvent.click(screen.getByRole("link", { name: "Session Setup" }));

    expect(await screen.findByLabelText("Learner name or nickname")).toHaveValue(edited);
  });

  it("guides a beginner from learner name to a recommended practice", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    const wizard = await screen.findByTestId("setup.wizard");
    expect(wizard).toHaveAttribute("data-semantic-id", "setup.wizard");
    expect(screen.getByTestId("setup.step_learner")).toHaveAttribute("aria-current", "step");
    expect(screen.getByTestId("setup.recommended_start")).toHaveAttribute(
      "data-semantic-id",
      "setup.recommended_start",
    );

    expect(screen.getByTestId("setup.layout")).toHaveStyle({
      gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 22rem), 1fr))",
    });
    expect(screen.getByLabelText("Learner name or nickname")).toHaveAttribute(
      "data-semantic-id",
      "setup.speaker_id",
    );

    fireEvent.change(screen.getByLabelText("Learner name or nickname"), {
      target: { value: "maria" },
    });
    fireEvent.click(screen.getByTestId("setup.recommended_start"));

    expect(screen.getByTestId("setup.step_practice")).toHaveAttribute("aria-current", "step");
    expect(screen.getByLabelText("Learning language")).toHaveValue("it");
    expect(screen.getByLabelText("Level for this practice")).toHaveValue("B1");
    expect(screen.getByLabelText("Speaking time goal")).toHaveValue("90");
    expect(screen.getByLabelText("Theme")).toHaveValue("Il mio ultimo viaggio all'estero");
    expect(screen.queryByLabelText("Save this topic for reuse")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Today's speaking brief" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Your practice summary" })).toBeVisible();
    expect(screen.getByTestId("setup.runtime_callout")).toHaveAttribute(
      "data-semantic-id",
      "setup.runtime_callout",
    );
    expect(screen.getByTestId("setup.runtime_callout")).toHaveTextContent("Ready to record");

    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });

    expect(store.getState().draft.speakerId).toBe("maria");
    expect(store.getState().draft.learningLanguage).toBe("it");
    expect(store.getState().draft.cefrLevel).toBe("B1");
    expect(store.getState().draft.themeLabel).toBe("Il mio ultimo viaggio all'estero");
    expect(store.getState().draft.durationSec).toBe(90);
  });

  it("keeps the recommended library theme authoritative over a populated prior draft", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        draft: {
          speakerId: "returning-learner",
          learningLanguage: "en",
          learningLanguageLabel: "English",
          cefrLevel: "B2",
          themeId: "prior-b2-theme",
          themeLabel: "The pros and cons of working from home",
          taskFamily: "opinion_monologue",
          durationSec: 120,
          promptText: "Prior prompt",
        },
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.click(await screen.findByTestId("setup.change_details"));
    fireEvent.click(screen.getByTestId("setup.recommended_start"));

    expect(screen.getByLabelText("Level for this practice")).toHaveValue("B1");
    expect(screen.getByLabelText("Theme")).toHaveValue("My last trip abroad");
    expect(screen.queryByLabelText("My own topic")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });
    expect(store.getState().draft.cefrLevel).toBe("B1");
    expect(store.getState().draft.themeLabel).toBe("My last trip abroad");
    expect(store.getState().draft.taskFamily).toBe("travel_narrative");
  });

  it("personalizes the recommended starter from recent learner history", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetHistory.mockResolvedValue({
      items: [
        {
          timestamp: new Date().toISOString(),
          session_id: "history-1",
          speaker_id: "maria",
          learning_language: "en",
          theme: "My last trip abroad",
          task_family: "travel_narrative",
          overall: 4.1,
          wpm: 112,
          report_path: "/tmp/history-1.json",
          requires_human_review: false,
          duration_pass: true,
          topic_pass: true,
          language_pass: true,
          min_words_pass: true,
          top_priorities: [],
          grammar_error_categories: [],
          coherence_issue_categories: [],
          final_score: 4.1,
          band: "4",
        },
      ],
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: "maria" },
    });

    await waitFor(() => {
      expect(screen.getByTestId("setup.recommendation_hint")).toHaveTextContent(
        "Picking up from your last English practice · B1 · 90 s",
      );
    });

    fireEvent.click(screen.getByTestId("setup.recommended_start"));

    expect(screen.getByLabelText("Learning language")).toHaveValue("en");
    expect(screen.getByLabelText("Level for this practice")).toHaveValue("B1");
    expect(screen.getByLabelText("Theme")).toHaveValue("My last trip abroad");
    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });
    expect(store.getState().draft.learningLanguage).toBe("en");
    expect(store.getState().draft.themeLabel).toBe("My last trip abroad");
    expect(store.getState().draft.taskFamily).toBe("travel_narrative");
    expect(store.getState().draft.durationSec).toBe(90);
  });

  it("keeps the generic starter when learner history is stale", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetHistory.mockResolvedValue({
      items: [
        {
          timestamp: "2025-01-01T12:00:00Z",
          session_id: "history-old",
          speaker_id: "maria",
          learning_language: "en",
          theme: "My last trip abroad",
          task_family: "travel_narrative",
          overall: 4.1,
          wpm: 112,
          report_path: "/tmp/history-old.json",
          requires_human_review: false,
          duration_pass: true,
          topic_pass: true,
          language_pass: true,
          min_words_pass: true,
          top_priorities: [],
          grammar_error_categories: [],
          coherence_issue_categories: [],
          final_score: 4.1,
          band: "4",
        },
      ],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: "maria" },
    });

    expect(screen.getByTestId("setup.recommendation_hint")).toHaveTextContent(
      "Recommended starter: Italiano · B1 · 90 s",
    );

    fireEvent.click(screen.getByTestId("setup.recommended_start"));

    expect(screen.getByLabelText("Learning language")).toHaveValue("it");
    expect(screen.getByLabelText("Theme")).toHaveValue("Il mio ultimo viaggio all'estero");
  });

  it("lets learners customize details and routes to runtime setup when no runtime connection exists", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: "bern" },
    });
    fireEvent.click(screen.getByTestId("setup.customize_details"));
    fireEvent.change(screen.getByLabelText("Learning language"), {
      target: { value: "en" },
    });
    fireEvent.change(screen.getByLabelText("Level for this practice"), {
      target: { value: "B2" },
    });
    fireEvent.change(screen.getByLabelText("Theme"), {
      target: { value: "The pros and cons of working from home" },
    });
    fireEvent.change(screen.getByLabelText("Speaking time goal"), {
      target: { value: "120" },
    });

    expect(
      await screen.findByText(
        "Give your opinion about 'The pros and cons of working from home' with at least two distinct arguments.",
      ),
    ).toBeVisible();
    expect(screen.getByTestId("setup.runtime_callout")).toHaveTextContent("Device setup needed");

    fireEvent.click(screen.getByRole("button", { name: "Save practice and set up device" }));

    expect(await screen.findByRole("heading", { name: "Section A · Whisper" })).toBeVisible();
    expect(store.getState().draft.speakerId).toBe("bern");
    expect(store.getState().draft.learningLanguage).toBe("en");
    expect(store.getState().draft.cefrLevel).toBe("B2");
    expect(store.getState().draft.themeLabel).toBe("The pros and cons of working from home");
    expect(store.getState().draft.taskFamily).toBe("opinion_monologue");
    expect(store.getState().draft.durationSec).toBe(120);
    expect(store.getState().draft.promptText).toContain("Give your opinion about");
  });

  it("saves a custom theme for reuse and routes to speak when runtime is ready", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: "alex" },
    });
    fireEvent.click(screen.getByTestId("setup.customize_details"));
    fireEvent.change(screen.getByLabelText("Learning language"), {
      target: { value: "en" },
    });
    fireEvent.click(screen.getByTestId("setup.advanced_topic"));
    expect(screen.getByTestId("setup.advanced_topic")).toHaveAttribute(
      "data-semantic-id",
      "setup.advanced_topic",
    );
    fireEvent.change(screen.getByLabelText("My own topic"), {
      target: { value: "Climate transitions" },
    });
    fireEvent.click(screen.getByLabelText("Save this topic for reuse"));

    expect(
      await screen.findByText("Speak in English about 'Climate transitions' with a simple but clear structure."),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });

    const storedLibrary = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}") as {
      en?: { themes?: Array<{ level: string; task_family: string; title: string }> };
    };
    expect(storedLibrary.en?.themes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          title: "Climate transitions",
          level: "B1",
          task_family: "free_monologue",
        }),
      ]),
    );

    expect(store.getState().draft.speakerId).toBe("alex");
    expect(store.getState().draft.themeLabel).toBe("Climate transitions");
    expect(store.getState().draft.taskFamily).toBe("free_monologue");
    expect(store.getState().draft.promptText).toContain("Climate transitions");
  });

  it("restores the visible library theme when advanced topic details close", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: "alex" },
    });
    fireEvent.click(screen.getByTestId("setup.customize_details"));
    fireEvent.change(screen.getByLabelText("Learning language"), {
      target: { value: "en" },
    });
    const visibleLibraryTheme = String((screen.getByLabelText("Theme") as HTMLSelectElement).value);

    fireEvent.click(screen.getByTestId("setup.advanced_topic"));
    fireEvent.change(screen.getByLabelText("My own topic"), {
      target: { value: "Hidden custom authority" },
    });
    fireEvent.click(screen.getByTestId("setup.advanced_topic"));

    expect(screen.queryByLabelText("My own topic")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Theme")).toHaveValue(visibleLibraryTheme);

    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });
    expect(store.getState().draft.themeLabel).toBe(visibleLibraryTheme);
    expect(store.getState().draft.themeLabel).not.toBe("Hidden custom authority");
  });

  it("announces validation errors and links them to invalid setup fields", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.click(await screen.findByTestId("setup.customize_details"));
    fireEvent.click(screen.getByTestId("setup.advanced_topic"));
    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Learner name or nickname is required.");
    expect(alert).toHaveTextContent("Choose or enter a theme.");
    expect(screen.getByLabelText("Learner name or nickname")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByLabelText("Learner name or nickname")).toHaveAttribute(
      "aria-describedby",
      expect.stringContaining("setup-error-speaker-id"),
    );
    expect(screen.getByLabelText("Theme")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByLabelText("My own topic")).toHaveAttribute("aria-invalid", "true");
  });

  it("ignores invalid CEFR and duration select values", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/session-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), {
      target: { value: "alex" },
    });
    fireEvent.click(screen.getByTestId("setup.customize_details"));
    fireEvent.change(screen.getByLabelText("Level for this practice"), {
      target: { value: "Z9" },
    });
    fireEvent.change(screen.getByLabelText("Speaking time goal"), {
      target: { value: "999" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });
    expect(store.getState().draft.cefrLevel).toBe("B1");
    expect(store.getState().draft.durationSec).toBe(90);
  });
});
