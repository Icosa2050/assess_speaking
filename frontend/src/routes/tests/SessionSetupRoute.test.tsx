import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor } from "@testing-library/react";
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
    getDiagnostics: vi.fn(),
    getRuntime: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";

const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
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
  });

  it("updates the draft preview and routes to runtime setup when no runtime connection exists", async () => {
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

    fireEvent.change(await screen.findByLabelText("Speaker ID"), {
      target: { value: "bern" },
    });
    fireEvent.change(screen.getByLabelText("Learning language"), {
      target: { value: "en" },
    });
    fireEvent.change(screen.getByLabelText("Target CEFR level"), {
      target: { value: "B2" },
    });
    fireEvent.change(screen.getByLabelText("Theme"), {
      target: { value: "The pros and cons of working from home" },
    });
    fireEvent.change(screen.getByLabelText("Target duration (seconds)"), {
      target: { value: "120" },
    });

    expect(
      await screen.findByText(
        "Give your opinion about 'The pros and cons of working from home' with at least two distinct arguments.",
      ),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Continue to speaking" }));

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

    fireEvent.change(await screen.findByLabelText("Speaker ID"), {
      target: { value: "alex" },
    });
    fireEvent.change(screen.getByLabelText("Learning language"), {
      target: { value: "en" },
    });
    fireEvent.change(screen.getByLabelText("Theme"), {
      target: { value: "__custom__" },
    });
    fireEvent.change(screen.getByLabelText("Custom theme text"), {
      target: { value: "Climate transitions" },
    });
    fireEvent.click(screen.getByLabelText("Save this custom theme for reuse"));

    expect(
      await screen.findByText("Speak in English about 'Climate transitions' with a simple but clear structure."),
    ).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Continue to speaking" }));

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

    fireEvent.change(await screen.findByLabelText("Theme"), {
      target: { value: "__custom__" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue to speaking" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Speaker ID is required.");
    expect(alert).toHaveTextContent("Choose or enter a theme.");
    expect(screen.getByLabelText("Speaker ID")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByLabelText("Speaker ID")).toHaveAttribute(
      "aria-describedby",
      expect.stringContaining("setup-error-speaker-id"),
    );
    expect(screen.getByLabelText("Theme")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByLabelText("Custom theme text")).toHaveAttribute("aria-invalid", "true");
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

    fireEvent.change(await screen.findByLabelText("Speaker ID"), {
      target: { value: "alex" },
    });
    fireEvent.change(screen.getByLabelText("Target CEFR level"), {
      target: { value: "Z9" },
    });
    fireEvent.change(screen.getByLabelText("Target duration (seconds)"), {
      target: { value: "999" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue to speaking" }));

    await waitFor(() => {
      expect(screen.getByTestId("speak.status_panel")).toBeVisible();
    });
    expect(store.getState().draft.cefrLevel).toBe("B1");
    expect(store.getState().draft.durationSec).toBe(90);
  });
});
