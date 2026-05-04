import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/renderWithProviders";

vi.mock("@/lib/api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    detail: string;

    constructor(detail: string) {
      super(detail);
      this.detail = detail;
    }
  },
  apiClient: {
    getDiagnostics: vi.fn(),
    getRuntime: vi.fn(),
    getSamples: vi.fn(),
  },
}));

import { AppFrame } from "@/App";
import { apiClient } from "@/lib/api/client";

const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetSamples = vi.mocked(apiClient.getSamples);

describe("Library and Guide routes", () => {
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
          title: "travel story",
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
    expect(await screen.findByRole("heading", { name: "Theme library" })).toBeVisible();

    fireEvent.click(screen.getByRole("link", { name: "Scoring Guide" }));
    expect(await screen.findByRole("heading", { name: "How scoring works" })).toBeVisible();
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

    expect(await screen.findByRole("heading", { name: "Theme library" })).toBeVisible();
    expect(screen.getByText("My last trip abroad")).toBeVisible();
    expect(await screen.findByText("Travel Story")).toBeVisible();

    fireEvent.click(screen.getByTestId("library.sample_prepare"));

    await waitFor(() => {
      expect(store.getState().draft.themeId).toBe("en_B1_travel_story");
      expect(store.getState().draft.themeLabel).toBe("Travel Story");
    });
  });

  it("renders the scoring guide sections from localized copy", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/guide"],
      locale: "en",
    });

    expect(await screen.findByRole("heading", { name: "How scoring works" })).toBeVisible();
    expect(screen.getByText("Final score formula")).toBeVisible();
    expect(screen.getByText("Speaking pace (WPM)")).toBeVisible();
    expect(screen.getByText("Provisional CEFR estimate")).toBeVisible();
  });
});
