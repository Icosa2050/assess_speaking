import "@testing-library/jest-dom/vitest";
import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/test/renderWithProviders";
import { useAppStore } from "@/lib/state/appStore";
import { SavedLocale } from "./SavedLocale";

vi.mock("@/lib/api/client", () => ({ apiClient: { getRuntimeSettings: vi.fn() } }));
import { apiClient } from "@/lib/api/client";
import type { RuntimeSettingsResponse } from "@/lib/api/types";
const mockedSettings = vi.mocked(apiClient.getRuntimeSettings);
function LocaleProbe() { return <output data-testid="locale">{useAppStore(state => state.preferences.uiLocale)}</output>; }

describe("saved interface language", () => {
  it("restores Italian from settings when opening history directly", async () => {
    mockedSettings.mockResolvedValue({ ui_locale: "it" } as RuntimeSettingsResponse);
    renderWithProviders(<><SavedLocale /><LocaleProbe /></>, { initialEntries: ["/history"], locale: "en" });
    await waitFor(() => expect(screen.getByTestId("locale")).toHaveTextContent("it"));
  });
});
