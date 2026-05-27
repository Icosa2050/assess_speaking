import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/renderWithProviders";

vi.mock("@/lib/api/client", () => ({
  apiClient: {
    getDiagnostics: vi.fn(),
    getRuntime: vi.fn(),
    getRuntimeSettings: vi.fn(),
    getWhisperModelStatus: vi.fn(),
    postWhisperModelDownload: vi.fn(),
    postRuntimeSettingsTestConnection: vi.fn(),
    putRuntimeSettings: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";
import { AppFrame } from "@/App";

const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetRuntimeSettings = vi.mocked(apiClient.getRuntimeSettings);
const mockedGetWhisperModelStatus = vi.mocked(apiClient.getWhisperModelStatus);
const mockedPostWhisperModelDownload = vi.mocked(apiClient.postWhisperModelDownload);
const mockedPostRuntimeSettingsTestConnection = vi.mocked(apiClient.postRuntimeSettingsTestConnection);
const mockedPutRuntimeSettings = vi.mocked(apiClient.putRuntimeSettings);

describe("Home and Runtime Setup routes", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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
        {
          key: "runtime",
          status: "warning",
          title_key: "diagnostics.runtime_title",
          detail_key: "diagnostics.runtime_warning_detail",
          detail_args: {},
        },
      ],
    });
    mockedGetWhisperModelStatus.mockResolvedValue({
      model: "medium",
      repo_id: "systran/faster-whisper-medium",
      cached: false,
      cached_path: "",
      recommended: true,
      recommendation_reason: "Practice tier",
    });
    mockedPostWhisperModelDownload.mockResolvedValue({
      model: "medium",
      repo_id: "systran/faster-whisper-medium",
      cached: true,
      cached_path: "/tmp/medium",
      recommended: true,
      recommendation_reason: "Practice tier",
    });
  });

  it("keeps configured Home focused on the next speaking action", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale: "en",
      appState: {
        preferences: {
          setupComplete: true,
        },
      },
    });

    const startNewButton = await screen.findByTestId("home.start_new");
    const diagnosticsHeading = await screen.findByText("Startup checks");

    expect(startNewButton).toBeVisible();
    expect(startNewButton.closest("section")?.className).toContain("primaryPracticeCard");
    expect(diagnosticsHeading.closest("section")?.className).toContain("supportCard");
    expect(screen.queryByTestId("home.runtime_setup_button")).not.toBeInTheDocument();
  });

  it("shows the runtime setup branch on Home and renders interactive runtime controls", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "medium",
      active_connection_id: "",
      connections: [],
    });
    mockedPutRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "medium",
      active_connection_id: "conn-ollama",
      connections: [
        {
          connection_id: "conn-ollama",
          provider_key: "ollama",
          provider_choice: "ollama_local",
          provider_label: "Ollama local",
          label: "Ollama local",
          model: "llama3.2:3b",
          base_url: "http://localhost:11434",
          is_default: true,
          is_local: true,
          requires_api_key: false,
          has_api_key: false,
          secret_state: "absent",
          last_test_status: "",
          last_tested_at: "",
          openrouter_http_referer: "",
          openrouter_app_title: "",
          provider_metadata: {},
        },
      ],
    });
    mockedPostRuntimeSettingsTestConnection.mockResolvedValue({
      provider: "ollama",
      base_url: "http://localhost:11434/v1",
      service_base_url: "http://localhost:11434",
      health_endpoint: "http://localhost:11434/api/tags",
      discovered_models: ["llama3.2:3b", "qwen2.5"],
      tested_at: "2026-04-30T12:00:00+00:00",
      content_preview: "ok",
    });
    mockedPutRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "medium",
      active_connection_id: "conn-ollama",
      connections: [
        {
          connection_id: "conn-ollama",
          provider_key: "ollama",
          provider_choice: "ollama_local",
          provider_label: "Ollama local",
          label: "Ollama local",
          model: "llama3.2:3b",
          base_url: "http://localhost:11434",
          is_default: true,
          is_local: true,
          requires_api_key: false,
          has_api_key: false,
          secret_state: "absent",
          last_test_status: "",
          last_tested_at: "",
          openrouter_http_referer: "",
          openrouter_app_title: "",
          provider_metadata: {},
        },
      ],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale: "en",
    });

    fireEvent.click(await screen.findByRole("button", { name: "Set up local AI" }));

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.getByTestId("runtime_connection.form")).toBeVisible();
    const providerField = screen.getByTestId("runtime_connection.provider");
    expect(within(providerField).getByRole("option", { name: "Ollama local" })).toBeInTheDocument();
    expect(within(providerField).getByRole("option", { name: "LM Studio local" })).toBeInTheDocument();
    expect(within(providerField).queryByRole("option", { name: "OpenRouter" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show cloud and advanced providers" })).toBeVisible();
    expect(screen.getByTestId("runtime_setup.detect_local_models")).toBeVisible();
    expect(screen.queryByTestId("runtime_connection.api_key")).not.toBeInTheDocument();
    const diagnosticsSummary = screen.getByText("Connection diagnostics");
    expect(diagnosticsSummary).toBeVisible();
    expect(diagnosticsSummary.closest("details")).not.toHaveAttribute("open");

    fireEvent.click(screen.getByTestId("runtime_setup.detect_local_models"));

    await waitFor(() => {
      expect(mockedPostRuntimeSettingsTestConnection).toHaveBeenCalled();
    });
    expect(screen.getByText("Detected 2 local model(s) via http://localhost:11434/api/tags.")).toBeVisible();

    const detectedModelField = screen.getByLabelText("Detected local models");
    expect(screen.getByTestId("runtime_connection.model")).toHaveValue("llama3.2:3b");
    expect(
      within(detectedModelField).queryByText("Local model discovery can populate this field from the running service."),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));
    const healthCheckMessage =
      "Health check passed at http://localhost:11434/api/tags. Smoke test reached http://localhost:11434/v1 with model llama3.2:3b. Preview: ok";
    expect(await screen.findByText(healthCheckMessage)).toBeVisible();

    fireEvent.change(screen.getByTestId("runtime_connection.provider"), {
      target: { value: "lmstudio_local" },
    });
    expect(screen.getByTestId("runtime_connection.model")).toHaveValue("");
    expect(screen.getByTestId("runtime_connection.base_url")).toHaveValue("http://localhost:1234/v1");
    expect(screen.queryByLabelText("Detected local models")).not.toBeInTheDocument();
    expect(screen.queryByText("Detected 2 local model(s) via http://localhost:11434/api/tags.")).not.toBeInTheDocument();
    expect(screen.queryByText(healthCheckMessage)).not.toBeInTheDocument();

    fireEvent.change(screen.getByTestId("runtime_connection.provider"), {
      target: { value: "ollama_local" },
    });
    fireEvent.change(screen.getByTestId("runtime_connection.model"), {
      target: { value: "llama3.2:3b" },
    });
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));

    await waitFor(() => {
      expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          ui_locale: "en",
          whisper_model: "medium",
          connection: expect.objectContaining({
            provider_choice: "ollama_local",
            model: "llama3.2:3b",
          }),
        }),
      );
    });
    expect(screen.getByText("Connection saved and set as active.")).toBeVisible();
  });

  it("keeps cloud providers behind the advanced toggle on runtime setup", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "medium",
      active_connection_id: "",
      connections: [],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"],
      locale: "en",
    });

    const providerField = await screen.findByTestId("runtime_connection.provider");
    expect(within(providerField).queryByRole("option", { name: "OpenRouter" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show cloud and advanced providers" }));

    expect(within(providerField).getByRole("option", { name: "OpenRouter" })).toBeInTheDocument();
    expect(within(providerField).getByRole("option", { name: "Generic OpenAI-compatible" })).toBeInTheDocument();

    fireEvent.change(providerField, {
      target: { value: "openrouter" },
    });

    expect(screen.queryByTestId("runtime_setup.detect_local_models")).not.toBeInTheDocument();
    expect(screen.getByTestId("runtime_connection.api_key")).toBeVisible();
    expect(screen.getByLabelText("OpenRouter HTTP-Referer")).toBeVisible();

    fireEvent.change(screen.getByTestId("runtime_connection.api_key"), {
      target: { value: "temporary-cloud-key" },
    });
    fireEvent.change(providerField, {
      target: { value: "ollama_local" },
    });
    expect(screen.queryByTestId("runtime_connection.api_key")).not.toBeInTheDocument();

    fireEvent.change(screen.getByTestId("runtime_connection.model"), {
      target: { value: "llama3.2:3b" },
    });
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));

    await waitFor(() => {
      expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          connection: expect.objectContaining({
            provider_choice: "ollama_local",
            api_key: "",
          }),
        }),
      );
    });
  });

  it("preserves saved-secret clear confirmation and save flow for existing connections", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "openrouter",
      model: "google/gemini-3.1-pro-preview",
      base_url: "https://openrouter.ai/api/v1",
      requires_api_key: true,
      has_api_key: true,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "large-v3",
      active_connection_id: "conn-primary",
      connections: [
        {
          connection_id: "conn-primary",
          provider_key: "openrouter",
          provider_choice: "openrouter",
          provider_label: "OpenRouter",
          label: "Primary cloud",
          model: "google/gemini-3.1-pro-preview",
          base_url: "https://openrouter.ai/api/v1",
          is_default: true,
          is_local: false,
          requires_api_key: true,
          has_api_key: true,
          secret_state: "present",
          last_test_status: "passed",
          last_tested_at: "2026-04-30T12:00:00+00:00",
          openrouter_http_referer: "https://example.test/app",
          openrouter_app_title: "Vostavo Desktop",
          provider_metadata: {
            http_referer: "https://example.test/app",
            app_title: "Vostavo Desktop",
          },
        },
      ],
    });
    mockedPostRuntimeSettingsTestConnection.mockResolvedValue({
      provider: "openrouter",
      base_url: "https://openrouter.ai/api/v1",
      service_base_url: "https://openrouter.ai/api",
      health_endpoint: "https://openrouter.ai/api/v1/models",
      discovered_models: ["google/gemini-3.1-pro-preview"],
      tested_at: "2026-04-30T12:00:00+00:00",
      content_preview: "ok",
    });
    mockedPutRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "large-v3",
      active_connection_id: "conn-primary",
      connections: [
        {
          connection_id: "conn-primary",
          provider_key: "openrouter",
          provider_choice: "openrouter",
          provider_label: "OpenRouter",
          label: "Primary cloud",
          model: "google/gemini-3.1-pro-preview",
          base_url: "https://openrouter.ai/api/v1",
          is_default: true,
          is_local: false,
          requires_api_key: true,
          has_api_key: false,
          secret_state: "missing",
          last_test_status: "passed",
          last_tested_at: "2026-04-30T12:00:00+00:00",
          openrouter_http_referer: "https://example.test/app",
          openrouter_app_title: "Vostavo Desktop",
          provider_metadata: {
            http_referer: "https://example.test/app",
            app_title: "Vostavo Desktop",
          },
        },
      ],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"],
      locale: "en",
      appState: {
        preferences: {
          activeConnectionId: "conn-primary",
          setupComplete: true,
        },
      },
    });

    const form = await screen.findByTestId("runtime_connection.form");
    expect(
      await within(form).findByText("A saved key is already available for this connection."),
    ).toBeVisible();
    expect(screen.queryByTestId("runtime_connection.api_key")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Replace saved key" }));

    expect(screen.getByTestId("runtime_connection.api_key")).toBeVisible();

    fireEvent.click(screen.getByTestId("runtime_connection.clear_saved_key"));
    expect(
      screen.getByText("Confirm that the saved key should be removed from this connection."),
    ).toBeVisible();

    fireEvent.click(screen.getByTestId("runtime_connection.clear_saved_key_confirm"));
    expect(screen.getByText("The saved key will be removed when you save this connection.")).toBeVisible();

    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));

    await waitFor(() => {
      expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          clear_saved_secret: true,
          connection: expect.objectContaining({
            connection_id: "conn-primary",
            provider_choice: "openrouter",
          }),
        }),
      );
    });
    expect(screen.getByText("Connection saved and set as active.")).toBeVisible();
  });

  it("blocks invalid OpenRouter referers before saving", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en",
      whisper_model: "medium",
      active_connection_id: "",
      connections: [],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"],
      locale: "en",
    });

    expect(await screen.findByTestId("runtime_connection.form")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Show cloud and advanced providers" }));
    fireEvent.change(screen.getByTestId("runtime_connection.provider"), {
      target: { value: "openrouter" },
    });
    fireEvent.change(screen.getByLabelText("OpenRouter HTTP-Referer"), {
      target: { value: "test-referer" },
    });

    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));

    expect(
      await screen.findByText(
        "OpenRouter HTTP-Referer must be a full URL starting with http:// or https://.",
      ),
    ).toBeVisible();
    expect(mockedPutRuntimeSettings).not.toHaveBeenCalled();
  });
});
