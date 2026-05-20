import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/renderWithProviders";

vi.mock("@/lib/api/client", () => ({
  apiClient: {
    createSupportBundle: vi.fn(),
    deleteRuntimeSettingsConnection: vi.fn(),
    downloadSupportBundle: vi.fn(),
    getDiagnostics: vi.fn(),
    getMaintenanceStorage: vi.fn(),
    getRuntime: vi.fn(),
    getRuntimeSettings: vi.fn(),
    getWhisperModelStatus: vi.fn(),
    postMaintenanceCleanup: vi.fn(),
    postRuntimeSettingsSetDefault: vi.fn(),
    postRuntimeSettingsTestConnection: vi.fn(),
    postWhisperModelDownload: vi.fn(),
    putRuntimeSettings: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";
import { AppFrame } from "@/App";
import type {
  RuntimeSettingsConnection,
  RuntimeSettingsResponse,
} from "@/lib/api/types";

const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetMaintenanceStorage = vi.mocked(apiClient.getMaintenanceStorage);
const mockedPostMaintenanceCleanup = vi.mocked(apiClient.postMaintenanceCleanup);
const mockedCreateSupportBundle = vi.mocked(apiClient.createSupportBundle);
const mockedDownloadSupportBundle = vi.mocked(apiClient.downloadSupportBundle);
const mockedGetRuntimeSettings = vi.mocked(apiClient.getRuntimeSettings);
const mockedGetWhisperModelStatus = vi.mocked(apiClient.getWhisperModelStatus);
const mockedPostRuntimeSettingsSetDefault = vi.mocked(apiClient.postRuntimeSettingsSetDefault);
const mockedPostRuntimeSettingsTestConnection = vi.mocked(apiClient.postRuntimeSettingsTestConnection);
const mockedPutRuntimeSettings = vi.mocked(apiClient.putRuntimeSettings);
const mockedDeleteRuntimeSettingsConnection = vi.mocked(apiClient.deleteRuntimeSettingsConnection);

const alphaConnection: RuntimeSettingsConnection = {
  connection_id: "conn-alpha",
  provider_key: "ollama",
  provider_choice: "ollama_local",
  provider_label: "Ollama local",
  label: "Alpha runtime",
  model: "llama3.2:3b",
  base_url: "http://localhost:11434",
  is_default: false,
  is_local: true,
  requires_api_key: false,
  has_api_key: false,
  secret_state: "absent",
  last_test_status: "",
  last_tested_at: "",
  openrouter_http_referer: "",
  openrouter_app_title: "",
  provider_metadata: {},
};

const bravoConnection: RuntimeSettingsConnection = {
  connection_id: "conn-bravo",
  provider_key: "openrouter",
  provider_choice: "openrouter",
  provider_label: "OpenRouter",
  label: "Bravo runtime",
  model: "gpt-4.1-mini",
  base_url: "https://openrouter.ai/api/v1",
  is_default: true,
  is_local: false,
  requires_api_key: true,
  has_api_key: true,
  secret_state: "present",
  last_test_status: "passed",
  last_tested_at: "2026-04-27T18:00:00Z",
  openrouter_http_referer: "https://example.test/app",
  openrouter_app_title: "Vostavo Desktop",
  provider_metadata: {
    http_referer: "https://example.test/app",
    app_title: "Vostavo Desktop",
  },
};

const runtimeSettings = (
  overrides: Partial<RuntimeSettingsResponse> = {},
): RuntimeSettingsResponse => ({
  ui_locale: "en",
  whisper_model: "large-v3",
  active_connection_id: "conn-bravo",
  connections: [alphaConnection, bravoConnection],
  ...overrides,
});

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
};

describe("Settings route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const currentLocalStorage = window.localStorage as
      | Storage
      | {
          clear?: () => void;
          getItem?: (key: string) => string | null;
          removeItem?: (key: string) => void;
          setItem?: (key: string, value: string) => void;
        }
      | undefined;
    if (
      !currentLocalStorage ||
      typeof currentLocalStorage.getItem !== "function" ||
      typeof currentLocalStorage.setItem !== "function" ||
      typeof currentLocalStorage.removeItem !== "function" ||
      typeof currentLocalStorage.clear !== "function"
    ) {
      const storage = new Map<string, string>();
      Object.defineProperty(window, "localStorage", {
        configurable: true,
        value: {
          clear: () => {
            storage.clear();
          },
          getItem: (key: string) => storage.get(key) ?? null,
          removeItem: (key: string) => {
            storage.delete(key);
          },
          setItem: (key: string, value: string) => {
            storage.set(key, value);
          },
        },
      });
    }
    window.localStorage.clear();
    Object.defineProperty(window.URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:settings-test"),
    });
    Object.defineProperty(window.URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });

    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: true,
    });
    mockedGetDiagnostics.mockResolvedValue({
      items: [],
    });
    mockedGetRuntimeSettings.mockResolvedValue(runtimeSettings());
    mockedGetWhisperModelStatus.mockResolvedValue({
      model: "large-v3",
      repo_id: "systran/faster-whisper-large-v3",
      cached: true,
      cached_path: "/tmp/whisper/large-v3",
      recommended: true,
      recommendation_reason: "Scoring baseline",
    });
  });

  it("updates the active saved connection through the runtime settings API and returns to Home", async () => {
    const defaultedSettings = runtimeSettings({
      active_connection_id: "conn-alpha",
      connections: [
        { ...alphaConnection, is_default: true },
        { ...bravoConnection, is_default: false },
      ],
    });
    mockedGetRuntimeSettings
      .mockResolvedValueOnce(runtimeSettings())
      .mockResolvedValue(defaultedSettings);
    mockedPostRuntimeSettingsSetDefault.mockResolvedValue(defaultedSettings);

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: [{ pathname: "/settings", state: { from: "home" } }],
      locale: "en",
    });

    expect(await screen.findByRole("heading", { name: "Settings" })).toBeVisible();
    expect(await screen.findAllByText("Alpha runtime")).toHaveLength(2);
    const selector = screen.getByTestId("settings.connection_id");
    fireEvent.change(selector, {
      target: {
        value: "conn-alpha",
      },
    });

    fireEvent.click(screen.getByTestId("settings.connection_row_set_default"));

    await waitFor(() => {
      expect(mockedPostRuntimeSettingsSetDefault).toHaveBeenCalledWith("conn-alpha");
      expect(store.getState().preferences.activeConnectionId).toBe("conn-alpha");
    });

    fireEvent.click(screen.getByTestId("settings.return"));

    expect(await screen.findByRole("button", { name: "Start new session" })).toBeVisible();
  });

  it("tests, saves, and deletes the selected connection through the runtime settings API", async () => {
    const savedSettings = runtimeSettings({
      connections: [
        alphaConnection,
        {
          ...bravoConnection,
          has_api_key: false,
          secret_state: "missing",
        },
      ],
    });
    const deletedSettings = runtimeSettings({
      active_connection_id: "conn-alpha",
      connections: [{ ...alphaConnection, is_default: true }],
    });
    mockedGetRuntimeSettings
      .mockResolvedValueOnce(runtimeSettings())
      .mockResolvedValue(savedSettings);
    mockedPostRuntimeSettingsTestConnection.mockResolvedValue({
      provider: "openrouter",
      base_url: "https://openrouter.ai/api/v1",
      service_base_url: "https://openrouter.ai/api",
      health_endpoint: "https://openrouter.ai/api/v1/models",
      discovered_models: ["gpt-4.1-mini"],
      tested_at: "2026-04-30T12:00:00+00:00",
      content_preview: "ok",
    });
    mockedPutRuntimeSettings.mockResolvedValue(savedSettings);
    mockedDeleteRuntimeSettingsConnection.mockResolvedValue(deletedSettings);

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findAllByText("Bravo runtime")).toHaveLength(2);
    expect(screen.getAllByText("A saved key is already available for this connection.")).toHaveLength(2);
    const providerField = screen.getByTestId("runtime_connection.provider");
    expect(within(providerField).getByRole("option", { name: "OpenRouter" })).toBeInTheDocument();
    expect(screen.getByTestId("runtime_connection.api_key")).toBeVisible();

    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));

    await waitFor(() => {
      expect(mockedPostRuntimeSettingsTestConnection).toHaveBeenCalledWith({
        connection: expect.objectContaining({
          connection_id: "conn-bravo",
          provider_choice: "openrouter",
        }),
      });
    });

    fireEvent.change(screen.getByTestId("runtime_connection.model"), {
      target: { value: "gpt-4.1-mini-updated" },
    });
    fireEvent.click(screen.getByTestId("runtime_connection.clear_saved_key"));
    fireEvent.click(screen.getByTestId("runtime_connection.clear_saved_key_confirm"));
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));

    await waitFor(() => {
      expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          ui_locale: "en",
          whisper_model: "large-v3",
          clear_saved_secret: true,
          connection: expect.objectContaining({
            connection_id: "conn-bravo",
            model: "gpt-4.1-mini-updated",
          }),
        }),
      );
    });

    fireEvent.click(screen.getByTestId("settings.connection_row_delete"));

    await waitFor(() => {
      expect(mockedDeleteRuntimeSettingsConnection).toHaveBeenCalledWith("conn-bravo");
    });
  });

  it("surfaces save failures without labeling them as test failures", async () => {
    mockedPutRuntimeSettings.mockRejectedValue(new Error("disk full"));

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findAllByText("Bravo runtime")).toHaveLength(2);
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));

    expect(await screen.findByText("Settings could not be saved: disk full")).toBeVisible();
    expect(screen.queryByText("Connection test failed: disk full")).not.toBeInTheDocument();
  });

  it("surfaces default connection failures without optimistic store updates", async () => {
    mockedPostRuntimeSettingsSetDefault.mockRejectedValue(new Error("backend offline"));

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findAllByText("Alpha runtime")).toHaveLength(2);
    fireEvent.change(screen.getByTestId("settings.connection_id"), {
      target: {
        value: "conn-alpha",
      },
    });
    fireEvent.click(screen.getByTestId("settings.connection_row_set_default"));

    expect(await screen.findByText("Default connection could not be changed: backend offline")).toBeVisible();
    expect(store.getState().preferences.activeConnectionId).not.toBe("conn-alpha");
  });

  it("surfaces delete connection failures without removing the selected connection", async () => {
    mockedDeleteRuntimeSettingsConnection.mockRejectedValue(new Error("locked"));

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findAllByText("Bravo runtime")).toHaveLength(2);
    fireEvent.click(screen.getByTestId("settings.connection_row_delete"));

    expect(await screen.findByText("Connection could not be deleted: locked")).toBeVisible();
    expect(screen.getAllByText("Bravo runtime")).toHaveLength(2);
  });

  it("resets auto-filled base URLs when the provider changes but preserves custom URLs", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findAllByText("Bravo runtime")).toHaveLength(2);
    const providerField = screen.getByTestId("runtime_connection.provider");
    const baseUrlField = screen.getByTestId("runtime_connection.base_url");
    const modelField = screen.getByTestId("runtime_connection.model");

    expect(providerField).toHaveValue("openrouter");
    expect(baseUrlField).toHaveValue("https://openrouter.ai/api/v1");
    expect(modelField).toHaveValue("gpt-4.1-mini");

    fireEvent.change(providerField, {
      target: { value: "ollama_local" },
    });
    expect(baseUrlField).toHaveValue("http://localhost:11434");
    expect(modelField).toHaveValue("");

    fireEvent.change(modelField, {
      target: { value: "llama3.2:3b" },
    });
    fireEvent.change(baseUrlField, {
      target: { value: "https://custom.example.test/v1" },
    });
    fireEvent.change(providerField, {
      target: { value: "lmstudio_local" },
    });
    expect(baseUrlField).toHaveValue("https://custom.example.test/v1");
    expect(modelField).toHaveValue("");
  });

  it("reports the edited draft provider when testing before save", async () => {
    mockedPostRuntimeSettingsTestConnection.mockResolvedValue({
      provider: "ollama",
      base_url: "http://localhost:11434/v1",
      service_base_url: "http://localhost:11434",
      health_endpoint: "http://localhost:11434/api/tags",
      discovered_models: ["llama3.2:3b"],
      tested_at: "2026-04-30T12:00:00+00:00",
      content_preview: "ok",
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findAllByText("Bravo runtime")).toHaveLength(2);

    fireEvent.change(screen.getByTestId("runtime_connection.provider"), {
      target: { value: "ollama_local" },
    });
    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));

    expect(
      await screen.findByText(
        "Connection test succeeded for Ollama local via http://localhost:11434/v1. Reply preview: ok",
      ),
    ).toBeVisible();
    expect(screen.queryByText(/Connection test succeeded for OpenRouter/)).not.toBeInTheDocument();
  });

  it("keeps the explicit create-new draft selected across runtime settings refreshes", async () => {
    const runtimeSettingsDeferred = createDeferred<RuntimeSettingsResponse>();
    mockedGetRuntimeSettings.mockReturnValue(runtimeSettingsDeferred.promise);

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    const selector = await screen.findByTestId("settings.connection_id");
    const labelField = screen.getByTestId("runtime_connection.connection_label");
    fireEvent.change(selector, {
      target: { value: "__new__" },
    });
    fireEvent.change(labelField, {
      target: { value: "New API target" },
    });

    await act(async () => {
      runtimeSettingsDeferred.resolve(runtimeSettings({ ui_locale: "de" }));
    });

    await waitFor(() => {
      expect(screen.getAllByText("Bravo runtime").length).toBeGreaterThan(0);
    });

    await waitFor(() => {
      expect(selector).toHaveValue("__new__");
      expect(labelField).toHaveValue("New API target");
    });
  });

  it("uses the support endpoints for storage, cleanup, and support bundles", async () => {
    mockedGetMaintenanceStorage.mockResolvedValue({
      app_data_root: "/tmp/vostavo-app",
      cache_root: "/tmp/vostavo-cache",
      areas: {
        tmp: {
          path: "/tmp/vostavo-app/tmp",
          size_bytes: 2048,
          file_count: 3,
        },
      },
    });
    mockedPostMaintenanceCleanup
      .mockResolvedValueOnce({
        target: "all_safe",
        dry_run: true,
        deleted_file_count: 4,
        freed_bytes: 4096,
        warnings: [],
      })
      .mockResolvedValueOnce({
        target: "all_safe",
        dry_run: false,
        deleted_file_count: 2,
        freed_bytes: 1024,
        warnings: ["rotated logs skipped"],
      });
    mockedCreateSupportBundle.mockResolvedValue({
      bundle_id: "bundle-123",
      filename: "support-bundle.zip",
      size_bytes: 8192,
      expires_at: "2026-04-28T08:30:00Z",
    });
    mockedDownloadSupportBundle.mockResolvedValue(
      new Blob(["bundle"], { type: "application/zip" }),
    );

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"],
      locale: "en",
    });

    expect(await screen.findByRole("heading", { name: "Troubleshooting and support" })).toBeVisible();

    fireEvent.click(screen.getByTestId("settings.support_refresh_storage"));
    expect(await screen.findByText("/tmp/vostavo-app/tmp")).toBeVisible();

    fireEvent.click(screen.getByTestId("settings.support_cleanup_preview"));
    expect(
      await screen.findByText("Cleanup preview found 4 file(s) and 4.0 KB safely removable."),
    ).toBeVisible();

    fireEvent.click(screen.getByTestId("settings.support_cleanup_run"));
    fireEvent.click(screen.getByTestId("settings.support_cleanup_run_confirm"));
    expect(
      await screen.findByText("Cleanup removed 2 file(s) and freed 1.0 KB."),
    ).toBeVisible();
    expect(screen.getByText("Support warning: rotated logs skipped")).toBeVisible();

    fireEvent.click(screen.getByLabelText("Include saved reports"));
    fireEvent.click(screen.getByLabelText("Include live runtime health check"));
    fireEvent.click(screen.getByTestId("settings.support_create_bundle"));

    await waitFor(() => {
      expect(mockedCreateSupportBundle).toHaveBeenCalledWith(
        expect.objectContaining({
          include_reports: true,
          include_recordings: false,
          include_runtime_health: true,
          include_uploads: false,
        }),
      );
    });
    expect(mockedDownloadSupportBundle).toHaveBeenCalledWith("bundle-123");
  });
});
