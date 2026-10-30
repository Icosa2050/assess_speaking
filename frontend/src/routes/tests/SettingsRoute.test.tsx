import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient } from "@tanstack/react-query";

import { renderWithProviders } from "@/test/renderWithProviders";
import { queryKeys } from "@/lib/query/queryClient";
import { readyReadiness, readinessWith } from "@/test/readinessFixtures";

vi.mock("@/lib/api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/api/client")>(),
  apiClient: {
    getSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getResumeSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    createSupportBundle: vi.fn(),
    deleteRuntimeSettingsConnection: vi.fn(),
    downloadSupportBundle: vi.fn(),
    getDiagnostics: vi.fn(),
    getReadiness: vi.fn().mockResolvedValue({ ready: true, checks: [
      "speech_recognition", "ai_tutor", "media_decoder", "storage",
    ].map(key => ({ key, state: "available", blocking: true, reason: "available", detail_args: {} })) }),
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

vi.mock("@/lib/rehearsal/maintenance", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/maintenance")>(),
  journalRecoveryStatus: vi.fn().mockResolvedValue({transaction: null, archived: [], completed: []}),
}));
vi.mock("@/lib/rehearsal/storage", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/storage")>(),
  listArchivedRehearsals: vi.fn().mockResolvedValue([]),
}));

import { apiClient } from "@/lib/api/client";
import { AppFrame } from "@/App";
import type {
  RuntimeSettingsConnection,
  RuntimeSettingsResponse,
} from "@/lib/api/types";

const mockedGetRuntime = vi.mocked(apiClient.getRuntime);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetReadiness = vi.mocked(apiClient.getReadiness);
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
  afterEach(() => vi.unstubAllGlobals());
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetReadiness.mockResolvedValue(readyReadiness);
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

  it("keeps provider fields in Setup while saved settings load", async () => {
    const pending = createDeferred<RuntimeSettingsResponse>();
    mockedGetRuntimeSettings.mockReturnValue(pending.promise);
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    expect(await screen.findByTestId("settings.open_setup")).toBeVisible();
    expect(screen.queryByTestId("runtime_connection.provider")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("settings.open_setup"));
    expect(screen.queryByTestId("runtime_connection.provider")).not.toBeInTheDocument();
    await act(async () => pending.resolve(runtimeSettings()));
    expect(await screen.findByTestId("runtime_connection.provider")).toHaveValue("ollama_local");
    expect(screen.getByRole("heading", { name: "Set up a provider" })).toHaveFocus();
  });

  it("shows load failure without exposing a Settings editor", async () => {
    mockedGetRuntimeSettings.mockRejectedValue(new Error("backend offline"));
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/settings"], locale: "en",
      queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("Saved settings could not be loaded");
    expect(screen.queryByTestId("runtime_connection.provider")).not.toBeInTheDocument();
    expect(screen.getByTestId("settings.open_setup")).toBeVisible();
  });

  it("opens a blank Setup editor and returns to Settings on cancel", async () => {
    mockedGetRuntimeSettings.mockResolvedValue(runtimeSettings({connections: [], active_connection_id: ""}));
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    fireEvent.click(await screen.findByTestId("settings.open_setup"));
    expect(await screen.findByTestId("runtime_connection.provider")).toHaveValue("ollama_local");
    fireEvent.click(screen.getByTestId("runtime_setup.back_home"));
    expect(await screen.findByTestId("settings.open_setup")).toBeVisible();
    expect(screen.queryByTestId("runtime_connection.provider")).not.toBeInTheDocument();
  });

  it("shows setup guidance when a saved connection is blocked by backend readiness", async () => {
    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "credential_missing"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    expect(await screen.findByText("Practice setup needs attention. Open Setup to resolve the failing check.")).toBeVisible();
  });

  it("returns to Review after opening Settings from the navigation", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/review"], locale: "en" });
    expect(await screen.findByTestId("review-guard-missing-review")).toBeVisible();
    fireEvent.click(screen.getByTestId("home.open_settings"));
    fireEvent.click(await screen.findByTestId("settings.return"));
    expect(await screen.findByTestId("review-guard-missing-review")).toBeVisible();
  });

  it("selects an active connection through Settings", async () => {
    const updated = runtimeSettings({active_connection_id: "conn-alpha", connections: [
      {...alphaConnection, is_default: true}, {...bravoConnection, is_default: false},
    ]});
    mockedGetRuntimeSettings.mockResolvedValueOnce(runtimeSettings()).mockResolvedValue(updated);
    mockedPostRuntimeSettingsSetDefault.mockResolvedValue(updated);
    const { store } = renderWithProviders(<AppFrame />, {initialEntries: ["/settings"], locale: "en"});
    await screen.findAllByText("Alpha runtime");
    fireEvent.change(screen.getByTestId("settings.connection_id"), {target: {value: "conn-alpha"}});
    fireEvent.click(screen.getByTestId("settings.connection_row_set_default"));
    await waitFor(() => expect(mockedPostRuntimeSettingsSetDefault).toHaveBeenCalledWith("conn-alpha"));
    await waitFor(() => expect(store.getState().preferences.activeConnectionId).toBe("conn-alpha"));
    expect(screen.getByTestId("settings.connection_id")).toHaveValue("conn-alpha");
  });

  it("surfaces active selection failures without changing the stored active connection", async () => {
    mockedPostRuntimeSettingsSetDefault.mockRejectedValue(new Error("backend offline"));
    const { store } = renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    await screen.findAllByText("Alpha runtime");
    fireEvent.change(screen.getByTestId("settings.connection_id"), { target: { value: "conn-alpha" } });
    fireEvent.click(screen.getByTestId("settings.connection_row_set_default"));
    expect(await screen.findByText("Default connection could not be changed: backend offline")).toBeVisible();
    expect(store.getState().preferences.activeConnectionId).not.toBe("conn-alpha");
  });

  it("refreshes credential diagnostics after selecting the local connection", async () => {
    let localActive = false;
    mockedGetReadiness.mockImplementation(async () => localActive ? readyReadiness : readinessWith("ai_tutor", "credential_missing"));
    const localSettings = runtimeSettings({ active_connection_id: "conn-alpha", connections: [
      { ...alphaConnection, is_default: true }, { ...bravoConnection, is_default: false },
    ] });
    const oldDiagnostics = { items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime_api_key", status: "error", title_key: "", detail_key: "", detail_args: {
        provider: "openrouter", model: "gpt-4.1-mini", base_url: "https://openrouter.ai/api/v1",
      } },
    ] };
    mockedGetRuntime.mockImplementation(async () => localActive
      ? { configured: true, provider: "ollama", model: "llama3.2:3b", base_url: "http://localhost:11434/v1", requires_api_key: false, has_api_key: false }
      : { configured: true, provider: "openrouter", model: "gpt-4.1-mini", base_url: "https://openrouter.ai/api/v1", requires_api_key: true, has_api_key: false });
    mockedGetRuntimeSettings.mockImplementation(async () => localActive ? localSettings : runtimeSettings());
    mockedGetDiagnostics.mockImplementation(async () => localActive
      ? { items: oldDiagnostics.items.filter((item) => item.key !== "runtime_api_key") }
      : oldDiagnostics);
    mockedPostRuntimeSettingsSetDefault.mockImplementation(async () => { localActive = true; return localSettings; });
    const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: false } } });
    queryClient.setQueryData(queryKeys.diagnostics, oldDiagnostics);
    renderWithProviders(<AppFrame />, { initialEntries: [{ pathname: "/settings", state: { from: "home" } }], locale: "en", queryClient });
    await screen.findAllByText("Alpha runtime");
    fireEvent.change(screen.getByTestId("settings.connection_id"), { target: { value: "conn-alpha" } });
    fireEvent.click(screen.getByTestId("settings.connection_row_set_default"));
    await waitFor(() => expect(queryClient.getQueryState(queryKeys.diagnostics)?.isInvalidated).toBe(true));
    fireEvent.click(screen.getByTestId("settings.return"));
    fireEvent.click(await screen.findByRole("button", { name: "Open Setup Guide" }));
    const tutor = await screen.findByTestId("runtime_setup.setup_guide.ai_tutor");
    await waitFor(() => expect(tutor).toHaveAttribute("data-status", "ready"));
  });

  it("edits a chosen connection in Setup and returns to Settings after saving", async () => {
    const edited = runtimeSettings({active_connection_id: "conn-alpha", connections: [{...alphaConnection, model: "new-local-model", is_default: true}, {...bravoConnection, is_default: false}]});
    const restored = runtimeSettings({connections: [{...alphaConnection, model: "new-local-model"}, bravoConnection]});
    mockedGetRuntimeSettings.mockResolvedValueOnce(runtimeSettings()).mockResolvedValue(restored);
    mockedPutRuntimeSettings.mockResolvedValue(edited);
    mockedPostRuntimeSettingsSetDefault.mockResolvedValue(restored);
    const { store } = renderWithProviders(<AppFrame />, {initialEntries: [{pathname: "/settings", state: {from: "home"}}], locale: "en"});
    const edit = await screen.findAllByTestId("settings.connection_row_edit");
    fireEvent.click(edit[1]);
    expect(await screen.findByTestId("runtime_connection.provider")).toHaveValue("ollama_local");
    expect(screen.getByRole("heading", {name: "Set up a provider"})).toHaveFocus();
    fireEvent.change(screen.getByTestId("runtime_connection.model"), {target: {value: "new-local-model"}});
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));
    await waitFor(() => expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(expect.objectContaining({
      connection: expect.objectContaining({connection_id: "conn-alpha", model: "new-local-model"}),
    })));
    await waitFor(() => expect(mockedPostRuntimeSettingsSetDefault).toHaveBeenCalledWith("conn-bravo"));
    expect(store.getState().preferences.activeConnectionId).toBe("conn-bravo");
    expect(await screen.findByTestId("settings.connection_id")).toHaveValue("conn-alpha");
    expect(screen.getByRole("heading", {name: "Saved connections"})).toHaveFocus();
    expect(screen.queryByTestId("runtime_connection.provider")).not.toBeInTheDocument();
  });

  it("shows a session-only warning after returning from Setup", async () => {
    const sessionOnly = runtimeSettings({ connections: [alphaConnection, { ...bravoConnection, provider_metadata: { persistent: false } }] });
    mockedGetRuntimeSettings.mockResolvedValueOnce(runtimeSettings()).mockResolvedValue(sessionOnly);
    mockedPutRuntimeSettings.mockResolvedValue(sessionOnly);
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    fireEvent.click((await screen.findAllByTestId("settings.connection_row_edit"))[0]);
    fireEvent.click(await screen.findByTestId("runtime_connection.save_connection"));
    expect(await screen.findByTestId("settings.connection_id")).toBeVisible();
    expect(screen.getAllByText(/credential is available for this session only/i)[0]).toBeVisible();
    expect(screen.queryByText("A saved key is already available for this connection.")).not.toBeInTheDocument();
  });

  it("keeps a failed provider save in Setup with its error", async () => {
    mockedPutRuntimeSettings.mockRejectedValue(new Error("disk full"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    fireEvent.click((await screen.findAllByTestId("settings.connection_row_edit"))[0]);
    fireEvent.click(await screen.findByTestId("runtime_connection.save_connection"));
    expect(await screen.findByText(/disk full/)).toBeVisible();
    expect(screen.getByTestId("runtime_connection.provider")).toBeVisible();
    expect(screen.queryByTestId("settings.connection_id")).not.toBeInTheDocument();
  });

  it("requires confirmation before removing a connection", async () => {
    mockedDeleteRuntimeSettingsConnection.mockResolvedValue(runtimeSettings({
      active_connection_id: "conn-alpha", connections: [{...alphaConnection, is_default: true}],
    }));
    renderWithProviders(<AppFrame />, {initialEntries: ["/settings"], locale: "en"});
    const remove = await screen.findAllByTestId("settings.connection_row_delete");
    fireEvent.click(remove[0]);
    expect(mockedDeleteRuntimeSettingsConnection).not.toHaveBeenCalled();
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("open");
    expect(remove[0]).toHaveAttribute("data-connection-id", "conn-bravo");
    expect(within(dialog).getByText(/Remove Bravo runtime/)).toBeVisible();
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(remove[0]).toHaveFocus();
    fireEvent.click(remove[0]);
    fireEvent.click(screen.getByTestId("settings.connection_row_delete_confirm"));
    await waitFor(() => expect(mockedDeleteRuntimeSettingsConnection).toHaveBeenCalledWith("conn-bravo"));
  });

  it("closes removal confirmation when its connection disappears", async () => {
    let exists = true;
    mockedGetRuntimeSettings.mockImplementation(async () => exists ? runtimeSettings() : runtimeSettings({
      active_connection_id: "conn-alpha", connections: [{ ...alphaConnection, is_default: true }],
    }));
    const { queryClient } = renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    fireEvent.click((await screen.findAllByTestId("settings.connection_row_delete"))[0]);
    expect(screen.getByRole("dialog")).toBeVisible();
    exists = false;
    await act(async () => { await queryClient.invalidateQueries({ queryKey: ["runtime", "settings"] }); });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "Saved connections" })).toHaveFocus();
  });

  it("reports deletion failures and keeps the connection selected", async () => {
    mockedDeleteRuntimeSettingsConnection.mockRejectedValue(new Error("locked"));
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    fireEvent.click((await screen.findAllByTestId("settings.connection_row_delete"))[0]);
    fireEvent.click(screen.getByTestId("settings.connection_row_delete_confirm"));
    expect(await screen.findByText("Connection could not be deleted: locked")).toBeVisible();
    expect(screen.getByTestId("settings.connection_id")).toHaveValue("conn-bravo");
    expect(screen.getAllByText("Bravo runtime")).toHaveLength(2);
  });

  it("explains why preferences cannot be saved before a provider is configured", async () => {
    mockedGetRuntimeSettings.mockResolvedValue(runtimeSettings({ connections: [], active_connection_id: "" }));
    renderWithProviders(<AppFrame />, { initialEntries: ["/settings"], locale: "en" });
    expect(await screen.findByText("Set up a provider before saving these preferences.")).toBeVisible();
    expect(screen.getByTestId("settings.save_preferences")).toBeDisabled();
  });

  it("keeps UI language and appearance in Settings", async () => {
    mockedPutRuntimeSettings.mockResolvedValue(runtimeSettings({ui_locale: "de"}));
    const { store } = renderWithProviders(<AppFrame />, {initialEntries: ["/settings"], locale: "en"});
    await screen.findAllByText("Bravo runtime");
    const localeField = screen.getByTestId("settings.ui_locale");
    fireEvent.change(localeField, {target: {value: "de"}});
    fireEvent.click(screen.getByRole("button", {name: "Save preferences"}));
    await waitFor(() => expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(expect.objectContaining({ui_locale: "de"})));
    await waitFor(() => expect(store.getState().preferences.uiLocale).toBe("de"));
    fireEvent.change(screen.getByTestId("settings.appearance"), {target: {value: "dark"}});
    await waitFor(() => expect(document.documentElement.dataset.theme).toBe("dark"));
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
      await screen.findByText("Files that can be removed: 4. Space to be freed: 4.0 KB."),
    ).toBeVisible();

    fireEvent.click(screen.getByTestId("settings.support_cleanup_run"));
    fireEvent.click(screen.getByTestId("settings.support_cleanup_run_confirm"));
    expect(
      await screen.findByText("Files removed: 2. Space freed: 1.0 KB."),
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
