import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/renderWithProviders";
import { queryKeys } from "@/lib/query/queryClient";
import { readyReadiness, readinessWith } from "@/test/readinessFixtures";

vi.mock("@/lib/api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    code = "runtime_error";
    detail: string;
    constructor(_status: number, error: { detail: string }) {
      super(error.detail);
      this.detail = error.detail;
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
const mockedGetReadiness = vi.mocked(apiClient.getReadiness);
const mockedGetRuntimeSettings = vi.mocked(apiClient.getRuntimeSettings);
const mockedGetWhisperModelStatus = vi.mocked(apiClient.getWhisperModelStatus);
const mockedPostWhisperModelDownload = vi.mocked(apiClient.postWhisperModelDownload);
const mockedPostRuntimeSettingsTestConnection = vi.mocked(apiClient.postRuntimeSettingsTestConnection);
const mockedPutRuntimeSettings = vi.mocked(apiClient.putRuntimeSettings);

const optionalTutorReadiness = () => {
  const response = readinessWith("ai_tutor", "not_configured");
  response.ready = true;
  response.checks[1].blocking = false;
  return response;
};

describe("Home and Runtime Setup routes", () => {
  afterEach(() => vi.unstubAllGlobals());
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetReadiness.mockResolvedValue(readyReadiness);
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "",
      connections: [], asr_provider: "local",
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
    const diagnosticsHeading = await screen.findByText("Please review these notices");
    const readinessMeter = screen.getByRole("progressbar", { name: "Practice readiness" });

    expect(startNewButton).toBeVisible();
    expect(startNewButton.closest("section")?.className).toContain("primaryPracticeCard");
    expect(readinessMeter).toHaveAttribute("aria-valuemin", "0");
    expect(readinessMeter).toHaveAttribute("aria-valuemax", "100");
    expect(readinessMeter).toHaveAttribute("aria-valuenow", "80");
    expect(diagnosticsHeading.closest("section")?.className).toContain("supportCard");
    expect(screen.getByRole("button", { name: "Open Setup Guide" })).toHaveAttribute(
      "data-semantic-id",
      "home.setup_guide_button",
    );
    expect(screen.queryByText("See what is ready and what still needs to be set up.")).not.toBeInTheDocument();
    expect(screen.queryByTestId("home.runtime_setup_button")).not.toBeInTheDocument();
  });

  it("keeps cached Home diagnostics visible after a background refresh fails", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "local-model", base_url: "http://localhost:11434",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    ] });
    const { queryClient } = renderWithProviders(<AppFrame />, { initialEntries: ["/"], locale: "en" });
    expect(await screen.findByText("Next step")).toBeVisible();

    mockedGetDiagnostics.mockRejectedValue(new Error("offline"));
    mockedGetReadiness.mockRejectedValue(new Error("offline"));
    await act(async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["diagnostics"] }),
        queryClient.invalidateQueries({ queryKey: queryKeys.readiness }),
      ]);
    });
    expect(queryClient.getQueryState(["diagnostics"])?.status).toBe("error");
    expect(queryClient.getQueryState(queryKeys.readiness)?.status).toBe("error");
    expect(screen.getByText("Next step")).toBeVisible();
    expect(screen.queryByText("The setup checks could not run. Open setup to check your connections.")).not.toBeInTheDocument();
  });

  it("opens timed rehearsal from configured Home", async () => {
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
      appState: { preferences: { setupComplete: true } },
    });

    const rehearsalLink = await screen.findByRole("link", { name: "Timed speaking rehearsal" });
    expect(rehearsalLink).toHaveAttribute("href", "/rehearsal");
    fireEvent.click(rehearsalLink);

    expect(await screen.findByTestId("rehearsal-screen")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Timed speaking rehearsal" })).toBeVisible();
    expect(screen.getByRole("link", { name: "Timed speaking rehearsal", current: "page" })).toBeVisible();
  });

  it("keeps timed rehearsal discoverable on first-run Home", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });

    renderWithProviders(<AppFrame />, { initialEntries: ["/"], locale: "en" });

    expect(await screen.findByRole("button", { name: "Get ready to practice" })).toBeVisible();
    expect(screen.getByTestId("first-run.preview")).toHaveTextContent("SAMPLE REPORT · ILLUSTRATIVE");
    expect(screen.getByTestId("first-run.preview")).toHaveTextContent("This is a sample, never your result or score.");
    expect(screen.getByRole("region", { name: /SAMPLE REPORT · ILLUSTRATIVE/ })).toHaveAccessibleDescription(/never your result or score/);
    expect(screen.getByRole("region", { name: /Example first task/ })).toBeVisible();
    fireEvent.click(screen.getByRole("link", { name: "Timed speaking rehearsal" }));
    expect(await screen.findByTestId("rehearsal-screen")).toBeVisible();
  });

  it("labels the German UI sample as an Italian response task", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false, provider: "", model: "", base_url: "",
      requires_api_key: false, has_api_key: false,
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/"], locale: "de" });

    const preview = await screen.findByTestId("first-run.preview");
    expect(preview).toHaveTextContent("Antworte auf Italienisch");
    expect(preview).toHaveTextContent("Beispiel · Italienisch · B1");
    expect(preview).toHaveTextContent("BEISPIELBERICHT · NUR ZUR VERANSCHAULICHUNG");
  });

  it("does not call stale saved preferences ready when the runtime is not configured", async () => {
    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "not_configured"));
    mockedGetRuntime.mockResolvedValue({ configured: false, provider: "", model: "", base_url: "", requires_api_key: false, has_api_key: false });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    ] });

    renderWithProviders(<AppFrame />, { initialEntries: ["/"], locale: "en", appState: {
      preferences: { setupComplete: true, activeConnectionId: "old-connection" },
      microphoneStatus: "ready", microphoneSetupPassed: true,
    } });

    expect(await screen.findByRole("button", { name: "Get ready to practice" })).toBeVisible();
    await waitFor(() => expect(screen.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "80"));
    expect(screen.queryByText("Ready on this device")).not.toBeInTheDocument();
  });

  it("keeps a cloud tutor without its key unavailable on Home and Runtime Setup", async () => {
    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "credential_missing"));
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "openrouter", model: "vendor/model", base_url: "https://example.test",
      requires_api_key: true, has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime_api_key", status: "error", title_key: "", detail_key: "", detail_args: {} },
    ] });
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });

    await waitFor(() => expect(screen.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "80"));
    expect(screen.queryByText("Ready on this device")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open Setup Guide" }));
    const tutor = await screen.findByTestId("runtime_setup.setup_guide.ai_tutor");
    await waitFor(() => expect(tutor).toHaveAttribute("data-status", "setup"));
    expect(tutor).toHaveTextContent("The selected service needs a saved account key.");
    expect(tutor).toHaveTextContent("Add the key to the selected connection");
  });

  it("treats ok and informational diagnostics as healthy after actual microphone capture", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({
      items: [
        {
          key: "whisper",
          status: "ok",
          title_key: "diagnostics.whisper_title",
          detail_key: "diagnostics.whisper_ok_detail",
          detail_args: {},
        },
        {
          key: "runtime",
          status: "ok",
          title_key: "diagnostics.runtime_title",
          detail_key: "diagnostics.runtime_ok_detail",
          detail_args: {},
        },
        {
          key: "microphone",
          status: "info",
          title_key: "diagnostics.microphone_title",
          detail_key: "diagnostics.microphone_info_detail",
          detail_args: {},
        },
      ],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale: "en",
      appState: {
        microphoneStatus: "ready",
        microphoneSetupPassed: true,
        preferences: {
          setupComplete: true,
        },
      },
    });

    expect(await screen.findByText("Ready on this device")).toBeVisible();
    expect(screen.queryByText("Please review these notices")).not.toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute(
      "aria-valuenow",
      "100",
    );
  });

  it("updates readiness only after confirming calibration while retaining unrelated notices", async () => {
    mockedGetRuntime.mockResolvedValue({ configured: true, provider: "ollama", model: "test", base_url: "", requires_api_key: false, has_api_key: false });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "maintenance", status: "warning", title_key: "", detail_key: "", detail_args: {} },
    ] });
    const { store } = renderWithProviders(<AppFrame />, { locale: "en" });
    await screen.findByText("Please review these notices");
    const meter = screen.getByRole("progressbar", { name: "Practice readiness" });
    expect(meter).toHaveAttribute("aria-valuenow", "80");
    act(() => store.getState().setMicrophoneStatus("ready"));
    expect(meter).toHaveAttribute("aria-valuenow", "80");
    act(() => store.getState().setMicrophoneSetupPassed(true));
    expect(meter).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText("Please review these notices")).toBeVisible();
    act(() => store.getState().setMicrophoneStatus("denied"));
    expect(meter).toHaveAttribute("aria-valuenow", "80");
  });

  it("counts only actionable diagnostics in the attention summary", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({
      items: [
        {
          key: "runtime",
          status: "ok",
          title_key: "diagnostics.runtime_title",
          detail_key: "diagnostics.runtime_ok_detail",
          detail_args: {},
        },
        {
          key: "microphone",
          status: "info",
          title_key: "diagnostics.microphone_title",
          detail_key: "diagnostics.microphone_info_detail",
          detail_args: {},
        },
        {
          key: "whisper",
          status: " DeGrAdEd ",
          title_key: "diagnostics.whisper_title",
          detail_key: "diagnostics.whisper_warning_detail",
          detail_args: {},
        },
      ],
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

    expect(await screen.findByText("Please review these notices")).toBeVisible();
    expect(screen.getByText("Items to check: 1. Open the setup guide to see your next steps.")).toBeVisible();
  });

  it.each([
    ["de", "Bitte prüfe die folgenden Hinweise"],
    ["en", "Please review these notices"],
    ["fr", "Points à vérifier"],
    ["es", "Avisos por revisar"],
    ["it", "Avvisi da verificare"],
  ] as const)("uses a neutral %s heading for maintenance-only warnings after setup", async (locale, heading) => {
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({
      items: [
        {
          key: "whisper",
          status: "ok",
          title_key: "diagnostics.whisper_title",
          detail_key: "diagnostics.whisper_ok_detail",
          detail_args: { model: "small", path: "/tmp/whisper" },
        },
        {
          key: "runtime",
          status: "ok",
          title_key: "diagnostics.runtime_title",
          detail_key: "diagnostics.runtime_ok_detail",
          detail_args: { provider: "ollama", model: "llama3.2:3b" },
        },
        {
          key: "maintenance_tmp",
          status: "warning",
          title_key: "diagnostics.maintenance_tmp_title",
          detail_key: "diagnostics.maintenance_tmp_warning_detail",
          detail_args: { file_count: 1, size_bytes: 1024, oldest_age_hours: 48 },
        },
      ],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale,
      appState: { preferences: { setupComplete: true } },
    });

    expect(await screen.findByRole("heading", { name: heading })).toBeVisible();
    expect(screen.getByTestId("home.start_new")).toBeEnabled();
    expect(screen.queryByTestId("home.runtime_setup_button")).not.toBeInTheDocument();
  });

  it("keeps UI language in Settings and avoids duplicate Home navigation copy", async () => {
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

    expect(await screen.findByRole("button", { name: "Start new session" })).toBeVisible();
    expect(screen.queryByLabelText("Interface language")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "English" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "Settings" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("Explore more")).not.toBeInTheDocument();
  });

  it("groups the configured navigation around practice without showing runtime setup as a peer route", async () => {
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

    const nav = screen.getByRole("navigation", { name: "Main navigation" });

    expect(await screen.findByRole("button", { name: "Start new session" })).toBeVisible();
    expect(within(nav).getByText("Practice")).toBeVisible();
    expect(within(nav).getByText("Progress")).toBeVisible();
    expect(within(nav).getByRole("link", { name: "Practice" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getByRole("link", { name: "Progress" })).toHaveAttribute("href", "/history");
    expect(within(nav).getByRole("link", { name: "Library" })).toHaveAttribute("href", "/library");
    expect(within(nav).getByRole("link", { name: "How scores work" })).toHaveAttribute("href", "/guide");
    expect(within(nav).getByRole("link", { name: "Settings" })).toHaveAttribute("href", "/settings");
    expect(within(nav).queryByRole("link", { name: "Session Setup" })).not.toBeInTheDocument();
    expect(within(nav).queryByRole("link", { name: "Speak" })).not.toBeInTheDocument();
    expect(within(nav).queryByRole("link", { name: "Review" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main-content");
    expect(within(nav).queryByRole("link", { name: "Set up local AI" })).not.toBeInTheDocument();
    expect(screen.getByTestId("home.open_history")).toHaveAttribute("href", "/history");
    expect(screen.getByTestId("home.open_library")).toHaveAttribute("href", "/library");
    expect(screen.getByTestId("home.open_guide")).toHaveAttribute("href", "/guide");
    expect(screen.getByTestId("home.open_settings")).toHaveAttribute("href", "/settings");
  });

  it("uses backend runtime readiness to remove runtime setup from primary navigation", async () => {
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
    });

    const nav = screen.getByRole("navigation", { name: "Main navigation" });

    expect(await screen.findByRole("button", { name: "Start new session" })).toBeVisible();
    await waitFor(() => {
      expect(within(nav).queryByRole("link", { name: "Set up local AI" })).not.toBeInTheDocument();
    });
  });

  it("keeps runtime setup reachable in navigation before setup is complete", async () => {
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

    const nav = screen.getByRole("navigation", { name: "Main navigation" });

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(document.title).toBe("Set up local AI");
    expect(within(nav).getByText("Practice")).toBeVisible();
    expect(within(nav).getByRole("link", { name: "Set up local AI" })).toHaveAttribute(
      "href",
      "/runtime-setup",
    );
    expect(within(nav).getByRole("link", { name: "Timed speaking rehearsal" })).toHaveAttribute("href", "/rehearsal");
    expect(nav.querySelectorAll('[aria-current="page"]')).toHaveLength(1);
  });

  it("shows the return step from route state on Home", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: [{ pathname: "/", state: { from: "/runtime-setup" } }],
      locale: "en",
    });

    const nav = screen.getByRole("navigation", { name: "Main navigation" });
    expect(await screen.findByRole("button", { name: "Get ready to practice" })).toBeVisible();
    expect(within(nav).getByRole("group", { name: "Return to step" })).toBeVisible();
    expect(within(nav).getByRole("link", { name: "Set up local AI" })).toHaveAttribute("href", "/runtime-setup");
    expect(nav.querySelectorAll('[aria-current="page"]')).toHaveLength(1);
  });

  it("shows a setup next step instead of an empty startup checklist", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false,
      provider: "",
      model: "",
      base_url: "",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [] });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale: "en",
    });

    expect(await screen.findByText("Next step")).toBeVisible();
    expect(screen.getByText("Run the guided checks to prepare your first recording.")).toBeVisible();
    expect(screen.queryByText("See what is ready and what still needs to be set up.")).not.toBeInTheDocument();
  });

  it("keeps startup checks in a loading state until readiness is known", () => {
    mockedGetRuntime.mockImplementation(() => new Promise(() => undefined));
    mockedGetDiagnostics.mockImplementation(() => new Promise(() => undefined));

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/"],
      locale: "en",
    });

    expect(screen.getByText("Running startup checks...")).toBeVisible();
    expect(screen.queryByText("Next step")).not.toBeInTheDocument();
    expect(screen.queryByText("Run the guided checks to prepare your first recording.")).not.toBeInTheDocument();
  });

  it("shows the setup guide readiness rows before runtime controls", async () => {
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

    expect(await screen.findByTestId("runtime_setup.setup_guide")).toBeVisible();
    expect(screen.getByTestId("runtime_setup.local_path")).toHaveTextContent("Recommended local path");
    expect(screen.getByRole("heading", { name: "Setup Guide" })).toBeVisible();
    expect(screen.getByTestId("runtime_setup.setup_guide.speech_recognition")).toHaveTextContent(
      "Speech recognition",
    );
    expect(screen.getByTestId("runtime_setup.setup_guide.ai_tutor")).toHaveTextContent("AI tutor");
    expect(screen.getByTestId("runtime_setup.setup_guide.microphone")).toHaveTextContent("Microphone");
    expect(screen.getByRole("button", { name: "Start a session" })).toBeDisabled();
    expect(screen.getByTestId("runtime_connection.form")).toBeVisible();
  });

  it("does not call cloud transcription ready without its selected Groq account", async () => {
    const readiness = readinessWith("speech_recognition", "credential_missing");
    readiness.checks[0].detail_args = { provider: "groq" };
    mockedGetReadiness.mockResolvedValue(readiness);
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "groq", model: "test", base_url: "https://example.test",
      requires_api_key: true, has_api_key: true,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "cloud-connection",
      connections: [], asr_provider: "groq",
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });

    expect(await screen.findByText("Follow the next action in the checklist below; a check is ready only after it passes.")).toBeVisible();
    expect(screen.getByTestId("runtime_setup.local_path")).toHaveTextContent("Your current setup");
    expect(screen.getByTestId("runtime_setup.local_path")).not.toHaveTextContent("Recommended local path");
    expect(screen.queryByRole("button", { name: "Choose a practice task" })).not.toBeInTheDocument();
    expect(screen.getByTestId("runtime_setup.setup_guide.speech_recognition.action")).toHaveTextContent("Configure cloud transcription");
  });

  it("marks Groq transcription ready only with its selected saved account and key", async () => {
    const cloudReady = structuredClone(readyReadiness);
    cloudReady.checks[0].detail_args = { provider: "groq", model: "whisper-large-v3" };
    mockedGetReadiness.mockResolvedValue(cloudReady);
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      settings: {
        version: 1, asr_provider: "groq", asr_connection_id: "groq-1", asr_model: "whisper-large-v3",
        openrouter_modes: {}, fallback_connection_id: "", paid_fallback_enabled: false,
        monthly_budget_usd: 5, max_output_tokens: 4096,
      },
      spending: { spent_usd: 0, reserved_usd: 0, unresolved_requests: [] },
    }), { status: 200, headers: { "Content-Type": "application/json" } })));
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "groq", model: "test", base_url: "https://example.test",
      requires_api_key: true, has_api_key: true,
    });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "groq-1", asr_provider: "groq",
      connections: [{
        connection_id: "groq-1", provider_key: "groq", provider_choice: "groq", provider_label: "Groq",
        label: "Groq speech", model: "test", base_url: "https://example.test", is_default: true,
        is_local: false, requires_api_key: true, has_api_key: true, secret_state: "present",
        last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
      }],
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });

    await waitFor(() => expect(screen.getByTestId("runtime_setup.setup_guide.speech_recognition")).toHaveAttribute("data-status", "ready"));
    expect(screen.getByTestId("runtime_setup.setup_guide.speech_recognition")).toHaveTextContent("Groq cloud transcription is selected and its saved key is present.");
    expect(screen.getByTestId("runtime_setup.setup_guide.speech_recognition.action")).toHaveTextContent("Manage cloud transcription");
    expect(screen.getByText("All required checks passed. You can choose a task.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Choose a practice task" })).toBeEnabled();
  });

  it("opens cloud rescue when Groq readiness is known but saved settings fail", async () => {
    const cloudReady = structuredClone(readyReadiness);
    cloudReady.checks[0].detail_args = { provider: "groq", model: "whisper-large-v3" };
    mockedGetReadiness.mockResolvedValue(cloudReady);
    mockedGetRuntimeSettings.mockRejectedValue(new Error("settings offline"));
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "test", base_url: "http://localhost:11434",
      requires_api_key: false, has_api_key: false,
    });
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      settings: { version: 1, asr_provider: "groq", asr_connection_id: "", asr_model: "whisper-large-v3",
        openrouter_modes: {}, fallback_connection_id: "", paid_fallback_enabled: false,
        monthly_budget_usd: 5, max_output_tokens: 4096 },
      spending: { spent_usd: 0, reserved_usd: 0, unresolved_requests: [] },
    }), { status: 200, headers: { "Content-Type": "application/json" } })));

    renderWithProviders(<AppFrame />, { initialEntries: ["/runtime-setup"], locale: "en" });
    const speech = await screen.findByTestId("runtime_setup.setup_guide.speech_recognition");
    await waitFor(() => expect(speech).toHaveAttribute("data-status", "ready"));
    expect(screen.getByTestId("runtime_setup.setup_guide.speech_recognition.action")).toHaveTextContent("Manage cloud transcription");
    expect(await screen.findByText("Saved cloud connections could not be loaded. Retry saved settings before changing Groq transcription.")).toBeVisible();
    const cloud = document.getElementById("runtime-setup-cloud");
    expect(cloud?.querySelector("details")).toHaveAttribute("open");
    const scrollCloud = vi.fn();
    cloud!.scrollIntoView = scrollCloud;
    const whisper = document.getElementById("runtime-setup-whisper");
    if (whisper) whisper.scrollIntoView = vi.fn();
    fireEvent.click(screen.getByTestId("runtime_setup.setup_guide.speech_recognition.action"));
    expect(scrollCloud).toHaveBeenCalledWith({ block: "start", behavior: "smooth" });
    expect(document.getElementById("runtime-setup-whisper")).toBeNull();
    expect(screen.getByRole("button", { name: "Retry saved settings" })).toBeVisible();
  });

  it("counts only required checks in Home progress when an optional check needs setup", async () => {
    mockedGetReadiness.mockResolvedValue(optionalTutorReadiness());
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "test", base_url: "http://localhost:11434",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [] });
    renderWithProviders(<AppFrame />, { initialEntries: ["/"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true } });

    expect(await screen.findByText("Ready on this device")).toBeVisible();
    expect(screen.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText("4 of 4 checks ready")).toBeVisible();
  });

  it("offers Speak return without calling an optional check a blocker", async () => {
    mockedGetReadiness.mockResolvedValue(optionalTutorReadiness());
    renderWithProviders(<AppFrame />, {
      initialEntries: [{ pathname: "/runtime-setup", state: { returnTo: "/speak" } }],
      locale: "en", appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });

    expect(await screen.findByRole("button", { name: "Return to the recording step" })).toBeVisible();
    expect(screen.getByText("The speaking checks now allow you to continue. Test your microphone before recording.")).toBeVisible();
    expect(screen.queryByText(/check needs attention before you can continue/)).not.toBeInTheDocument();
  });

  it("refreshes backend readiness across Setup and Home after a local connection test", async () => {
    mockedGetReadiness.mockResolvedValueOnce(readinessWith("ai_tutor", "local_service_unreachable")).mockResolvedValue(readyReadiness);
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "llama3.2:3b", base_url: "http://localhost:11434/v1",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    ] });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "local-1", asr_provider: "local",
      connections: [{
        connection_id: "local-1", provider_key: "ollama", provider_choice: "ollama_local", provider_label: "Ollama local",
        label: "Local tutor", model: "llama3.2:3b", base_url: "http://localhost:11434", is_default: true,
        is_local: true, requires_api_key: false, has_api_key: false, secret_state: "absent",
        last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
      }],
    });
    mockedGetWhisperModelStatus.mockResolvedValue({
      model: "small", repo_id: "systran/faster-whisper-small", cached: true, cached_path: "/tmp/small",
      recommended: true, recommendation_reason: "Practice tier",
    });
    mockedPostRuntimeSettingsTestConnection.mockResolvedValue({
      provider: "ollama", base_url: "http://localhost:11434/v1", service_base_url: "http://localhost:11434",
      health_endpoint: "http://localhost:11434/api/tags", discovered_models: ["llama3.2:3b"],
      tested_at: "2026-10-27T09:00:00Z", content_preview: "ok",
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });

    const tutor = await screen.findByTestId("runtime_setup.setup_guide.ai_tutor");
    await waitFor(() => expect(tutor).toHaveAttribute("data-status", "setup"));
    expect(screen.getByText("Follow the next action in the checklist below; a check is ready only after it passes.")).toBeVisible();
    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));
    await waitFor(() => expect(tutor).toHaveAttribute("data-status", "ready"));
    expect(screen.getByText("All required checks passed. You can choose a task.")).toBeVisible();

    fireEvent.click(screen.getByRole("link", { name: "Practice" }));
    expect(await screen.findByText("Ready on this device")).toBeVisible();
    expect(screen.getByRole("progressbar", { name: "Practice readiness" })).toHaveAttribute("aria-valuenow", "100");
    fireEvent.click(screen.getByTestId("home.start_new"));
    expect(await screen.findByText("Your checks are ready")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Learner name or nickname"), { target: { value: "maria" } });
    fireEvent.click(screen.getByTestId("setup.recommended_start"));
    expect(screen.getByTestId("setup.runtime_callout")).toHaveTextContent("Ready to record");
    expect(screen.getByRole("button", { name: "Start speaking" })).toBeVisible();
  });

  it("lets a learner choose a task and enter Speak before testing the microphone", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "local-model", base_url: "http://localhost:11434",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    ] });
    mockedGetWhisperModelStatus.mockResolvedValue({
      model: "small", repo_id: "systran/faster-whisper-small", cached: true, cached_path: "/tmp/small",
      recommended: true, recommendation_reason: "Practice tier",
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/runtime-setup"], locale: "en" });

    await waitFor(() => expect(screen.getByTestId("runtime_setup.setup_guide.microphone")).toHaveAttribute("data-status", "setup"));
    expect(screen.queryByText("All three checks passed. You can choose a task.")).not.toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Choose a practice task" }));
    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), { target: { value: "maria" } });
    fireEvent.click(screen.getByTestId("setup.recommended_start"));
    expect(screen.getByTestId("setup.runtime_callout")).toHaveTextContent("Continue to the recording step");
    fireEvent.click(screen.getByRole("button", { name: "Start speaking" }));
    expect(await screen.findByTestId("speak.recording_panel")).toBeVisible();
    expect(screen.getByTestId("speak.record_start")).toBeDisabled();
  });

  it("keeps a successful test on a new connection after saving and activating it", async () => {
    mockedGetReadiness.mockResolvedValue(readinessWith("ai_tutor", "not_configured"));
    mockedGetRuntime.mockResolvedValue({
      configured: false, provider: "", model: "", base_url: "",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetDiagnostics.mockResolvedValue({ items: [
      { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
    ] });
    mockedGetRuntimeSettings.mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [], asr_provider: "local",
    });
    mockedGetWhisperModelStatus.mockResolvedValue({
      model: "small", repo_id: "systran/faster-whisper-small", cached: true, cached_path: "/tmp/small",
      recommended: true, recommendation_reason: "Practice tier",
    });
    mockedPostRuntimeSettingsTestConnection.mockResolvedValue({
      provider: "ollama", base_url: "http://localhost:11434/v1", service_base_url: "http://localhost:11434",
      health_endpoint: "http://localhost:11434/api/tags", discovered_models: ["local-model"],
      tested_at: "2026-10-27T09:00:00Z", content_preview: "ok",
    });
    mockedPutRuntimeSettings.mockImplementation(async () => {
      mockedGetReadiness.mockResolvedValue(readyReadiness);
      mockedGetRuntime.mockResolvedValue({
        configured: true, provider: "ollama", model: "local-model", base_url: "http://localhost:11434/v1",
        requires_api_key: false, has_api_key: false,
      });
      return {
        ui_locale: "en", whisper_model: "small", active_connection_id: "new-local", asr_provider: "local",
        connections: [{
          connection_id: "new-local", provider_key: "ollama", provider_choice: "ollama_local", provider_label: "Ollama local",
          label: "Ollama local", model: "local-model", base_url: "http://localhost:11434", is_default: true,
          is_local: true, requires_api_key: false, has_api_key: false, secret_state: "absent",
          last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
        }],
      };
    });
    renderWithProviders(<AppFrame />, { initialEntries: ["/runtime-setup"], locale: "en" });

    fireEvent.change(await screen.findByTestId("runtime_connection.model"), { target: { value: "local-model" } });
    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));
    await screen.findByText(/Connection to .* succeeded/);
    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));
    await waitFor(() => expect(screen.getByTestId("runtime_setup.setup_guide.ai_tutor")).toHaveAttribute("data-status", "ready"));
    expect(screen.getByText("Start speaking opens the recording step. Finish the device checks before you record; the microphone check is required there.")).toBeVisible();
  });

  it("refreshes speech diagnostics after a model download before entering Session Setup", async () => {
    mockedGetReadiness.mockResolvedValue(readinessWith("speech_recognition", "model_not_cached"));
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "llama3.2:3b", base_url: "http://localhost:11434/v1",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetDiagnostics
      .mockResolvedValueOnce({ items: [
        { key: "whisper", status: "warning", title_key: "", detail_key: "", detail_args: {} },
        { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      ] })
      .mockResolvedValue({ items: [
        { key: "whisper", status: "ok", title_key: "", detail_key: "", detail_args: {} },
        { key: "runtime", status: "ok", title_key: "", detail_key: "", detail_args: {} },
      ] });
    mockedGetWhisperModelStatus
      .mockResolvedValueOnce({
        model: "small", repo_id: "systran/faster-whisper-small", cached: false, cached_path: "",
        recommended: true, recommendation_reason: "Practice tier",
      })
      .mockResolvedValue({
        model: "small", repo_id: "systran/faster-whisper-small", cached: true, cached_path: "/tmp/small",
        recommended: true, recommendation_reason: "Practice tier",
      });
    mockedPostWhisperModelDownload.mockImplementation(async () => {
      mockedGetReadiness.mockResolvedValue(readyReadiness);
      return { model: "small", repo_id: "systran/faster-whisper-small", cached: true, cached_path: "/tmp/small", recommended: true, recommendation_reason: "Practice tier" };
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });
    const speech = await screen.findByTestId("runtime_setup.setup_guide.speech_recognition");
    await waitFor(() => expect(speech).toHaveAttribute("data-status", "setup"));
    fireEvent.click(screen.getByTestId("runtime_setup.download_model"));
    await waitFor(() => expect(speech).toHaveAttribute("data-status", "ready"));
    expect(mockedGetDiagnostics).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole("button", { name: "Choose a practice task" }));
    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), { target: { value: "maria" } });
    fireEvent.click(screen.getByTestId("setup.recommended_start"));
    expect(screen.getByTestId("setup.runtime_callout")).toHaveTextContent("Ready to record");
    expect(screen.getByRole("button", { name: "Start speaking" })).toBeVisible();
  });

  it("does not continue with a downloaded speech model until that model is saved", async () => {
    let savedModel = "small";
    mockedGetReadiness.mockImplementation(async () => savedModel === "medium" ? readyReadiness : readinessWith("speech_recognition", "model_not_cached"));
    let mediumCached = false;
    const connection = {
      connection_id: "local-1", provider_key: "ollama", provider_choice: "ollama_local", provider_label: "Ollama local",
      label: "Local tutor", model: "llama3.2:3b", base_url: "http://localhost:11434", is_default: true,
      is_local: true, requires_api_key: false, has_api_key: false, secret_state: "absent" as const,
      last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
    };
    mockedGetRuntime.mockResolvedValue({
      configured: true, provider: "ollama", model: "llama3.2:3b", base_url: "http://localhost:11434/v1",
      requires_api_key: false, has_api_key: false,
    });
    mockedGetRuntimeSettings.mockImplementation(async () => ({
      ui_locale: "en", whisper_model: savedModel, active_connection_id: "local-1",
      asr_provider: "local", connections: [connection],
    }));
    mockedGetDiagnostics.mockImplementation(async () => ({ items: [
      { key: "whisper", status: savedModel === "small" ? "warning" as const : "ok" as const, title_key: "", detail_key: "", detail_args: {} },
      { key: "runtime", status: "ok" as const, title_key: "", detail_key: "", detail_args: {} },
    ] }));
    mockedGetWhisperModelStatus.mockImplementation(async (model) => ({
      model, repo_id: `systran/faster-whisper-${model}`, cached: model === "medium" && mediumCached,
      cached_path: model === "medium" && mediumCached ? "/tmp/medium" : "",
      recommended: true, recommendation_reason: "Practice tier",
    }));
    mockedPostWhisperModelDownload.mockImplementation(async (model) => {
      mediumCached = true;
      return { model, repo_id: `systran/faster-whisper-${model}`, cached: true,
        cached_path: "/tmp/medium", recommended: true, recommendation_reason: "Practice tier" };
    });
    mockedPutRuntimeSettings.mockImplementation(async (request) => {
      savedModel = request.whisper_model;
      return { ui_locale: "en", whisper_model: savedModel, active_connection_id: "local-1",
        asr_provider: "local", connections: [connection] };
    });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"], locale: "en",
      appState: { microphoneStatus: "ready", microphoneSetupPassed: true },
    });
    const speech = await screen.findByTestId("runtime_setup.setup_guide.speech_recognition");
    await waitFor(() => expect(speech).toHaveAttribute("data-status", "setup"));
    fireEvent.change(screen.getByLabelText("Whisper model"), { target: { value: "medium" } });
    expect(screen.getByText(/Practice still uses the saved speech model/)).toHaveTextContent("medium");
    fireEvent.click(screen.getByTestId("runtime_setup.download_model"));
    await waitFor(() => expect(screen.getByTestId("runtime_setup.whisper_status")).toHaveTextContent("/tmp/medium"));
    expect(mockedPostWhisperModelDownload).toHaveBeenCalledWith("medium");
    expect(speech).toHaveAttribute("data-status", "setup");
    expect(screen.queryByRole("button", { name: "Choose a practice task" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("runtime_connection.save_connection"));
    await waitFor(() => expect(mockedPutRuntimeSettings).toHaveBeenCalledWith(expect.objectContaining({ whisper_model: "medium" })));
    await waitFor(() => expect(speech).toHaveAttribute("data-status", "ready"));
    expect(screen.getByRole("button", { name: "Choose a practice task" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Choose a practice task" }));
    fireEvent.change(await screen.findByLabelText("Learner name or nickname"), { target: { value: "maria" } });
    fireEvent.click(screen.getByTestId("setup.recommended_start"));
    expect(screen.getByTestId("setup.runtime_callout")).toHaveTextContent("Ready to record");
    expect(screen.getByRole("button", { name: "Start speaking" })).toBeVisible();
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

    fireEvent.click(await screen.findByRole("button", { name: "Get ready to practice" }));

    expect(await screen.findByTestId("runtime_setup.screen")).toBeVisible();
    expect(screen.getByTestId("runtime_connection.form")).toBeVisible();
    const providerField = screen.getByTestId("runtime_connection.provider");
    expect(within(providerField).getByRole("option", { name: "Ollama local" })).toBeInTheDocument();
    expect(within(providerField).queryByRole("option", { name: "LM Studio local" })).not.toBeInTheDocument();
    expect(within(providerField).queryByRole("option", { name: "OpenRouter" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show cloud and advanced providers" })).toBeVisible();
    const otherRoutes = screen.getByText("Choose another route: cloud or expert setup").closest("details");
    expect(otherRoutes).not.toHaveAttribute("open");
    expect(screen.queryByRole("region", { name: "Cloud services" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Choose another route: cloud or expert setup"));
    expect(otherRoutes).toHaveAttribute("open");
    expect(await screen.findByRole("region", { name: "Cloud services" })).toBeVisible();
    expect(screen.getByTestId("runtime_setup.detect_local_models")).toBeVisible();
    expect(screen.queryByTestId("runtime_connection.api_key")).not.toBeInTheDocument();
    const diagnosticsSummary = screen.getByText("Connection diagnostics");
    expect(diagnosticsSummary).toBeVisible();
    expect(diagnosticsSummary.closest("details")).not.toHaveAttribute("open");

    fireEvent.click(screen.getByTestId("runtime_setup.detect_local_models"));

    await waitFor(() => {
      expect(mockedPostRuntimeSettingsTestConnection).toHaveBeenCalled();
    });
    expect(screen.getByText("Local models found via http://localhost:11434/api/tags: 2.")).toBeVisible();

    const detectedModelField = screen.getByLabelText("Detected local models");
    await waitFor(() => expect(screen.getByTestId("runtime_connection.model")).toHaveValue("llama3.2:3b"));
    expect(
      within(detectedModelField).queryByText("Local model discovery can populate this field from the running service."),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("runtime_connection.test_connection"));
    const healthCheckMessage =
      "Connection to http://localhost:11434/api/tags succeeded. Model llama3.2:3b responded via http://localhost:11434/v1. Preview: ok";
    expect(await screen.findByText(healthCheckMessage)).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Show cloud and advanced providers" }));
    fireEvent.change(screen.getByTestId("runtime_connection.provider"), {
      target: { value: "lmstudio_local" },
    });
    expect(screen.getByTestId("runtime_connection.model")).toHaveValue("");
    expect(screen.getByTestId("runtime_connection.base_url")).toHaveValue("http://localhost:1234/v1");
    expect(screen.queryByLabelText("Detected local models")).not.toBeInTheDocument();
    expect(screen.queryByText("Local models found via http://localhost:11434/api/tags: 2.")).not.toBeInTheDocument();
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

    expect(within(providerField).getByRole("option", { name: "LM Studio local" })).toBeInTheDocument();
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

  it("finishes OpenRouter authorization after the cloud disclosure is collapsed", async () => {
    mockedGetRuntime.mockResolvedValue({
      configured: false, provider: "", model: "", base_url: "",
      requires_api_key: false, has_api_key: false,
    });
    let accountConnected = false;
    mockedGetRuntimeSettings.mockImplementation(async () => ({
      ui_locale: "en", whisper_model: "small", active_connection_id: "",
      asr_provider: "local", connections: accountConnected ? [{
        connection_id: "openrouter-account", provider_key: "openrouter",
        provider_choice: "openrouter", provider_label: "OpenRouter",
        label: "New OpenRouter account", model: "vendor/model:free",
        base_url: "", is_default: false, is_local: false,
        requires_api_key: true, has_api_key: true, secret_state: "present",
        last_test_status: "", last_tested_at: "",
        openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
      }] : [],
    }));
    let finishPoll: ((response: Response) => void) | undefined;
    const pollResponse = new Promise<Response>((resolve) => { finishPoll = resolve; });
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const path = String(url);
      if (path.endsWith("/openrouter/sign-in")) {
        return new Response(JSON.stringify({ attempt_id: "pending-account", authorization_url: "https://openrouter.ai/auth" }));
      }
      if (path.endsWith("/openrouter/attempts/pending-account/open-browser")) return new Response("{}");
      if (path.endsWith("/openrouter/attempts/pending-account")) return pollResponse;
      return new Response(JSON.stringify({ settings: {
        version: 1, asr_provider: "local", asr_connection_id: "", asr_model: "whisper-large-v3",
        openrouter_modes: {}, fallback_connection_id: "", paid_fallback_enabled: false,
        monthly_budget_usd: 5, max_output_tokens: 4096,
      }, spending: { spent_usd: 0, reserved_usd: 0, unresolved_requests: [] } }));
    }));

    const { queryClient } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/runtime-setup"], locale: "en",
    });
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");
    const disclosure = (await screen.findByText("Choose another route: cloud or expert setup")).closest("details");
    expect(disclosure).not.toHaveAttribute("open");
    expect(screen.queryByRole("region", { name: "Cloud services" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("Choose another route: cloud or expert setup"));
    fireEvent.click(await screen.findByRole("button", { name: "Connect OpenRouter" }));
    await waitFor(() => expect(finishPoll).toBeDefined());
    fireEvent.click(screen.getByText("Choose another route: cloud or expert setup"));
    expect(disclosure).not.toHaveAttribute("open");
    expect(disclosure?.querySelector('section[aria-label="Cloud services"]')).toBeInTheDocument();

    accountConnected = true;
    await act(async () => {
      finishPoll?.(new Response(JSON.stringify({ status: "connected" })));
    });
    await waitFor(() => expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["runtime"] }));
    await waitFor(() => expect(mockedGetRuntimeSettings).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByText("Choose another route: cloud or expert setup"));
    expect(await screen.findByText("OpenRouter access: New OpenRouter account")).toBeVisible();
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

    fireEvent.click(await screen.findByRole("button", { name: "Replace saved key" }));

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
