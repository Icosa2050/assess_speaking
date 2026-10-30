import "@testing-library/jest-dom/vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { RehearsalRoute } from "../RehearsalRoute";
import { renderWithProviders } from "@/test/renderWithProviders";
import { createRehearsal } from "@/lib/rehearsal/session";
import { getRehearsal, listRehearsals } from "@/lib/rehearsal/storage";

vi.mock("@/lib/rehearsal/storage", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/rehearsal/storage")>(),
  listRehearsals: vi.fn(),
  getRehearsal: vi.fn(),
}));
vi.mock("@/components/speak/RecorderPanel", () => ({
  RecorderPanel: ({ onFileSelected, onRecordingStarted, onRecordingActiveChange, previewUrl }: {
    onFileSelected: (file: File) => void;
    onRecordingStarted?: () => void;
    onRecordingActiveChange?: (active: boolean) => void;
    previewUrl: string;
  }) => <div>
    <button onClick={() => onFileSelected(new File(["audio"], "first.webm", { type: "audio/webm" }))}>Finish first take</button>
    <button onClick={() => { onRecordingStarted?.(); onRecordingActiveChange?.(true); }}>Start replacement</button>
    <button onClick={() => undefined}>Permission denied</button>
    <span data-testid="rehearsal-preview-url">{previewUrl}</span>
  </div>,
}));
vi.mock("@/lib/api/client", () => ({
  ApiClientError: class extends Error {},
  apiClient: {
    getRuntime: vi.fn().mockResolvedValue({ configured: true, provider: "ollama_local", model: "test", base_url: "http://localhost:11434" }),
    getRuntimeSettings: vi.fn().mockResolvedValue({ whisper_model: "small" }),
    getSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64),
      audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" },
      analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  const session = { ...createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  }), phase: "speaking" as const };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
});

it("clears an old rehearsal part and disables Save part when re-recording begins", async () => {
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "en",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Finish first take" }));
  await waitFor(() => expect(screen.getByTestId("rehearsal-save-part")).toBeEnabled());
  expect(screen.getByTestId("rehearsal-preview-url")).not.toBeEmptyDOMElement();
  fireEvent.click(screen.getByRole("button", { name: "Permission denied" }));
  expect(screen.getByTestId("rehearsal-save-part")).toBeEnabled();
  expect(screen.getByTestId("rehearsal-preview-url")).not.toBeEmptyDOMElement();
  fireEvent.click(screen.getByRole("button", { name: "Start replacement" }));
  expect(screen.getByTestId("rehearsal-save-part")).toBeDisabled();
  await waitFor(() => expect(screen.getByTestId("rehearsal-preview-url")).toBeEmptyDOMElement());
});
