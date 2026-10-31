import "@testing-library/jest-dom/vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { RehearsalRoute } from "../RehearsalRoute";
import { renderWithProviders } from "@/test/renderWithProviders";
import { createRehearsal } from "@/lib/rehearsal/session";
import { getRehearsal, listRehearsals } from "@/lib/rehearsal/storage";
import { apiClient } from "@/lib/api/client";

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
    getHistoryDetail: vi.fn(),
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

it("shows a saved compact coaching summary and next step for a rehearsal part", async () => {
  const session = createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  });
  session.phase = "review";
  session.parts[0] = { ...session.parts[0], recorded: true, reportId: "compact-part" };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
  vi.mocked(apiClient.getHistoryDetail).mockResolvedValue({ payload: { report: {
    session_id: "compact-part", metrics: { duration_sec: 180, wpm: 110 }, scores: { final: 4 },
    rubric: {}, warnings: [], coaching: {
      summary: "Your story has a clear sequence.", fixes: [],
      next_step: "Record a 180 seconds retry with one clear ending.",
      feedback_language: "en", practice_language: "en",
    },
  } } } as Awaited<ReturnType<typeof apiClient.getHistoryDetail>>);
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "en",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  const result = await screen.findByTestId("rehearsal-result-0");
  expect(result).toHaveTextContent("Your story has a clear sequence.");
  expect(result).toHaveTextContent("Record a 180 seconds retry with one clear ending.");
  expect(result).not.toHaveTextContent("Report unavailable");
});

it("bounds legacy rehearsal feedback at word boundaries and offers its full text", async () => {
  const summary = "This legacy summary repeats one useful point for the learner. ".repeat(6).trim();
  const focus = "Connect every example to the main idea before you close. ".repeat(6).trim();
  const session = createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  });
  session.phase = "review";
  session.parts[0] = { ...session.parts[0], recorded: true, reportId: "legacy-part" };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
  vi.mocked(apiClient.getHistoryDetail).mockResolvedValue({ payload: { report: {
    session_id: "legacy-part", metrics: {}, scores: {}, rubric: {}, warnings: [],
    coaching: { coach_summary: summary, next_focus: focus },
  } } } as Awaited<ReturnType<typeof apiClient.getHistoryDetail>>);
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "en",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  const preview = (await screen.findByTestId("rehearsal-coaching-summary-0")).textContent || "";
  expect(preview.length).toBeLessThanOrEqual(160);
  expect(preview.endsWith("…")).toBe(true);
  expect(summary.startsWith(`${preview.slice(0, -1)} `)).toBe(true);
  const disclosure = screen.getByTestId("rehearsal-full-feedback-0");
  expect(disclosure).not.toHaveAttribute("open");
  fireEvent.click(screen.getByText("Show full original feedback"));
  expect(disclosure).toHaveTextContent(summary);
  expect(disclosure).toHaveTextContent(focus);
});
