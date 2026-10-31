import "@testing-library/jest-dom/vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { RehearsalRoute } from "../RehearsalRoute";
import { renderWithProviders } from "@/test/renderWithProviders";
import { createRehearsal } from "@/lib/rehearsal/session";
import { getRehearsal, listRehearsals } from "@/lib/rehearsal/storage";
import { apiClient } from "@/lib/api/client";

const assessableChecks = { language_pass: true, topic_pass: true, content_validity_pass: true,
  duration_pass: true, min_words_pass: true };
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((onResolve, onReject) => { resolve = onResolve; reject = onReject; });
  return { promise, resolve, reject };
};

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

it("shows neutral loading text before a saved compact report arrives", async () => {
  const session = createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  });
  session.phase = "review";
  session.parts[0] = { ...session.parts[0], recorded: true, reportId: "compact-part" };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
  const detail = deferred<Awaited<ReturnType<typeof apiClient.getHistoryDetail>>>();
  vi.mocked(apiClient.getHistoryDetail).mockReturnValue(detail.promise);
  const payload = { payload: { report: {
    session_id: "compact-part", metrics: { duration_sec: 180, wpm: 110 }, scores: { final: 4 },
    checks: assessableChecks, eligibility: { state: "assessable" },
    rubric: {}, warnings: [], coaching: {
      summary: "Your story has a clear sequence.",
      fixes: [{ you_said: "I goes to Rome", better: "I go to Rome.", why: "Use the base verb with I." }],
      next_step: "Record a 180 seconds retry with one clear ending.",
      feedback_language: "en", practice_language: "en",
    },
  } } } as Awaited<ReturnType<typeof apiClient.getHistoryDetail>>;
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "it",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  const result = await screen.findByTestId("rehearsal-result-0");
  const loading = screen.getByTestId("rehearsal-report-loading-0");
  expect(loading).toHaveTextContent("Caricamento del feedback salvato");
  expect(loading).not.toHaveAttribute("role");
  expect(result.querySelector('[role="status"]')).toBeNull();
  expect(screen.queryAllByRole("status").filter(node => node.textContent?.includes("feedback salvato"))).toHaveLength(0);
  expect(screen.queryByTestId("rehearsal-report-missing-0")).not.toBeInTheDocument();
  await act(async () => { detail.resolve(payload); });
  await waitFor(() => expect(screen.queryByTestId("rehearsal-report-loading-0")).not.toBeInTheDocument());
  expect(result).toHaveTextContent("Your story has a clear sequence.");
  expect(result).toHaveTextContent("Record a 180 seconds retry with one clear ending.");
  expect(screen.getByTestId("rehearsal-coaching-summary-0")).toHaveAttribute("lang", "en");
  expect(screen.getByTestId("rehearsal-next-step-0")).toHaveAttribute("lang", "en");
  const fixes = screen.getByTestId("rehearsal-compact-fixes-0");
  expect(fixes.querySelector("h4")).toBeInTheDocument();
  expect(fixes).toHaveTextContent("I goes to Rome");
  expect(fixes).toHaveTextContent("I go to Rome.");
  expect(fixes).toHaveTextContent("Use the base verb with I.");
  expect(screen.getByText("I goes to Rome")).toHaveAttribute("lang", "en");
  expect(result).not.toHaveTextContent("Report unavailable");
});

it("withholds retained compact corrections when a rehearsal part is ineligible", async () => {
  const session = createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  });
  session.phase = "review";
  session.parts[0] = { ...session.parts[0], recorded: true, reportId: "short-part" };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
  vi.mocked(apiClient.getHistoryDetail).mockResolvedValue({ payload: { report: {
    session_id: "short-part", metrics: { duration_sec: 20, wpm: 110 }, scores: { final: 4 },
    checks: assessableChecks, eligibility: { state: "insufficient_speech" },
    rubric: {}, warnings: [], coaching: {
      summary: "Your story has a clear sequence.",
      fixes: [{ you_said: "I goes to Rome", better: "I go to Rome.", why: "Use the base verb with I." }],
      next_step: "Record a 180 seconds retry with one clear ending.",
      feedback_language: "en", practice_language: "en",
    },
  } } } as Awaited<ReturnType<typeof apiClient.getHistoryDetail>>);
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "en",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  const result = await screen.findByTestId("rehearsal-result-0");
  expect(screen.getByTestId("rehearsal-ineligible-0")).toBeVisible();
  expect(result).toHaveTextContent("Not enough speech");
  expect(result).not.toHaveTextContent("Your story has a clear sequence.");
  expect(result).not.toHaveTextContent("I goes to Rome");
  expect(result).not.toHaveTextContent("Record a 180 seconds retry");
  expect(screen.queryByTestId("rehearsal-compact-fixes-0")).not.toBeInTheDocument();
  expect(screen.queryByTestId("rehearsal-coaching-summary-0")).not.toBeInTheDocument();
  expect(result.querySelectorAll("dd")[2]).toHaveTextContent("—");
});

it("shows unavailable feedback only after a saved report fetch fails", async () => {
  const session = createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  });
  session.phase = "review";
  session.parts[0] = { ...session.parts[0], recorded: true, reportId: "missing-part" };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
  const detail = deferred<Awaited<ReturnType<typeof apiClient.getHistoryDetail>>>();
  vi.mocked(apiClient.getHistoryDetail).mockReturnValue(detail.promise);
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "en",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  const result = await screen.findByTestId("rehearsal-result-0");
  expect(screen.getByTestId("rehearsal-report-loading-0")).toBeVisible();
  expect(result.querySelector('[role="status"]')).toBeNull();
  expect(screen.queryAllByRole("status").filter(node => node.textContent?.includes("Saved feedback"))).toHaveLength(0);
  expect(screen.queryByTestId("rehearsal-report-missing-0")).not.toBeInTheDocument();
  await act(async () => { detail.reject(new Error("History detail unavailable")); });
  await waitFor(() => expect(screen.queryByTestId("rehearsal-report-loading-0")).not.toBeInTheDocument());
  expect(screen.getByTestId("rehearsal-report-missing-0")).toBeVisible();
  expect(result).toHaveTextContent("Saved feedback is unavailable");
  expect(screen.queryByTestId("rehearsal-ineligible-0")).not.toBeInTheDocument();
  expect(result).not.toHaveTextContent("could not be verified");
});

it("settles each saved part independently", async () => {
  const session = createRehearsal("en", "B1", "Alex", {
    provider: "ollama_local", model: "test", baseUrl: "http://localhost:11434", whisper: "small", feedbackLanguage: "en",
  });
  session.phase = "review";
  session.parts[0] = { ...session.parts[0], recorded: true, reportId: "first-part" };
  session.parts[1] = { ...session.parts[1], recorded: true, reportId: "second-part" };
  vi.mocked(listRehearsals).mockResolvedValue([session]);
  vi.mocked(getRehearsal).mockResolvedValue(session);
  const first = deferred<Awaited<ReturnType<typeof apiClient.getHistoryDetail>>>();
  const second = deferred<Awaited<ReturnType<typeof apiClient.getHistoryDetail>>>();
  vi.mocked(apiClient.getHistoryDetail).mockImplementation(id => id === "first-part" ? first.promise : second.promise);
  renderWithProviders(<RehearsalRoute />, { initialEntries: ["/rehearsal"], locale: "en",
    appState: { draft: { speakerId: "Alex" }, microphoneSetupPassed: true } });
  fireEvent.click(await screen.findByRole("button", { name: /EN.*Alex/ }));
  await screen.findByTestId("rehearsal-report-loading-0");
  expect(screen.getByTestId("rehearsal-report-loading-1")).toBeVisible();
  await act(async () => { first.resolve({ payload: { report: {
    session_id: "first-part", metrics: {}, scores: {}, checks: assessableChecks,
    eligibility: { state: "insufficient_speech" }, rubric: {}, warnings: [], coaching: {},
  } } } as Awaited<ReturnType<typeof apiClient.getHistoryDetail>>); });
  expect(screen.queryByTestId("rehearsal-report-loading-0")).not.toBeInTheDocument();
  expect(screen.getByTestId("rehearsal-ineligible-0")).toBeVisible();
  expect(screen.getByTestId("rehearsal-report-loading-1")).toBeVisible();
  expect(screen.queryByTestId("rehearsal-report-missing-1")).not.toBeInTheDocument();
  await act(async () => { second.reject(new Error("History detail unavailable")); });
  expect(screen.getByTestId("rehearsal-report-missing-1")).toBeVisible();
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
    session_id: "legacy-part", metrics: { duration_sec: 180 }, scores: {},
    checks: assessableChecks, eligibility: { state: "assessable" }, rubric: {}, warnings: [],
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
