import "@testing-library/jest-dom/vitest";

import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { createQueryClient, queryKeys } from "@/lib/query/queryClient";

import { AppFrame } from "@/App";
import { createTranslator } from "@/lib/i18n";
import { formatTrendSummary } from "@/routes/HistoryRoute";
import { renderWithProviders } from "@/test/renderWithProviders";

vi.mock("@/lib/api/client", () => ({
  apiClient: {
    getHistory: vi.fn(),
    getHistoryDetail: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";

const mockedGetHistory = vi.mocked(apiClient.getHistory);
const mockedGetHistoryDetail = vi.mocked(apiClient.getHistoryDetail);

const validDraftState = {
  draft: {
    speakerId: "bern",
    learningLanguage: "it",
    learningLanguageLabel: "Italian",
    cefrLevel: "B2" as const,
    themeId: "trip-b2",
    themeLabel: "Holiday trip",
    taskFamily: "travel_narrative" as const,
    durationSec: 120 as const,
    promptId: "trip-b2",
    promptText: "Tell the story of a holiday trip.",
  },
  preferences: {
    activeConnectionId: "conn-primary",
    setupComplete: true,
  },
};

const historyRows = [
  {
    timestamp: "2026-04-20T09:00:00Z",
    session_id: "s1",
    speaker_id: "anna",
    learning_language: "it",
    theme: "Weekend trip",
    task_family: "travel_narrative",
    overall: 3.3,
    wpm: 95.9,
    report_path: "/tmp/s1.json",
    requires_human_review: false,
    duration_pass: true,
    topic_pass: true,
    language_pass: true,
    min_words_pass: true,
    top_priorities: ["More detail"],
    grammar_error_categories: ["preposition_choice"],
    coherence_issue_categories: ["missing_sequence_markers"],
    final_score: 3.6,
    band: "4",
  },
  {
    timestamp: "2026-04-21T09:00:00Z",
    session_id: "s2",
    speaker_id: "bern",
    learning_language: "it",
    theme: "Spring trip",
    task_family: "travel_narrative",
    overall: 3.7,
    wpm: 110.2,
    report_path: "/tmp/s2.json",
    requires_human_review: false,
    duration_pass: true,
    topic_pass: true,
    language_pass: true,
    min_words_pass: true,
    top_priorities: ["More detail", "Clearer ending"],
    grammar_error_categories: ["preposition_choice"],
    coherence_issue_categories: ["missing_sequence_markers"],
    final_score: 4.0,
    band: "4",
  },
  {
    timestamp: "2026-04-22T09:00:00Z",
    session_id: "s3",
    speaker_id: "bern",
    learning_language: "en",
    theme: "Remote work",
    task_family: "opinion_monologue",
    overall: 3.2,
    wpm: 118.0,
    report_path: "/tmp/s3.json",
    requires_human_review: true,
    duration_pass: true,
    topic_pass: false,
    language_pass: true,
    min_words_pass: true,
    top_priorities: ["Stay on topic"],
    grammar_error_categories: ["verb_tense"],
    coherence_issue_categories: ["weak_closing"],
    final_score: 3.7,
    band: "4",
  },
  {
    timestamp: "2026-04-23T09:00:00Z",
    session_id: "s4",
    speaker_id: "bern",
    learning_language: "it",
    theme: "Holiday return",
    task_family: "travel_narrative",
    overall: 4.0,
    wpm: 132.0,
    report_path: "/tmp/s4.json",
    requires_human_review: false,
    duration_pass: true,
    topic_pass: true,
    language_pass: true,
    min_words_pass: true,
    top_priorities: ["More precision"],
    grammar_error_categories: ["article_choice"],
    coherence_issue_categories: ["weak_closing"],
    final_score: 4.2,
    band: "5",
  },
];

const detailPayload = (sessionId: string, options: { review?: boolean; topicPass?: boolean } = {}) => ({
  meta: {
    label: `Label ${sessionId}`,
    learning_language: sessionId === "s3" ? "en" : "it",
  },
  notes: `Notes for ${sessionId}`,
  transcript_full: `Transcript for ${sessionId}`,
  report: {
    session_id: sessionId,
    input: {
      expected_language: sessionId === "s3" ? "en" : "it",
    },
    scores: {
      final: sessionId === "s4" ? 4.2 : sessionId === "s2" ? 4.0 : 3.7,
      band: sessionId === "s4" ? "5" : "4",
      mode: "hybrid",
      llm: sessionId === "s4" ? 4.1 : 3.8,
      deterministic: sessionId === "s4" ? 4.0 : 3.7,
    },
    checks: {
      language_pass: true,
      topic_pass: options.topicPass ?? true,
      duration_pass: true,
      min_words_pass: true,
    },
    coaching: {
      coach_summary: `Coach summary for ${sessionId}`,
      strengths: ["Strong opening"],
      top_3_priorities: ["Add one more concrete detail"],
      next_focus: "Sharpen the structure.",
      next_exercise: "Repeat the story in two minutes.",
    },
    warnings: options.review ? ["llm_invalid_schema"] : [],
    requires_human_review: Boolean(options.review),
    rubric: {
      recurring_grammar_errors: [{ type: "verb_tense" }],
      coherence_issues: [{ type: "weak_closing" }],
    },
    progress_delta: {
      score_delta: {
        final: 0.2,
        overall: 0.1,
        wpm: 4.2,
      },
      previous_session_id: "prev-1",
      new_priorities: ["Add detail"],
    },
  },
});

describe("History route", () => {
  it("formats score and pace trend summaries for the UI locale", () => {
    const translate = createTranslator("de");

    expect(formatTrendSummary([4, 4.2], "de", translate)).toBe("4,0 bis 4,2");
    expect(formatTrendSummary([110.2, 132], "de", translate)).toBe("110,2 bis 132,0");
    expect(formatTrendSummary([], "de", translate)).toBe("");
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetHistory.mockResolvedValue({ items: historyRows });
    mockedGetHistoryDetail.mockImplementation(async (sessionId: string) => ({
      payload: detailPayload(sessionId, {
        review: sessionId === "s3",
        topicPass: sessionId !== "s3",
      }),
    }));
  });

  it("shows the localized empty state when there are no saved rows", async () => {
    mockedGetHistory.mockResolvedValueOnce({ items: [] });

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("history-empty")).toBeVisible();
    expect(screen.getByText("Your practice history starts here.")).toBeVisible();
    expect(screen.getByText("Each completed speaking session adds a saved result to this page.")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Start your first session" }));

    expect(await screen.findByRole("heading", { name: "Set up today's speaking practice" })).toBeVisible();
  });

  it("shows all languages on refresh and preserves an explicit filter", async () => {
    const queryClient = createQueryClient();
    queryClient.setQueryData(queryKeys.history, { items: historyRows.filter(row => row.learning_language === "en") });
    await queryClient.invalidateQueries({ queryKey: queryKeys.history });
    let finish!: (value: { items: typeof historyRows }) => void;
    mockedGetHistory.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"], locale: "en", appState: validDraftState, queryClient,
    });
    await waitFor(() => expect(mockedGetHistory).toHaveBeenCalled());
    expect(mockedGetHistoryDetail).not.toHaveBeenCalled();
    await act(async () => finish({ items: historyRows }));
    await waitFor(() => expect(mockedGetHistoryDetail).toHaveBeenCalledWith("s4"));
    await waitFor(() => expect(screen.getByTestId("history-language-filter")).toHaveValue("__all__"));
    // Explicit filtering must survive subsequent refreshes.
    fireEvent.change(screen.getByTestId("history-language-filter"), { target: { value: "it" } });
    await act(async () => { await queryClient.invalidateQueries({ queryKey: queryKeys.history }); });
    expect(screen.getByTestId("history-language-filter")).toHaveValue("it");
  });

  it("shows every saved review first and allows explicit learner and language filters", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    await screen.findByTestId("history-attempts-list");
    expect(screen.getByTestId("history-language-filter")).toHaveValue("__all__");
    expect(screen.getByTestId("history-speaker-filter")).toHaveValue("");
    expect(within(screen.getByTestId("history-attempts-list")).getAllByRole("listitem")).toHaveLength(4);
    expect(screen.getByTestId("history-attempt-card-s1")).toHaveTextContent("anna");
    expect(screen.getByTestId("history-attempt-card-s3")).toHaveTextContent("English");
    fireEvent.change(screen.getByTestId("history-speaker-filter"), { target: { value: "bern" } });
    fireEvent.change(screen.getByTestId("history-language-filter"), { target: { value: "it" } });
    expect(screen.getByTestId("history-scope-caption")).toHaveTextContent(
      "Saved attempts for bern in Italian: 2.",
    );
    expect(screen.getByTestId("practice-progress")).toHaveTextContent("Attempts: 2");
    expect(screen.getByTestId("practice-legacy")).toBeVisible();
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("No earlier comparable attempt");
    expect(screen.queryByRole("img", { name: "Overall performance" })).not.toBeInTheDocument();
    expect(screen.getByTestId("history-attempt-card-s4")).toBeVisible();
    expect(screen.queryByTestId("history-attempt-card-s3")).not.toBeInTheDocument();
    expect(screen.getByTestId("history-attempts-list")).toBeVisible();
    expect(screen.getByTestId("history-attempt-card-s4")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("history-attempt-card-s4")).toHaveTextContent("Holiday return");
    expect(screen.getByTestId("history-attempt-card-s4")).toHaveTextContent("Score 4.2");
    expect(screen.getByTestId("history-attempt-card-s4")).toHaveTextContent("Done");
    expect(screen.getByTestId("history-attempt-card-s2")).toHaveTextContent("Spring trip");
  });

  it("keeps filters available when nothing matches and can restore the complete list", async () => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/history"], locale: "en", appState: validDraftState });
    await screen.findByTestId("history-attempts-list");
    fireEvent.change(screen.getByTestId("history-speaker-filter"), { target: { value: "anna" } });
    fireEvent.change(screen.getByTestId("history-language-filter"), { target: { value: "en" } });
    expect(screen.getByTestId("history-empty-filtered")).toBeVisible();
    expect(screen.getByTestId("history-speaker-filter")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(within(screen.getByTestId("history-attempts-list")).getAllByRole("listitem")).toHaveLength(4);
    expect(await screen.findByTestId("history-detail-panel")).toBeVisible();
  });

  it("does not pull other learners or languages into the journal through blank legacy IDs", async () => {
    mockedGetHistory.mockResolvedValueOnce({ items: [
      { ...historyRows[0], session_id: "", speaker_id: "bern", learning_language: "it", report_path: "" },
      { ...historyRows[0], session_id: "", speaker_id: "anna", learning_language: "it", report_path: "" },
      { ...historyRows[0], session_id: "", speaker_id: "bern", learning_language: "en", report_path: "" },
    ] });
    renderWithProviders(<AppFrame />, { initialEntries: ["/history"], locale: "en", appState: validDraftState });
    await screen.findByTestId("history-attempts-list");
    fireEvent.change(screen.getByTestId("history-speaker-filter"), { target: { value: "bern" } });
    fireEvent.change(screen.getByTestId("history-language-filter"), { target: { value: "it" } });
    expect(screen.getByTestId("practice-progress")).toHaveTextContent("Attempts: 1");
    expect(mockedGetHistoryDetail).not.toHaveBeenCalled();
  });

  it("explains an unavailable retry parent without calling it a first attempt", async () => {
    mockedGetHistory.mockResolvedValueOnce({ items: [{ ...historyRows[3], practice: {
      version: 1, goal: "B2", prompt_id: "travel", prompt_text: "Describe your trip", retry_of_session_id: "missing",
      target_duration_sec: 120, scoring_version: "scorer-v1", scoring_mode: "hybrid", analysis_signature: "v1",
      provider: "ollama", model: "model", asr_provider: "faster_whisper", whisper_model: "small", dry_run: false,
    } }] });
    renderWithProviders(<AppFrame />, { initialEntries: ["/history"], locale: "en", appState: validDraftState });
    await waitFor(() => expect(screen.getByTestId("practice-comparison")).toHaveTextContent("This retry has a saved parent"));
    expect(screen.getByTestId("practice-comparison")).not.toHaveTextContent("No earlier comparable attempt");
    expect(within(screen.getByRole("img", { name: "Overall performance" })).getByText("5", { exact: true })).toBeInTheDocument();
  });

  it("opens the most recent saved detail and lets the learner jump to another attempt", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("history-detail-panel")).toBeVisible();
    expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s4");
    expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Attempt summary");
    expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("4.2");
    expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Coach summary for s4");
    expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Strong opening");
    expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Add one more concrete detail");
    expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Sharpen the structure.");
    expect(screen.queryByTestId("review-warnings")).not.toBeInTheDocument();
    const fullReport = screen.getByTestId("history-detail-full-report");
    expect(fullReport).not.toHaveAttribute("open");
    expect(screen.getByTestId("history-detail-full-report-summary")).toHaveTextContent("Show full breakdown");

    fireEvent.click(screen.getByTestId("history-detail-full-report-summary"));

    expect(fullReport).toHaveAttribute("open");
    expect(screen.getByTestId("review-summary")).toBeVisible();

    fireEvent.click(screen.getByTestId("history-attempt-card-s2"));

    await waitFor(() => {
      expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s2");
      expect(screen.getByTestId("history-attempt-card-s2")).toHaveAttribute("aria-pressed", "true");
      expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Coach summary for s2");
    });

    fireEvent.click(screen.getByTestId("history-attempt-card-s4"));

    await waitFor(() => {
      expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s4");
      expect(screen.getByTestId("history-attempt-card-s4")).toHaveAttribute("aria-pressed", "true");
    });

    fireEvent.click(screen.getByTestId("history-attempt-card-s2"));

    await waitFor(() => {
      expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s2");
      expect(screen.getByTestId("history-detail-digest")).toHaveTextContent("Coach summary for s2");
    });
  });

  it("resets the opened detail when the learner switches to another filtered language", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("history-detail-digest-coach-summary")).toHaveTextContent(
      "Coach summary for s4",
    );

    fireEvent.change(screen.getByTestId("history-language-filter"), {
      target: { value: "en" },
    });

    await waitFor(() => {
      expect(screen.getByTestId("practice-progress")).toHaveTextContent("Attempts: 1");
      expect(screen.getByTestId("practice-progress")).toHaveTextContent("Stay on topic");
    });
    await waitFor(() => {
      expect(screen.getByTestId("history-detail-digest-coach-summary")).toHaveTextContent(
        "Coach summary for s3",
      );
    });
    expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s3");
    expect(screen.getByTestId("review-requires-human-review")).toBeVisible();
  });

  it("shows the localized detail error when a saved report cannot be loaded", async () => {
    mockedGetHistoryDetail.mockRejectedValue(
      Object.assign(new Error("missing"), {
        responseStatus: 404,
      }),
    );

    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("history-detail-error")).toBeVisible();
    expect(screen.getByText("The saved report for this attempt could not be loaded.")).toBeVisible();
  });
});
