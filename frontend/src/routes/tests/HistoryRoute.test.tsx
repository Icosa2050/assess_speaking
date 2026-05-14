import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppFrame } from "@/App";
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
    expect(screen.getByText("No history rows are available yet.")).toBeVisible();
  });

  it("defaults the language filter to the current draft language and scopes to the current speaker", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("history-language-filter")).toHaveValue("it");
    expect(screen.getByTestId("history-scope-caption")).toHaveTextContent(
      "Showing 2 saved runs for speaker bern in Italian.",
    );
    expect(screen.getByTestId("history-metric-runs")).toHaveTextContent("2");
    expect(screen.getByTestId("history-chart-score")).toBeVisible();
    expect(screen.getByTestId("history-attempts-row-s4")).toBeVisible();
    expect(screen.queryByTestId("history-attempts-row-s3")).not.toBeInTheDocument();
  });

  it("opens the most recent saved detail and lets the learner jump to another attempt", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("history-detail-panel")).toBeVisible();
    expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s4");
    expect(screen.getByText("Coach summary for s4")).toBeVisible();

    fireEvent.click(screen.getByTestId("history-jump-1"));

    await waitFor(() => {
      expect(screen.getByTestId("history-detail-caption")).toHaveTextContent("Showing saved report s2");
      expect(screen.getByText("Coach summary for s2")).toBeVisible();
    });
  });

  it("resets the opened detail when the learner switches to another filtered language", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/history"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByText("Coach summary for s4")).toBeVisible();

    fireEvent.change(screen.getByTestId("history-language-filter"), {
      target: { value: "en" },
    });

    expect(await screen.findByText("Coach summary for s3")).toBeVisible();
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
