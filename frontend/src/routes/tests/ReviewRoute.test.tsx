import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppFrame } from "@/App";
import { renderWithProviders } from "@/test/renderWithProviders";

vi.mock("@/lib/api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    code = "runtime_error";
    detail: string;
    responseStatus = 500;

    constructor(responseStatus: number, error: { detail: string }) {
      super(error.detail);
      this.detail = error.detail;
      this.responseStatus = responseStatus;
    }
  },
  apiClient: {
    getAssessmentStatus: vi.fn(),
    getDiagnostics: vi.fn(),
    getRuntime: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";

const mockedGetAssessmentStatus = vi.mocked(apiClient.getAssessmentStatus);
const mockedGetDiagnostics = vi.mocked(apiClient.getDiagnostics);
const mockedGetRuntime = vi.mocked(apiClient.getRuntime);

const reviewPayload = {
  meta: {
    label: "Morning run",
    learning_language: "en",
  },
  notes: "Mention two concrete examples.",
  transcript_full: "Full transcript text",
  baseline_comparison: {
    level: "B2",
    targets: {
      fluency: {
        expected: "Sustains clear speech",
        actual: "Mostly sustained",
        ok: true,
      },
    },
  },
  report: {
    session_id: "report-1",
    transcript_preview: "Transcript preview",
    input: {
      expected_language: "en",
    },
    scores: {
      final: 3.5,
      band: "B2",
      mode: "hybrid",
      llm: 3.7,
      deterministic: 3.2,
    },
    checks: {
      language_pass: true,
      topic_pass: false,
      duration_pass: true,
      min_words_pass: true,
    },
    coaching: {
      coach_summary: "Focus on connectors.",
      strengths: ["Good range of vocabulary."],
      top_3_priorities: ["Link ideas more clearly."],
      next_focus: "Connect your examples more clearly.",
      next_exercise: "Repeat the task with explicit transitions.",
    },
    warnings: ["llm_skipped_low_word_count"],
    requires_human_review: true,
    rubric: {
      recurring_grammar_errors: [{ type: "verb_tense" }],
      coherence_issues: [{ type: "weak_closing" }],
    },
    progress_delta: {
      items: [{ kind: "delta_final", value: 0.5 }],
    },
  },
};

const validDraftState = {
  draft: {
    speakerId: "bern",
    learningLanguage: "en",
    learningLanguageLabel: "English",
    cefrLevel: "B2" as const,
    themeId: "work-home-b2",
    themeLabel: "The pros and cons of working from home",
    taskFamily: "opinion_monologue" as const,
    durationSec: 120 as const,
    promptId: "work-home-b2",
    promptText: "Give your opinion on working from home.",
  },
  preferences: {
    activeConnectionId: "conn-primary",
    setupComplete: true,
  },
};

describe("Review route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetDiagnostics.mockResolvedValue({ items: [] });
    mockedGetRuntime.mockResolvedValue({
      configured: true,
      provider: "ollama_local",
      model: "llama3.2:3b",
      base_url: "http://localhost:11434",
      requires_api_key: false,
      has_api_key: false,
    });
    mockedGetAssessmentStatus.mockResolvedValue({
      assessment_id: "asmt-1",
      status: "running",
      phase: "transcribing",
      progress: 0.4,
      error: null,
      payload: null,
      report_path: null,
      summary: null,
    });
  });

  it("renders an existing review summary with warnings and evidence", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: {
        ...validDraftState,
        review: {
          reportId: "report-1",
          transcript: "Full transcript text",
          scoreOverall: 3.5,
          band: "B2",
          summary: "Focus on connectors.",
          payload: reviewPayload,
        },
      },
    });

    expect(await screen.findByTestId("review-summary")).toBeVisible();
    expect(screen.getByText("Focus on connectors.")).toBeVisible();
    expect(screen.getByTestId("review-requires-human-review")).toBeVisible();
    expect(screen.getByTestId("review-warning-item-0")).toHaveTextContent(
      "AI scoring was skipped because the response was too short.",
    );
    expect(screen.getByDisplayValue("Full transcript text")).toBeVisible();
  });

  it("shows the still-assessing guard when the recording job is still running", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: {
        ...validDraftState,
        recording: {
          status: "assessing",
          assessmentState: "running",
          job: {
            assessmentId: "asmt-1",
            status: "running",
            phase: "transcribing",
            progress: 0.4,
            error: "",
            reportPath: "",
          },
        },
      },
    });

    expect(await screen.findByTestId("review-guard-still-assessing")).toBeVisible();
    expect(screen.getByRole("button", { name: "Back to Speak" })).toBeVisible();
  });

  it("shows the missing-review guard when there is no report to show", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("review-guard-missing-review")).toBeVisible();
    expect(screen.getByRole("button", { name: "Go to Speak" })).toBeVisible();
  });

  it("restores a completed review from the assessment status endpoint", async () => {
    mockedGetAssessmentStatus.mockResolvedValueOnce({
      assessment_id: "asmt-1",
      status: "completed",
      phase: "finalizing_report",
      progress: 1,
      error: null,
      payload: reviewPayload,
      report_path: "/tmp/report-1.json",
      summary: {
        score_overall: 3.5,
        band: "B2",
        next_focus: "Connect your examples more clearly.",
      },
    });

    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: {
        ...validDraftState,
        recording: {
          status: "submitted",
          assessmentState: "completed",
          job: {
            assessmentId: "asmt-1",
            status: "completed",
            phase: "finalizing_report",
            progress: 1,
            error: "",
            reportPath: "",
          },
        },
      },
    });

    expect(await screen.findByTestId("review-summary")).toBeVisible();
    expect(screen.getByText("Focus on connectors.")).toBeVisible();
    expect(mockedGetAssessmentStatus).toHaveBeenCalledTimes(1);
    expect(store.getState().review.reportId).toBe("report-1");
    expect(store.getState().review.transcript).toBe("Full transcript text");
  });

  it("keeps setup details when the learner chooses try again", async () => {
    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: {
        ...validDraftState,
        review: {
          reportId: "report-1",
          transcript: "Full transcript text",
          scoreOverall: 3.5,
          band: "B2",
          summary: "Focus on connectors.",
          payload: reviewPayload,
        },
      },
    });

    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));

    expect(await screen.findByTestId("speak.status_panel")).toBeVisible();
    expect(store.getState().review.reportId).toBe("");
    expect(store.getState().draft.themeLabel).toBe("The pros and cons of working from home");
  });

  it("clears task-specific setup when the learner chooses change task", async () => {
    const { store } = renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: {
        ...validDraftState,
        review: {
          reportId: "report-1",
          transcript: "Full transcript text",
          scoreOverall: 3.5,
          band: "B2",
          summary: "Focus on connectors.",
          payload: reviewPayload,
        },
      },
    });

    fireEvent.click(await screen.findByRole("button", { name: "Change task" }));

    expect(await screen.findByRole("heading", { name: "Prepare one speaking session" })).toBeVisible();
    await waitFor(() => {
      expect(store.getState().draft.themeLabel).toBe("");
      expect(store.getState().draft.promptText).toBe("");
    });
  });
});
