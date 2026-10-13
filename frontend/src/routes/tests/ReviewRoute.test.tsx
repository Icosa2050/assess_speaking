import "@testing-library/jest-dom/vitest";

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppFrame } from "@/App";
import { createQueryClient, queryKeys } from "@/lib/query/queryClient";
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
    getSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getResumeSharingRoute: vi.fn().mockResolvedValue({ version: 1, available: true, fingerprint: "a".repeat(64), audio: { provider: "local", model: "small", connection_id: "", host: "", local: true, mode: "" }, analysis: { provider: "ollama", model: "test", connection_id: "", host: "localhost", local: true, mode: "" }, fallback: null }),
    getAssessmentStatus: vi.fn(),
    resumeHistory: vi.fn(),
    getDiagnostics: vi.fn(),
    getRuntime: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api/client";

const mockedResumeHistory = vi.mocked(apiClient.resumeHistory);
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
      cohesion_markers: {
        expected: ">=0",
        actual: 0,
        ok: null,
        status: "observed",
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
      content_validity_pass: false,
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
    warnings: ["coaching_unavailable"],
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

const expectDecorativeIcon = (element: HTMLElement) => {
  expect(element.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument();
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

  it("does not turn a three-word attempt into a scored result", async () => {
    const payload = { ...reviewPayload, report: { ...reviewPayload.report,
      rubric: undefined,
      checks: { language_pass: true, topic_pass: null, content_validity_pass: null,
        duration_pass: false, min_words_pass: false },
      warnings: ["llm_skipped_low_word_count"],
    } };
    renderWithProviders(<AppFrame />, { initialEntries: ["/review"], locale: "en", appState: {
      ...validDraftState, review: { reportId: "short", transcript: "bla bla bla", scoreOverall: 3.5,
        band: "B2", summary: "Try again", payload },
    } });
    expect(await screen.findByText(/Not enough speech to assess/)).toBeVisible();
    expect(screen.getByTestId("review-next-step-score")).not.toHaveTextContent("3.5");
    expect(screen.getByTestId("review-metric-score-overall")).not.toHaveTextContent("3.5");
    expect(screen.getByTestId("review-baseline")).not.toHaveTextContent("Pass");
    expect(screen.queryByRole("button", {name: "Resume unfinished analysis"})).not.toBeInTheDocument();
  });

  it.each(["content_unverified", "invalid_content"])("withholds stale saved grades for %s reports", async state => {
    renderWithProviders(<AppFrame />, { initialEntries: ["/review"], locale: "en", appState: {
      ...validDraftState, review: {reportId: "report-1", transcript: "Full transcript text", scoreOverall: 3.5, band: "B2", summary: "Saved report", payload: {
        ...reviewPayload, report: {...reviewPayload.report, rubric: undefined, warnings: [], eligibility: {version: 1, state, metrics_reliable: false, reasons: [], task_complete: null}},
      }},
    }});
    await screen.findByTestId("review-next-step-card");
    expect(screen.getByTestId("review-next-step-score")).not.toHaveTextContent("3.5");
    expect(screen.getByTestId("review-metric-score-overall")).not.toHaveTextContent("3.5");
    expect(screen.queryByRole("button", {name: "Resume unfinished analysis"})).not.toBeInTheDocument();
  });

  it("recovers a lost resume response without creating a different submission", async () => {
    mockedResumeHistory.mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue({assessment_id: "asmt-resume", status: "queued"});
    mockedGetAssessmentStatus.mockResolvedValue({assessment_id:"asmt-resume",status:"running",phase:"scoring_rubric",progress:.5,error:null,payload:null,report_path:null,summary:null});
    const {store} = renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"], locale: "en", appState: {
        ...validDraftState,
        review: {reportId:"report-1",transcript:"Full transcript text",scoreOverall:3.5,band:"B2",summary:"Saved partial report",payload:{...reviewPayload,report:{...reviewPayload.report,rubric:undefined,eligibility:{state:"content_unverified"},warnings:["llm_unavailable"]}}},
      },
    });
    await waitFor(() => expect(screen.getByRole("button", {name:"Resume unfinished analysis"})).toBeEnabled());
    fireEvent.click(screen.getByRole("button", {name:"Resume unfinished analysis"}));
    await waitFor(() => expect(mockedResumeHistory).toHaveBeenCalledTimes(2));
    expect(mockedResumeHistory.mock.calls[0][0]).toBe("report-1");
    expect(mockedResumeHistory.mock.calls[0][1]).toBeTruthy();
    expect(mockedResumeHistory.mock.calls[1]).toEqual(mockedResumeHistory.mock.calls[0]);
    await waitFor(() => expect(store.getState().recording.job.assessmentId).toBe("asmt-resume"));
  });

  it("does not offer unfinished-analysis recovery for a fully successful report", async () => {
    renderWithProviders(<AppFrame />, {initialEntries:["/review"],locale:"en",appState:{...validDraftState,
      review:{reportId:"report-1",transcript:"Full transcript text",scoreOverall:3.5,band:"B2",summary:"Ready",
        payload:{...reviewPayload,report:{...reviewPayload.report,rubric:{overall:4},warnings:[]}}},
    }});
    await screen.findByTestId("review-next-step-card");
    expect(screen.queryByRole("button",{name:"Resume unfinished analysis"})).not.toBeInTheDocument();
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

    const nextStepCard = await screen.findByTestId("review-next-step-card");
    expect(nextStepCard).toBeVisible();
    expect(within(nextStepCard).getByText("Coach note")).toBeVisible();
    expect(within(nextStepCard).getByRole("heading", { name: "Your next step" })).toBeVisible();
    const coachTakeaway = within(nextStepCard).getByTestId("review-coach-takeaway");
    expect(coachTakeaway).toHaveAttribute("data-semantic-id", "review-coach-takeaway");
    expect(coachTakeaway).toHaveTextContent("Focus on connectors.");
    expect(within(nextStepCard).getByText("Choose one thing to work on in your next attempt.")).toBeVisible();
    const scoreVisual = within(nextStepCard).getByTestId("review-next-step-score");
    expect(scoreVisual).toHaveAttribute("data-semantic-id", "review-next-step-score");
    const scoreRing = within(scoreVisual).getByRole("progressbar", { name: "Current result" });
    expect(scoreRing).toHaveAttribute("aria-valuenow", "70");
    expect(scoreRing).toHaveAttribute("aria-valuetext", "B2 · 3.5");
    expect(scoreVisual).toHaveTextContent("B2 · 3.5");
    const focusChip = within(nextStepCard).getByTestId("review-next-step-focus");
    expect(focusChip).toHaveAttribute("data-semantic-id", "review-next-step-focus");
    expect(focusChip).toHaveTextContent(
      "Focus next on: Connect your examples more clearly.",
    );
    expectDecorativeIcon(focusChip);
    const exerciseChip = within(nextStepCard).getByTestId("review-next-step-exercise");
    expect(exerciseChip).toHaveAttribute("data-semantic-id", "review-next-step-exercise");
    expect(exerciseChip).toHaveTextContent(
      "Next exercise: Repeat the task with explicit transitions.",
    );
    expectDecorativeIcon(exerciseChip);
    expect(within(nextStepCard).getByRole("status")).toHaveTextContent(
      "Manual review is recommended before treating this score as final.",
    );
    const tryAgainButton = within(nextStepCard).getByRole("button", { name: "Try again" });
    const changeTaskButton = within(nextStepCard).getByRole("button", { name: "Change task" });
    const historyButton = within(nextStepCard).getByRole("button", { name: "Open history" });
    expect(tryAgainButton).toHaveAttribute("data-semantic-id", "review-action-try-again");
    expect(changeTaskButton).toHaveAttribute("data-semantic-id", "review-action-new-setup");
    expect(historyButton).toHaveAttribute("data-semantic-id", "review-action-view-history");
    expectDecorativeIcon(tryAgainButton);
    expectDecorativeIcon(changeTaskButton);
    expectDecorativeIcon(historyButton);
    expect(screen.getAllByRole("button", { name: "Try again" })).toHaveLength(1);

    expect(await screen.findByTestId("review-summary")).toBeVisible();
    expect(screen.queryByTestId("review-coach-summary")).not.toBeInTheDocument();
    expect(screen.getByTestId("review-requires-human-review")).toBeVisible();
    expect(screen.getByTestId("review-warning-item-0")).toHaveTextContent(
      "AI feedback was unavailable or took too long. General practice tips are shown instead.",
    );
    expect(screen.getByText("Warnings")).toBeVisible();
    expect(screen.getByTestId("review-failed-gates")).toHaveTextContent("Failed quality checks");
    expect(screen.getByText("Result")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Score details" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Feedback summary" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Quality checks" })).toBeVisible();
    expect(screen.getByTestId("review-quality-summary")).toHaveTextContent("3 of 5 checks passed");
    expect(screen.getByTestId("review-gates-disclosure")).toHaveAttribute("open");
    expect(screen.getByTestId("review-validation-toggle")).toHaveTextContent("Show quality check details");
    expect(screen.getByTestId("review-gate-content-validity")).toHaveTextContent("Content validity");
    expect(screen.getByTestId("review-gate-content-validity")).toHaveTextContent("Failed");
    expect(screen.getAllByText("Not assessed")).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "Transcript and reference" })).toBeVisible();
    expect(screen.getByTestId("review-evidence-disclosure")).not.toHaveAttribute("open");
    expect(screen.getByTestId("review-evidence-toggle")).toHaveTextContent(
      "Show transcript and reference details",
    );
    fireEvent.click(screen.getByTestId("review-evidence-toggle"));
    expect(screen.getByTestId("review-evidence-disclosure")).toHaveAttribute("open");
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

  it("shows an actionable empty review state when there is no report to show", async () => {
    renderWithProviders(<AppFrame />, {
      initialEntries: ["/review"],
      locale: "en",
      appState: validDraftState,
    });

    expect(await screen.findByTestId("review-guard-missing-review")).toBeVisible();
    expect(screen.getByText("Nothing to review yet.")).toBeVisible();
    expect(screen.getByText("Complete a speaking session and your results will appear here.")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Start a session" }));

    expect(await screen.findByRole("heading", { name: "Set up today's speaking practice" })).toBeVisible();
  });

  it("restores a completed review from the assessment status endpoint", async () => {
    const queryClient = createQueryClient();
    queryClient.setQueryData(queryKeys.history, { items: [] });
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
      queryClient,
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
    expect(queryClient.getQueryState(queryKeys.history)?.isInvalidated).toBe(true);
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

    expect(await screen.findByRole("heading", { name: "Set up today's speaking practice" })).toBeVisible();
    await waitFor(() => {
      expect(store.getState().draft.themeLabel).toBe("");
      expect(store.getState().draft.promptText).toBe("");
    });
  });
});
