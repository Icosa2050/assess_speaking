import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createTranslator } from "@/lib/i18n";
import { ReviewSummary, selectReviewSummary } from "./ReviewSummary";

const summarize = (report: Record<string, unknown>) => selectReviewSummary({
  band: "B2", reportId: "saved", scoreOverall: 3, summary: "", transcript: "Actual words",
  payload: { report },
});

describe("Review quality safeguards", () => {
  it("keeps legacy comparison evidence but hides its unverified deltas", () => {
    const report = { progress_delta: { previous_session_id: "older", score_delta: { final: 1 } } };
    const summary = summarize(report);
    expect(summary.progressItems).toEqual([]);
    expect(summary.payload.report).toBe(report);
  });

  it("displays only explicitly verified comparisons", () => {
    const summary = summarize({ progress_delta: {
      comparison_verified: true, previous_session_id: "older", score_delta: { final: 1 },
    } });
    expect(summary.progressItems).toContainEqual({ kind: "delta_final", value: 1 });
  });

  it("labels fallback coaching and uncertain transcription at the coaching section", () => {
    const summary = summarize({ warnings: ["coaching_unavailable", "transcript_uncertain"],
      coaching: { coach_summary: "General tips", next_exercise: "Try a 30-second drill",
        next_attempt_instruction: "Repeat the full task for 90 seconds" },
    });
    render(<ReviewSummary summary={summary} translate={createTranslator("en")} />);
    expect(screen.getByTestId("review-general-practice-tips")).toHaveTextContent("General practice tips");
    expect(screen.getByTestId("review-transcript-uncertain")).toHaveTextContent("Listen to the recording");
    expect(screen.getByTestId("review-next-attempt-instruction")).toHaveTextContent("90 seconds");
    expect(screen.getByTestId("review-next-exercise")).toHaveTextContent("30-second drill");
  });

  it("preserves historical coaching without new optional instruction fields", () => {
    render(<ReviewSummary summary={summarize({ coaching: { coach_summary: "Saved feedback" } })}
      translate={createTranslator("en")} />);
    expect(screen.getByTestId("review-coach-summary")).toHaveTextContent("Saved feedback");
    expect(screen.queryByTestId("review-next-attempt-instruction")).not.toBeInTheDocument();
    expect(screen.queryByTestId("review-general-practice-tips")).not.toBeInTheDocument();
  });
});

describe("Optional style in saved feedback", () => {
  it.each(["en", "it"] as const)("keeps optional alternatives separate from grammar and priorities in %s", (locale) => {
    const report = { rubric: { recurring_grammar_errors: [], style_suggestions: [{
      original: "The trip was good", suggestion: "The trip was enjoyable", explanation: "Optional specificity.",
    }] }, coaching: { top_3_priorities: ["Add supporting examples"] } };
    const summary = summarize(JSON.parse(JSON.stringify(report)));
    render(<ReviewSummary summary={summary} translate={createTranslator(locale)} />);
    expect(screen.getByTestId("review-style-suggestions")).toHaveTextContent("The trip was good → The trip was enjoyable");
    expect(screen.getByTestId("review-style-suggestions")).toHaveTextContent(locale === "en" ? "Optional wording suggestions" : "Suggerimenti facoltativi");
    expect(screen.getByTestId("review-recurring-grammar")).not.toHaveTextContent("The trip was good");
    expect(screen.getByTestId("review-priorities")).not.toHaveTextContent("The trip was good");
    expect(summary.payload.report).toEqual(report);
  });

  it("preserves old reports without adding an empty style section", () => {
    render(<ReviewSummary summary={summarize({ rubric: { recurring_grammar_errors: [] } })} translate={createTranslator("en")} />);
    expect(screen.queryByTestId("review-style-suggestions")).not.toBeInTheDocument();
  });
});
