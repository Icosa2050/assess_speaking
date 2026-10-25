import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { createTranslator } from "@/lib/i18n";
import { ReviewSummary, selectReviewSummary } from "./ReviewSummary";
import { ReviewResult } from "./ReviewResult";

const eligibleChecks = { language_pass: true, topic_pass: true, content_validity_pass: true,
  duration_pass: true, min_words_pass: true };
const summarize = (report: Record<string, unknown>, metrics: Record<string, unknown> = { duration_sec: 120 }) => selectReviewSummary({
  band: "3", reportId: "saved", scoreOverall: null, summary: "", transcript: "Actual words",
  payload: { metrics, report: { checks: eligibleChecks, ...report } },
});

describe("Review quality safeguards", () => {
  const legacyBaseline = {
    level: "B1", valid: true, passed: true,
    targets: {
      wpm: { expected: "≥80", actual: 80, ok: true, status: "pass" },
      fillers: { expected: "≤6", actual: 0, ok: true, status: "pass" },
    },
  };
  const baselineSummary = (checks: Record<string, unknown>, baseline = legacyBaseline) => selectReviewSummary({
    band: "3", reportId: "saved", scoreOverall: 3, summary: "", transcript: "bla bla bla",
    payload: { metrics: { duration_sec: 120 }, baseline_comparison: baseline, report: { checks, scores: { deterministic: 3, final: 3 } } },
  });

  it("withholds grades and stale baseline passes for a three-word recording", () => {
    const summary = baselineSummary({ ...eligibleChecks, min_words_pass: false, duration_pass: false,
      content_validity_pass: null, topic_pass: null });
    expect(summary.scoreOverall).toBeNull();
    expect(summary.deterministicScore).toBeNull();
    expect(summary.band).toBe("");
    expect(summary.cefrLevel).toBe("");
    expect(summary.dimensions).toEqual([]);
    render(<ReviewSummary summary={summary} translate={createTranslator("en")} />);
    expect(screen.getByText(/Not enough speech to assess/)).toBeVisible();
    expect(screen.getByTestId("review-baseline")).not.toHaveTextContent("Pass");
    expect(screen.getAllByText("Not assessed")).toHaveLength(2);
    expect(legacyBaseline.targets.wpm.ok).toBe(true);
  });

  it("uses reported provisional CEFR and dimension confidence without deriving uncertainty", () => {
    const summary = summarize({ eligibility: { state: "assessable" }, scores: {
      final: 4.1, cefr_estimate: { level: "B2", calibrated: false },
      dimensions: { pronunciation_intelligibility: { score: 3.7, confidence: "low", method: "asr_proxy_v1" } },
    } });
    expect(summary.cefrLevel).toBe("B2");
    expect(summary.dimensions).toEqual([{ key: "pronunciation_intelligibility", score: 3.7, confidence: "low", method: "asr_proxy_v1" }]);
    const invalid = summarize({ eligibility: { state: "invalid_content" }, scores: {
      final: 4.1, cefr_estimate: { level: "B2" },
      dimensions: { pronunciation_intelligibility: { score: 3.7, confidence: "low" } },
    } });
    expect(invalid.scoreOverall).toBeNull();
    expect(invalid.cefrLevel).toBe("");
    expect(invalid.dimensions).toEqual([]);
    for (const failedCheck of ["language_pass", "topic_pass", "content_validity_pass"]) {
      const invalidLegacyReport = summarize({ checks: { [failedCheck]: false }, scores: {
        final: 4.1, cefr_estimate: { level: "B2" },
        dimensions: { pronunciation_intelligibility: { score: 3.7 } },
      } });
      expect(invalidLegacyReport.scoreOverall).toBeNull();
      expect(invalidLegacyReport.cefrLevel).toBe("");
      expect(invalidLegacyReport.dimensions).toEqual([]);
      expect(invalidLegacyReport.eligibilityState).toBe("invalid_content");
    }
  });

  it.each([
    { checks: { ...eligibleChecks, language_pass: null }, metrics: { duration_sec: 120 } },
    { checks: { ...eligibleChecks, content_validity_pass: undefined }, metrics: { duration_sec: 120 } },
    { checks: eligibleChecks, metrics: {} },
    { checks: eligibleChecks, metrics: { duration_sec: Number.NaN } },
  ])("withholds legacy grades when content is unverified: %j", ({ checks, metrics }) => {
    const summary = summarize({ checks, scores: { final: 4.1, cefr_estimate: { level: "B2" },
      dimensions: { pronunciation_intelligibility: { score: 3.7 } } } }, metrics);
    expect(summary.eligibilityState).toBe("content_unverified");
    expect(summary.scoreOverall).toBeNull();
    expect(summary.cefrLevel).toBe("");
    expect(summary.dimensions).toEqual([]);
  });

  it("withholds legacy grades for a known recording shorter than 30 seconds", () => {
    const summary = summarize({ scores: { final: 4.1 } }, { duration_sec: 29 });
    expect(summary.eligibilityState).toBe("insufficient_speech");
    expect(summary.scoreOverall).toBeNull();
  });

  it("retains an assessable legacy grade when topic completion alone is unknown", () => {
    const summary = summarize({ checks: { ...eligibleChecks, topic_pass: null }, scores: { final: 4.1 } });
    expect(summary.eligibilityState).toBe("assessable");
    expect(summary.scoreOverall).toBe(4.1);
  });

  it.each([
    { ...eligibleChecks, duration_pass: false },
    { ...eligibleChecks, content_validity_pass: false },
    { ...eligibleChecks, content_validity_pass: null },
    { ...eligibleChecks, language_pass: null },
  ])("withholds baseline judgments when speech eligibility is not established: %j", (checks) => {
    render(<ReviewSummary summary={baselineSummary(checks)} translate={createTranslator("en")} />);
    expect(screen.getByTestId("review-baseline")).not.toHaveTextContent("Pass");
    expect(screen.getAllByText("Not assessed")).toHaveLength(2);
  });

  it("retains judgments for eligible speech and distinguishes targets from measurements", () => {
    render(<ReviewSummary summary={baselineSummary(eligibleChecks)} translate={createTranslator("en")} />);
    expect(screen.getByTestId("review-baseline")).toHaveTextContent("Expected");
    expect(screen.getByTestId("review-baseline")).toHaveTextContent("Actual");
    expect(screen.getByTestId("review-baseline")).toHaveTextContent("Pass");
    expect(screen.queryByTestId("review-baseline-unassessed")).not.toBeInTheDocument();
  });

  it("keeps legacy comparison evidence but hides its unverified deltas", () => {
    const report = { progress_delta: { previous_session_id: "older", score_delta: { final: 1 } } };
    const summary = summarize(report);
    expect(summary.progressItems).toEqual([]);
    expect(summary.payload.report).toMatchObject(report);
  });

  it("displays only explicitly verified comparisons", () => {
    const summary = summarize({ progress_delta: {
      comparison_verified: true, previous_session_id: "older", score_delta: { final: 1 },
    } });
    expect(summary.progressItems).toContainEqual({ kind: "delta_final", value: 1 });
  });

  it("withholds even a marked verified legacy comparison when its language gate failed", () => {
    const summary = summarize({ checks: { ...eligibleChecks, language_pass: false }, progress_delta: {
      comparison_verified: true, previous_session_id: "older", score_delta: { final: 1 },
    } });
    expect(summary.progressItems).toEqual([]);
  });

  it.each([
    ["hybrid", "Rule-based speech signals and a model rubric both contributed."],
    ["deterministic_only", "Only rule-based speech signals contributed"],
    ["unexpected_mode", "The scoring method was not reported"],
    ["", "The scoring method was not reported"],
  ])("describes scoring mode %s without guessing", (mode, expected) => {
    const summary = summarize({ scores: { final: 4.1, mode } });
    render(<MemoryRouter><ReviewResult summary={summary} translate={createTranslator("en")} /></MemoryRouter>);
    expect(screen.getByText(new RegExp(expected, "i"))).toBeVisible();
  });

  it.each([[4.1, 4], [4.9, 5], [3.5, 4]])("rounds scale ticks for %.1f to %i", (value, count) => {
    const summary = summarize({ scores: { final: value, mode: "hybrid" } });
    render(<MemoryRouter><ReviewResult summary={summary} translate={createTranslator("en")} /></MemoryRouter>);
    const ticks = screen.getAllByText(/^[1-5]$/, { selector: "li" });
    expect(ticks.filter((tick) => tick.className.includes("filled"))).toHaveLength(count);
    expect(screen.getByTestId("review-estimate-text")).toHaveTextContent(`${value.toFixed(1)}/5`);
  });

  it("exposes the reported dimension source next to its value", () => {
    const summary = summarize({ scores: { final: 4.9, mode: "hybrid", dimensions: {
      pronunciation_intelligibility: { score: 3.7, confidence: "low", method: "asr_proxy_v1" },
    } } });
    render(<MemoryRouter><ReviewResult summary={summary} translate={createTranslator("en")} /></MemoryRouter>);
    fireEvent.click(screen.getByText("Show dimension signals"));
    expect(screen.getByText("Source: ASR confidence and pausing proxy; not acoustic pronunciation scoring.")).toBeVisible();
  });

  it("does not infer a rules-only method from an unrecognized mode with a rubric score", () => {
    const summary = summarize({ scores: { final: 4.1, llm: 4.2, mode: "legacy_v0" } });
    render(<MemoryRouter><ReviewResult summary={summary} translate={createTranslator("en")} /></MemoryRouter>);
    expect(screen.getByText("The scoring method was not reported for this result.")).toBeVisible();
    expect(screen.queryByText(/Only rule-based speech signals contributed/)).not.toBeInTheDocument();
  });

  it.each(["insufficient_speech", "content_unverified", "invalid_content", "assessable"])(
    "does not revive saved language praise or score progress for %s", (state) => {
      const report = { eligibility: { version: 1, state },
        coaching: { coach_summary: "You spoke in the required language.", strengths: ["Verified Italian"] },
        progress_delta: { comparison_verified: true, previous_session_id: "older", score_delta: { final: 1 } },
      };
      const summary = summarize(report);
      render(<ReviewSummary summary={summary} translate={createTranslator("en")} />);
      if (state === "assessable") {
        expect(screen.getByTestId("review-coach-summary")).toHaveTextContent("You spoke in the required language.");
        expect(summary.progressItems).toContainEqual({ kind: "delta_final", value: 1 });
      } else {
        expect(screen.queryByText("You spoke in the required language.")).not.toBeInTheDocument();
        expect(screen.queryByText("Verified Italian")).not.toBeInTheDocument();
        expect(summary.progressItems).toEqual([]);
      }
      expect(summary.payload.report).toMatchObject(report);
      expect(report.coaching.coach_summary).toBe("You spoke in the required language.");
    },
  );

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
    expect(summary.payload.report).toMatchObject(report);
  });

  it("preserves old reports without adding an empty style section", () => {
    render(<ReviewSummary summary={summarize({ rubric: { recurring_grammar_errors: [] } })} translate={createTranslator("en")} />);
    expect(screen.queryByTestId("review-style-suggestions")).not.toBeInTheDocument();
  });
});
