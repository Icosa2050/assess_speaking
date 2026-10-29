import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { HistoryRow } from "@/lib/api/types";
import { PracticeProgress } from "./PracticeProgress";

const row = (id: string, score: number | null, day: string): HistoryRow => ({
  timestamp: `2026-09-${day}T12:00:00Z`, session_id: id, speaker_id: "learner", learning_language: "it",
  theme: `Task ${id}`, task_family: "travel_narrative", overall: score, final_score: score,
  wpm: 100, report_path: `/tmp/${id}.json`, band: "3", requires_human_review: false,
  duration_pass: true, topic_pass: true, language_pass: true, min_words_pass: true,
  top_priorities: [], grammar_error_categories: [], coherence_issue_categories: [], duration_sec: 90,
  practice: { version: 1, goal: "B2", prompt_id: "travel", prompt_text: "Describe travel", retry_of_session_id: "",
    target_duration_sec: 90, scoring_version: "v1", scoring_mode: "hybrid", analysis_signature: "settings-v1",
    provider: "ollama", model: "model", whisper_model: "small", asr_provider: "faster_whisper", dry_run: false },
});

describe("PracticeProgress chart alternative", () => {
  it("exposes every comparable chart row as a labelled table and switches the values with the metric", () => {
    const first = { ...row("first", 3, "02"), duration_sec: null };
    const selected = row("selected", 4, "03");
    render(<PracticeProgress rows={[selected, first]} selected={selected} verifiedPreviousId="first" locale="en" onSelect={() => {}} onRetry={() => {}} />);
    const table = within(screen.getByTestId("practice-chart-table")).getByRole("table", { name: "Chart data: Overall performance" });
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    expect(within(table).getByRole("row", { name: /Task first/ })).toHaveTextContent("3");
    expect(within(table).getByRole("row", { name: /Task selected/ })).toHaveTextContent("4");
    expect(within(table).getByRole("columnheader", { name: "Overall performance (out of 5)" })).toBeVisible();
    expect(within(table).getByRole("row", { name: /Task selected/ })).toHaveAttribute("aria-current", "true");
    fireEvent.change(screen.getByRole("combobox", { name: "Measurement" }), { target: { value: "duration_sec" } });
    const duration = within(screen.getByTestId("practice-chart-table")).getByRole("table", { name: "Chart data: Recording duration (seconds)" });
    expect(within(duration).getByRole("row", { name: /Task first/ })).toHaveTextContent("—");
    expect(within(duration).getByRole("row", { name: /Task selected/ })).toHaveTextContent("90");
    expect(screen.getByRole("img", { name: "Recording duration (seconds)" }).querySelector("desc")).toHaveTextContent(/^1 plotted point:/);
  });

  it("withholds change claims without a verified eligible parent", () => {
    const first = row("first", 3, "01");
    const selected = row("selected", 4, "03");
    const { rerender } = render(<PracticeProgress rows={[first, selected]} selected={selected} locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.queryByText(/since your first/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("Earlier take is not verified for comparison");
    rerender(<PracticeProgress rows={[first, selected]} selected={selected} verifiedPreviousId="first" locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("Task first");
    rerender(<PracticeProgress rows={[{ ...first, eligibility: { version: 1, state: "insufficient_speech", reasons: [], metrics_reliable: false, task_complete: false } }, selected]} selected={selected} verifiedPreviousId="first" locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("No earlier comparable attempt");
  });

  it("labels the score change against the verified parent rather than the first matching take", () => {
    const first = row("first", 2, "01");
    const middle = row("middle", 4, "02");
    const selected = row("selected", 3, "03");
    render(<PracticeProgress rows={[first, middle, selected]} selected={selected} verifiedPreviousId="middle"
      locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.getByText("Score change from verified previous take: -1")).toBeVisible();
    const table = within(screen.getByTestId("practice-chart-table")).getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(4);
    expect(within(table).getByRole("row", { name: /Task first/ })).toHaveTextContent("2");
    expect(within(table).getByRole("row", { name: /Task middle/ })).toHaveTextContent("4");
    expect(within(table).getByRole("row", { name: /Task selected/ })).toHaveTextContent("3");
    expect(screen.getByText(/^3 comparable attempts/)).toBeVisible();
    expect(screen.getByRole("img", { name: "Overall performance" }).querySelector("desc")).toHaveTextContent(/^3 plotted points/);
  });

  it("shows an unscored selected take as progress without implying an estimate or comparison", () => {
    const first = row("first", 3, "01");
    const selected = { ...row("selected", 4, "02"), eligibility: {
      version: 1, state: "insufficient_speech" as const, reasons: [], metrics_reliable: false, task_complete: false,
    } };
    render(<PracticeProgress rows={[first, selected]} selected={selected} verifiedPreviousId="first"
      locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.getByTestId("practice-no-estimate")).toHaveTextContent("recording still counts as practice");
    expect(screen.queryByTestId("practice-legacy")).not.toBeInTheDocument();
    const table = within(screen.getByTestId("practice-chart-table")).getByRole("table");
    expect(within(table).getByRole("row", { name: /Task selected/ })).toHaveTextContent("—");
    expect(within(table).getAllByRole("row")).toHaveLength(2);
    expect(screen.getByText("This measurement is not available for these attempts.")).toBeVisible();
    expect(screen.queryByText("4", { selector: "p" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Measurement" }), { target: { value: "duration_sec" } });
    expect(within(screen.getByTestId("practice-chart-table")).getByRole("row", { name: /Task selected/ })).toHaveTextContent("—");
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("no score comparison");
  });

  it("does not invent a zero delta when the verified parent has no score", () => {
    const first = row("first", null, "01");
    const selected = row("selected", 4, "02");
    render(<PracticeProgress rows={[first, selected]} selected={selected} verifiedPreviousId="first"
      locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.queryByText(/Score change from verified previous take/)).not.toBeInTheDocument();
  });

  it("distinguishes a pending or failed report from an unverified comparison", () => {
    const first = row("first", 3, "01");
    const selected = row("selected", 4, "02");
    const { rerender } = render(<PracticeProgress rows={[first, selected]} selected={selected} comparisonState="loading"
      locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("Checking whether");
    rerender(<PracticeProgress rows={[first, selected]} selected={selected} comparisonState="error"
      locale="en" onSelect={() => {}} onRetry={() => {}} />);
    expect(screen.getByTestId("practice-comparison")).toHaveTextContent("Comparison unavailable");
    expect(within(screen.getByTestId("practice-chart-table")).getAllByRole("row")).toHaveLength(3);
  });
});
