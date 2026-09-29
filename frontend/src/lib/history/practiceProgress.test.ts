import { describe, expect, it } from "vitest";
import type { HistoryRow } from "@/lib/api/types";
import { comparableAttempts, comparisonAttempt, measurement, retryDraft } from "./practiceProgress";

export const attempt = (id: string, overrides: Partial<HistoryRow> = {}): HistoryRow => ({
  timestamp: `2026-09-${id === "first" ? "01" : "02"}T12:00:00Z`, session_id: id,
  speaker_id: "learner", learning_language: "it", theme: "Travel", task_family: "travel_narrative",
  overall: 3, final_score: 3, wpm: 120, report_path: `/tmp/${id}.json`, band: "3",
  requires_human_review: false, duration_pass: true, topic_pass: true, language_pass: true,
  min_words_pass: true, top_priorities: ["Give one concrete example"], grammar_error_categories: [], coherence_issue_categories: [],
  duration_sec: 90, elapsed_wpm: 100, pause_total_sec: 10,
  practice: { version: 1, goal: "B2", prompt_id: "travel-b2", prompt_text: "Describe a trip", retry_of_session_id: "",
    target_duration_sec: 90, scoring_version: "scorer-v1", scoring_mode: "hybrid", analysis_signature: "settings-v1", provider: "ollama", model: "model",
    whisper_model: "small", asr_provider: "faster_whisper", dry_run: false },
  ...overrides,
});

describe("practice progress", () => {
  it("keeps missing and invalid measurements distinct from a real zero", () => {
    for (const input of [null, undefined, "", "  ", false, NaN, Infinity]) expect(measurement(input)).toBeNull();
    expect(measurement(0)).toBe(0);
    expect(measurement("3.5")).toBe(3.5);
  });
  it("orders matching attempts by date and excludes changed conditions and legacy reports", () => {
    const first = attempt("first"); const next = attempt("next");
    const mismatches = [
      { speaker_id: "other" }, { learning_language: "en" }, { task_family: "free_monologue" }, { practice: null },
      ...[{ goal: "C1" }, { target_duration_sec: 120 }, { model: "other" }, { scoring_version: "v3" }, { dry_run: true },
        { scoring_mode: "deterministic_only" }, { analysis_signature: "different-settings" }, { analysis_signature: undefined }]
        .map((change) => ({ practice: { ...first.practice!, ...change } })),
    ];
    const rows = [next, ...mismatches.map((change) => attempt("excluded", change)), first];
    expect(comparableAttempts(rows, next).map((row) => row.session_id)).toEqual(["first", "next"]);
    expect(comparisonAttempt(rows, next)?.session_id).toBe("first");
  });
  it("uses the explicit retry parent and never substitutes a different or changed prompt", () => {
    const first = attempt("first");
    const next = attempt("next", { practice: { ...first.practice!, retry_of_session_id: "first" } });
    expect(comparisonAttempt([first, next], next)?.session_id).toBe("first");
    expect(comparisonAttempt([attempt("unrelated"), next], next)).toBeNull();
    expect(comparisonAttempt([first, { ...next, practice: { ...next.practice!, prompt_text: "Changed" } }],
      { ...next, practice: { ...next.practice!, prompt_text: "Changed" } })).toBeNull();
  });
  it("restores the saved goal and exact prompt for a retry without inferring them from the score", () => {
    expect(retryDraft(attempt("first"))).toMatchObject({ cefrLevel: "B2", promptText: "Describe a trip", retryOfSessionId: "first", durationSec: 90 });
    expect(retryDraft(attempt("first", { practice: null }))).toBeNull();
  });
});
