import { describe, expect, it } from "vitest";
import type { HistoryRow } from "@/lib/api/types";
import { localDayKey, practiceRhythm } from "./practiceRhythm";

const row = (timestamp: string, overrides: Partial<HistoryRow> = {}): HistoryRow => ({
  timestamp, speaker_id: "alex", learning_language: "it", session_id: timestamp,
  ...overrides,
} as HistoryRow);

describe("weekly practice rhythm", () => {
  it("counts local practice days, not repeat submissions; isolates learner, language and real attempts", () => {
    const now = new Date(2026, 9, 7, 14);
    const monday = new Date(2026, 9, 5, 10).toISOString();
    const wednesday = new Date(2026, 9, 7, 11).toISOString();
    const result = practiceRhythm([
      row(monday), row(monday), row(wednesday), row(wednesday, { speaker_id: "someone-else" }),
      row(wednesday, { learning_language: "en" }), row("invalid"),
      row(new Date(2026, 9, 9).toISOString()),
      row(wednesday, { practice: { dry_run: true } as HistoryRow["practice"] }),
    ], "alex", "it", now);
    expect(result.completed).toBe(2);
    expect(result.attempts).toBe(3);
    expect(result.week.map(day => day.completed)).toEqual([true, false, true, false, false, false, false]);
  });

  it("starts a fresh week on Monday while keeping earlier practice achievements", () => {
    const now = new Date(2026, 9, 5, 9);
    const result = practiceRhythm([row(new Date(2026, 9, 4, 23, 59).toISOString())], "alex", "it", now);
    expect(result.completed).toBe(0);
    expect(result.days).toBe(1);
    expect(localDayKey(result.week[0].date)).toBe("2026-10-5");
  });

  it("keeps consecutive calendar dates across a daylight-saving boundary", () => {
    const now = new Date(2026, 9, 25, 23);
    const result = practiceRhythm([], "alex", "it", now);
    expect(result.week.map(day => day.date.getDate())).toEqual([19, 20, 21, 22, 23, 24, 25]);
  });
});
