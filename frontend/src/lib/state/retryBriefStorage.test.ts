import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildSessionDraft } from "./sessionDraft";
import { clearRetryDraft, persistRetryDraft, readRetryDraft, RETRY_DRAFT_STORAGE_KEY, RETRY_DRAFT_TTL_MS } from "./retryBriefStorage";

const retry = () => buildSessionDraft({
  speakerId: "learner", themeId: "trip", themeLabel: "A trip", promptText: "Describe a trip",
  retryOfSessionId: "first", coachingGoal: "End with one reflection", retryGoalEligible: true,
});

const stored = () => JSON.parse(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY) || "null") as Record<string, unknown>;

describe("local retry draft", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => vi.useRealTimers());

  it("restores the task and editable goal after a reload, including a cleared and long goal", () => {
    persistRetryDraft(retry());
    expect(readRetryDraft()).toMatchObject({ promptText: "Describe a trip", coachingGoal: "End with one reflection" });
    persistRetryDraft(buildSessionDraft({ ...retry(), coachingGoal: "" }));
    expect(readRetryDraft()?.coachingGoal).toBe("");
    const longSuggestion = "End with a personal reflection. ".repeat(12);
    persistRetryDraft(buildSessionDraft({ ...retry(), coachingGoal: longSuggestion }));
    expect(readRetryDraft()).toMatchObject({ promptText: "Describe a trip", coachingGoal: longSuggestion });
  });

  it("normalizes oversized setup fields on write and preserves the retry", () => {
    persistRetryDraft(buildSessionDraft({ ...retry(), speakerId: "s".repeat(257),
      themeLabel: "t".repeat(1001), promptText: "p".repeat(12001) }));
    expect(readRetryDraft()).toMatchObject({ speakerId: "s".repeat(256),
      themeLabel: "t".repeat(1000), promptText: "p".repeat(12000), coachingGoal: "End with one reflection" });
  });

  it("keeps a 12,000-character goal and caps one character beyond it", () => {
    persistRetryDraft(buildSessionDraft({ ...retry(), coachingGoal: "g".repeat(12000) }));
    expect(readRetryDraft()?.coachingGoal).toHaveLength(12000);
    persistRetryDraft(buildSessionDraft({ ...retry(), coachingGoal: "g".repeat(12001) }));
    expect(readRetryDraft()?.coachingGoal).toHaveLength(12000);
  });

  it("expires an abandoned retry after seven days and removes the stale key", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-01T12:00:00Z"));
    persistRetryDraft(retry());
    vi.setSystemTime(new Date(Date.now() + RETRY_DRAFT_TTL_MS + 1));
    expect(readRetryDraft()).toBeNull();
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
  });

  it("removes non-eligible or abandoned retry data", () => {
    persistRetryDraft(retry());
    persistRetryDraft(buildSessionDraft({ ...retry(), retryGoalEligible: false }));
    expect(readRetryDraft()).toBeNull();
    persistRetryDraft(retry());
    clearRetryDraft();
    expect(readRetryDraft()).toBeNull();
    window.localStorage.setItem("vostavo.retry-draft.v1", "stale");
    clearRetryDraft();
    expect(window.localStorage.getItem("vostavo.retry-draft.v1")).toBeNull();
  });

  it.each([
    ["broken JSON", "{broken"],
    ["unsupported language", (value: Record<string, unknown>) => ({ ...value, draft: { ...(value.draft as object), learningLanguage: "de" } })],
    ["wrong field type", (value: Record<string, unknown>) => ({ ...value, draft: { ...(value.draft as object), speakerId: 12 } })],
    ["oversized prompt", (value: Record<string, unknown>) => ({ ...value, draft: { ...(value.draft as object), promptText: "x".repeat(12001) } })],
    ["invalid duration", (value: Record<string, unknown>) => ({ ...value, draft: { ...(value.draft as object), durationSec: 999 } })],
    ["mixed learner", (value: Record<string, unknown>) => ({ ...value, draft: { ...(value.draft as object), speakerId: "another" } })],
  ])("rejects %s without keeping corrupt data", (_name, mutate) => {
    persistRetryDraft(retry());
    const value = typeof mutate === "string" ? mutate : JSON.stringify(mutate(stored()));
    window.localStorage.setItem(RETRY_DRAFT_STORAGE_KEY, value);
    expect(readRetryDraft()).toBeNull();
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
  });
});
