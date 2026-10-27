import { beforeEach, describe, expect, it } from "vitest";

import { buildSessionDraft } from "./sessionDraft";
import { persistRetryDraft, readRetryDraft, RETRY_DRAFT_STORAGE_KEY } from "./retryBriefStorage";

const retry = () => buildSessionDraft({
  speakerId: "learner", themeId: "trip", themeLabel: "A trip", promptText: "Describe a trip",
  retryOfSessionId: "first", coachingGoal: "End with one reflection",
});

describe("local retry draft", () => {
  beforeEach(() => window.localStorage.clear());

  it("restores the task and editable goal after a reload, including a cleared goal", () => {
    persistRetryDraft(retry());
    expect(readRetryDraft()).toMatchObject({ promptText: "Describe a trip", coachingGoal: "End with one reflection" });
    persistRetryDraft(buildSessionDraft({ ...retry(), coachingGoal: "" }));
    expect(readRetryDraft()?.coachingGoal).toBe("");
    const longSuggestion = "End with a personal reflection. ".repeat(12);
    persistRetryDraft(buildSessionDraft({ ...retry(), coachingGoal: longSuggestion }));
    expect(readRetryDraft()).toMatchObject({ promptText: "Describe a trip", coachingGoal: longSuggestion });
  });

  it("removes stale retry data when starting a different task and ignores damaged storage", () => {
    persistRetryDraft(retry());
    persistRetryDraft(buildSessionDraft());
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
    window.localStorage.setItem(RETRY_DRAFT_STORAGE_KEY, "{broken");
    expect(readRetryDraft()).toBeNull();
  });
});
