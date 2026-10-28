import "@testing-library/jest-dom/vitest";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "@/test/renderWithProviders";
import { RETRY_DRAFT_STORAGE_KEY } from "./retryBriefStorage";

describe("retry draft lifecycle", () => {
  it("clears the saved brief when the assessment completes", () => {
    const { store } = renderWithProviders(<div />, { appState: { draft: {
      speakerId: "learner", themeId: "trip", promptText: "Describe a trip", retryOfSessionId: "first",
      coachingGoal: "Add a closing sentence", retryGoalEligible: true,
    } } });
    store.getState().updateDraft({ coachingGoal: "Add one clear closing sentence" });
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toContain("Add one clear closing sentence");
    store.getState().setRecordingJob({ status: "completed" });
    expect(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)).toBeNull();
    expect(store.getState().draft.coachingGoal).toBe("Add one clear closing sentence");
  });
});
