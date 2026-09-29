import { describe, expect, it } from "vitest";

import { createAppStore, selectCanSubmitAssessment } from "./appStore";

describe("app store recording jobs", () => {
  it("links each retry to the saved report session, and clears the link for a new task", () => {
    const store = createAppStore({
      draft: { promptText: "Explain your choice", cefrLevel: "C1" },
      review: { reportId: "/tmp/report.json", payload: { report: { session_id: "saved-first" } } },
    });
    const originalDraftId = store.getState().draft.sessionId;
    store.getState().clearAttempt({ keepSetup: true });
    expect(store.getState().draft.retryOfSessionId).toBe("saved-first");
    expect(store.getState().draft.sessionId).not.toBe(originalDraftId);
    expect(store.getState().draft.cefrLevel).toBe("C1");
    expect(store.getState().draft.promptText).toBe("Explain your choice");
    store.getState().updateReview({ payload: { report: { session_id: "saved-second" } } });
    store.getState().clearAttempt({ keepSetup: true });
    expect(store.getState().draft.retryOfSessionId).toBe("saved-second");
    store.getState().clearAttempt({ keepSetup: false });
    expect(store.getState().draft.retryOfSessionId).toBe("");
  });

  it.each(["upload", "assessment"])("allows retry after an %s failure without losing the recording", (phase) => {
    const store = createAppStore({
      draft: { speakerId: "learner", themeId: "travel", promptText: "Describe a trip", cefrLevel: "B1" },
      recording: { audioPath: "/tmp/take.wav", inputMethod: "upload" },
    });
    expect(selectCanSubmitAssessment(store.getState())).toBe(true);

    if (phase === "upload") {
      store.getState().setRecordingError("Connection dropped");
    } else {
      store.getState().setRecordingJob({ status: "failed", error: "Provider unavailable" });
    }

    expect(store.getState().recording.audioPath).toBe("/tmp/take.wav");
    expect(selectCanSubmitAssessment(store.getState())).toBe(true);
    store.getState().clearRecording();
    expect(selectCanSubmitAssessment(store.getState())).toBe(false);
  });

  const assessingSeed = (error = "") =>
    ({
      recording: {
        audioPath: "/tmp/audio.wav",
        inputDigest: "abc",
        inputMethod: "upload",
        status: "assessing",
        assessmentState: "running",
        error,
        job: {
          assessmentId: "asmt-1",
          status: "running",
          phase: "running",
          progress: 0.5,
          error: "",
          reportPath: "",
        },
      },
    }) as const;

  it("does not store undefined when a failed job omits an error", () => {
    const store = createAppStore(assessingSeed());

    store.getState().setRecordingJob({ status: "failed", error: undefined });

    expect(store.getState().recording.error).toBe("");
  });

  it("preserves the visible error when a partial failed job omits an error", () => {
    const store = createAppStore(assessingSeed("Connection dropped"));

    store.getState().setRecordingJob({ status: "failed", error: undefined });

    expect(store.getState().recording.error).toBe("Connection dropped");
  });

  it("preserves the visible error when a partial cancelled job omits an error", () => {
    const store = createAppStore(assessingSeed("Cancelled while offline"));

    store.getState().setRecordingJob({ status: "cancelled", error: undefined });

    expect(store.getState().recording.error).toBe("Cancelled while offline");
  });
});
