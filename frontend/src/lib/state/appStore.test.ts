import { describe, expect, it } from "vitest";

import { createAppStore, selectCanSubmitAssessment } from "./appStore";

describe("app store recording jobs", () => {
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
