import { describe, expect, it } from "vitest";

import { createAppStore } from "./appStore";

describe("app store recording jobs", () => {
  it("does not store undefined when a failed job omits an error", () => {
    const store = createAppStore({
      recording: {
        audioPath: "/tmp/audio.wav",
        inputDigest: "abc",
        inputMethod: "upload",
        status: "assessing",
        assessmentState: "running",
        error: "",
        job: {
          assessmentId: "asmt-1",
          status: "running",
          phase: "running",
          progress: 0.5,
          error: "",
          reportPath: "",
        },
      },
    });

    store.getState().setRecordingJob({ status: "failed", error: undefined });

    expect(store.getState().recording.error).toBe("");
  });
});
