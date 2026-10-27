import { describe, expect, it } from "vitest";

import { selectReviewSummary } from "@/components/review/ReviewSummary";
import { buildSessionDraft } from "./state/sessionDraft";
import { followUpBrief } from "./retryBrief";

const checks = { language_pass: true, topic_pass: true, content_validity_pass: true, min_words_pass: true };
const draft = buildSessionDraft({ retryOfSessionId: "first", coachingGoal: "Connect your examples", retryGoalEligible: true });

const summary = (progress: unknown, eligibility = "assessable", priorities = ["Connect examples with one transition."]) => selectReviewSummary({
  reportId: "second", band: "", scoreOverall: null, summary: "", transcript: "",
  payload: {
    metrics: { duration_sec: 90 },
    meta: { practice: { retry_of_session_id: "first" } },
    report: {
      session_id: "second", checks, eligibility: { state: eligibility },
      scores: { final: 3.5 },
      coaching: { top_3_priorities: priorities },
      progress_delta: progress,
    },
  },
});

describe("follow-up retry brief", () => {
  it("shows a related observation and only a verified parent delta", () => {
    expect(followUpBrief(summary({ comparison_verified: true, previous_session_id: "first", score_delta: { final: 0.5 } }), draft))
      .toMatchObject({ observation: "Connect examples with one transition.", comparable: true, overallChange: 0.5 });
  });

  it("withholds a change for a different parent or an unassessable report", () => {
    expect(followUpBrief(summary({ comparison_verified: true, previous_session_id: "other", score_delta: { final: 0.5 } }), draft)?.overallChange).toBeNull();
    expect(followUpBrief(summary({ comparison_verified: true, previous_session_id: "first", score_delta: { final: 0.5 } }, "invalid_content"), draft)?.overallChange).toBeNull();
    expect(followUpBrief(summary({ comparison_verified: false, previous_session_id: "first", score_delta: { final: 0.5 } }), draft)?.overallChange).toBeNull();
  });

  it("does not call a generic shared word a goal-related observation", () => {
    const goal = buildSessionDraft({ retryOfSessionId: "first", coachingGoal: "Speak more slowly and clearly", retryGoalEligible: true });
    expect(followUpBrief(summary(null, "assessable", ["Use more varied vocabulary"]), goal)?.observation).toBe("");
  });
});
