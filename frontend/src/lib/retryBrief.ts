import type { ReviewDisplaySummary } from "@/components/review/ReviewSummary";
import type { SessionDraft } from "./state/sessionDraft";

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};

const commonWords = new Set([
  "with", "your", "this", "that", "from", "into", "more", "speak", "make", "each", "about", "some", "when", "also",
  "para", "cada", "sobre", "algo", "mucho", "avec", "pour", "plus", "dans", "dass", "eine", "einen", "mehr", "sehr",
  "questo", "questa", "della", "delle", "molto", "anche",
]);
const words = (value: string): Set<string> => new Set(
  (value.toLocaleLowerCase().match(/[\p{L}\p{N}]{4,}/gu) ?? []).filter(word => !commonWords.has(word)),
);

export type FollowUpBrief = {
  goal: string;
  observation: string;
  comparable: boolean;
  overallChange: number | null;
};

export const followUpBrief = (summary: ReviewDisplaySummary, draft: SessionDraft): FollowUpBrief | null => {
  const goal = draft.coachingGoal.trim();
  const parentId = draft.retryOfSessionId;
  const report = record(summary.payload.report);
  const practice = record(record(summary.payload.meta).practice);
  if (!goal || !parentId || !summary.reportId || summary.reportId === parentId ||
      practice.retry_of_session_id !== parentId) return null;

  const goalWords = words(goal);
  const matching = [...summary.priorities, ...summary.strengths].map(text => ({
    text,
    overlap: [...words(text)].filter(word => goalWords.has(word)).length,
  })).sort((a, b) => b.overlap - a.overlap)[0];
  const progress = record(report.progress_delta);
  const comparable = summary.eligibilityState === "assessable" &&
    progress.comparison_verified === true && progress.previous_session_id === parentId;
  const value = record(progress.score_delta).final;
  return {
    goal,
    observation: matching && matching.overlap >= 2 && matching.overlap / goalWords.size >= 0.5
      ? matching.text : "",
    comparable,
    overallChange: comparable && typeof value === "number" && Number.isFinite(value) ? value : null,
  };
};
