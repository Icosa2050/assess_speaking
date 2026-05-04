import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { ReviewSummary, selectReviewSummary } from "@/components/review/ReviewSummary";
import { WarningsPanel } from "@/components/review/WarningsPanel";
import { ApiClientError, apiClient } from "@/lib/api/client";
import { createTranslator } from "@/lib/i18n";
import { pollingIntervals, queryKeys } from "@/lib/query/queryClient";
import { useAppStore } from "@/lib/state/appStore";
import { hasReviewState } from "@/lib/state/sessionDraft";

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const primaryButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid rgba(15, 118, 110, 0.2)",
  backgroundColor: "#d7ebe5",
  color: "#10201c",
  fontWeight: 600,
  font: "inherit",
  cursor: "pointer",
} as const;

const secondaryButtonStyle = {
  ...primaryButtonStyle,
  backgroundColor: "rgba(255, 255, 255, 0.95)",
  border: "1px solid rgba(18, 61, 55, 0.12)",
} as const;

const buildStoredReviewState = (payload: Record<string, unknown>, fallbackReportId: string) => {
  const summary = selectReviewSummary({
    band: "",
    payload,
    reportId: fallbackReportId,
    scoreOverall: null,
    summary: "",
    transcript: "",
  });

  return {
    band: summary.band,
    payload,
    reportId: summary.reportId || fallbackReportId,
    scoreOverall: summary.scoreOverall,
    summary: summary.coachSummary,
    transcript: summary.transcript,
  };
};

const GuardCard = ({
  body,
  buttonLabel,
  onClick,
  testId,
  title,
}: {
  body: string;
  buttonLabel: string;
  onClick: () => void;
  testId: string;
  title: string;
}) => (
  <section
    style={cardStyle}
    data-testid={testId}
    data-semantic-id={testId}
  >
    <h2
      style={{
        margin: 0,
        fontSize: "1.5rem",
        color: "#10201c",
      }}
    >
      {title}
    </h2>
    <p
      style={{
        margin: 0,
        lineHeight: 1.6,
        color: "#33514b",
      }}
    >
      {body}
    </p>
    <button
      type="button"
      onClick={onClick}
      style={primaryButtonStyle}
      data-testid="review-guard-cta"
      data-semantic-id="review-guard-cta"
    >
      {buttonLabel}
    </button>
  </section>
);

export const ReviewRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const recording = useAppStore((state) => state.recording);
  const review = useAppStore((state) => state.review);
  const clearAttempt = useAppStore((state) => state.clearAttempt);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setRecordingError = useAppStore((state) => state.setRecordingError);
  const setRecordingJob = useAppStore((state) => state.setRecordingJob);
  const setReturnTo = useAppStore((state) => state.setReturnTo);
  const updateReview = useAppStore((state) => state.updateReview);

  const translate = createTranslator(locale);
  const hasReview = hasReviewState(review);
  const assessmentId = recording.job.assessmentId;
  const shouldPoll = !hasReview && Boolean(assessmentId);
  const isStillAssessing = !hasReview && recording.status === "assessing" && Boolean(assessmentId);

  const assessmentQuery = useQuery({
    queryKey: queryKeys.assessment(assessmentId || "review"),
    queryFn: () => apiClient.getAssessmentStatus(assessmentId),
    enabled: shouldPoll,
    refetchInterval: isStillAssessing ? pollingIntervals.assessmentMs : false,
  });

  useEffect(() => {
    setCurrentPage("review");
  }, [setCurrentPage]);

  useEffect(() => {
    if (!assessmentQuery.data) {
      return;
    }

    const response = assessmentQuery.data;
    setRecordingJob({
      assessmentId: response.assessment_id,
      error: response.error?.detail ?? "",
      phase: response.phase,
      progress: response.progress,
      reportPath: response.report_path ?? "",
      status: response.status,
    });

    if (response.payload) {
      updateReview(buildStoredReviewState(response.payload, response.report_path ?? response.assessment_id));
    }
  }, [assessmentQuery.data, setRecordingJob, updateReview]);

  useEffect(() => {
    if (assessmentQuery.error instanceof ApiClientError) {
      setRecordingError(assessmentQuery.error.detail);
    }
  }, [assessmentQuery.error, setRecordingError]);

  const summary = useMemo(() => selectReviewSummary(review), [review]);

  if (!hasReview && assessmentId && assessmentQuery.isPending && !isStillAssessing) {
    return (
      <section style={cardStyle} data-testid="review-route" data-semantic-id="review-route">
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("review.title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>
          {translate("review.answer_result_pending")}
        </p>
      </section>
    );
  }

  if (!hasReview && isStillAssessing) {
    return (
      <GuardCard
        body={translate("review.guard_still_assessing")}
        buttonLabel={translate("review.go_back_speak")}
        onClick={() => {
          setReturnTo("review");
          navigate("/speak");
        }}
        testId="review-guard-still-assessing"
        title={translate("review.title")}
      />
    );
  }

  if (!hasReview) {
    return (
      <GuardCard
        body={translate("review.guard_missing_review")}
        buttonLabel={translate("review.go_speak")}
        onClick={() => {
          setReturnTo("review");
          navigate("/speak");
        }}
        testId="review-guard-missing-review"
        title={translate("review.title")}
      />
    );
  }

  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      data-testid="review-route"
      data-semantic-id="review-route"
    >
      <ReviewSummary
        summary={summary}
        translate={translate}
        warningsSlot={
          <WarningsPanel
            failedGates={summary.failedGates}
            requiresHumanReview={summary.requiresHumanReview}
            translate={translate}
            warnings={summary.warnings}
          />
        }
      />

      <section style={cardStyle}>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "0.75rem",
          }}
        >
          <button
            type="button"
            onClick={() => {
              clearAttempt({ keepSetup: true });
              setReturnTo("review");
              navigate("/speak");
            }}
            style={primaryButtonStyle}
            data-testid="review-action-try-again"
            data-semantic-id="review-action-try-again"
          >
            {translate("review.try_again")}
          </button>
          <button
            type="button"
            onClick={() => {
              clearAttempt({ keepSetup: false });
              setReturnTo("review");
              navigate("/session-setup");
            }}
            style={secondaryButtonStyle}
            data-testid="review-action-new-setup"
            data-semantic-id="review-action-new-setup"
          >
            {translate("review.new_setup")}
          </button>
          <button
            type="button"
            onClick={() => navigate("/history")}
            style={secondaryButtonStyle}
            data-testid="review-action-view-history"
            data-semantic-id="review-action-view-history"
          >
            {translate("review.view_history")}
          </button>
        </div>
      </section>
    </div>
  );
};
