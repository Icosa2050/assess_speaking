import { SharingSummary } from "@/lib/setup/sharing";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { ReviewSummary, selectReviewSummary } from "@/components/review/ReviewSummary";
import { WarningsPanel } from "@/components/review/WarningsPanel";
import { Icon } from "@/components/ui/Icon";
import { ProgressRing } from "@/components/ui/ProgressRing";
import { ApiClientError, apiClient } from "@/lib/api/client";
import { createTranslator } from "@/lib/i18n";
import { pollingIntervals, queryKeys } from "@/lib/query/queryClient";
import { useAppStore } from "@/lib/state/appStore";
import { hasReviewState } from "@/lib/state/sessionDraft";

import styles from "./ReviewRoute.module.css";

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
  gap: "0.45rem",
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

const tertiaryButtonStyle = {
  ...secondaryButtonStyle,
  backgroundColor: "transparent",
  border: "1px solid transparent",
  color: "#33514b",
} as const;

const nextStepCardStyle = {
  ...cardStyle,
  border: "1px solid rgba(15, 118, 110, 0.22)",
  backgroundColor: "rgba(250, 253, 252, 0.98)",
} as const;

const scoreVisualStyle = {
  display: "grid",
  justifyItems: "start",
  width: "fit-content",
  maxWidth: "100%",
  padding: "1rem",
  border: "1px solid rgba(15, 118, 110, 0.16)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.72)",
} as const;

const focusChipGroupStyle = {
  display: "grid",
  gap: "0.55rem",
} as const;

const focusChipStyle = {
  display: "flex",
  alignItems: "center",
  gap: "0.5rem",
  width: "fit-content",
  maxWidth: "100%",
  padding: "0.6rem 0.75rem",
  border: "1px solid rgba(15, 118, 110, 0.16)",
  borderRadius: "8px",
  backgroundColor: "rgba(215, 235, 229, 0.62)",
  color: "#10201c",
  fontWeight: 650,
  lineHeight: 1.45,
  textWrap: "pretty",
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
  const [resumeError, setResumeError] = useState("");
  const [resuming, setResuming] = useState(false);
  const resumeSubmission = useRef<{sessionId: string; requestId: string} | null>(null);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const recording = useAppStore((state) => state.recording);
  const review = useAppStore((state) => state.review);
  const retainedReport = (review.payload.report || {}) as Record<string, unknown>;
  const warnings = Array.isArray(retainedReport.warnings) ? retainedReport.warnings : [];
  const unfinishedAnalysis = !retainedReport.rubric || warnings.some(warning =>
    ["coaching_unavailable", "llm_unavailable", "llm_invalid_schema"].includes(String(warning)));
  const resumeSessionId = String(retainedReport.session_id || review.reportId);
  const sharingQuery = useQuery({ queryKey: ["runtime", "resume-sharing", resumeSessionId], queryFn: () => apiClient.getResumeSharingRoute(resumeSessionId), enabled: Boolean(resumeSessionId) && unfinishedAnalysis, retry: false });
  const clearAttempt = useAppStore((state) => state.clearAttempt);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setRecordingError = useAppStore((state) => state.setRecordingError);
  const setRecordingAssessing = useAppStore(state => state.setRecordingAssessing);
  const setRecordingJob = useAppStore((state) => state.setRecordingJob);
  const setReturnTo = useAppStore((state) => state.setReturnTo);
  const updateReview = useAppStore((state) => state.updateReview);

  const translate = createTranslator(locale);
  const hasReview = hasReviewState(review);
  useEffect(() => {
    if (review.reportId) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.history });
    }
  }, [queryClient, review.reportId]);
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
  const scoreStatus = translate("review.next_step_score", {
    band: summary.band || "-",
    score: summary.scoreOverall !== null ? summary.scoreOverall.toFixed(1) : "-",
  });
  const scoreRingValue = summary.scoreOverall !== null ? summary.scoreOverall * 20 : 0;
  const coachTakeaway = summary.coachSummary || translate("review.coach_takeaway_placeholder");

  const handleTryAgain = () => {
    clearAttempt({ keepSetup: true });
    setReturnTo("review");
    navigate("/speak");
  };

  const handleChangeTask = () => {
    clearAttempt({ keepSetup: false });
    setReturnTo("review");
    navigate("/session-setup");
  };

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
        body={translate("review.empty_body")}
        buttonLabel={translate("review.empty_cta")}
        onClick={() => {
          navigate("/session-setup");
        }}
        testId="review-guard-missing-review"
        title={translate("review.empty_title")}
      />
    );
  }


  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      data-testid="review-route"
      data-semantic-id="review-route"
    >
      {unfinishedAnalysis && <section style={cardStyle}>
        <SharingSummary route={sharingQuery.data} locale={locale} />
        <button type="button" disabled={resuming || !sharingQuery.data?.available} onClick={() => {
          setResuming(true); setResumeError("");
          const report = (review.payload.report || {}) as Record<string, unknown>;
          const id = String(report.session_id || review.reportId);
          if (resumeSubmission.current?.sessionId !== id) resumeSubmission.current = {sessionId: id, requestId: crypto.randomUUID()};
          const requestId = resumeSubmission.current.requestId;
          const submit = async () => {
            const refreshed = await sharingQuery.refetch();
            if (!refreshed.data?.available || refreshed.data.fingerprint !== sharingQuery.data?.fingerprint) throw new Error(translate("sharing.changed"));
            try { return await apiClient.resumeHistory(id, requestId, refreshed.data.fingerprint); }
            catch (error) {
              if (!(error instanceof TypeError)) throw error;
              return apiClient.resumeHistory(id, requestId, refreshed.data.fingerprint);
            }
          };
          void submit().then(created => {
            resumeSubmission.current = null;
            updateReview({payload: {}, reportId: "", summary: "", transcript: "", scoreOverall: null, band: ""});
            setRecordingAssessing({assessmentId: created.assessment_id, status: created.status, phase: created.status, progress: 0, error: "", reportPath: ""});
          }).catch(error => setResumeError(error instanceof Error ? error.message : String(error))).finally(() => setResuming(false));
        }}>{({en: "Resume unfinished analysis", it: "Riprendi l’analisi incompleta", de: "Unvollständige Analyse fortsetzen", es: "Continuar el análisis pendiente", fr: "Reprendre l’analyse inachevée"})[locale]}</button>
        {resumeError && <p role="alert">{resumeError}</p>}
      </section>}

      <section
        style={nextStepCardStyle}
        data-testid="review-next-step-card"
        data-semantic-id="review-next-step-card"
      >
        <div className={styles.coachCardGrid}>
          <div className={styles.coachMain}>
            <p
              style={{
                margin: 0,
                fontSize: "0.875rem",
                fontWeight: 700,
                color: "#0f766e",
              }}
            >
              {translate("review.coach_note_eyebrow")}
            </p>
            <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>
              {translate("review.next_step_title")}
            </h2>
            <p
              className={styles.coachTakeaway}
              data-testid="review-coach-takeaway"
              data-semantic-id="review-coach-takeaway"
            >
              {coachTakeaway}
            </p>
            <p className={styles.coachHelper}>
              {translate("review.coach_takeaway_body")}
            </p>
            <div style={focusChipGroupStyle}>
              <div
                style={focusChipStyle}
                data-testid="review-next-step-focus"
                data-semantic-id="review-next-step-focus"
              >
                <Icon name="target" size={18} />
                {summary.nextFocus
                  ? translate("review.answer_next_focus", { value: summary.nextFocus })
                  : translate("review.answer_next_placeholder")}
              </div>
              {summary.nextExercise ? (
                <div
                  style={{
                    ...focusChipStyle,
                    backgroundColor: "rgba(255, 255, 255, 0.72)",
                    color: "#33514b",
                  }}
                  data-testid="review-next-step-exercise"
                  data-semantic-id="review-next-step-exercise"
                >
                  <Icon name="play" size={18} />
                  {translate("review.next_exercise", { value: summary.nextExercise })}
                </div>
              ) : null}
            </div>
            {summary.requiresHumanReview ? (
              <p
                role="status"
                style={{
                  margin: 0,
                  color: "#8f1f14",
                  lineHeight: 1.5,
                  fontWeight: 650,
                }}
                data-testid="review-human-review-guidance"
                data-semantic-id="review-human-review-guidance"
              >
                {translate("review.human_review_guidance")}
              </p>
            ) : null}
            <div
              className={styles.actionRow}
            >
              <button
                type="button"
                onClick={handleTryAgain}
                style={primaryButtonStyle}
                data-testid="review-action-try-again"
                data-semantic-id="review-action-try-again"
              >
                <Icon name="play" size={18} />
                {translate("review.try_again")}
              </button>
              <button
                type="button"
                onClick={handleChangeTask}
                style={secondaryButtonStyle}
                data-testid="review-action-new-setup"
                data-semantic-id="review-action-new-setup"
              >
                <Icon name="target" size={18} />
                {translate("review.new_setup")}
              </button>
              <button
                type="button"
                onClick={() => navigate("/history")}
                style={tertiaryButtonStyle}
                data-testid="review-action-view-history"
                data-semantic-id="review-action-view-history"
              >
                <Icon name="history" size={18} />
                {translate("review.view_history")}
              </button>
            </div>
          </div>
          <div className={styles.coachAside}>
            <div
              style={scoreVisualStyle}
              data-testid="review-next-step-score"
              data-semantic-id="review-next-step-score"
            >
              <ProgressRing
                label={translate("review.answer_result_title")}
                status={scoreStatus}
                value={scoreRingValue}
              />
            </div>
          </div>
        </div>
      </section>
      <ReviewSummary
        hideCoachSummary
        hideNextStep
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
    </div>
  );
};
