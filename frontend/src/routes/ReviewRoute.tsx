import { SharingSummary } from "@/lib/setup/sharing";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { ReviewSummary, reviewEligibilityState, selectReviewSummary } from "@/components/review/ReviewSummary";
import { ReviewResult } from "@/components/review/ReviewResult";
import { WarningsPanel } from "@/components/review/WarningsPanel";
import { Icon } from "@/components/ui/Icon";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Field } from "@/components/ui/Field";
import { ApiClientError, apiClient } from "@/lib/api/client";
import { createTranslator } from "@/lib/i18n";
import { followUpBrief } from "@/lib/retryBrief";
import { pollingIntervals, queryKeys } from "@/lib/query/queryClient";
import { useAppStore, useAppStoreApi } from "@/lib/state/appStore";
import { useDraftRecovery } from "@/lib/state/DraftRecoveryProvider";
import { hasReviewState } from "@/lib/state/sessionDraft";

import styles from "./ReviewRoute.module.css";

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid var(--color-border-decorative)",
  borderRadius: "8px",
  backgroundColor: "var(--color-surface)",
  boxShadow: "0 18px 40px var(--color-shadow-ambient)",
} as const;

const primaryButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "0.45rem",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-selected-border)",
  backgroundColor: "var(--color-selected)",
  color: "var(--color-text)",
  fontWeight: 600,
  font: "inherit",
  cursor: "pointer",
} as const;

const secondaryButtonStyle = {
  ...primaryButtonStyle,
  backgroundColor: "var(--color-surface)",
  border: "1px solid var(--color-border-control)",
} as const;

const tertiaryButtonStyle = {
  ...secondaryButtonStyle,
  backgroundColor: "transparent",
  border: "1px solid transparent",
  color: "var(--color-text-muted)",
} as const;

const nextStepCardStyle = {
  ...cardStyle,
  border: "1px solid var(--color-selected-border)",
  backgroundColor: "var(--color-surface-subtle)",
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
  border: "1px solid var(--color-selected-border)",
  borderRadius: "8px",
  backgroundColor: "var(--color-selected)",
  color: "var(--color-text)",
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
        color: "var(--color-text)",
      }}
    >
      {title}
    </h2>
    <p
      style={{
        margin: 0,
        lineHeight: 1.6,
        color: "var(--color-text-muted)",
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
  const appStore = useAppStoreApi();
  const draftRecovery = useDraftRecovery();
  const [resumeError, setResumeError] = useState("");
  const [resuming, setResuming] = useState(false);
  const resumeSubmission = useRef<{sessionId: string; requestId: string} | null>(null);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const recording = useAppStore((state) => state.recording);
  const review = useAppStore((state) => state.review);
  const draft = useAppStore((state) => state.draft);
  const retainedReport = (review.payload.report || {}) as Record<string, unknown>;
  const warnings = Array.isArray(retainedReport.warnings) ? retainedReport.warnings : [];
  const interruptedAnalysis = warnings.some(warning =>
    ["coaching_unavailable", "llm_unavailable", "llm_invalid_schema"].includes(String(warning)));
  const eligibilityState = reviewEligibilityState(review.payload);
  // An unavailable rubric can itself make content unverified. Keep recovery for
  // that interruption, while excluding deliberate skips and invalid speech.
  const intentionallySkipped = warnings.some(warning => String(warning).startsWith("llm_skipped_")) ||
    ["insufficient_speech", "invalid_content"].includes(eligibilityState) ||
    (eligibilityState === "content_unverified" && !interruptedAnalysis);
  const unfinishedAnalysis = !intentionallySkipped && (!retainedReport.rubric || interruptedAnalysis);
  const resumeSessionId = String(retainedReport.session_id || review.reportId);
  const sharingQuery = useQuery({ queryKey: ["runtime", "resume-sharing", resumeSessionId], queryFn: () => apiClient.getResumeSharingRoute(resumeSessionId), enabled: Boolean(resumeSessionId) && unfinishedAnalysis, retry: false });
  const clearAttempt = useAppStore((state) => state.clearAttempt);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setRecordingError = useAppStore((state) => state.setRecordingError);
  const setRecordingAssessing = useAppStore(state => state.setRecordingAssessing);
  const setRecordingJob = useAppStore((state) => state.setRecordingJob);
  const setReturnTo = useAppStore((state) => state.setReturnTo);
  const updateReview = useAppStore((state) => state.updateReview);
  const updateDraft = useAppStore((state) => state.updateDraft);

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
  const [chosenGoal, setChosenGoal] = useState("");
  useEffect(() => setChosenGoal(summary.nextFocus.slice(0, 12000)), [summary.reportId, summary.nextFocus]);
  const goalOptions = [...new Set([summary.nextFocus, ...summary.priorities].map(value => value.trim()).filter(Boolean))];
  const followUp = followUpBrief(summary, draft);
  const coachTakeaway = summary.coachSummary || translate("review.coach_takeaway_placeholder");

  const handleTryAgain = async () => {
    if (!await draftRecovery.prepareNewAttempt(draft.speakerId)) return;
    // A failed retry does not create new guidance. Keep the learner's existing
    // goal tied to its last assessable parent so the next take can continue it.
    const continueEarlierGoal = summary.eligibilityState !== "assessable" && followUp !== null;
    clearAttempt({ keepSetup: true });
    updateDraft({
      ...(continueEarlierGoal ? { retryOfSessionId: draft.retryOfSessionId } : {}),
      coachingGoal: summary.eligibilityState === "assessable" ? chosenGoal.trim() : continueEarlierGoal ? followUp.goal : "",
      retryGoalEligible: summary.eligibilityState === "assessable" || continueEarlierGoal,
    });
    try { await draftRecovery.saveSetup(appStore.getState().draft); } catch { /* Speak shows the local storage failure. */ }
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
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "var(--color-text)" }}>{translate("review.title")}</h2>
        <p style={{ margin: 0, color: "var(--color-text-muted)", lineHeight: 1.6 }}>
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

      <ReviewResult summary={summary} translate={translate} />
      {followUp && (
        <Card as="section" className={styles.followUp} data-testid="review-retry-follow-up">
          <p className={styles.eyebrow}>{translate("retry.follow_up_eyebrow")}</p>
          <h2>{translate("retry.follow_up_title")}</h2>
          <p>{translate("retry.goal_value", { goal: followUp.goal })}</p>
          {summary.eligibilityState !== "assessable" ? (
            <p data-testid="review-retry-incompatible">{translate("retry.not_assessable")}</p>
          ) : (
            <>
              <p data-testid="review-retry-observation">
                {followUp.observation
                  ? translate("retry.observation", { value: followUp.observation })
                  : summary.transcript
                    ? translate("retry.transcript_check", { value: summary.transcript.slice(0, 220) })
                    : translate("retry.no_direct_observation")}
              </p>
              {followUp.comparable ? (
                followUp.overallChange !== null ? <p data-testid="review-retry-change">
                  {translate("retry.overall_change", { value: `${followUp.overallChange >= 0 ? "+" : ""}${followUp.overallChange.toFixed(2)}` })}
                </p> : <p>{translate("retry.no_numeric_change")}</p>
              ) : <p data-testid="review-retry-incompatible">{translate("retry.incompatible")}</p>}
            </>
          )}
        </Card>
      )}
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
                color: "var(--color-brand)",
              }}
            >
              {translate("review.coach_note_eyebrow")}
            </p>
            <h2 style={{ margin: 0, fontSize: "1.5rem", color: "var(--color-text)" }}>
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
                    backgroundColor: "var(--color-surface)",
                    color: "var(--color-text-muted)",
                  }}
                  data-testid="review-next-step-exercise"
                  data-semantic-id="review-next-step-exercise"
                >
                  <Icon name="play" size={18} />
                  {translate("review.next_exercise", { value: summary.nextExercise })}
                </div>
              ) : null}
            </div>
            {summary.eligibilityState === "assessable" ? (
              <div className={styles.goalPicker} data-testid="review-goal-picker">
                <strong>{translate("retry.choose_goal")}</strong>
                <div className={styles.goalOptions} role="group" aria-label={translate("retry.choose_goal")}>
                  {goalOptions.map((option, index) => (
                    <Button key={option} variant={chosenGoal === option ? "primary" : "secondary"}
                      aria-pressed={chosenGoal === option} onClick={() => setChosenGoal(option.slice(0, 12000))}>
                      {index === 0 ? translate("retry.suggested_goal", { goal: option }) : option}
                    </Button>
                  ))}
                </div>
                <Field label={translate("retry.goal_label")} value={chosenGoal} maxLength={12000}
                  onChange={event => setChosenGoal(event.target.value)} hint={translate("retry.goal_hint")}
                  data-testid="review-goal-input" />
                <Button variant="quiet" onClick={() => setChosenGoal("")}>{translate("retry.clear_goal")}</Button>
              </div>
            ) : null}
            {summary.requiresHumanReview ? (
              <p
                role="status"
                style={{
                  margin: 0,
                  color: "var(--color-error)",
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
