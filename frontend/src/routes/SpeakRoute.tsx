import { useSharingRoute, SharingSummary } from "@/lib/setup/sharing";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { AssessmentStatusPanel } from "@/components/speak/AssessmentStatusPanel";
import { RecorderPanel } from "@/components/speak/RecorderPanel";
import { SpeakStatusRail, type SpeakConfidencePhase } from "@/components/speak/SpeakStatusRail";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Field } from "@/components/ui/Field";
import { apiClient, ApiClientError } from "@/lib/api/client";
import type {
  AssessmentStatusResponse,
  RuntimeResponse,
  RuntimeSettingsConnection,
  RuntimeSettingsResponse,
} from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { MIN_REVIEW_SECONDS } from "@/lib/recordingPolicy";
import { capturedAudioSeconds } from "@/lib/setup/audioCapture";
import { useDraftRecovery } from "@/lib/state/DraftRecoveryProvider";
import { confirmLocalDraft } from "@/lib/runtime/confirmLocalDraft";
import { sameDraftAttempt } from "@/lib/state/draftStore";
import { clearRetryDraft, readRetryDraft } from "@/lib/state/retryBriefStorage";
import { hasFirstTakeMilestone, recordFirstTakeMilestone } from "@/lib/history/firstTakeMilestone";
import { StatusBanner } from "@/components/ui/StatusBanner";
import { pollingIntervals, queryKeys } from "@/lib/query/queryClient";
import { buildSetupReadinessRows, canEnterSpeak, isBlockingSetupReadinessRow } from "@/lib/setup/readiness";
import { useReadiness } from "@/lib/setup/useReadiness";
import {
  selectAssessmentLifecycleState,
  selectCanSubmitAssessment,
  useAppStore,
} from "@/lib/state/appStore";
import {
  hasSetupDraft,
  hasRecordingAttachment,
  type RecordingInputMethod,
} from "@/lib/state/sessionDraft";

const DEFAULT_SCORING_WHISPER_MODEL = "large-v3";

const isIndeterminateAssessmentResponse = (error: unknown): boolean =>
  error instanceof TypeError ||
  (error instanceof DOMException && ["AbortError", "NetworkError", "TimeoutError"].includes(error.name));

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid var(--color-border-decorative)",
  borderRadius: "8px",
  backgroundColor: "var(--color-surface)",
  boxShadow: "0 18px 40px var(--color-shadow-ambient)",
} as const;

const phaseMessage = (
  translate: ReturnType<typeof createTranslator>,
  phase: string,
  lifecycleState: string,
): string => {
  if (!phase) {
    return "";
  }

  const translated = translate(`speak.job_phase_${phase}`);
  const resolvedPhase = translated.startsWith("[") ? phase.replaceAll("_", " ") : translated;
  if (lifecycleState === "failed" || lifecycleState === "cancelled") {
    return ["failed", "cancelled", "done"].includes(phase) ? "" : translate("speak.wait_stopped_phase", { phase: resolvedPhase });
  }
  return translate("speak.job_phase", { phase: resolvedPhase });
};

const selectRuntimeSettingsConnection = (
  settings: RuntimeSettingsResponse | undefined,
): RuntimeSettingsConnection | null => {
  if (!settings) {
    return null;
  }
  const connections = settings.connections ?? [];
  return (
    connections.find((connection) => connection.connection_id === settings.active_connection_id) ??
    connections.find((connection) => connection.is_default) ??
    null
  );
};

const buildStatusMessage = ({
  hasAttachment,
  hasMissingAttachment,
  lifecycleState,
  provider,
  requiresApiKey,
  model,
  errorMessage,
  translate,
}: {
  errorMessage: string;
  hasAttachment: boolean;
  hasMissingAttachment: boolean;
  lifecycleState: string;
  model: string;
  provider: string;
  requiresApiKey: boolean;
  translate: ReturnType<typeof createTranslator>;
}): string => {
  if (lifecycleState === "queued") {
    return translate("speak.job_status_queued_provider", { provider, model });
  }

  if (lifecycleState === "running") {
    return translate(
      requiresApiKey ? "speak.job_status_running_remote_provider" : "speak.job_status_running_local_provider",
      { provider, model },
    );
  }

  if (lifecycleState === "failed" && errorMessage) {
    return translate("speak.assessment_error", { detail: errorMessage });
  }

  if (lifecycleState === "cancelled") {
    return translate("speak.job_status_cancelled");
  }

  if (hasMissingAttachment) {
    return translate("speak.status_missing_file");
  }

  if (hasAttachment) {
    return translate("speak.status_ready");
  }

  return translate("speak.status_idle");
};

const toReviewState = (
  response: AssessmentStatusResponse,
): {
  band: string;
  payload: Record<string, unknown>;
  reportId: string;
  scoreOverall: number | null;
  summary: string;
  transcript: string;
} => {
  const payload = (response.payload ?? {}) as Record<string, unknown>;
  const report = (payload.report ?? {}) as Record<string, unknown>;
  const transcriptPayload = (report.transcript ?? {}) as Record<string, unknown>;
  const summaryPayload = (report.coaching ?? {}) as Record<string, unknown>;

  return {
    band: response.summary?.band ?? String(report.band ?? ""),
    payload,
    reportId: response.report_path ?? response.assessment_id,
    scoreOverall: response.summary?.score_overall ?? null,
    summary: String(response.summary?.next_focus ?? summaryPayload.coach_summary ?? ""),
    transcript: String(transcriptPayload.text ?? report.transcript ?? ""),
  };
};

export const SpeakRoute = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const setMicrophoneStatus = useAppStore(state => state.setMicrophoneStatus);
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const draft = useAppStore((state) => state.draft);
  const retryRestorePending = useAppStore((state) => state.retryRestorePending);
  const confirmRetryRestore = useAppStore((state) => state.confirmRetryRestore);
  const discardRetryRestore = useAppStore((state) => state.discardRetryRestore);
  const recording = useAppStore((state) => state.recording);
  const review = useAppStore((state) => state.review);
  const updateDraft = useAppStore((state) => state.updateDraft);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const updateRecording = useAppStore((state) => state.updateRecording);
  const updateRecordingInputs = useAppStore((state) => state.updateRecordingInputs);
  const setRecordingError = useAppStore((state) => state.setRecordingError);
  const setRecordingAssessing = useAppStore((state) => state.setRecordingAssessing);
  const setRecordingJob = useAppStore((state) => state.setRecordingJob);
  const clearRecording = useAppStore((state) => state.clearRecording);
  const updateReview = useAppStore((state) => state.updateReview);
  const draftRecovery = useDraftRecovery();

  const translate = useMemo(() => createTranslator(locale), [locale]);
  const lifecycleState = useAppStore(selectAssessmentLifecycleState);
  const canSubmitAssessment = useAppStore(selectCanSubmitAssessment);
  const retryBriefHeadingRef = useRef<HTMLHeadingElement>(null);
  const focusedRetryBriefRef = useRef<string | null>(null);
  const microphoneSetupPassed = useAppStore(state => state.microphoneSetupPassed);
  const microphoneDeviceId = useAppStore(state => state.microphoneDeviceId);
  const microphoneVoiceProcessing = useAppStore(state => state.microphoneVoiceProcessing);

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
  });

  const runtimeSettingsQuery = useQuery({
    queryKey: ["runtime", "settings"],
    queryFn: () => apiClient.getRuntimeSettings(),
  });
  const readinessQuery = useReadiness();
  const speakingChecks = buildSetupReadinessRows({
    readiness: readinessQuery.data,
    readinessError: readinessQuery.isError && !readinessQuery.data,
    readinessPending: readinessQuery.isPending,
  });
  const speakingChecksPending = !readinessQuery.data && readinessQuery.isPending;

  const effectiveWhisperModel =
    String(runtimeSettingsQuery.data?.whisper_model || "").trim() || DEFAULT_SCORING_WHISPER_MODEL;
  const activeRuntimeConnection = useMemo(
    () => selectRuntimeSettingsConnection(runtimeSettingsQuery.data),
    [runtimeSettingsQuery.data],
  );

  useEffect(() => {
    setCurrentPage("speak");
  }, [setCurrentPage]);

  const [inputMode, setInputMode] = useState<RecordingInputMethod>(recording.inputMethod || "record");
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [recordedDurationSec, setRecordedDurationSec] = useState<number | null>(null);
  const [firstTakeJustRecorded, setFirstTakeJustRecorded] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [recordingActive, setRecordingActive] = useState(false);
  const [localSaveState, setLocalSaveState] = useState<"none" | "saving" | "saved" | "failed">(draftRecovery.recoveredFile ? "saved" : "none");
  const [localSaveProgress, setLocalSaveProgress] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadStartedAt, setUploadStartedAt] = useState<number | null>(null);
  const [uploadUpdatedAt, setUploadUpdatedAt] = useState<number | null>(null);
  const [uploadUnconfirmed, setUploadUnconfirmed] = useState(false);
  const [cancelError, setCancelError] = useState("");
  const [retryError, setRetryError] = useState("");
  const [retryResponseUnknown, setRetryResponseUnknown] = useState(false);
  const [pollError, setPollError] = useState("");
  const uploadController = useRef<AbortController | null>(null);
  const submittingRef = useRef(false);
  const retryingRef = useRef(false);
  const submission = useRef<{ signature: string; requestId: string } | null>(null);
  const savedUpload = useRef<{ file: File; audioId: string } | null>(null);
  const uploadIdentity = useRef<{ file: File; requestId: string } | null>(null);
  const previewUrlRef = useRef("");
  const saveInFlight = useRef(false);
  const activeMatches = sameDraftAttempt(draftRecovery.active, draft);
  const readinessBlocked = !speakingChecksPending && !canEnterSpeak(speakingChecks);
  const ownsTake = recordingActive || Boolean(attachedFile) || Boolean(draftRecovery.recoveredFile) || hasRecordingAttachment(recording) ||
    isSubmitting || uploadProgress !== null || lifecycleState === "queued" || lifecycleState === "running";
  const guardTarget = !hasSetupDraft(draft) ? "/session-setup"
    : readinessBlocked && !ownsTake ? "/runtime-setup" : null;
  const blockedReadinessCheck = speakingChecks.find((row) =>
    row.key !== "microphone" && row.key !== "sample_check" && isBlockingSetupReadinessRow(row) && row.status !== "ready");

  const clearAttachment = () => {
    savedUpload.current = null;
    uploadIdentity.current = null;
    setUploadUnconfirmed(false);
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }
    setPreviewUrl("");
    setAttachedFile(null);
    setRecordedDurationSec(null);
    setFirstTakeJustRecorded(false);
    setLocalSaveState("none");
    clearRecording({ preserveInputs: true });
  };

  useEffect(() => {
    const file = draftRecovery.recoveredFile;
    if (!file || !activeMatches) return;
    const url = URL.createObjectURL(file);
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = url;
    setPreviewUrl(url);
    setAttachedFile(file);
    setInputMode(draftRecovery.active?.take?.inputMethod || "record");
    setRecordedDurationSec(draftRecovery.active?.take?.durationSec ?? null);
    setLocalSaveState("saved");
    updateRecording({ status: "ready", ...(["queued", "running", "uncertain"].includes(lifecycleState) ? {} : { assessmentState: "idle" }), audioPath: url,
      durationSec: draftRecovery.active?.take?.durationSec ?? 0,
      inputDigest: `${file.name}:${file.size}:${file.lastModified}`,
      inputMethod: draftRecovery.active?.take?.inputMethod || "record", error: "" });
    draftRecovery.clearRecoveredFile();
  }, [draftRecovery.recoveredFile, activeMatches]);

  useEffect(() => {
    if (activeMatches && draftRecovery.active?.take && !draftRecovery.recoveredFile && !attachedFile) {
      void draftRecovery.reloadActiveTake();
    }
  }, [draftRecovery.active?.draftId, activeMatches]);

  const confirmReplace = () => !saveInFlight.current && (!draftRecovery.supported || (!attachedFile && !draftRecovery.active?.take) || confirmLocalDraft(translate("draft.replace_confirm")));
  const confirmRemove = () => !saveInFlight.current && (localSaveState !== "saved" || !draftRecovery.active?.take ||
    confirmLocalDraft(translate("draft.discard_saved_confirm", {
      learner: draftRecovery.active.setup.speakerId,
      task: draftRecovery.active.setup.themeLabel || draftRecovery.active.setup.promptText,
      size: `${(draftRecovery.active.take.size / 1024 / 1024).toFixed(1)} MiB`,
    })));

  const handleRemove = async (): Promise<boolean> => {
    if (saveInFlight.current) return false;
    try {
      if (draftRecovery.active?.take && activeMatches && localSaveState === "saved") {
        await draftRecovery.discardActive();
        await draftRecovery.saveSetup(draft);
      }
      clearAttachment();
      if (draftRecovery.active?.take && activeMatches && localSaveState !== "saved") {
        await draftRecovery.reloadActiveTake();
      }
      return true;
    } catch { setLocalSaveState("failed"); return false; }
  };

  useEffect(
    () => () => {
      uploadController.current?.abort();
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (!draftRecovery.supported || !attachedFile || (localSaveState !== "saving" && localSaveState !== "failed")) return;
    const beforeUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    const blockLink = (event: MouseEvent) => {
      const target = event.target;
      const link = target instanceof Element ? target.closest<HTMLAnchorElement>("a[href]:not([download])") : null;
      if (!link) return;
      event.preventDefault();
      if (localSaveState === "saving") return;
      const destination = link.getAttribute("href");
      void Promise.resolve(confirmLocalDraft(translate("draft.save_failed"))).then(confirmed => {
        if (!confirmed || !destination) return;
        if (destination.startsWith("/")) navigate(destination);
        else window.location.assign(link.href);
      });
    };
    window.addEventListener("beforeunload", beforeUnload);
    document.addEventListener("click", blockLink, true);
    return () => { window.removeEventListener("beforeunload", beforeUnload); document.removeEventListener("click", blockLink, true); };
  }, [attachedFile, localSaveState, draftRecovery.supported, navigate, translate]);

  const handleInputModeChange = (nextMode: RecordingInputMethod) => {
    if (nextMode === inputMode || saveInFlight.current) return;
    if (!draftRecovery.active?.take) clearAttachment();
    setInputMode(nextMode);
  };
  const saveRecordedTake = async (file: File, durationSec: number) => {
    if (!draftRecovery.supported || saveInFlight.current || durationSec < MIN_REVIEW_SECONDS) return;
    saveInFlight.current = true;
    setLocalSaveState("saving");
    try {
      if (!activeMatches) await draftRecovery.saveSetup(draft);
      await draftRecovery.saveTake(draft, file, durationSec, "record", setLocalSaveProgress);
      setLocalSaveState("saved");
    } catch { setLocalSaveState("failed"); }
    finally { saveInFlight.current = false; }
  };

  const handleFileSelected = async (file: File | null, durationSec?: number) => {
    if (!file || saveInFlight.current) return;
    if (file.size > 100 * 1024 * 1024 || file.size === 0) {
      setRecordingError(translate("speak.upload_size_error"));
      return;
    }
    clearAttachment();

    const nextPreviewUrl = URL.createObjectURL(file);
    previewUrlRef.current = nextPreviewUrl;
    setPreviewUrl(nextPreviewUrl);
    setAttachedFile(file);
    setRecordedDurationSec(durationSec ?? null);
    if (inputMode === "record" && durationSec !== undefined && !hasFirstTakeMilestone()) {
      recordFirstTakeMilestone();
      setFirstTakeJustRecorded(hasFirstTakeMilestone());
    }
    updateRecording({
      status: "ready",
      assessmentState: "idle",
      audioPath: nextPreviewUrl,
      inputDigest: `${file.name}:${file.size}:${file.lastModified}`,
      inputMethod: inputMode,
      error: "",
      job: {
        assessmentId: "",
        status: "",
        phase: "",
        progress: 0,
        error: "",
        reportPath: "",
      },
    });
    if (inputMode === "record" && draftRecovery.supported && durationSec !== undefined && durationSec >= MIN_REVIEW_SECONDS) {
      await saveRecordedTake(file, durationSec);
    }
  };

  const keepSelectedFile = async () => {
    if (!attachedFile || !draftRecovery.supported || inputMode !== "upload" || saveInFlight.current) return;
    saveInFlight.current = true;
    setLocalSaveState("saving");
    try {
      const duration = recordedDurationSec ?? await capturedAudioSeconds(attachedFile);
      if (!duration) throw new Error("Could not decode duration");
      if (!activeMatches) await draftRecovery.saveSetup(draft);
      await draftRecovery.saveTake(draft, attachedFile, duration, "upload", setLocalSaveProgress);
      setRecordedDurationSec(duration);
      setLocalSaveState("saved");
    } catch { setLocalSaveState("failed"); }
    finally { saveInFlight.current = false; }
  };

  const sharingSelection = { provider: runtimeQuery.data?.provider || "", llm_model: runtimeQuery.data?.model || "", llm_base_url: runtimeQuery.data?.base_url || "", whisper: effectiveWhisperModel };
  const sharingQuery = useSharingRoute(sharingSelection, Boolean(runtimeQuery.data?.configured));
  const assessmentStatusQuery = useQuery({
    queryKey: queryKeys.assessment(recording.job.assessmentId || "pending"),
    queryFn: ({ signal }) => apiClient.getAssessmentStatus(recording.job.assessmentId, { signal }),
    enabled:
      Boolean(recording.job.assessmentId) &&
      (lifecycleState === "queued" || lifecycleState === "running"),
    refetchInterval:
      lifecycleState === "queued" || lifecycleState === "running"
        ? pollingIntervals.assessmentMs
        : false,
  });

  useEffect(() => {
    if (!assessmentStatusQuery.data) {
      return;
    }

    const nextStatus = assessmentStatusQuery.data;
    setPollError("");
    if (nextStatus.status === "completed" || nextStatus.status === "failed" || nextStatus.status === "cancelled") {
      setCancelError("");
    }
    setRecordingJob({
      assessmentId: nextStatus.assessment_id,
      status: nextStatus.status,
      phase: nextStatus.status === "failed" || nextStatus.status === "cancelled" ? nextStatus.last_phase || nextStatus.phase : nextStatus.phase,
      progress: nextStatus.progress,
      error: nextStatus.error?.code === "recording_too_short" ? translate("speak.recording_too_short") : nextStatus.error?.detail ?? "",
      reportPath: nextStatus.report_path ?? "",
    });

    if (nextStatus.status === "completed" && nextStatus.payload) {
      updateReview(toReviewState(nextStatus));
    }
  }, [assessmentStatusQuery.data, navigate, setRecordingJob, updateReview, translate]);

  useEffect(() => {
    if (assessmentStatusQuery.error instanceof ApiClientError) {
      setPollError(assessmentStatusQuery.error.detail);
    } else if (assessmentStatusQuery.error) {
      setPollError(assessmentStatusQuery.error.message || translate("speak.job_status_unknown"));
    } else if (!assessmentStatusQuery.error) {
      setPollError("");
    }
  }, [assessmentStatusQuery.error, translate]);

  useEffect(() => {
    if (lifecycleState === "completed" && review.reportId) {
      void (async () => {
        const completedDraft = draftRecovery.active;
        if (sameDraftAttempt(completedDraft, draft)) {
          try {
            await draftRecovery.discardActive();
            const retryBrief = readRetryDraft();
            if (retryBrief && sameDraftAttempt(completedDraft, retryBrief)) clearRetryDraft();
          } catch { /* Review remains available if local cleanup fails. */ }
        }
        navigate("/review");
      })();
    }
  }, [lifecycleState, navigate, review.reportId]);

  useEffect(() => {
    const focusKey = `${draft.sessionId}:${draft.retryOfSessionId}`;
    if (draft.retryOfSessionId && draft.retryGoalEligible && !retryRestorePending &&
        !runtimeQuery.isPending && !runtimeSettingsQuery.isPending && !speakingChecksPending &&
        !guardTarget && lifecycleState !== "completed" &&
        focusedRetryBriefRef.current !== focusKey && retryBriefHeadingRef.current) {
      retryBriefHeadingRef.current.focus();
      focusedRetryBriefRef.current = focusKey;
    }
  }, [draft.sessionId, draft.retryOfSessionId, draft.retryGoalEligible, retryRestorePending,
    runtimeQuery.isPending, runtimeSettingsQuery.isPending, speakingChecksPending, guardTarget, lifecycleState]);

  if (lifecycleState === "completed") {
    return <Card as="section" role="status" data-testid="speak-completed-no-report">
        <h2>{translate("review.status_done")}</h2>
        {review.reportId ? <p>{translate("journey.loading")}</p> : <Link to="/history">{translate("review.view_history")}</Link>}
      </Card>;
  }

  if (retryRestorePending) {
    return <Card as="section" data-testid="speak-retry-resume" style={{ display: "grid", gap: "var(--space-3)" }}>
      <h2>{translate("retry.resume_title")}</h2>
      <p>{translate("retry.resume_context", { speaker: draft.speakerId, task: draft.themeLabel || draft.promptText })}</p>
      <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
        <Button variant="primary" onClick={confirmRetryRestore}>{translate("retry.resume_action")}</Button>
        <Button onClick={() => { discardRetryRestore(); navigate("/session-setup"); }}>{translate("retry.discard_action")}</Button>
      </div>
    </Card>;
  }

  if (runtimeQuery.isPending || runtimeSettingsQuery.isPending || speakingChecksPending) {
    return draft.retryOfSessionId ? <Card as="section" role="status" data-testid="speak-retry-runtime-check">
      <h2>{translate("retry.brief_title")}</h2>
      {draft.retryGoalEligible && <p>{draft.coachingGoal.trim()
        ? translate("retry.goal_value", { goal: draft.coachingGoal.trim() })
        : translate("retry.no_goal")}</p>}
      <p>{translate("retry.runtime_check")}</p>
    </Card> : <p role="status">{translate("journey.loading")}</p>;
  }
  if (guardTarget) {
    return <Navigate replace to={guardTarget} state={guardTarget === "/runtime-setup" ? { returnTo: "/speak" } : undefined} />;
  }

  const runtime = runtimeQuery.data as RuntimeResponse | undefined;
  const isOpenRouterRuntime = String(runtime?.provider || "").trim().toLowerCase() === "openrouter";
  const storeHasAttachment = hasRecordingAttachment(recording);
  const assessmentOwnsAttachment =
    lifecycleState === "queued" || lifecycleState === "running";
  const hasAttachment = assessmentOwnsAttachment ? storeHasAttachment : Boolean(attachedFile);
  const shortRecording = recordedDurationSec !== null && recordedDurationSec < MIN_REVIEW_SECONDS;
  const canSubmitLiveAttachment = !readinessBlocked && !recordingActive && canSubmitAssessment && Boolean(attachedFile) &&
    localSaveState !== "saving" && !shortRecording && Boolean(sharingQuery.data?.available);
  const statusMessage = cancelError ? translate("speak.wait_cancel_failed", { detail: cancelError })
    : retryError ? translate(retryResponseUnknown && !recording.job.assessmentId ? "draft.submit_uncertain"
      : retryResponseUnknown ? "speak.wait_retry_unconfirmed" : "speak.wait_retry_not_ready", { detail: retryError })
    : pollError ? translate("speak.wait_status_failed", { detail: pollError })
    : attachedFile && draftRecovery.supported && lifecycleState === "idle" && !isSubmitting && localSaveState === "saving" ? translate("draft.save_progress", { percent: localSaveProgress })
    : shortRecording ? translate("speak.recording_too_short")
    : attachedFile && draftRecovery.supported && lifecycleState === "idle" && !isSubmitting && localSaveState === "saved" ? translate("draft.saved_on_mac")
    : attachedFile && draftRecovery.supported && lifecycleState === "idle" && !isSubmitting && localSaveState === "failed" ? translate("draft.save_failed")
    : buildStatusMessage({
    errorMessage: recording.error || recording.job.error,
    hasAttachment,
    hasMissingAttachment: storeHasAttachment && !attachedFile && !assessmentOwnsAttachment,
    lifecycleState,
    model: runtime?.model || "-",
    provider: runtime?.provider || "-",
    requiresApiKey: Boolean(runtime?.requires_api_key),
    translate,
  });
  const sessionSummary = translate("speak.session_summary", {
    language: draft.learningLanguageLabel || "-",
    cefr: draft.cefrLevel || "-",
    duration: `${draft.durationSec} s`,
    speaker: draft.speakerId || "-",
  });
  const runtimeDetail = translate("speak.runtime_detail", {
    provider: runtime?.provider || "-",
    model: runtime?.model || "-",
    whisper: effectiveWhisperModel,
  });
  const assessmentContextMode = hasAttachment ? "optional" : "inline";
  const submitDisabledHelp = readinessBlocked ? translate("first_run.readiness_changed_body")
    : shortRecording ? translate("speak.recording_too_short")
      : !hasAttachment ? translate("speak.submit_disabled_no_audio") : null;
  const confidencePhase: SpeakConfidencePhase =
    lifecycleState === "failed" || lifecycleState === "cancelled"
      ? "failed"
      : lifecycleState === "queued" || lifecycleState === "running"
        ? "assess"
        : hasAttachment
          ? "submit"
          : "record";
  const handoffHint =
    shortRecording
      ? translate("speak.recording_too_short")
      : lifecycleState === "cancelled"
      ? translate("speak.handoff_cancelled")
      : lifecycleState === "failed"
        ? translate("speak.handoff_failed")
        : lifecycleState === "queued" || lifecycleState === "running"
          ? translate("speak.handoff_assessing")
          : isSubmitting
            ? translate("speak.handoff_submitting")
            : hasAttachment
              ? translate("speak.handoff_ready")
              : translate("speak.handoff_idle");

  const uploaded = Boolean(savedUpload.current || recording.job.assessmentId);
  const mediaMessage = recording.job.assessmentId
    ? translate("speak.wait_media_retained")
    : savedUpload.current
      ? translate(isSubmitting ? "speak.wait_media_uploaded" : "speak.wait_media_uploaded_ready")
      : uploadUnconfirmed
        ? translate("speak.wait_media_upload_unconfirmed")
      : attachedFile
        ? translate("speak.wait_media_selected")
        : "";

  const handleSubmit = async (resumeAssessmentId?: string) => {
    if (readinessBlocked || recordingActive || saveInFlight.current || submittingRef.current || !attachedFile || shortRecording || !sharingQuery.data?.available) {
      return;
    }

    let uploadAttempted = false;
    let assessmentRequestAttempted = false;
    const priorTerminalJob = recording.job.assessmentId && (recording.job.status === "failed" || recording.job.status === "cancelled")
      ? recording.job : null;
    try {
      submittingRef.current = true;
      setCancelError("");
      setRetryError("");
      setRetryResponseUnknown(false);
      setPollError("");
      setIsSubmitting(true);
      const refreshed = await sharingQuery.refetch();
      if (!refreshed.data?.available || refreshed.data.fingerprint !== sharingQuery.data?.fingerprint) {
        setRecordingError(translate("sharing.changed"));
        await Promise.all([runtimeQuery.refetch(), runtimeSettingsQuery.refetch()]); return;
      }
      updateRecording(priorTerminalJob ? { error: "" } : { error: "", assessmentState: "idle" });
      if (savedUpload.current?.file !== attachedFile) {
        const replayingUpload = uploadUnconfirmed && uploadIdentity.current?.file === attachedFile;
        if (uploadIdentity.current?.file !== attachedFile) uploadIdentity.current = { file: attachedFile, requestId: crypto.randomUUID() };
        const controller = new AbortController();
        uploadController.current = controller;
        const startedAt = Date.now();
        setUploadStartedAt(startedAt);
        setUploadUpdatedAt(startedAt);
        setUploadProgress(0);
        const limits = await apiClient.getUploadLimits({ signal: controller.signal });
        if (attachedFile.size > limits.max_bytes) {
          throw new Error(translate("speak.upload_size_error"));
        }
        if (!replayingUpload && attachedFile.size > limits.available_bytes) {
          throw new Error(translate("speak.upload_disk_error"));
        }
        uploadAttempted = true;
        const upload = await apiClient.uploadAudio(attachedFile, { signal: controller.signal,
          requestId: uploadIdentity.current.requestId,
          onProgress: (percent) => { setUploadProgress(percent); setUploadUpdatedAt(Date.now()); } });
        savedUpload.current = { file: attachedFile, audioId: upload.audio_id };
        uploadAttempted = false;
        setUploadUnconfirmed(false);
        uploadController.current = null;
        setUploadProgress(null);
      }
      const openrouterHttpReferer = String(
        activeRuntimeConnection?.openrouter_http_referer || "",
      ).trim();
      const openrouterAppTitle = String(
        activeRuntimeConnection?.openrouter_app_title || "",
      ).trim();
      const request = {
        sharing_fingerprint: sharingQuery.data.fingerprint,
        ...((resumeAssessmentId || (recording.job.assessmentId && ["failed", "cancelled"].includes(recording.job.status) ? recording.job.assessmentId : ""))
          ? { resume_assessment_id: resumeAssessmentId || recording.job.assessmentId } : {}),
        audio_id: savedUpload.current.audioId,
        whisper: effectiveWhisperModel,
        provider: runtime?.provider || "",
        llm_model: runtime?.model || "",
        expected_language: draft.learningLanguage,
        feedback_language: locale,
        speaker_id: draft.speakerId,
        task_family: draft.taskFamily,
        theme: draft.themeLabel,
        target_duration_sec: draft.durationSec,
        target_cefr: draft.cefrLevel,
        prompt_id: draft.promptId,
        prompt_text: draft.promptText,
        retry_of_session_id: draft.retryOfSessionId,
        label: recording.labelInput || undefined,
        notes: recording.notesInput || undefined,
        llm_base_url: runtime?.base_url || undefined,
        ...(isOpenRouterRuntime
          ? {
              openrouter_http_referer: openrouterHttpReferer || undefined,
              openrouter_app_title: openrouterAppTitle || undefined,
            }
          : {}),
      };
      const signature = JSON.stringify(request);
      if (submission.current?.signature !== signature) submission.current = { signature, requestId: crypto.randomUUID() };
      const body = { ...request, request_id: submission.current.requestId };
      assessmentRequestAttempted = true;
      const created = await apiClient.createAssessment(body);
      setRecordingAssessing({
        assessmentId: created.assessment_id,
        status: created.status,
        phase: created.status,
        progress: 0,
        error: "",
        reportPath: "",
      });
    } catch (error) {
      if (uploadAttempted && (error instanceof TypeError ||
          (error instanceof DOMException && (error.name === "NetworkError" || error.name === "TimeoutError")))) {
        setUploadUnconfirmed(true);
      }
      if (error instanceof ApiClientError && error.code === "sharing_changed") {
        await Promise.all([runtimeQuery.refetch(), runtimeSettingsQuery.refetch(), sharingQuery.refetch()]);
      }
      if (error instanceof ApiClientError && error.responseStatus === 409 && recording.job.assessmentId &&
          (recording.job.status === "failed" || recording.job.status === "cancelled")) {
        if (priorTerminalJob) setRecordingJob(priorTerminalJob);
        setRetryError(error.detail);
        return;
      }
      if (error instanceof ApiClientError && error.responseStatus === 404) savedUpload.current = null;
      if (assessmentRequestAttempted && isIndeterminateAssessmentResponse(error)) {
        if (priorTerminalJob) setRecordingJob(priorTerminalJob);
        else updateRecording({ assessmentState: "uncertain" });
        setRetryResponseUnknown(true);
        setRetryError(error instanceof Error || error instanceof DOMException
          ? error.message || translate("speak.job_status_unknown")
          : translate("speak.job_status_unknown"));
        return;
      }
      setRecordingError(
        error instanceof DOMException && error.name === "AbortError"
          ? translate("speak.upload_cancelled")
          : error instanceof DOMException && error.name === "NetworkError" ? translate("speak.upload_connection_lost")
          : error instanceof DOMException && error.name === "TimeoutError" ? translate("speak.upload_timeout")
          : error instanceof ApiClientError ? (error.code === "sharing_changed" ? translate("sharing.changed") : error.detail) : error instanceof Error ? error.message : String(error || translate("speak.job_status_unknown")),
      );
    } finally {
      submittingRef.current = false;
      setIsSubmitting(false);
      setUploadProgress(null);
      uploadController.current = null;
    }
  };

  const handleRetry = async () => {
    if (recordingActive || saveInFlight.current || retryingRef.current || submittingRef.current || !attachedFile || !recording.job.assessmentId ||
        (lifecycleState !== "failed" && lifecycleState !== "cancelled")) return;
    retryingRef.current = true;
    setCancelError("");
    try {
      await handleSubmit(recording.job.assessmentId || undefined);
    } finally {
      retryingRef.current = false;
    }
  };

  const handleCancel = async () => {
    if (uploadController.current) {
      uploadController.current.abort();
      return;
    }
    if (!recording.job.assessmentId) {
      return;
    }

    try {
      setCancelError("");
      const response = await apiClient.cancelAssessment(recording.job.assessmentId);
      await queryClient.cancelQueries({ queryKey: queryKeys.assessment(response.assessment_id) });
      queryClient.setQueryData(queryKeys.assessment(response.assessment_id), response);
      setPollError("");
      setRecordingJob({
        assessmentId: response.assessment_id,
        status: response.status,
        phase: response.status === "failed" || response.status === "cancelled" ? response.last_phase || response.phase : response.phase,
        progress: response.progress,
        error: response.status === "failed" ? response.error?.detail ?? ""
          : response.status === "cancelled" ? translate("speak.job_status_cancelled") : "",
        reportPath: response.report_path ?? "",
      });
    } catch (error) {
      setCancelError(error instanceof ApiClientError ? error.detail : String(error || translate("speak.job_status_unknown")));
    }
  };

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {draftRecovery.supported && draftRecovery.error ? (
        <StatusBanner title={translate(draftRecovery.error.kind === "keychain" ? "draft.keychain_error"
          : draftRecovery.error.kind === "limit" ? "draft.limit_error"
          : draftRecovery.error.kind === "format" ? "draft.format_error"
          : draftRecovery.error.kind === "uncertain" ? "draft.save_uncertain" : "draft.storage_error")} tone="warning">
          <p>{translate(draftRecovery.error.kind === "uncertain" ? "draft.save_uncertain" : "draft.save_failed")}</p>
          <Button onClick={draftRecovery.manageDrafts}>{translate("draft.manage_saved")}</Button>
        </StatusBanner>
      ) : null}
      {readinessBlocked && ownsTake ? (
        <div data-testid="speak.readiness_changed">
          <StatusBanner title={translate("first_run.readiness_changed_title")} tone="warning">
            <p>{translate(blockedReadinessCheck?.detailKey ?? "first_run.readiness_backend_unavailable")}</p>
            <p>{translate(blockedReadinessCheck?.rescueKey ?? "first_run.readiness_retry_backend")}</p>
            <p>{translate("first_run.readiness_changed_body")}</p>
            <Button onClick={() => void readinessQuery.refetch()}>{translate("first_run.check_readiness")}</Button>
          </StatusBanner>
        </div>
      ) : null}
      <section style={cardStyle}>
        <h2
          style={{
            margin: 0,
            fontSize: "1.45rem",
            color: "var(--color-text)",
          }}
        >
          {translate("speak.prompt_title")}
        </h2>
        <p
          style={{
            margin: 0,
            color: "var(--color-text-muted)",
            fontWeight: 600,
          }}
        >
          {draft.themeLabel || translate("speak.prompt_title")}
        </p>
        <p
          style={{
            margin: 0,
            color: "var(--color-text-muted)",
            fontWeight: 600,
          }}
          data-testid="speak.session_summary"
          data-semantic-id="speak.session_summary"
        >
          {sessionSummary}
        </p>
        <blockquote
          style={{
            margin: 0,
            padding: "1rem",
            borderLeft: "4px solid var(--color-selected-border)",
            borderRadius: "0 8px 8px 0",
            backgroundColor: "var(--color-surface-subtle)",
            color: "var(--color-text)",
            fontSize: "1.05rem",
            lineHeight: 1.55,
            fontWeight: 600,
          }}
        >
          {draft.promptText || translate("speak.prompt_placeholder")}
        </blockquote>
      </section>

      {draft.retryOfSessionId && draft.retryGoalEligible && (
        <Card as="section" data-testid="speak-retry-brief" style={{ display: "grid", gap: "var(--space-3)" }}>
          <h2 ref={retryBriefHeadingRef} tabIndex={-1} style={{ margin: 0, fontSize: "var(--font-size-title)" }}>{translate("retry.brief_title")}</h2>
          <p style={{ margin: 0 }}>{translate("retry.brief_context", { task: draft.themeLabel || draft.promptText })}</p>
          <p style={{ margin: 0 }} data-testid="speak-retry-goal-text">
            {draft.coachingGoal.trim()
              ? translate("retry.goal_value", { goal: draft.coachingGoal.trim() })
              : translate("retry.no_goal")}
          </p>
          <Field label={translate("retry.goal_label")} value={draft.coachingGoal} maxLength={12000}
            disabled={recordingActive || isSubmitting || lifecycleState === "queued" || lifecycleState === "running"}
            onChange={event => {
              const coachingGoal = event.target.value;
              updateDraft({ coachingGoal });
              if (draftRecovery.supported) void draftRecovery.saveSetup({ ...draft, coachingGoal }).catch(() => undefined);
            }}
            hint={translate("retry.brief_hint")} data-testid="speak-retry-goal" />
          <Button variant="quiet" disabled={recordingActive || isSubmitting || lifecycleState === "queued" || lifecycleState === "running"}
            onClick={() => {
              updateDraft({ coachingGoal: "" });
              if (draftRecovery.supported) void draftRecovery.saveSetup({ ...draft, coachingGoal: "" }).catch(() => undefined);
            }}>
            {translate("retry.clear_goal")}
          </Button>
        </Card>
      )}

      <SpeakStatusRail phase={confidencePhase} translate={translate} />

      {firstTakeJustRecorded && recordedDurationSec !== null && inputMode === "record" ? (
        <div data-testid="speak.first_take_milestone">
          <StatusBanner title={translate("first_run.milestone_title")} tone="success">
            {translate("first_run.milestone_body")}
          </StatusBanner>
        </div>
      ) : null}

      <div
        style={{
          display: "grid",
          gap: "1rem",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 22rem), 1fr))",
          alignItems: "start",
        }}
      >
        <fieldset disabled={isSubmitting || localSaveState === "saving" || lifecycleState === "queued" || lifecycleState === "running"} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
        <p data-testid="speak.minimum_duration">{translate("speak.recording_too_short")}</p>
        <SharingSummary route={sharingQuery.data} locale={locale} />
      <RecorderPanel onMicrophoneStatusChange={setMicrophoneStatus}
          onRecordingActiveChange={setRecordingActive}
          microphoneSetupPassed={microphoneSetupPassed}
          microphoneDeviceId={microphoneDeviceId}
          microphoneVoiceProcessing={microphoneVoiceProcessing}
          onSetupMicrophone={() => navigate("/runtime-setup#runtime-setup-microphone")}
          canRemove={Boolean(attachedFile)}
          downloadName={attachedFile?.name}
          inputMode={inputMode}
          onFileSelected={handleFileSelected}
          onBeforeReplace={confirmReplace}
          onBeforeRemove={confirmRemove}
          onInputModeChange={handleInputModeChange}
          onRemove={() => { void handleRemove(); }}
          previewUrl={previewUrl}
          preferStatusMessage={draftRecovery.supported && localSaveState === "saved" && lifecycleState === "idle" && !isSubmitting}
          showReadyCheckpoint={(!draftRecovery.supported || localSaveState === "saved") && hasAttachment && lifecycleState === "idle" && !isSubmitting}
          statusMessage={statusMessage}
          statusTone={
            cancelError || retryError || pollError || localSaveState === "failed" || lifecycleState === "failed"
              ? "error"
              : lifecycleState === "cancelled"
                ? "warning"
                : shortRecording ? "warning"
                : localSaveState === "saved" || (!draftRecovery.supported && hasAttachment)
                  ? "success"
                  : "info"
          }
          translate={translate}
        />
        {!draftRecovery.supported && attachedFile ? <p role="note">{translate("draft.browser_unsupported")}</p> : null}
        {draftRecovery.supported && inputMode === "upload" && attachedFile && localSaveState !== "saved" && localSaveState !== "saving" ? (
          <div style={{ display: "grid", gap: "var(--space-2)" }}>
            <p>{translate("draft.save_upload_note")}</p>
            <Button onClick={() => void keepSelectedFile()}>{translate("draft.save_upload_action")}</Button>
          </div>
        ) : null}
        {draftRecovery.supported && inputMode === "record" && attachedFile && localSaveState === "failed" && recordedDurationSec !== null ? (
          <Button onClick={() => void saveRecordedTake(attachedFile, recordedDurationSec)}>{translate("draft.retry_save_action")}</Button>
        ) : null}
        </fieldset>
        <AssessmentStatusPanel
          assessmentStatus={lifecycleState === "queued" || lifecycleState === "running" ? assessmentStatusQuery.data : undefined}
          pollFailed={Boolean(pollError)}
          uploadStartedAt={uploadStartedAt}
          uploadUpdatedAt={uploadUpdatedAt}
          uploading={uploadProgress !== null}
          uploaded={uploaded}
          canRetry={!readinessBlocked && Boolean(recording.job.assessmentId) && Boolean(attachedFile) && !shortRecording && Boolean(sharingQuery.data?.available) && (lifecycleState === "failed" || lifecycleState === "cancelled")}
          retryDisabled={recordingActive || localSaveState === "saving" || isSubmitting}
          mediaMessage={mediaMessage}
          canCancel={uploadProgress !== null || lifecycleState === "queued" || lifecycleState === "running"}
          contextMode={assessmentContextMode}
          handoffHint={handoffHint}
          labelValue={recording.labelInput}
          lifecycleState={lifecycleState}
          notesValue={recording.notesInput}
          onCancel={handleCancel}
          onRetry={handleRetry}
          onLabelChange={(value) => updateRecordingInputs(value, recording.notesInput)}
          onNotesChange={(value) => updateRecordingInputs(recording.labelInput, value)}
          onSubmit={() => void handleSubmit()}
          phaseMessage={phaseMessage(translate, recording.job.phase, lifecycleState)}
          runtimeDetail={runtimeDetail}
          statusMessage={uploadProgress !== null ? translate("speak.upload_progress", { percent: uploadProgress }) : isSubmitting ? translate("speak.assessing") : statusMessage}
          submitDisabled={!canSubmitLiveAttachment || isSubmitting}
          submitDisabledHelp={submitDisabledHelp}
          translate={translate}
          warningMessage={
            runtime?.requires_api_key && !runtime.has_api_key
              ? translate(runtime.provider === "chatgpt" || runtime.provider === "xai" || runtime.provider === "groq" ? "chatgpt.missing_access" : "speak.openrouter_missing_key")
              : null
          }
        />
      </div>
    </div>
  );
};
