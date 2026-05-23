import { useEffect, useMemo, useRef, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { AssessmentStatusPanel } from "@/components/speak/AssessmentStatusPanel";
import { RecorderPanel } from "@/components/speak/RecorderPanel";
import { apiClient, ApiClientError } from "@/lib/api/client";
import type {
  AssessmentStatusResponse,
  RuntimeResponse,
  RuntimeSettingsConnection,
  RuntimeSettingsResponse,
} from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { pollingIntervals, queryKeys } from "@/lib/query/queryClient";
import {
  resolveSpeakRouteGuardTarget,
  selectAssessmentLifecycleState,
  selectCanSubmitAssessment,
  useAppStore,
} from "@/lib/state/appStore";
import {
  hasRecordingAttachment,
  type RecordingInputMethod,
} from "@/lib/state/sessionDraft";

const DEFAULT_SCORING_WHISPER_MODEL = "large-v3";

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const detailGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
} as const;

const taskFamilyLabel = (
  translate: ReturnType<typeof createTranslator>,
  taskFamily: string,
): string => {
  const translated = translate(`task_family.${taskFamily}`);
  return translated.startsWith("[") ? taskFamily.replaceAll("_", " ") : translated;
};

const phaseMessage = (
  translate: ReturnType<typeof createTranslator>,
  phase: string,
): string => {
  if (!phase) {
    return "";
  }

  const translated = translate(`speak.job_phase_${phase}`);
  const resolvedPhase = translated.startsWith("[") ? phase.replaceAll("_", " ") : translated;
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
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const draft = useAppStore((state) => state.draft);
  const recording = useAppStore((state) => state.recording);
  const review = useAppStore((state) => state.review);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const updateRecording = useAppStore((state) => state.updateRecording);
  const updateRecordingInputs = useAppStore((state) => state.updateRecordingInputs);
  const setRecordingError = useAppStore((state) => state.setRecordingError);
  const setRecordingAssessing = useAppStore((state) => state.setRecordingAssessing);
  const setRecordingJob = useAppStore((state) => state.setRecordingJob);
  const clearRecording = useAppStore((state) => state.clearRecording);
  const updateReview = useAppStore((state) => state.updateReview);

  const translate = createTranslator(locale);
  const lifecycleState = useAppStore(selectAssessmentLifecycleState);
  const canSubmitAssessment = useAppStore(selectCanSubmitAssessment);

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
  });

  const runtimeSettingsQuery = useQuery({
    queryKey: ["runtime", "settings"],
    queryFn: () => apiClient.getRuntimeSettings(),
  });

  const effectiveWhisperModel =
    String(runtimeSettingsQuery.data?.whisper_model || "").trim() || DEFAULT_SCORING_WHISPER_MODEL;
  const activeRuntimeConnection = useMemo(
    () => selectRuntimeSettingsConnection(runtimeSettingsQuery.data),
    [runtimeSettingsQuery.data],
  );

  const effectivePreferences = useMemo(
    () => ({
      ...preferences,
      activeConnectionId: runtimeQuery.data?.configured
        ? preferences.activeConnectionId ||
          `${runtimeQuery.data.provider || "runtime"}:${runtimeQuery.data.model || "active"}`
        : preferences.activeConnectionId,
      setupComplete: Boolean(runtimeQuery.data?.configured) || preferences.setupComplete,
    }),
    [preferences, runtimeQuery.data],
  );

  const guardTarget = resolveSpeakRouteGuardTarget({
    draft,
    preferences: effectivePreferences,
  });

  useEffect(() => {
    setCurrentPage("speak");
  }, [setCurrentPage]);

  const [inputMode, setInputMode] = useState<RecordingInputMethod>(recording.inputMethod || "record");
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const previewUrlRef = useRef("");

  const clearAttachment = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }
    setPreviewUrl("");
    setAttachedFile(null);
    clearRecording({ preserveInputs: true });
  };

  useEffect(
    () => () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    },
    [],
  );

  const handleInputModeChange = (nextMode: RecordingInputMethod) => {
    if (nextMode === inputMode) {
      return;
    }
    clearAttachment();
    setInputMode(nextMode);
  };

  const handleFileSelected = (file: File | null) => {
    clearAttachment();
    if (!file) {
      return;
    }

    const nextPreviewUrl = URL.createObjectURL(file);
    previewUrlRef.current = nextPreviewUrl;
    setPreviewUrl(nextPreviewUrl);
    setAttachedFile(file);
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
  };

  const assessmentStatusQuery = useQuery({
    queryKey: queryKeys.assessment(recording.job.assessmentId || "pending"),
    queryFn: () => apiClient.getAssessmentStatus(recording.job.assessmentId),
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
    setRecordingJob({
      assessmentId: nextStatus.assessment_id,
      status: nextStatus.status,
      phase: nextStatus.phase,
      progress: nextStatus.progress,
      error: nextStatus.error?.detail ?? "",
      reportPath: nextStatus.report_path ?? "",
    });

    if (nextStatus.status === "completed" && nextStatus.payload) {
      updateReview(toReviewState(nextStatus));
      navigate("/review");
    }
  }, [assessmentStatusQuery.data, navigate, setRecordingJob, updateReview]);

  useEffect(() => {
    if (assessmentStatusQuery.error instanceof ApiClientError) {
      setRecordingError(assessmentStatusQuery.error.detail);
    }
  }, [assessmentStatusQuery.error, setRecordingError]);

  useEffect(() => {
    if (lifecycleState === "completed" && review.reportId) {
      navigate("/review");
    }
  }, [lifecycleState, navigate, review.reportId]);

  if (guardTarget) {
    return <Navigate replace to={guardTarget} />;
  }

  const runtime = runtimeQuery.data as RuntimeResponse | undefined;
  const isOpenRouterRuntime = String(runtime?.provider || "").trim().toLowerCase() === "openrouter";
  const statusMessage = buildStatusMessage({
    errorMessage: recording.error || recording.job.error,
    hasAttachment: hasRecordingAttachment(recording),
    hasMissingAttachment: Boolean(recording.audioPath) && !attachedFile,
    lifecycleState,
    model: runtime?.model || "-",
    provider: runtime?.provider || "-",
    requiresApiKey: Boolean(runtime?.requires_api_key),
    translate,
  });

  const handleSubmit = async () => {
    if (!attachedFile) {
      return;
    }

    try {
      setIsSubmitting(true);
      updateRecording({
        error: "",
        assessmentState: "idle",
      });
      const upload = await apiClient.uploadAudio(attachedFile);
      const openrouterHttpReferer = String(
        activeRuntimeConnection?.openrouter_http_referer || "",
      ).trim();
      const openrouterAppTitle = String(
        activeRuntimeConnection?.openrouter_app_title || "",
      ).trim();
      const created = await apiClient.createAssessment({
        audio_id: upload.audio_id,
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
        label: recording.labelInput || undefined,
        notes: recording.notesInput || undefined,
        llm_base_url: runtime?.base_url || undefined,
        ...(isOpenRouterRuntime
          ? {
              openrouter_http_referer: openrouterHttpReferer || undefined,
              openrouter_app_title: openrouterAppTitle || undefined,
            }
          : {}),
      });
      setRecordingAssessing({
        assessmentId: created.assessment_id,
        status: created.status,
        phase: created.status,
        progress: 0,
        error: "",
        reportPath: "",
      });
    } catch (error) {
      setRecordingError(
        error instanceof ApiClientError ? error.detail : String(error || translate("speak.job_status_unknown")),
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = async () => {
    if (!recording.job.assessmentId) {
      return;
    }

    try {
      const response = await apiClient.cancelAssessment(recording.job.assessmentId);
      setRecordingJob({
        assessmentId: response.assessment_id,
        status: response.status,
        phase: response.phase,
        progress: response.progress,
        error: translate("speak.job_status_cancelled"),
        reportPath: response.report_path ?? "",
      });
    } catch (error) {
      setRecordingError(
        error instanceof ApiClientError ? error.detail : String(error || translate("speak.job_status_unknown")),
      );
    }
  };

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <section style={cardStyle}>
        <h2
          style={{
            margin: 0,
            fontSize: "1.35rem",
            color: "#10201c",
          }}
        >
          {translate("speak.prompt_title")}
        </h2>
        <p
          style={{
            margin: 0,
            color: "#33514b",
            fontWeight: 600,
          }}
        >
          {draft.themeLabel || translate("speak.prompt_title")}
        </p>
        <blockquote
          style={{
            margin: 0,
            padding: "1rem",
            borderLeft: "3px solid rgba(15, 118, 110, 0.32)",
            backgroundColor: "rgba(248, 251, 250, 0.96)",
            color: "#10201c",
          }}
        >
          {draft.promptText || translate("speak.prompt_placeholder")}
        </blockquote>
        <div style={detailGridStyle}>
          {[
            [translate("setup.speaker_id"), draft.speakerId || "-"],
            [translate("setup.learning_language"), draft.learningLanguageLabel || "-"],
            [translate("setup.cefr"), draft.cefrLevel || "-"],
            [translate("setup.duration"), `${draft.durationSec} s`],
            [translate("history.task_family_name"), taskFamilyLabel(translate, draft.taskFamily)],
            [translate("setup.theme"), draft.themeLabel || "-"],
          ].map(([label, value]) => (
            <div key={label} style={cardStyle}>
              <strong
                style={{
                  color: "#33514b",
                }}
              >
                {label}
              </strong>
              <span
                style={{
                  color: "#10201c",
                }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
      </section>

      <div
        style={{
          display: "grid",
          gap: "1rem",
          gridTemplateColumns: "minmax(0, 1.08fr) minmax(0, 0.92fr)",
        }}
      >
        <RecorderPanel
          canRemove={Boolean(attachedFile)}
          inputMode={inputMode}
          onFileSelected={handleFileSelected}
          onInputModeChange={handleInputModeChange}
          onRemove={clearAttachment}
          previewUrl={previewUrl}
          statusMessage={statusMessage}
          statusTone={
            lifecycleState === "failed"
              ? "error"
              : lifecycleState === "cancelled"
                ? "warning"
                : hasRecordingAttachment(recording)
                  ? "success"
                  : "info"
          }
          translate={translate}
        />
        <AssessmentStatusPanel
          canCancel={lifecycleState === "queued" || lifecycleState === "running"}
          cefrSummary={draft.cefrLevel || "-"}
          labelValue={recording.labelInput}
          learningLanguageSummary={draft.learningLanguageLabel || "-"}
          lifecycleState={lifecycleState}
          model={runtime?.model || "-"}
          notesValue={recording.notesInput}
          onCancel={handleCancel}
          onLabelChange={(value) => updateRecordingInputs(value, recording.notesInput)}
          onNotesChange={(value) => updateRecordingInputs(recording.labelInput, value)}
          onSubmit={handleSubmit}
          phaseMessage={phaseMessage(translate, recording.job.phase)}
          provider={runtime?.provider || "-"}
          speakerSummary={draft.speakerId || "-"}
          statusMessage={isSubmitting ? translate("speak.assessing") : statusMessage}
          submitDisabled={!canSubmitAssessment || isSubmitting}
          targetDurationSummary={`${draft.durationSec} s`}
          translate={translate}
          warningMessage={
            runtime?.requires_api_key && !runtime.has_api_key
              ? translate("speak.openrouter_missing_key")
              : null
          }
          whisperModel={effectiveWhisperModel}
        />
      </div>
      {lifecycleState === "completed" ? (
        <span
          hidden
          data-testid="speak.review_auto_nav"
          data-semantic-id="speak.review_auto_nav"
        />
      ) : null}
    </div>
  );
};
