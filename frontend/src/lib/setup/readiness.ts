import type { ReadinessCheck, ReadinessResponse } from "@/lib/api/types";
import type { MicrophoneStatus } from "./microphone";

export type SetupReadinessStatus = "loading" | "ready" | "setup" | "unavailable";
export type SetupReadinessKey = ReadinessCheck["key"] | "microphone" | "sample_check";

export type SetupReadinessRow = {
  actionKey: string;
  backendReady?: boolean;
  blocking?: boolean;
  detailKey: string;
  disabled: boolean;
  key: SetupReadinessKey;
  rescueKey?: string;
  status: SetupReadinessStatus;
  titleKey: string;
};

export type SetupReadinessInput = {
  readiness?: ReadinessResponse;
  readinessError?: boolean;
  readinessPending?: boolean;
  microphoneStatus?: MicrophoneStatus;
  microphoneSetupPassed?: boolean;
};

export type SetupReadinessActionTarget =
  | { kind: "route"; value: "/session-setup" }
  | { kind: "section"; value: "runtime-setup-connection" | "runtime-setup-whisper" | "runtime-setup-microphone" }
  | { kind: "refresh" };

const backendKeys: ReadinessCheck["key"][] = [
  "speech_recognition", "ai_tutor", "media_decoder", "storage",
];

export const isCoreSetupReadinessKey = (key: SetupReadinessKey): boolean => key !== "sample_check";
export const isBlockingSetupReadinessRow = (row: SetupReadinessRow): boolean => row.blocking !== false;
export const canContinueWithCheck = (status: SetupReadinessStatus): boolean => status === "ready";

// The recording page performs its own microphone test. Backend prerequisites
// must already be available before entering it.
export const canEnterSpeak = (rows: SetupReadinessRow[]): boolean =>
  backendKeys.every((key) => rows.some((row) => row.key === key && row.backendReady === true &&
    (!row.blocking || canContinueWithCheck(row.status))));

export const canRecord = (rows: SetupReadinessRow[]): boolean =>
  canEnterSpeak(rows) && rows.some((row) => row.key === "microphone" && canContinueWithCheck(row.status));

export const resolveSetupReadinessAction = (key: SetupReadinessKey): SetupReadinessActionTarget => {
  if (key === "speech_recognition") return { kind: "section", value: "runtime-setup-whisper" };
  if (key === "ai_tutor") return { kind: "section", value: "runtime-setup-connection" };
  if (key === "microphone") return { kind: "section", value: "runtime-setup-microphone" };
  if (key === "media_decoder" || key === "storage") return { kind: "refresh" };
  return { kind: "route", value: "/session-setup" };
};

const reasonCopy: Record<ReadinessCheck["reason"], { detail: string; rescue?: string }> = {
  available: { detail: "" },
  not_configured: { detail: "first_run.readiness_not_configured", rescue: "first_run.readiness_configure" },
  credential_missing: { detail: "first_run.readiness_credential_missing", rescue: "first_run.readiness_add_credential" },
  local_service_unreachable: { detail: "first_run.readiness_service_unreachable", rescue: "first_run.readiness_start_service" },
  model_not_cached: { detail: "first_run.readiness_model_not_cached", rescue: "first_run.readiness_download_model" },
  model_unavailable: { detail: "first_run.readiness_model_unavailable", rescue: "first_run.readiness_select_model" },
  media_decoder_missing: { detail: "first_run.readiness_media_missing", rescue: "first_run.readiness_install_media" },
  storage_not_writable: { detail: "first_run.readiness_storage_unwritable", rescue: "first_run.readiness_fix_storage" },
};

const rowMeta: Record<ReadinessCheck["key"], { title: string; ready: string; action: string }> = {
  speech_recognition: {
    title: "runtime_setup.setup_guide_speech_title",
    ready: "runtime_setup.setup_guide_speech_ready",
    action: "runtime_setup.setup_guide_download_model",
  },
  ai_tutor: {
    title: "runtime_setup.setup_guide_ai_title",
    ready: "runtime_setup.setup_guide_ai_ready",
    action: "runtime_setup.setup_guide_connect_ai",
  },
  media_decoder: {
    title: "first_run.readiness_media_title",
    ready: "first_run.readiness_media_ready",
    action: "first_run.check_readiness",
  },
  storage: {
    title: "first_run.readiness_storage_title",
    ready: "first_run.readiness_storage_ready",
    action: "first_run.check_readiness",
  },
};

export const buildSetupReadinessRows = ({
  readiness,
  readinessError = false,
  readinessPending = false,
  microphoneStatus = "unknown",
  microphoneSetupPassed = false,
}: SetupReadinessInput): SetupReadinessRow[] => {
  const checks = new Map(readiness?.checks.map((check) => [check.key, check]));
  const backendReady = readiness?.ready === true && readiness.checks.every((check) =>
    !check.blocking || check.state === "available");
  const backendRows: SetupReadinessRow[] = backendKeys.map((key) => {
    const check = checks.get(key);
    const meta = rowMeta[key];
    const cloudSpeech = key === "speech_recognition" && check?.detail_args.provider === "groq";
    const status: SetupReadinessStatus = !readiness && readinessPending ? "loading"
      : !readiness && readinessError ? "unavailable"
        : check?.state === "available" ? "ready" : "setup";
    const copy = check ? reasonCopy[check.reason] : undefined;
    return {
      key, status,
      backendReady, blocking: check?.blocking ?? true,
      titleKey: meta.title,
      detailKey: status === "ready" ? cloudSpeech ? "first_run.cloud_speech_ready" : meta.ready : status === "loading"
        ? "runtime_setup.setup_guide_loading" : status === "unavailable"
          ? "first_run.readiness_backend_unavailable" : cloudSpeech
            ? "first_run.cloud_speech_missing" : copy?.detail || "first_run.readiness_not_configured",
      rescueKey: status === "setup" ? cloudSpeech
        ? "first_run.rescue_cloud_speech" : copy?.rescue || "first_run.readiness_configure"
        : status === "unavailable" ? "first_run.readiness_retry_backend" : undefined,
      actionKey: cloudSpeech ? status === "ready"
        ? "first_run.readiness_manage_cloud_speech" : "first_run.readiness_configure_cloud_speech" : meta.action,
      disabled: status === "loading",
    };
  });
  const microphoneReadiness: SetupReadinessStatus = microphoneSetupPassed && microphoneStatus === "ready"
    ? "ready" : ["unknown", "needs_review", "ready"].includes(microphoneStatus) ? "setup" : "unavailable";
  const microphoneRow: SetupReadinessRow = {
    key: "microphone", status: microphoneReadiness,
    titleKey: "runtime_setup.setup_guide_microphone_title",
    detailKey: microphoneReadiness === "ready" ? "runtime_setup.setup_guide_microphone_ready" : "runtime_setup.setup_guide_microphone_setup",
    rescueKey: microphoneReadiness === "unavailable" ? "first_run.rescue_microphone" : undefined,
    actionKey: "runtime_setup.setup_guide_check_microphone", disabled: false,
  };
  const requiredRows = [...backendRows, microphoneRow];
  const sampleReady = canRecord(requiredRows);
  const sampleLoading = !readiness && readinessPending;
  return [...requiredRows, {
    key: "sample_check", status: sampleLoading ? "loading" : sampleReady ? "ready" : "setup",
    titleKey: "runtime_setup.setup_guide_sample_title",
    detailKey: sampleLoading ? "runtime_setup.setup_guide_loading"
      : sampleReady ? "runtime_setup.setup_guide_sample_ready" : "runtime_setup.setup_guide_sample_blocked",
    actionKey: "runtime_setup.setup_guide_run_sample", disabled: !sampleReady,
  }];
};
