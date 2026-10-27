import type { MicrophoneStatus } from "./microphone";
import type {
  DiagnosticItem,
  RuntimeResponse,
} from "@/lib/api/types";

export type SetupReadinessStatus = "loading" | "ready" | "configured" | "setup" | "unavailable";

export type SetupReadinessKey =
  | "speech_recognition"
  | "ai_tutor"
  | "microphone"
  | "sample_check";

export type SetupReadinessRow = {
  actionKey: string;
  detailKey: string;
  disabled: boolean;
  key: SetupReadinessKey;
  rescueKey?: string;
  status: SetupReadinessStatus;
  titleKey: string;
};

export type SetupReadinessInput = {
  diagnostics: DiagnosticItem[];
  microphoneStatus?: MicrophoneStatus;
  microphoneSetupPassed?: boolean;
  diagnosticsError?: boolean;
  diagnosticsPending?: boolean;
  runtime?: RuntimeResponse;
  runtimeError?: boolean;
  runtimePending?: boolean;
  connectionTestPassed?: boolean;
  cloudSpeechSelected?: boolean;
  cloudSpeechConnected?: boolean;
  cloudSpeechPending?: boolean;
  cloudSpeechError?: boolean;
  whisperCached?: boolean;
  whisperChecked?: boolean;
  whisperPending?: boolean;
};

export type SetupReadinessActionTarget =
  | { kind: "route"; value: "/session-setup" }
  | { kind: "section"; value: "runtime-setup-connection" | "runtime-setup-whisper" | "runtime-setup-microphone" };

export const isCoreSetupReadinessKey = (key: SetupReadinessKey): boolean =>
  key === "speech_recognition" || key === "ai_tutor" || key === "microphone";

export const canContinueWithCheck = (status: SetupReadinessStatus): boolean =>
  status === "ready" || status === "configured";

export const resolveSetupReadinessAction = (
  key: SetupReadinessKey,
): SetupReadinessActionTarget => {
  if (key === "speech_recognition") {
    return { kind: "section", value: "runtime-setup-whisper" };
  }
  if (key === "ai_tutor") {
    return { kind: "section", value: "runtime-setup-connection" };
  }
  if (key === "microphone") return { kind: "section", value: "runtime-setup-microphone" };
  return { kind: "route", value: "/session-setup" };
};

const findDiagnostic = (items: DiagnosticItem[], key: string): DiagnosticItem | undefined =>
  items.find((item) => item.key === key);

const statusFromDiagnostic = (item: DiagnosticItem | undefined): SetupReadinessStatus => {
  if (!item) {
    return "setup";
  }
  if (item.status === "ok") {
    return "ready";
  }
  if (item.status === "error") {
    return "unavailable";
  }
  return "setup";
};

export const buildSetupReadinessRows = ({
  diagnostics,
  microphoneStatus = "unknown",
  microphoneSetupPassed = false,
  diagnosticsError = false,
  diagnosticsPending = false,
  runtime,
  runtimeError = false,
  runtimePending = false,
  connectionTestPassed = false,
  cloudSpeechSelected = false,
  cloudSpeechConnected = false,
  cloudSpeechPending = false,
  cloudSpeechError = false,
  whisperCached = false,
  whisperChecked = false,
  whisperPending = false,
}: SetupReadinessInput): SetupReadinessRow[] => {
  const allPending = diagnosticsPending || runtimePending || whisperPending || cloudSpeechPending;
  if (allPending) {
    return [
      {
        key: "speech_recognition",
        status: "loading",
        titleKey: "runtime_setup.setup_guide_speech_title",
        detailKey: "runtime_setup.setup_guide_loading",
        actionKey: "runtime_setup.setup_guide_download_model",
        disabled: true,
      },
      {
        key: "ai_tutor",
        status: "loading",
        titleKey: "runtime_setup.setup_guide_ai_title",
        detailKey: "runtime_setup.setup_guide_loading",
        actionKey: "runtime_setup.setup_guide_connect_ai",
        disabled: true,
      },
      {
        key: "microphone",
        status: "loading",
        titleKey: "runtime_setup.setup_guide_microphone_title",
        detailKey: "runtime_setup.setup_guide_loading",
        actionKey: "runtime_setup.setup_guide_check_microphone",
        disabled: true,
      },
      {
        key: "sample_check",
        status: "loading",
        titleKey: "runtime_setup.setup_guide_sample_title",
        detailKey: "runtime_setup.setup_guide_loading",
        actionKey: "runtime_setup.setup_guide_run_sample",
        disabled: true,
      },
    ];
  }

  const whisperItem = findDiagnostic(diagnostics, "whisper");
  const runtimeItem = findDiagnostic(diagnostics, "runtime");
  const credentialItem = findDiagnostic(diagnostics, "runtime_api_key");
  const credentialDiagnosticMatchesRuntime = Boolean(
    runtime?.configured &&
    credentialItem?.detail_args.provider === runtime.provider &&
    credentialItem?.detail_args.model === runtime.model &&
    credentialItem?.detail_args.base_url === runtime.base_url,
  );
  const credentialMissing = Boolean(
    runtime?.requires_api_key &&
    (!runtime.has_api_key || credentialDiagnosticMatchesRuntime && credentialItem?.status === "error"),
  );

  const speechStatus =
    diagnosticsError || cloudSpeechError ? "unavailable"
      : cloudSpeechSelected ? (cloudSpeechConnected ? "ready" : "setup")
        : whisperChecked ? (whisperCached ? "ready" : "setup")
          : whisperCached || whisperItem?.status === "ok" ? "ready"
            : whisperItem ? statusFromDiagnostic(whisperItem) : runtime?.configured ? "configured" : "setup";
  const aiStatus =
    diagnosticsError || runtimeError
      ? "unavailable"
      : credentialMissing || runtimeItem?.status === "error"
        ? "unavailable"
        : runtime?.configured
          ? runtime.requires_api_key && runtime.has_api_key || connectionTestPassed ? "ready" : "configured"
          : runtimeItem?.status === "error" ? "unavailable" : "setup";
  const microphoneReadiness = microphoneSetupPassed && microphoneStatus === "ready" ? "ready"
    : ["unknown", "needs_review", "ready"].includes(microphoneStatus) ? "setup" : "unavailable";
  const coreStatuses: SetupReadinessStatus[] = [speechStatus, aiStatus, microphoneReadiness];
  const coreReady = coreStatuses.every(canContinueWithCheck);

  return [
    {
      key: "speech_recognition",
      status: speechStatus,
      titleKey: "runtime_setup.setup_guide_speech_title",
      detailKey:
        speechStatus === "ready"
          ? "runtime_setup.setup_guide_speech_ready"
          : speechStatus === "configured" ? "first_run.speech_configured" : "runtime_setup.setup_guide_speech_setup",
      rescueKey: cloudSpeechSelected ? "first_run.rescue_cloud_speech" : undefined,
      actionKey: "runtime_setup.setup_guide_download_model",
      disabled: false,
    },
    {
      key: "ai_tutor",
      status: aiStatus,
      titleKey: "runtime_setup.setup_guide_ai_title",
      detailKey:
        credentialMissing
          ? "first_run.ai_missing_key"
          : aiStatus === "configured"
          ? "first_run.ai_configured"
          : aiStatus === "ready"
          ? "runtime_setup.setup_guide_ai_ready"
          : "runtime_setup.setup_guide_ai_setup",
      rescueKey: credentialMissing ? "first_run.rescue_ai_credential" : undefined,
      actionKey: "runtime_setup.setup_guide_connect_ai",
      disabled: false,
    },
    {
      key: "microphone",
      status: microphoneReadiness,
      titleKey: "runtime_setup.setup_guide_microphone_title",
      detailKey:
        microphoneReadiness === "ready"
          ? "runtime_setup.setup_guide_microphone_ready"
          : "runtime_setup.setup_guide_microphone_setup",
      actionKey: "runtime_setup.setup_guide_check_microphone",
      disabled: false,
    },
    {
      key: "sample_check",
      status: coreReady ? "configured" : "setup",
      titleKey: "runtime_setup.setup_guide_sample_title",
      detailKey: coreReady
        ? "runtime_setup.setup_guide_sample_ready"
        : "runtime_setup.setup_guide_sample_blocked",
      actionKey: "runtime_setup.setup_guide_run_sample",
      disabled: !coreReady,
    },
  ];
};
