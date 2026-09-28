import type {
  DiagnosticItem,
  RuntimeResponse,
} from "@/lib/api/types";

export type SetupReadinessStatus = "loading" | "ready" | "setup" | "unavailable";

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
  status: SetupReadinessStatus;
  titleKey: string;
};

export type SetupReadinessInput = {
  diagnostics: DiagnosticItem[];
  diagnosticsError?: boolean;
  diagnosticsPending?: boolean;
  runtime?: RuntimeResponse;
  runtimeError?: boolean;
  runtimePending?: boolean;
  whisperCached?: boolean;
  whisperPending?: boolean;
};

export type SetupReadinessActionTarget =
  | { kind: "route"; value: "/session-setup" }
  | { kind: "section"; value: "runtime-setup-connection" | "runtime-setup-whisper" };

export const isCoreSetupReadinessKey = (key: SetupReadinessKey): boolean =>
  key === "speech_recognition" || key === "ai_tutor";

export const resolveSetupReadinessAction = (
  key: SetupReadinessKey,
): SetupReadinessActionTarget => {
  if (key === "speech_recognition") {
    return { kind: "section", value: "runtime-setup-whisper" };
  }
  if (key === "ai_tutor") {
    return { kind: "section", value: "runtime-setup-connection" };
  }
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
  diagnosticsError = false,
  diagnosticsPending = false,
  runtime,
  runtimeError = false,
  runtimePending = false,
  whisperCached = false,
  whisperPending = false,
}: SetupReadinessInput): SetupReadinessRow[] => {
  const allPending = diagnosticsPending || runtimePending || whisperPending;
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
  const microphoneItem = findDiagnostic(diagnostics, "microphone");

  const speechStatus =
    diagnosticsError ? "unavailable" : whisperCached || whisperItem?.status === "ok" ? "ready" : statusFromDiagnostic(whisperItem);
  const aiStatus =
    diagnosticsError || runtimeError
      ? "unavailable"
      : runtime?.configured || runtimeItem?.status === "ok"
        ? "ready"
        : statusFromDiagnostic(runtimeItem);
  const microphoneStatus = diagnosticsError ? "unavailable" : statusFromDiagnostic(microphoneItem);
  const coreReady = speechStatus === "ready" && aiStatus === "ready";

  return [
    {
      key: "speech_recognition",
      status: speechStatus,
      titleKey: "runtime_setup.setup_guide_speech_title",
      detailKey:
        speechStatus === "ready"
          ? "runtime_setup.setup_guide_speech_ready"
          : "runtime_setup.setup_guide_speech_setup",
      actionKey: "runtime_setup.setup_guide_download_model",
      disabled: false,
    },
    {
      key: "ai_tutor",
      status: aiStatus,
      titleKey: "runtime_setup.setup_guide_ai_title",
      detailKey:
        aiStatus === "ready"
          ? "runtime_setup.setup_guide_ai_ready"
          : "runtime_setup.setup_guide_ai_setup",
      actionKey: "runtime_setup.setup_guide_connect_ai",
      disabled: false,
    },
    {
      key: "microphone",
      status: microphoneStatus,
      titleKey: "runtime_setup.setup_guide_microphone_title",
      detailKey:
        microphoneStatus === "ready"
          ? "runtime_setup.setup_guide_microphone_ready"
          : "runtime_setup.setup_guide_microphone_setup",
      actionKey: "runtime_setup.setup_guide_check_microphone",
      disabled: false,
    },
    {
      key: "sample_check",
      status: "setup",
      titleKey: "runtime_setup.setup_guide_sample_title",
      detailKey: coreReady
        ? "runtime_setup.setup_guide_sample_ready"
        : "runtime_setup.setup_guide_sample_blocked",
      actionKey: "runtime_setup.setup_guide_run_sample",
      disabled: !coreReady,
    },
  ];
};
