import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { CloudSettingsPanel } from "@/components/setup/CloudSettingsPanel";
import { RuntimeConnectionForm } from "@/components/setup/RuntimeConnectionForm";
import { ConnectionStatusPanel } from "@/components/setup/ConnectionStatusPanel";
import { MicrophoneTestPanel, useMicrophoneTest } from "@/components/setup/MicrophoneTestPanel";
import { SetupReadinessPanel } from "@/components/setup/SetupReadinessPanel";
import { FirstRunPreview } from "@/components/setup/FirstRunPreview";
import { Button } from "@/components/ui/Button";
import { Card, Disclosure } from "@/components/ui/Card";
import { StatusBanner } from "@/components/ui/StatusBanner";
import { apiClient, ApiClientError } from "@/lib/api/client";
import type {
  ConnectionSecretState,
  DiagnosticItem,
  RuntimeConnectionDraft,
  RuntimeSettingsConnection,
} from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import {
  buildSetupReadinessRows,
  canEnterSpeak,
  canRecord,
  canContinueWithCheck,
  resolveSetupReadinessAction,
  type SetupReadinessKey,
} from "@/lib/setup/readiness";
import { useReadiness } from "@/lib/setup/useReadiness";
import { useAppStore } from "@/lib/state/appStore";
import styles from "./SetupRoute.module.css";

const cardStyle = {
  display: "grid",
  gridTemplateColumns: "minmax(0, 1fr)",
  gap: "0.875rem",
  minWidth: 0,
  maxWidth: "100%",
  overflowWrap: "anywhere",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const actionButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  backgroundColor: "rgba(255, 255, 255, 0.95)",
  color: "#10201c",
  fontWeight: 600,
  font: "inherit",
} as const;

const inputStyle = {
  width: "100%",
  minWidth: 0,
  minHeight: "44px",
  borderRadius: "8px",
  border: "1px solid var(--color-border-control)",
  padding: "0.7rem 0.8rem",
  font: "inherit",
  color: "#10201c",
  backgroundColor: "rgba(255, 255, 255, 0.96)",
} as const;

const findDiagnostic = (items: DiagnosticItem[], key: string): DiagnosticItem | undefined =>
  items.find((item) => item.key === key);

const renderDiagnosticText = (
  item: DiagnosticItem | undefined,
  translate: ReturnType<typeof createTranslator>,
): string | null => {
  if (!item) {
    return null;
  }

  return translate(item.detail_key, item.detail_args as Record<string, string | number>);
};

const WHISPER_MODEL_ORDER: Record<string, number> = {
  tiny: 0,
  base: 1,
  small: 2,
  medium: 3,
  "large-v3-turbo": 4,
  "large-v3": 5,
};

// Derived from docs/vostavo_whisper_model_guide.md
const PRACTICE_WHISPER_GUIDE: Record<string, string> = {
  en: "small",
  de: "medium",
  it: "medium",
  fr: "medium",
  es: "medium",
  pt: "medium",
  nl: "medium",
  pl: "large-v3",
  cs: "large-v3",
  sk: "large-v3",
};

const recommendWhisperModel = (languageCode: string, cefrLevel: string): string => {
  const normalizedLanguage = String(languageCode || "").trim().toLowerCase();
  const normalizedLevel = String(cefrLevel || "").trim().toUpperCase();
  if (normalizedLevel === "B2" || normalizedLevel === "C1") {
    return "large-v3";
  }
  return PRACTICE_WHISPER_GUIDE[normalizedLanguage] || "large-v3";
};

const compareWhisperModelTier = (candidate: string, baseline: string): number =>
  (WHISPER_MODEL_ORDER[candidate] ?? -1) - (WHISPER_MODEL_ORDER[baseline] ?? -1);

const buildDefaultDraft = (
  translate: ReturnType<typeof createTranslator>,
): RuntimeConnectionDraft => ({
  provider_choice: "ollama_local",
  label: translate("runtime_setup.provider_options.ollama_local.label"),
  model: "",
  base_url: "http://localhost:11434",
  api_key: "",
  openrouter_http_referer: "",
  openrouter_app_title: "",
});

const buildDraftFromConnection = (
  connection: RuntimeSettingsConnection | null,
  translate: ReturnType<typeof createTranslator>,
): RuntimeConnectionDraft =>
  connection
    ? {
        connection_id: connection.connection_id,
        provider_choice: connection.provider_choice,
        label: connection.label,
        model: connection.model,
        base_url: connection.base_url,
        api_key: "",
        openrouter_http_referer: connection.openrouter_http_referer,
        openrouter_app_title: connection.openrouter_app_title,
      }
    : buildDefaultDraft(translate);

const formStateColor = (tone: "success" | "warning" | "error"): string => {
  if (tone === "success") {
    return "#166534";
  }
  if (tone === "error") {
    return "#b42318";
  }
  return "#9a6700";
};

export const SetupRoute = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { hash } = location;
  const returningToSpeak = (location.state as { returnTo?: string } | null)?.returnTo === "/speak";
  useEffect(() => {
    if (hash === "#runtime-setup-microphone" || hash === "#runtime-setup-cloud") {
      document.getElementById(hash.slice(1))?.scrollIntoView?.({ block: "start" });
    }
  }, [hash]);
  const queryClient = useQueryClient();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const draft = useAppStore((state) => state.draft);
  const setActiveConnectionId = useAppStore((state) => state.setActiveConnectionId);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setSetupComplete = useAppStore((state) => state.setSetupComplete);

  const translate = createTranslator(locale);
  const microphoneTest = useMicrophoneTest();

  const [selectedWhisperModel, setSelectedWhisperModel] = useState("small");
  const [whisperMessage, setWhisperMessage] = useState("");
  const [detectedModelMessage, setDetectedModelMessage] = useState("");
  const [detectedModels, setDetectedModels] = useState<string[]>([]);
  const [formStatus, setFormStatus] = useState<
    "idle" | "testing" | "test-succeeded" | "test-failed" | "saving" | "save-failed" | "saved"
  >("idle");
  const [formStatusMessage, setFormStatusMessage] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [cloudOptionsOpen, setCloudOptionsOpen] = useState(hash === "#runtime-setup-cloud");
  const [cloudPanelOpened, setCloudPanelOpened] = useState(hash === "#runtime-setup-cloud");

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
  });

  const diagnosticsQuery = useQuery({
    queryKey: queryKeys.diagnostics,
    queryFn: () => apiClient.getDiagnostics(),
  });

  const runtimeSettingsQuery = useQuery({
    queryKey: ["runtime", "settings"],
    queryFn: () => apiClient.getRuntimeSettings(),
  });

  const whisperStatusQuery = useQuery({
    queryKey: ["runtime", "whisper", selectedWhisperModel],
    queryFn: () => apiClient.getWhisperModelStatus(selectedWhisperModel),
    enabled: Boolean(selectedWhisperModel) && runtimeSettingsQuery.data?.asr_provider !== "groq",
  });
  const savedWhisperModel = String(runtimeSettingsQuery.data?.whisper_model || "").trim();

  useEffect(() => {
    setCurrentPage("runtime_setup");
  }, [setCurrentPage]);

  useEffect(() => {
    if (savedWhisperModel) {
      setSelectedWhisperModel(savedWhisperModel);
    }
  }, [savedWhisperModel]);

  const diagnosticsItems = diagnosticsQuery.data?.items ?? [];
  const whisperItem = findDiagnostic(diagnosticsItems, "whisper");
  const runtime = runtimeQuery.data;
  const runtimeSettings = runtimeSettingsQuery.data;
  const readinessQuery = useReadiness();

  const activeConnection = useMemo(() => {
    if (!runtimeSettings) {
      return null;
    }
    return (
      runtimeSettings.connections.find(
        (connection) => connection.connection_id === runtimeSettings.active_connection_id,
      ) ??
      runtimeSettings.connections.find((connection) => connection.is_default) ??
      null
    );
  }, [runtimeSettings]);

  const recommendedWhisperModel = recommendWhisperModel(draft.learningLanguage, draft.cefrLevel);
  const recommendedModelText = translate("runtime_setup.suggested_defaults", {
    values: translate("runtime_setup.suggested_model", {
      value: recommendedWhisperModel,
    }),
  });
  const recommendedModelTone =
    compareWhisperModelTier(selectedWhisperModel, recommendedWhisperModel) >= 0 ? "success" : "warning";

  const whisperMessageFromDiagnostics =
    renderDiagnosticText(whisperItem, translate) ?? translate("runtime_setup.cache_missing");
  const whisperStatusMessage = whisperStatusQuery.data?.cached
    ? translate("runtime_setup.cache_ready", { path: whisperStatusQuery.data.cached_path })
    : whisperMessage || whisperMessageFromDiagnostics;
  const readinessRows = buildSetupReadinessRows({
    readiness: readinessQuery.data,
    readinessError: readinessQuery.isError && !readinessQuery.data,
    readinessPending: readinessQuery.isPending,
    microphoneStatus: microphoneTest.status,
    microphoneSetupPassed: microphoneTest.passed,
  });

  const cloudAsr = runtimeSettings?.asr_provider === "groq";
  const cloudDisclosureOpen = cloudAsr || hash === "#runtime-setup-cloud" || cloudOptionsOpen;
  useEffect(() => {
    if (cloudDisclosureOpen) setCloudPanelOpened(true);
  }, [cloudDisclosureOpen]);
  const recommendedLocalPath = !cloudAsr && (!runtime?.configured ||
    ["ollama", "ollama_local"].includes(activeConnection?.provider_choice ?? runtime.provider));
  const requiredRows = readinessRows.filter((row) => row.key !== "sample_check");
  const setupReady = canRecord(requiredRows);
  const setupCanContinue = canEnterSpeak(requiredRows);
  const returnCheck = requiredRows.find((row) => row.key !== "microphone" && !canContinueWithCheck(row.status));

  const initialDraft = buildDraftFromConnection(activeConnection, translate);
  const formResetToken = JSON.stringify({
    connectionId: activeConnection?.connection_id || "",
    label: activeConnection?.label || "",
    model: activeConnection?.model || "",
    baseUrl: activeConnection?.base_url || "",
    secretState: activeConnection?.secret_state || "absent",
    lastTestedAt: activeConnection?.last_tested_at || "",
  });
  const initialSecretState = (activeConnection?.secret_state || "absent") as ConnectionSecretState;

  const handleInvalidateRuntime = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.runtime }),
      queryClient.invalidateQueries({ queryKey: queryKeys.diagnostics }),
      queryClient.invalidateQueries({ queryKey: queryKeys.readiness }),
      queryClient.invalidateQueries({ queryKey: ["runtime", "settings"] }),
      queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] }),
    ]);
  };

  const handleDownloadModel = async () => {
    setIsBusy(true);
    setWhisperMessage("");
    try {
      const status = await apiClient.postWhisperModelDownload(selectedWhisperModel);
      setWhisperMessage(
        translate("runtime_setup.whisper_ready", {
          path: status.cached_path || status.model,
        }),
      );
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] }),
        queryClient.invalidateQueries({ queryKey: queryKeys.diagnostics }),
        queryClient.invalidateQueries({ queryKey: queryKeys.readiness }),
      ]);
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setWhisperMessage(translate("runtime_setup.whisper_download_failed", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleDetectLocalModels = async (connection: RuntimeConnectionDraft) => {
    setIsBusy(true);
    setDetectedModelMessage(translate("runtime_setup.detecting_local_models"));
    try {
      const response = await apiClient.postRuntimeSettingsTestConnection({
        connection: {
          ...connection,
          model: "",
        },
      });
      setDetectedModels(response.discovered_models);
      if (response.discovered_models.length > 0) {
        setDetectedModelMessage(
          translate("runtime_setup.detected_local_models_message", {
            count: response.discovered_models.length,
            endpoint: response.health_endpoint,
          }),
        );
      } else {
        setDetectedModelMessage(
          translate("runtime_setup.detected_local_models_empty", {
            endpoint: response.health_endpoint,
          }),
        );
      }
    } catch (caught) {
      const detail =
        caught instanceof ApiClientError && caught.code === "validation_error"
          ? caught.detail
          : caught instanceof Error
            ? caught.message
            : String(caught);
      setDetectedModels([]);
      setDetectedModelMessage(translate("runtime_setup.detected_local_models_failed", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleProviderChange = () => {
    setDetectedModels([]);
    setDetectedModelMessage("");
    setFormStatus("idle");
    setFormStatusMessage("");
  };

  const scrollToSetupSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({
      block: "start",
      behavior: "smooth",
    });
  };

  const handleSetupReadinessAction = (key: SetupReadinessKey) => {
    if (key === "microphone") {
      scrollToSetupSection("runtime-setup-microphone");
      if (microphoneTest.phase === "idle") void microphoneTest.start();
      return;
    }
    if (key === "speech_recognition" && cloudAsr) { scrollToSetupSection("runtime-setup-cloud"); return; }
    const target = resolveSetupReadinessAction(key);
    if (target.kind === "refresh") { void readinessQuery.refetch(); return; }
    if (target.kind === "section") {
      scrollToSetupSection(target.value);
      return;
    }
    navigate(target.value);
  };

  const handleTestConnection = async (connection: RuntimeConnectionDraft) => {
    setIsBusy(true);
    setFormStatus("testing");
    setFormStatusMessage(translate("runtime_setup.testing_connection"));
    try {
      const response = await apiClient.postRuntimeSettingsTestConnection({ connection });
      setDetectedModels(response.discovered_models);
      await queryClient.invalidateQueries({ queryKey: queryKeys.readiness });
      setFormStatus("test-succeeded");
      setFormStatusMessage(
        translate("runtime_setup.test_message", {
          endpoint: response.health_endpoint,
          base_url: response.base_url,
          model: connection.model || response.discovered_models[0] || "-",
          preview: response.content_preview || "-",
        }),
      );
    } catch (caught) {
      const detail =
        caught instanceof ApiClientError && caught.code === "validation_error"
          ? caught.detail
          : caught instanceof Error
            ? caught.message
            : String(caught);
      setFormStatus("test-failed");
      await queryClient.invalidateQueries({ queryKey: queryKeys.readiness });
      setFormStatusMessage(translate("runtime_setup.test_error", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleSave = async ({
    clearSavedSecret,
    draft: connection,
  }: {
    clearSavedSecret: boolean;
    draft: RuntimeConnectionDraft;
  }) => {
    setIsBusy(true);
    setFormStatus("saving");
    setFormStatusMessage(translate("runtime_setup.save_connection"));
    try {
      const response = await apiClient.putRuntimeSettings({
        ui_locale: locale,
        whisper_model: selectedWhisperModel,
        clear_saved_secret: clearSavedSecret,
        connection,
      });
      setActiveConnectionId(response.active_connection_id);
      setSetupComplete(response.connections.length > 0);
      setFormStatus("saved");
      const saved = response.connections.find(c => c.connection_id === response.active_connection_id);
      setFormStatusMessage(translate(saved?.provider_metadata.persistent === false ? "runtime_setup.save_success_session_only" : "runtime_setup.save_success"));
      await handleInvalidateRuntime();
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setFormStatus("save-failed");
      setFormStatusMessage(translate("runtime_setup.save_error", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr)",
        gap: "1rem",
        minWidth: 0,
        maxWidth: "100%",
      }}
      {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.screen)}
    >
      {returningToSpeak ? <StatusBanner title={translate(setupCanContinue ? "first_run.continue_to_speak_title" : "first_run.speak_return_title")} tone={setupCanContinue ? "success" : "warning"}>
        {returnCheck ? translate("first_run.speak_return_body", { check: translate(returnCheck.titleKey) }) : translate("first_run.speak_return_ready_body")}
        {setupCanContinue ? <Button onClick={() => navigate("/speak")}>{translate("first_run.return_to_speak")}</Button> : null}
      </StatusBanner> : null}
      <Card as="section" className={styles.path} data-testid="runtime_setup.local_path">
        <p className={styles.eyebrow}>{translate(recommendedLocalPath ? "first_run.local_path_eyebrow" : "first_run.current_path_eyebrow")}</p>
        <h2>{translate(recommendedLocalPath ? "first_run.local_path_title" : "first_run.current_path_title")}</h2>
        <p>{translate(recommendedLocalPath ? "first_run.local_path_body" : "first_run.current_path_body")}</p>
        {recommendedLocalPath ? <ol>
          <li>{translate("first_run.prerequisite_mac")}</li>
          <li>{translate("first_run.prerequisite_download")}</li>
          <li>{translate("first_run.prerequisite_microphone")}</li>
        </ol> : null}
        <div className={styles.pathActions}>
          <Button variant="primary" onClick={() => void handleInvalidateRuntime()}>
            {translate("first_run.check_readiness")}
          </Button>
          {setupCanContinue ? <Button onClick={() => navigate("/session-setup")}>{translate("first_run.choose_task")}</Button> : null}
        </div>
        <p role="status" className={styles.checkStatus}>
          {translate(setupReady ? "first_run.checks_ready" : setupCanContinue ? "first_run.continue_to_speak_body" : requiredRows.some((row) => row.status === "loading") ? "first_run.checks_loading" : "first_run.checks_next")}
        </p>
      </Card>

      <SetupReadinessPanel
        rows={readinessRows}
        translate={translate}
        onAction={handleSetupReadinessAction}
      />

      <FirstRunPreview locale={locale} />

      <MicrophoneTestPanel test={microphoneTest} translate={translate} />

      <div id="runtime-setup-cloud">
        <Disclosure title={translate("first_run.other_routes")} open={cloudDisclosureOpen} onToggle={(event) => {
          setCloudOptionsOpen(event.currentTarget.open);
          if (event.currentTarget.open) setCloudPanelOpened(true);
        }}>
          <p>{translate("first_run.other_routes_body")}</p>
          {(cloudDisclosureOpen || cloudPanelOpened) ? <CloudSettingsPanel locale={locale} connections={runtimeSettings?.connections ?? []}/> : null}
        </Disclosure>
      </div>

      {!cloudAsr && <section
        id="runtime-setup-whisper"
        style={cardStyle}
      >
        <h2
          style={{
            margin: 0,
            fontSize: "1.2rem",
            color: "#10201c",
          }}
        >
          {translate("runtime_setup.section_whisper")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("runtime_setup.whisper_guidance")}
        </p>
        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate("runtime_setup.whisper_model")}
          </span>
          <select
            value={selectedWhisperModel}
            onChange={(event) => setSelectedWhisperModel(event.target.value)}
            style={inputStyle}
          >
            {["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"].map((option) => (
              <option
                key={option}
                value={option}
              >
                {option}
              </option>
            ))}
          </select>
        </label>
        {savedWhisperModel && selectedWhisperModel !== savedWhisperModel ? <p role="status" style={{ margin: 0, lineHeight: 1.55, color: "var(--color-warning)" }}>
          {translate("first_run.save_speech_selection", { model: selectedWhisperModel })}
        </p> : null}
        <p
          style={{
            margin: 0,
            lineHeight: 1.55,
            color: whisperStatusQuery.data?.cached ? "#166534" : "#9a6700",
          }}
          {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.whisperStatus, {
            cached: Boolean(whisperStatusQuery.data?.cached),
            model: selectedWhisperModel,
          })}
        >
          {whisperStatusMessage}
        </p>
        <p
          style={{
            margin: 0,
            lineHeight: 1.55,
            color: formStateColor(recommendedModelTone),
          }}
        >
          {recommendedModelText}
        </p>
        <button
          type="button"
          disabled={isBusy}
          onClick={() => {
            void handleDownloadModel();
          }}
          style={{ ...actionButtonStyle, width: "fit-content" }}
          {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.downloadModel)}
        >
          {translate("runtime_setup.download_model")}
        </button>
      </section>}

      <div id="runtime-setup-connection">
        {formStatus === "test-failed" ? (
          <StatusBanner title={translate("first_run.connection_retry_title")} tone="warning">
            {translate("first_run.connection_retry_body")}
          </StatusBanner>
        ) : null}
        <RuntimeConnectionForm
          onAccountConnected={(_id, notice) => { if (notice) setFormStatusMessage(notice); }}
          detectedModelMessage={detectedModelMessage}
          detectedModels={detectedModels}
          initialDraft={initialDraft}
          initialSecretState={initialSecretState}
          isBusy={isBusy}
          locale={locale}
          onBack={() => navigate("/")}
          onDetectLocalModels={(connection) => {
            void handleDetectLocalModels(connection);
          }}
          onProviderChange={handleProviderChange}
          onSave={(payload) => {
            void handleSave(payload);
          }}
          onTest={(connection) => {
            void handleTestConnection(connection);
          }}
          resetToken={formResetToken}
          status={formStatus}
          statusMessage={formStatusMessage}
          variant="runtime-setup"
        />
      </div>

      <details style={cardStyle}>
        <summary
          style={{
            cursor: "pointer",
            fontWeight: 700,
            color: "#10201c",
          }}
        >
          {translate("runtime_setup.connection_diagnostics")}
        </summary>
        <ConnectionStatusPanel
          activeConnection={activeConnection}
          diagnostics={diagnosticsItems}
          runtime={runtime}
        />
      </details>
    </div>
  );
};
