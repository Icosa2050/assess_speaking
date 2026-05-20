import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { RuntimeConnectionForm } from "@/components/setup/RuntimeConnectionForm";
import { ConnectionStatusPanel } from "@/components/setup/ConnectionStatusPanel";
import { apiClient, ApiClientError } from "@/lib/api/client";
import type {
  ConnectionSecretState,
  DiagnosticItem,
  RuntimeConnectionDraft,
  RuntimeSettingsConnection,
} from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import { useAppStore } from "@/lib/state/appStore";

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
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
  minHeight: "44px",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
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
  const queryClient = useQueryClient();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const draft = useAppStore((state) => state.draft);
  const setActiveConnectionId = useAppStore((state) => state.setActiveConnectionId);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setSetupComplete = useAppStore((state) => state.setSetupComplete);

  const translate = createTranslator(locale);

  const [selectedWhisperModel, setSelectedWhisperModel] = useState("small");
  const [whisperMessage, setWhisperMessage] = useState("");
  const [detectedModelMessage, setDetectedModelMessage] = useState("");
  const [detectedModels, setDetectedModels] = useState<string[]>([]);
  const [formStatus, setFormStatus] = useState<
    "idle" | "testing" | "test-succeeded" | "test-failed" | "saving" | "save-failed" | "saved"
  >("idle");
  const [formStatusMessage, setFormStatusMessage] = useState("");
  const [isBusy, setIsBusy] = useState(false);

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
    enabled: Boolean(selectedWhisperModel),
  });

  useEffect(() => {
    setCurrentPage("runtime_setup");
  }, [setCurrentPage]);

  useEffect(() => {
    const nextModel = String(runtimeSettingsQuery.data?.whisper_model || "").trim();
    if (nextModel) {
      setSelectedWhisperModel(nextModel);
    }
  }, [runtimeSettingsQuery.data?.whisper_model]);

  const diagnosticsItems = diagnosticsQuery.data?.items ?? [];
  const whisperItem = findDiagnostic(diagnosticsItems, "whisper");
  const runtime = runtimeQuery.data;
  const runtimeSettings = runtimeSettingsQuery.data;

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
      await queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] });
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

  const handleTestConnection = async (connection: RuntimeConnectionDraft) => {
    setIsBusy(true);
    setFormStatus("testing");
    setFormStatusMessage(translate("runtime_setup.testing_connection"));
    try {
      const response = await apiClient.postRuntimeSettingsTestConnection({ connection });
      setDetectedModels(response.discovered_models);
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
      setFormStatusMessage(translate("runtime_setup.save_success"));
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
      style={{ display: "grid", gap: "1rem" }}
      {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.screen)}
    >
      <section style={cardStyle}>
        <h2
          style={{
            margin: 0,
            fontSize: "1.75rem",
            color: "#10201c",
          }}
        >
          {translate("runtime_setup.title")}
        </h2>
        <p
          style={{
            margin: 0,
            lineHeight: 1.6,
            color: "#33514b",
          }}
        >
          {translate("runtime_setup.body")}
        </p>
        {runtime?.configured ? (
          <p
            style={{
              margin: 0,
              color: "#166534",
              fontWeight: 600,
            }}
          >
            {translate("runtime_setup.current_connection_ready")}
          </p>
        ) : null}
      </section>

      <section style={cardStyle}>
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
      </section>

      <RuntimeConnectionForm
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
