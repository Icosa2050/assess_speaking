import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { RuntimeConnectionForm } from "@/components/setup/RuntimeConnectionForm";
import { SavedConnectionsPanel } from "@/components/settings/SavedConnectionsPanel";
import { SupportPanel } from "@/components/settings/SupportPanel";
import { apiClient } from "@/lib/api/client";
import type {
  ConnectionSecretState,
  RuntimeConnectionDraft,
  RuntimeSettingsConnection,
} from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import {
  toSavedConnectionRecords,
} from "@/lib/settings/connectionRepository";
import { queryKeys } from "@/lib/query/queryClient";
import { selectRuntimeReadiness, useAppStore } from "@/lib/state/appStore";
import { SUPPORTED_UI_LOCALES, type UiLocale } from "@/lib/state/sessionDraft";

type SettingsOrigin =
  | "home"
  | "runtime-setup"
  | "session-setup"
  | "speak"
  | "review"
  | "history";

type SettingsLocationState = {
  from?: SettingsOrigin;
};

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

const returnPathMap: Record<SettingsOrigin, string> = {
  home: "/",
  "runtime-setup": "/runtime-setup",
  "session-setup": "/session-setup",
  speak: "/speak",
  review: "/review",
  history: "/history",
};

const localeLabel = (
  locale: UiLocale,
  displayLocale: UiLocale,
): string => {
  if (typeof Intl.DisplayNames !== "function") {
    return locale.toUpperCase();
  }

  const labels = new Intl.DisplayNames([displayLocale], { type: "language" });
  return labels.of(locale) ?? locale.toUpperCase();
};

const buildDefaultDraft = (
  translate: ReturnType<typeof createTranslator>,
  fallbackProviderChoice: string,
): RuntimeConnectionDraft => ({
  provider_choice: fallbackProviderChoice || "ollama_local",
  label: translate(`runtime_setup.provider_options.${fallbackProviderChoice || "ollama_local"}.label`),
  model: "",
  base_url:
    fallbackProviderChoice === "openrouter"
      ? "https://openrouter.ai/api/v1"
      : fallbackProviderChoice === "lmstudio_local"
        ? "http://localhost:1234/v1"
        : fallbackProviderChoice === "ollama_cloud"
          ? "https://ollama.com/api"
          : "http://localhost:11434",
  api_key: "",
  openrouter_http_referer: "",
  openrouter_app_title: "",
});

const buildDraftFromConnection = (
  connection: RuntimeSettingsConnection | null,
  translate: ReturnType<typeof createTranslator>,
  fallbackProviderChoice: string,
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
    : buildDefaultDraft(translate, fallbackProviderChoice);

const providerLabelForDraft = (
  draft: RuntimeConnectionDraft,
  selectedConnection: RuntimeSettingsConnection | null,
  translate: ReturnType<typeof createTranslator>,
): string => {
  const providerChoice = String(draft.provider_choice || "").trim();
  const translated = providerChoice
    ? translate(`runtime_setup.provider_options.${providerChoice}.label`, { value: providerChoice })
    : "";
  if (translated && !translated.startsWith("[")) {
    return translated;
  }
  return selectedConnection?.provider_label || providerChoice || "-";
};

export const SettingsRoute = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const setUiLocale = useAppStore((state) => state.setUiLocale);
  const setActiveConnectionId = useAppStore((state) => state.setActiveConnectionId);
  const setSetupComplete = useAppStore((state) => state.setSetupComplete);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const translate = useMemo(() => createTranslator(locale), [locale]);

  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [newConnectionInitialDraft, setNewConnectionInitialDraft] =
    useState<RuntimeConnectionDraft | null>(null);
  const [selectedUiLocale, setSelectedUiLocale] = useState<UiLocale>(locale);
  const [selectedWhisperModel, setSelectedWhisperModel] = useState("small");
  const [formStatus, setFormStatus] = useState<
    "idle" | "testing" | "test-succeeded" | "test-failed" | "saving" | "saved"
  >("idle");
  const [formStatusMessage, setFormStatusMessage] = useState("");
  const [whisperMessage, setWhisperMessage] = useState("");
  const [isBusy, setIsBusy] = useState(false);

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
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
    setCurrentPage("settings");
  }, [setCurrentPage]);

  useEffect(() => {
    if (!runtimeSettingsQuery.data) {
      return;
    }
    const nextLocale = runtimeSettingsQuery.data.ui_locale as UiLocale;
    setSelectedUiLocale(nextLocale);
    setSelectedWhisperModel(runtimeSettingsQuery.data.whisper_model || "small");
    if (nextLocale && nextLocale !== locale) {
      setUiLocale(nextLocale);
    }
    if (selectedConnectionId === null) {
      const nextConnectionId =
        runtimeSettingsQuery.data.active_connection_id ||
        runtimeSettingsQuery.data.connections.find((connection) => connection.is_default)?.connection_id ||
        "__new__";
      setSelectedConnectionId(nextConnectionId);
      setNewConnectionInitialDraft(
        nextConnectionId === "__new__" ? buildDefaultDraft(translate, "ollama_local") : null,
      );
    }
  }, [locale, runtimeSettingsQuery.data, selectedConnectionId, setUiLocale, translate]);

  const connections = useMemo(
    () => toSavedConnectionRecords(runtimeSettingsQuery.data?.connections ?? []),
    [runtimeSettingsQuery.data?.connections],
  );

  const selectedConnection = useMemo(() => {
    if (!runtimeSettingsQuery.data || !selectedConnectionId || selectedConnectionId === "__new__") {
      return null;
    }
    return (
      runtimeSettingsQuery.data.connections.find(
        (connection) => connection.connection_id === selectedConnectionId,
      ) ?? null
    );
  }, [runtimeSettingsQuery.data, selectedConnectionId]);

  const activeConnection = useMemo(() => {
    if (!runtimeSettingsQuery.data) {
      return null;
    }
    return (
      runtimeSettingsQuery.data.connections.find(
        (connection) => connection.connection_id === runtimeSettingsQuery.data.active_connection_id,
      ) ??
      runtimeSettingsQuery.data.connections.find((connection) => connection.is_default) ??
      null
    );
  }, [runtimeSettingsQuery.data]);

  const effectivePreferences = useMemo(
    () => ({
      ...preferences,
      activeConnectionId:
        runtimeSettingsQuery.data?.active_connection_id || preferences.activeConnectionId,
      setupComplete:
        Boolean(runtimeQuery.data?.configured) || (runtimeSettingsQuery.data?.connections.length ?? 0) > 0,
    }),
    [preferences, runtimeQuery.data, runtimeSettingsQuery.data],
  );

  useEffect(() => {
    if (runtimeSettingsQuery.data?.active_connection_id) {
      setActiveConnectionId(runtimeSettingsQuery.data.active_connection_id);
    }
    setSetupComplete(
      Boolean(runtimeQuery.data?.configured) || (runtimeSettingsQuery.data?.connections.length ?? 0) > 0,
    );
  }, [runtimeQuery.data, runtimeSettingsQuery.data, setActiveConnectionId, setSetupComplete]);

  const runtimeReadiness = selectRuntimeReadiness({
    preferences: effectivePreferences,
  });

  const newConnectionFallbackProviderChoice =
    selectedConnectionId === "__new__" ? "ollama_local" : activeConnection?.provider_choice || "ollama_local";
  const initialDraft =
    selectedConnectionId === "__new__" && newConnectionInitialDraft
      ? newConnectionInitialDraft
      : buildDraftFromConnection(
          selectedConnection,
          translate,
          newConnectionFallbackProviderChoice,
        );
  const initialSecretState = (selectedConnection?.secret_state || "absent") as ConnectionSecretState;
  const formResetToken = JSON.stringify({
    selectedConnectionId: selectedConnectionId ?? "",
    secretState: selectedConnection?.secret_state || "absent",
    lastTestedAt: selectedConnection?.last_tested_at || "",
  });
  const candidateReturnOrigin = (location.state as SettingsLocationState | null)?.from;
  const returnOrigin =
    candidateReturnOrigin && candidateReturnOrigin in returnPathMap
      ? candidateReturnOrigin
      : undefined;
  const returnPath = returnOrigin ? returnPathMap[returnOrigin] : "/";

  const handleInvalidateRuntime = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.runtime }),
      queryClient.invalidateQueries({ queryKey: ["runtime", "settings"] }),
      queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] }),
    ]);
  };

  const handleSave = async ({
    clearSavedSecret,
    draft,
  }: {
    clearSavedSecret: boolean;
    draft: RuntimeConnectionDraft;
  }) => {
    setIsBusy(true);
    setFormStatus("saving");
    setFormStatusMessage(translate("settings.save"));
    try {
      const response = await apiClient.putRuntimeSettings({
        ui_locale: selectedUiLocale,
        whisper_model: selectedWhisperModel,
        clear_saved_secret: clearSavedSecret,
        connection: draft,
      });
      setUiLocale(response.ui_locale);
      setActiveConnectionId(response.active_connection_id);
      setSetupComplete(response.connections.length > 0);
      setSelectedConnectionId(response.active_connection_id || draft.connection_id || "__new__");
      setNewConnectionInitialDraft(null);
      setFormStatus("saved");
      setFormStatusMessage(translate("settings.saved"));
      await handleInvalidateRuntime();
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setFormStatus("test-failed");
      setFormStatusMessage(translate("settings.test_failed", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleTest = async (draft: RuntimeConnectionDraft) => {
    setIsBusy(true);
    setFormStatus("testing");
    setFormStatusMessage(translate("settings.testing_connection"));
    try {
      const response = await apiClient.postRuntimeSettingsTestConnection({ connection: draft });
      setFormStatus("test-succeeded");
      setFormStatusMessage(
        translate("settings.test_success", {
          provider: providerLabelForDraft(draft, selectedConnection, translate),
          base_url: response.base_url,
          preview: response.content_preview || "-",
        }),
      );
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setFormStatus("test-failed");
      setFormStatusMessage(translate("settings.test_failed", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleSetDefault = async (connectionId: string) => {
    setIsBusy(true);
    try {
      const response = await apiClient.postRuntimeSettingsSetDefault(connectionId);
      setActiveConnectionId(response.active_connection_id);
      setSetupComplete(response.connections.length > 0);
      setSelectedConnectionId(connectionId);
      await handleInvalidateRuntime();
    } finally {
      setIsBusy(false);
    }
  };

  const handleSelectConnection = (connectionId: string) => {
    setSelectedConnectionId(connectionId);
    setNewConnectionInitialDraft(
      connectionId === "__new__" ? buildDefaultDraft(translate, "ollama_local") : null,
    );
  };

  const handleDelete = async (connectionId: string) => {
    setIsBusy(true);
    try {
      const response = await apiClient.deleteRuntimeSettingsConnection(connectionId);
      setActiveConnectionId(response.active_connection_id);
      setSetupComplete(response.connections.length > 0);
      const nextConnectionId = response.active_connection_id || "__new__";
      setSelectedConnectionId(nextConnectionId);
      setNewConnectionInitialDraft(
        nextConnectionId === "__new__" ? buildDefaultDraft(translate, "ollama_local") : null,
      );
      await handleInvalidateRuntime();
    } finally {
      setIsBusy(false);
    }
  };

  const handleDownloadModel = async () => {
    setIsBusy(true);
    setWhisperMessage("");
    try {
      const status = await apiClient.postWhisperModelDownload(selectedWhisperModel);
      setWhisperMessage(
        translate("settings.whisper_downloaded", {
          path: status.cached_path || status.model,
        }),
      );
      await queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] });
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setWhisperMessage(translate("settings.whisper_download_failed", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const whisperStatusMessage = whisperStatusQuery.data?.cached
    ? translate("settings.whisper_cached", { path: whisperStatusQuery.data.cached_path })
    : whisperMessage || translate("settings.whisper_not_cached");

  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      {...semanticAttributes(SEMANTIC_IDS.settings.screen)}
    >
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.75rem", color: "#10201c" }}>
          {translate("settings.title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("settings.body")}
        </p>
        {!runtimeReadiness.ready ? (
          <p style={{ margin: 0, color: "#9a6700", lineHeight: 1.6 }}>
            {translate("settings.needs_setup_info")}
          </p>
        ) : null}
      </section>

      <SavedConnectionsPanel
        connections={connections}
        locale={locale}
        onDelete={(connectionId) => {
          void handleDelete(connectionId);
        }}
        onOpenSetup={() =>
          navigate("/runtime-setup", {
            state: returnOrigin ? { from: returnOrigin } : undefined,
          })
        }
        onSelectConnection={handleSelectConnection}
        onSetDefault={(connectionId) => {
          void handleSetDefault(connectionId);
        }}
        selectedConnectionId={selectedConnectionId ?? "__new__"}
      />

      <section
        style={cardStyle}
        {...semanticAttributes(SEMANTIC_IDS.settings.sectionRuntimeDefaults)}
      >
        <div style={{ display: "grid", gap: "0.35rem" }}>
          <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
            {translate("settings.ui_locale")}
          </h2>
          <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
            {translate("settings.body")}
          </p>
        </div>

        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate("settings.ui_locale")}
          </span>
          <select
            value={selectedUiLocale}
            onChange={(event) => setSelectedUiLocale(event.target.value as UiLocale)}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.uiLocale)}
          >
            {SUPPORTED_UI_LOCALES.map((option) => (
              <option
                key={option}
                value={option}
              >
                {localeLabel(option, locale)}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate("settings.whisper_model")}
          </span>
          <select
            value={selectedWhisperModel}
            onChange={(event) => setSelectedWhisperModel(event.target.value)}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.whisperModel)}
          >
            {["tiny", "base", "small", "medium", "large-v3"].map((option) => (
              <option
                key={option}
                value={option}
              >
                {option}
              </option>
            ))}
          </select>
        </label>

        <p style={{ margin: 0, lineHeight: 1.55, color: whisperStatusQuery.data?.cached ? "#166534" : "#9a6700" }}>
          {whisperStatusMessage}
        </p>

        <button
          type="button"
          disabled={isBusy}
          onClick={() => {
            void handleDownloadModel();
          }}
          style={{ ...actionButtonStyle, width: "fit-content" }}
        >
          {translate("settings.whisper_download")}
        </button>

        <div
          style={{
            display: "grid",
            gap: "0.75rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          }}
        >
          <article style={cardStyle}>
            <strong style={{ color: "#33514b" }}>{translate("settings.provider")}</strong>
            <span style={{ color: "#10201c" }}>{activeConnection?.provider_label || runtimeQuery.data?.provider || "-"}</span>
          </article>
          <article style={cardStyle}>
            <strong style={{ color: "#33514b" }}>{translate("settings.model")}</strong>
            <span style={{ color: "#10201c" }}>{activeConnection?.model || runtimeQuery.data?.model || "-"}</span>
          </article>
          <article style={cardStyle}>
            <strong style={{ color: "#33514b" }}>{translate("settings.base_url")}</strong>
            <span style={{ color: "#10201c", lineHeight: 1.5 }}>
              {activeConnection?.base_url || runtimeQuery.data?.base_url || "-"}
            </span>
          </article>
        </div>
      </section>

      <RuntimeConnectionForm
        initialDraft={initialDraft}
        initialSecretState={initialSecretState}
        isBusy={isBusy}
        locale={locale}
        onSave={(payload) => {
          void handleSave(payload);
        }}
        onTest={(connection) => {
          void handleTest(connection);
        }}
        resetToken={formResetToken}
        status={formStatus}
        statusMessage={formStatusMessage}
        variant="settings"
      />

      <SupportPanel
        activeConnectionId={runtimeSettingsQuery.data?.active_connection_id || preferences.activeConnectionId}
        locale={locale}
      />

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("settings.return_title")}
        </h2>
        <button
          type="button"
          onClick={() => navigate(returnPath)}
          style={{ ...actionButtonStyle, width: "fit-content" }}
          {...semanticAttributes(SEMANTIC_IDS.settings.return)}
        >
          {translate("settings.back")}
        </button>
      </section>
    </div>
  );
};
