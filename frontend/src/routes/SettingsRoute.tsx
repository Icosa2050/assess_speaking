import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { SavedConnectionsPanel } from "@/components/settings/SavedConnectionsPanel";
import { Button } from "@/components/ui/Button";
import { JournalPanel } from "@/components/settings/JournalPanel";
import { SupportPanel } from "@/components/settings/SupportPanel";
import { apiClient } from "@/lib/api/client";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { toSavedConnectionRecords } from "@/lib/settings/connectionRepository";
import { queryKeys } from "@/lib/query/queryClient";
import { buildSetupReadinessRows, canEnterSpeak } from "@/lib/setup/readiness";
import { useReadiness } from "@/lib/setup/useReadiness";
import { useAppStore } from "@/lib/state/appStore";
import { SUPPORTED_UI_LOCALES, type UiLocale } from "@/lib/state/sessionDraft";

type SettingsOrigin = "home" | "runtime-setup" | "session-setup" | "speak" | "review" | "library" | "guide" | "history";
type SettingsLocationState = { from?: SettingsOrigin; focusConnectionId?: string; fromSetup?: boolean; saveNotice?: "session-only"; connectionNotice?: string };

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid var(--color-border-decorative)",
  borderRadius: "8px",
  backgroundColor: "var(--color-surface)",
  boxShadow: "0 18px 40px var(--color-shadow-ambient)",
} as const;

const actionButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-control)",
  backgroundColor: "var(--color-surface)",
  color: "var(--color-text)",
  fontWeight: 600,
  font: "inherit",
} as const;

const inputStyle = {
  minHeight: "44px",
  borderRadius: "8px",
  border: "1px solid var(--color-border-control)",
  padding: "0.7rem 0.8rem",
  font: "inherit",
  color: "var(--color-text)",
  backgroundColor: "var(--color-surface)",
} as const;

const returnPathMap: Record<SettingsOrigin, string> = {
  home: "/",
  "runtime-setup": "/runtime-setup",
  "session-setup": "/session-setup",
  speak: "/speak",
  review: "/review",
  history: "/history",
  library: "/library",
  guide: "/guide",
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

export const SettingsRoute = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const setUiLocale = useAppStore((state) => state.setUiLocale);
  const setAppearance = useAppStore((state) => state.setAppearance);
  const setActiveConnectionId = useAppStore((state) => state.setActiveConnectionId);
  const setSetupComplete = useAppStore((state) => state.setSetupComplete);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const translate = useMemo(() => createTranslator(locale), [locale]);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [selectedUiLocale, setSelectedUiLocale] = useState<UiLocale>(locale);
  const [selectedWhisperModel, setSelectedWhisperModel] = useState("small");
  const [connectionMessage, setConnectionMessage] = useState("");
  const [preferencesMessage, setPreferencesMessage] = useState("");
  const [whisperMessage, setWhisperMessage] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const focusedNavigationKey = useRef<string | null>(null);

  const runtimeQuery = useQuery({ queryKey: queryKeys.runtime, queryFn: () => apiClient.getRuntime() });
  const runtimeSettingsQuery = useQuery({ queryKey: ["runtime", "settings"], queryFn: () => apiClient.getRuntimeSettings() });
  const readinessQuery = useReadiness();
  const whisperStatusQuery = useQuery({
    queryKey: ["runtime", "whisper", selectedWhisperModel],
    queryFn: () => apiClient.getWhisperModelStatus(selectedWhisperModel),
    enabled: Boolean(selectedWhisperModel),
  });

  useEffect(() => { setCurrentPage("settings"); }, [setCurrentPage]);
  useEffect(() => {
    if (!runtimeSettingsQuery.data || !(location.state as SettingsLocationState | null)?.fromSetup || focusedNavigationKey.current === location.key) return;
    focusedNavigationKey.current = location.key;
    const heading = document.getElementById("settings-connections-heading");
    heading?.focus();
    heading?.scrollIntoView?.({ block: "start" });
  }, [runtimeSettingsQuery.data, location.key, location.state]);
  useEffect(() => {
    const data = runtimeSettingsQuery.data;
    if (!data) return;
    const nextLocale = data.ui_locale as UiLocale;
    setSelectedUiLocale(nextLocale);
    setSelectedWhisperModel(data.whisper_model || "small");
    if (nextLocale && nextLocale !== locale) setUiLocale(nextLocale);
    setSelectedConnectionId((current) => {
      if (current && data.connections.some((connection) => connection.connection_id === current)) return current;
      const requested = (location.state as SettingsLocationState | null)?.focusConnectionId;
      if (requested && data.connections.some((connection) => connection.connection_id === requested)) return requested;
      return data.active_connection_id || data.connections.find((connection) => connection.is_default)?.connection_id || "";
    });
  }, [runtimeSettingsQuery.data, setUiLocale, location.state]);

  const connections = useMemo(() => toSavedConnectionRecords(runtimeSettingsQuery.data?.connections ?? []), [runtimeSettingsQuery.data?.connections]);
  const activeConnection = useMemo(() => {
    const data = runtimeSettingsQuery.data;
    return data?.connections.find((connection) => connection.connection_id === data.active_connection_id) ??
      data?.connections.find((connection) => connection.is_default) ?? null;
  }, [runtimeSettingsQuery.data]);

  useEffect(() => {
    if (runtimeSettingsQuery.data?.active_connection_id) setActiveConnectionId(runtimeSettingsQuery.data.active_connection_id);
    setSetupComplete(Boolean(runtimeQuery.data?.configured) || (runtimeSettingsQuery.data?.connections.length ?? 0) > 0);
  }, [runtimeQuery.data, runtimeSettingsQuery.data, setActiveConnectionId, setSetupComplete]);

  const runtimeReady = canEnterSpeak(buildSetupReadinessRows({
    readiness: readinessQuery.data,
    readinessError: readinessQuery.isError && !readinessQuery.data,
    readinessPending: readinessQuery.isPending,
  }));
  const candidateReturnOrigin = (location.state as SettingsLocationState | null)?.from;
  const returnOrigin = candidateReturnOrigin && candidateReturnOrigin in returnPathMap ? candidateReturnOrigin : undefined;
  const returnPath = returnOrigin ? returnPathMap[returnOrigin] : "/";
  const setupNotice = (location.state as SettingsLocationState | null)?.saveNotice === "session-only"
    && connections.some((connection) => connection.connectionId === (location.state as SettingsLocationState).focusConnectionId)
    ? translate("settings.session_only_saved_notice") : "";
  const connectionNotice = (location.state as SettingsLocationState | null)?.connectionNotice || "";
  const openSetup = (connectionId: string) => navigate("/runtime-setup", {
    state: { connectionId, returnTo: "/settings", settingsFrom: returnOrigin, from: returnOrigin },
  });

  const handleInvalidateRuntime = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.runtime }),
      queryClient.invalidateQueries({ queryKey: queryKeys.diagnostics }),
      queryClient.invalidateQueries({ queryKey: queryKeys.readiness }),
      queryClient.invalidateQueries({ queryKey: ["runtime", "settings"] }),
      queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] }),
    ]);
  };
  const handleSavePreferences = async () => {
    if (!activeConnection) return;
    setIsBusy(true);
    setPreferencesMessage("");
    try {
      const response = await apiClient.putRuntimeSettings({
        ui_locale: selectedUiLocale,
        whisper_model: selectedWhisperModel,
        connection: {
          connection_id: activeConnection.connection_id,
          provider_choice: activeConnection.provider_choice,
          label: activeConnection.label,
          model: activeConnection.model,
          base_url: activeConnection.base_url,
          api_key: "",
          openrouter_http_referer: activeConnection.openrouter_http_referer,
          openrouter_app_title: activeConnection.openrouter_app_title,
        },
      });
      setUiLocale(response.ui_locale as UiLocale);
      setPreferencesMessage(translate("settings.saved"));
      await handleInvalidateRuntime();
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setPreferencesMessage(translate("settings.save_failed", { detail }));
    } finally { setIsBusy(false); }
  };
  const handleSetDefault = async (connectionId: string) => {
    setIsBusy(true);
    setConnectionMessage("");
    try {
      const response = await apiClient.postRuntimeSettingsSetDefault(connectionId);
      setActiveConnectionId(response.active_connection_id);
      setSetupComplete(response.connections.length > 0);
      setSelectedConnectionId(connectionId);
      await handleInvalidateRuntime();
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setConnectionMessage(translate("settings.default_failed", { detail }));
    } finally { setIsBusy(false); }
  };
  const handleDelete = async (connectionId: string) => {
    setIsBusy(true);
    setConnectionMessage("");
    try {
      const response = await apiClient.deleteRuntimeSettingsConnection(connectionId);
      setActiveConnectionId(response.active_connection_id);
      setSetupComplete(response.connections.length > 0);
      setSelectedConnectionId(response.active_connection_id || "");
      await handleInvalidateRuntime();
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setConnectionMessage(translate("settings.delete_failed", { detail }));
    } finally { setIsBusy(false); }
  };
  const handleDownloadModel = async () => {
    setIsBusy(true);
    setWhisperMessage("");
    try {
      const status = await apiClient.postWhisperModelDownload(selectedWhisperModel);
      setWhisperMessage(translate("settings.whisper_downloaded", { path: status.cached_path || status.model }));
      await queryClient.invalidateQueries({ queryKey: ["runtime", "whisper", selectedWhisperModel] });
      await queryClient.invalidateQueries({ queryKey: queryKeys.readiness });
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setWhisperMessage(translate("settings.whisper_download_failed", { detail }));
    } finally { setIsBusy(false); }
  };
  const whisperStatusMessage = whisperStatusQuery.data?.cached
    ? translate("settings.whisper_cached", { path: whisperStatusQuery.data.cached_path })
    : whisperMessage || translate("settings.whisper_not_cached");

  return (
    <div style={{ display: "grid", gap: "1rem" }} {...semanticAttributes(SEMANTIC_IDS.settings.screen)}>
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.75rem", color: "var(--color-text)" }}>{translate("settings.title")}</h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "var(--color-text-muted)" }}>{translate("settings.body")}</p>
        {!readinessQuery.isPending && !runtimeReady ? <p style={{ margin: 0, color: "var(--color-warning)", lineHeight: 1.6 }}>{translate("settings.needs_setup_info")}</p> : null}
      </section>
      <JournalPanel locale={locale} />
      <SavedConnectionsPanel
        connections={connections}
        locale={locale}
        onDelete={(connectionId) => { void handleDelete(connectionId); }}
        onEdit={openSetup}
        onOpenSetup={() => openSetup("__new__")}
        onSelectConnection={setSelectedConnectionId}
        onSetDefault={(connectionId) => { void handleSetDefault(connectionId); }}
        selectedConnectionId={selectedConnectionId ?? ""}
      />
      {connectionMessage ? <p role="alert">{connectionMessage}</p> : null}
      {setupNotice ? <p role="status">{setupNotice}</p> : null}
      {connectionNotice ? <p role="status">{connectionNotice}</p> : null}
      {runtimeSettingsQuery.isError ? <p role="alert">{translate("settings.load_failed")}</p> : null}
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
          {translate("settings.appearance")}
        </h2>
        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("settings.appearance")}
          </span>
          <select
            value={preferences.appearance}
            onChange={(event) => setAppearance(event.currentTarget.value as "system" | "light" | "dark")}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.appearance)}
          >
            <option value="system">{translate("settings.appearance_system")}</option>
            <option value="light">{translate("settings.appearance_light")}</option>
            <option value="dark">{translate("settings.appearance_dark")}</option>
          </select>
        </label>
      </section>
      <section
        style={cardStyle}
        {...semanticAttributes(SEMANTIC_IDS.settings.sectionRuntimeDefaults)}
      >
        <div style={{ display: "grid", gap: "0.35rem" }}>
          <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
            {translate("settings.ui_locale")}
          </h2>
          <p style={{ margin: 0, lineHeight: 1.6, color: "var(--color-text-muted)" }}>
            {translate("settings.body")}
          </p>
        </div>

        <label style={{ display: "grid", gap: "0.35rem" }}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
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
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("settings.whisper_model")}
          </span>
          <select
            value={selectedWhisperModel}
            onChange={(event) => setSelectedWhisperModel(event.target.value)}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.whisperModel)}
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

        <Button disabled={isBusy || !activeConnection} onClick={() => void handleSavePreferences()} style={{ width: "fit-content" }} {...semanticAttributes(SEMANTIC_IDS.settings.savePreferences)}>
          {translate("settings.save_preferences")}
        </Button>
        {!activeConnection ? <p role="status">{translate("settings.preferences_need_connection")}</p> : null}
        {preferencesMessage ? <p role="status">{preferencesMessage}</p> : null}

        <p style={{ margin: 0, lineHeight: 1.55, color: whisperStatusQuery.data?.cached ? "var(--color-success)" : "var(--color-warning)" }}>
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
            <strong style={{ color: "var(--color-text-muted)" }}>{translate("settings.provider")}</strong>
            <span style={{ color: "var(--color-text)" }}>{activeConnection?.provider_label || runtimeQuery.data?.provider || "-"}</span>
          </article>
          <article style={cardStyle}>
            <strong style={{ color: "var(--color-text-muted)" }}>{translate("settings.model")}</strong>
            <span style={{ color: "var(--color-text)" }}>{activeConnection?.model || runtimeQuery.data?.model || "-"}</span>
          </article>
          <article style={cardStyle}>
            <strong style={{ color: "var(--color-text-muted)" }}>{translate("settings.base_url")}</strong>
            <span style={{ color: "var(--color-text)", lineHeight: 1.5 }}>
              {activeConnection?.base_url || runtimeQuery.data?.base_url || "-"}
            </span>
          </article>
        </div>
      </section>

      <SupportPanel
        activeConnectionId={runtimeSettingsQuery.data?.active_connection_id || preferences.activeConnectionId}
        locale={locale}
      />

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
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
