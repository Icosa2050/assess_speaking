import { useEffect, useMemo, useState } from "react";

import type { ConnectionSecretState, RuntimeConnectionDraft } from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";

type RuntimeConnectionFormState =
  | "idle"
  | "editing"
  | "testing"
  | "test-succeeded"
  | "test-failed"
  | "saving"
  | "saved";

type RuntimeConnectionFormVariant = "runtime-setup" | "settings";

const PROVIDER_CHOICES = [
  "ollama_local",
  "ollama_cloud",
  "lmstudio_local",
  "openrouter",
  "openai_compatible",
] as const;

const PROVIDER_DEFAULT_BASE_URLS: Record<(typeof PROVIDER_CHOICES)[number], string> = {
  ollama_local: "http://localhost:11434",
  ollama_cloud: "https://ollama.com/api",
  lmstudio_local: "http://localhost:1234/v1",
  openrouter: "https://openrouter.ai/api/v1",
  openai_compatible: "",
};

const providerDefaultBaseUrl = (providerChoice: string): string =>
  PROVIDER_DEFAULT_BASE_URLS[providerChoice as keyof typeof PROVIDER_DEFAULT_BASE_URLS] ?? "";

const normalizeBaseUrlForComparison = (value: string): string =>
  String(value || "").trim().replace(/\/+$/, "");

const LOCAL_PROVIDER_CHOICES = new Set<string>(["ollama_local", "lmstudio_local"]);

const sectionStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1rem",
  border: "1px solid rgba(18, 61, 55, 0.1)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.92)",
} as const;

const fieldStyle = {
  display: "grid",
  gap: "0.35rem",
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

const normalizeDraft = (draft: RuntimeConnectionDraft): RuntimeConnectionDraft => ({
  connection_id: String(draft.connection_id || "").trim(),
  provider_choice: String(draft.provider_choice || "").trim(),
  label: String(draft.label || "").trim(),
  model: String(draft.model || "").trim(),
  base_url: String(draft.base_url || "").trim(),
  api_key: String(draft.api_key || ""),
  openrouter_http_referer: String(draft.openrouter_http_referer || "").trim(),
  openrouter_app_title: String(draft.openrouter_app_title || "").trim(),
});

const providerLabel = (
  translate: ReturnType<typeof createTranslator>,
  providerChoice: string,
): string => translate(`runtime_setup.provider_options.${providerChoice}.label`, { value: providerChoice });

const providerHint = (
  translate: ReturnType<typeof createTranslator>,
  providerChoice: string,
): string => translate(`runtime_setup.provider_options.${providerChoice}.hint`, { value: providerChoice });

const formStateColor = (state: RuntimeConnectionFormState): string => {
  if (state === "saved" || state === "test-succeeded") {
    return "#166534";
  }
  if (state === "test-failed") {
    return "#b42318";
  }
  if (state === "testing" || state === "saving" || state === "editing") {
    return "#9a6700";
  }
  return "#33514b";
};

type RuntimeConnectionFormProps = {
  detectedModelMessage?: string;
  detectedModels?: string[];
  initialDraft: RuntimeConnectionDraft;
  initialSecretState: ConnectionSecretState;
  isBusy?: boolean;
  locale: UiLocale;
  onBack?: () => void;
  onDetectLocalModels?: (draft: RuntimeConnectionDraft) => void;
  onSave: (payload: { clearSavedSecret: boolean; draft: RuntimeConnectionDraft }) => void;
  onTest: (draft: RuntimeConnectionDraft) => void;
  resetToken?: string;
  status?: RuntimeConnectionFormState;
  statusMessage?: string;
  variant?: RuntimeConnectionFormVariant;
};

export const RuntimeConnectionForm = ({
  detectedModelMessage = "",
  detectedModels = [],
  initialDraft,
  initialSecretState,
  isBusy = false,
  locale,
  onBack,
  onDetectLocalModels,
  onSave,
  onTest,
  resetToken = "",
  status = "idle",
  statusMessage = "",
  variant = "runtime-setup",
}: RuntimeConnectionFormProps) => {
  const translate = createTranslator(locale);
  const initialDraftSeed = useMemo(() => JSON.stringify(normalizeDraft(initialDraft)), [initialDraft]);

  const [draft, setDraft] = useState<RuntimeConnectionDraft>(() => normalizeDraft(initialDraft));
  const [clearStage, setClearStage] = useState<"idle" | "confirming" | "undo">("idle");

  useEffect(() => {
    setDraft(normalizeDraft(initialDraft));
    setClearStage("idle");
  }, [initialDraftSeed, initialSecretState, resetToken]);

  const normalizedInitialDraft = useMemo(
    () => normalizeDraft(initialDraft),
    [initialDraftSeed],
  );
  const normalizedDraft = useMemo(() => normalizeDraft(draft), [draft]);
  const isDirty =
    JSON.stringify(normalizedDraft) !== JSON.stringify(normalizedInitialDraft) || clearStage !== "idle";

  const effectiveState: RuntimeConnectionFormState =
    status === "testing" || status === "saving"
      ? status
      : isDirty
        ? "editing"
        : status;

  const providerChoice = normalizedDraft.provider_choice || "ollama_local";
  const requiresApiKey = providerChoice === "openrouter";
  const copyRoot = variant === "settings" ? "settings" : "runtime_setup";
  const providerFieldLabel =
    variant === "settings"
      ? translate("settings.provider")
      : translate("runtime_setup.provider_label");
  const modelSelectEnabled = LOCAL_PROVIDER_CHOICES.has(providerChoice);
  const secretStateValue =
    clearStage === "confirming"
      ? "confirming-clear"
      : clearStage === "undo"
        ? "cleared-undoable"
        : initialSecretState;

  const updateDraft = (updates: Partial<RuntimeConnectionDraft>) => {
    setDraft((current) => ({
      ...current,
      ...updates,
    }));
  };

  const handleProviderChange = (nextProviderChoice: string) => {
    setDraft((current) => {
      const previousProviderChoice = current.provider_choice || "ollama_local";
      const previousLabel = providerLabel(translate, previousProviderChoice);
      const nextLabel = providerLabel(translate, nextProviderChoice);
      const keepAutoLabel = !current.label || current.label === previousLabel;
      const currentBaseUrl = String(current.base_url || "").trim();
      const keepAutoBaseUrl =
        !currentBaseUrl ||
        normalizeBaseUrlForComparison(currentBaseUrl) ===
          normalizeBaseUrlForComparison(providerDefaultBaseUrl(previousProviderChoice));

      return {
        ...current,
        provider_choice: nextProviderChoice,
        label: keepAutoLabel ? nextLabel : current.label,
        base_url: keepAutoBaseUrl ? providerDefaultBaseUrl(nextProviderChoice) : current.base_url,
      };
    });
    setClearStage("idle");
  };

  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.form, { state: effectiveState })}
    >
      <section style={sectionStyle}>
        {variant === "runtime-setup" ? (
          <>
            <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
              {translate("runtime_setup.section_provider")}
            </h2>
            <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
              {translate("runtime_setup.local_first_intro")}
            </p>
          </>
        ) : null}

        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>{providerFieldLabel}</span>
          <select
            value={providerChoice}
            onChange={(event) => handleProviderChange(event.target.value)}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.provider)}
          >
            {PROVIDER_CHOICES.map((option) => (
              <option
                key={option}
                value={option}
              >
                {providerLabel(translate, option)}
              </option>
            ))}
          </select>
        </label>

        <p style={{ margin: 0, lineHeight: 1.55, color: "#33514b" }}>
          {providerHint(translate, providerChoice)}
        </p>

        {providerChoice === "ollama_cloud" ? (
          <p style={{ margin: 0, lineHeight: 1.55, color: "#33514b" }}>
            {translate("runtime_setup.ollama_cloud_note")}
          </p>
        ) : null}

        {providerChoice === "openai_compatible" ? (
          <p style={{ margin: 0, lineHeight: 1.55, color: "#33514b" }}>
            {translate("runtime_setup.openai_compatible_note")}
          </p>
        ) : null}
      </section>

      <section style={sectionStyle}>
        {variant === "runtime-setup" ? (
          <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
            {translate("runtime_setup.section_connection")}
          </h2>
        ) : null}

        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate(`${copyRoot}.connection_label`)}
          </span>
          <input
            value={normalizedDraft.label}
            onChange={(event) => updateDraft({ label: event.target.value })}
            style={inputStyle}
            type="text"
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.connectionLabel)}
          />
        </label>

        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate(`${copyRoot}.model`)}
          </span>
          <input
            value={normalizedDraft.model}
            onChange={(event) => updateDraft({ model: event.target.value })}
            style={inputStyle}
            type="text"
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.model)}
          />
        </label>

        {detectedModels.length > 0 ? (
          <label style={fieldStyle}>
            <span style={{ fontWeight: 600, color: "#33514b" }}>
              {translate("runtime_setup.detected_local_models_label")}
            </span>
            <select
              value={detectedModels.includes(normalizedDraft.model) ? normalizedDraft.model : ""}
              onChange={(event) => updateDraft({ model: event.target.value })}
              style={inputStyle}
            >
              <option value="">{translate("runtime_setup.detected_local_models_help")}</option>
              {detectedModels.map((item) => (
                <option
                  key={item}
                  value={item}
                >
                  {item}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {detectedModelMessage ? (
          <p style={{ margin: 0, lineHeight: 1.55, color: "#33514b" }}>{detectedModelMessage}</p>
        ) : null}

        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate(`${copyRoot}.base_url`)}
          </span>
          <input
            value={normalizedDraft.base_url}
            onChange={(event) => updateDraft({ base_url: event.target.value })}
            placeholder={translate("runtime_setup.base_url_placeholder")}
            style={inputStyle}
            type="text"
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.baseUrl)}
          />
        </label>

        {onDetectLocalModels ? (
          <button
            type="button"
            disabled={!modelSelectEnabled || isBusy}
            onClick={() => onDetectLocalModels(normalizedDraft)}
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.detectLocalModels)}
          >
            {translate("runtime_setup.detect_local_models")}
          </button>
        ) : null}

        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate(`${copyRoot}.api_key`)}
          </span>
          <input
            value={normalizedDraft.api_key}
            onChange={(event) => updateDraft({ api_key: event.target.value })}
            style={inputStyle}
            type="password"
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.apiKey)}
          />
        </label>

        {providerChoice === "openrouter" ? (
          <>
            <label style={fieldStyle}>
              <span style={{ fontWeight: 600, color: "#33514b" }}>
                {translate(`${copyRoot}.openrouter_http_referer`)}
              </span>
              <input
                value={normalizedDraft.openrouter_http_referer || ""}
                onChange={(event) => updateDraft({ openrouter_http_referer: event.target.value })}
                style={inputStyle}
                type="text"
              />
            </label>

            <label style={fieldStyle}>
              <span style={{ fontWeight: 600, color: "#33514b" }}>
                {translate(`${copyRoot}.openrouter_app_title`)}
              </span>
              <input
                value={normalizedDraft.openrouter_app_title || ""}
                onChange={(event) => updateDraft({ openrouter_app_title: event.target.value })}
                style={inputStyle}
                type="text"
              />
            </label>
          </>
        ) : null}

        <div
          style={{ display: "grid", gap: "0.5rem" }}
          {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.secretState, { secretState: secretStateValue })}
        >
          {clearStage === "confirming" ? (
            <>
              <p style={{ margin: 0, lineHeight: 1.55, color: "#9a6700" }}>
                {translate(`${copyRoot}.clear_saved_key_confirm`)}
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
                <button
                  type="button"
                  onClick={() => setClearStage("undo")}
                  style={actionButtonStyle}
                  {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.clearSavedKeyConfirm)}
                >
                  {translate(`${copyRoot}.clear_saved_key_confirm_button`)}
                </button>
                <button
                  type="button"
                  onClick={() => setClearStage("idle")}
                  style={actionButtonStyle}
                  {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.clearSavedKeyCancel)}
                >
                  {translate(`${copyRoot}.clear_saved_key_cancel`)}
                </button>
              </div>
            </>
          ) : clearStage === "undo" ? (
            <>
              <p style={{ margin: 0, lineHeight: 1.55, color: "#9a6700" }}>
                {translate(`${copyRoot}.secret_clear_pending`)}
              </p>
              <button
                type="button"
                onClick={() => setClearStage("idle")}
                style={{ ...actionButtonStyle, width: "fit-content" }}
                {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.clearSavedKeyUndo)}
              >
                {translate(`${copyRoot}.clear_saved_key_undo`)}
              </button>
            </>
          ) : initialSecretState === "present" ? (
            <>
              <p style={{ margin: 0, lineHeight: 1.55, color: "#33514b" }}>
                {translate(`${copyRoot}.secret_saved_state`)}
              </p>
              <p style={{ margin: 0, lineHeight: 1.55, color: "#33514b" }}>
                {translate(`${copyRoot}.secret_keep_blank_hint`)}
              </p>
            </>
          ) : initialSecretState === "missing" ? (
            <p style={{ margin: 0, lineHeight: 1.55, color: "#9a6700" }}>
              {translate(`${copyRoot}.secret_missing_state`)}
            </p>
          ) : null}

          {(initialSecretState === "present" || initialSecretState === "missing") && clearStage === "idle" ? (
            <button
              type="button"
              onClick={() => setClearStage("confirming")}
              style={{ ...actionButtonStyle, width: "fit-content" }}
              {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.clearSavedKey)}
            >
              {translate(`${copyRoot}.clear_saved_key`)}
            </button>
          ) : null}
        </div>
      </section>

      <section style={sectionStyle}>
        {variant === "runtime-setup" ? (
          <>
            <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
              {translate("runtime_setup.section_actions")}
            </h2>
            <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
              {translate("runtime_setup.actions_body")}
            </p>
          </>
        ) : null}

        {statusMessage ? (
          <p
            style={{
              margin: 0,
              lineHeight: 1.55,
              color: formStateColor(effectiveState),
            }}
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.formStatus, { state: effectiveState })}
          >
            {statusMessage}
          </p>
        ) : null}

        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
          <button
            type="button"
            disabled={isBusy}
            onClick={() => onTest(normalizedDraft)}
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.testConnection)}
          >
            {translate(`${copyRoot}.test_connection`)}
          </button>
          <button
            type="button"
            disabled={isBusy}
            onClick={() =>
              onSave({
                draft: normalizedDraft,
                clearSavedSecret: clearStage === "undo",
              })
            }
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.runtimeConnection.saveConnection)}
          >
            {translate(copyRoot === "settings" ? "settings.save" : "runtime_setup.save_connection")}
          </button>
          {onBack ? (
            <button
              type="button"
              disabled={isBusy}
              onClick={onBack}
              style={actionButtonStyle}
              {...semanticAttributes(SEMANTIC_IDS.runtimeSetup.backHome)}
            >
              {translate("runtime_setup.back_home")}
            </button>
          ) : null}
        </div>
      </section>
    </div>
  );
};
