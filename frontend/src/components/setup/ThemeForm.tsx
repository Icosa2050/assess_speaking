import { CEFR_LEVELS, DURATION_OPTIONS, type CefrLevel, type DurationOption } from "@/lib/state/sessionDraft";

export interface ThemeOption {
  title: string;
}

type Translate = (key: string, vars?: Record<string, string | number>) => string;
export type SessionSetupStep = "learner" | "practice";

const sectionStyle = {
  display: "grid",
  gap: "1rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const labelStyle = {
  display: "grid",
  gap: "0.375rem",
  minWidth: 0,
  color: "#10201c",
  fontWeight: 600,
} as const;

const controlStyle = {
  width: "100%",
  boxSizing: "border-box",
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  font: "inherit",
} as const;

const checkboxRowStyle = {
  display: "flex",
  alignItems: "start",
  gap: "0.625rem",
  color: "#10201c",
} as const;

const fieldsetStyle = {
  display: "grid",
  gap: "0.875rem",
  width: "100%",
  boxSizing: "border-box",
  minInlineSize: 0,
  margin: 0,
  padding: "0.25rem 0 0.25rem 0.875rem",
  border: "0",
  borderLeft: "3px solid rgba(15, 118, 110, 0.18)",
} as const;

const legendStyle = {
  padding: 0,
  marginBottom: "0.125rem",
  color: "#10201c",
  fontWeight: 700,
} as const;

const groupBodyStyle = {
  margin: 0,
  color: "#48645e",
  lineHeight: 1.5,
} as const;

const helperTextStyle = {
  margin: 0,
  color: "#48645e",
  fontSize: "0.92rem",
  lineHeight: 1.45,
} as const;

const stepperStyle = {
  display: "grid",
  gap: "0.75rem",
  padding: "0.875rem",
  borderRadius: "8px",
  backgroundColor: "rgba(239, 248, 245, 0.96)",
  border: "1px solid rgba(15, 118, 110, 0.12)",
} as const;

const stepListStyle = {
  display: "grid",
  gap: "0.5rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(8.75rem, 1fr))",
} as const;

const actionRowStyle = {
  display: "flex",
  flexWrap: "wrap",
  gap: "0.75rem",
  alignItems: "center",
} as const;

const secondaryButtonStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.95rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  fontWeight: 700,
  font: "inherit",
  cursor: "pointer",
} as const;

const primaryButtonStyle = {
  minHeight: "44px",
  padding: "0.875rem 1rem",
  borderRadius: "8px",
  border: "1px solid rgba(15, 118, 110, 0.2)",
  backgroundColor: "#d7ebe5",
  color: "#10201c",
  fontWeight: 700,
  font: "inherit",
  cursor: "pointer",
} as const;

const semanticAttributes = (id: string): Record<string, string> => ({
  "data-testid": id,
  "data-semantic-id": id,
});

const SETUP_ERRORS_ID = "setup-errors";
const SETUP_SPEAKER_ERROR_ID = "setup-error-speaker-id";
const SETUP_THEME_ERROR_ID = "setup-error-theme";

const isCefrLevel = (value: string): value is CefrLevel =>
  (CEFR_LEVELS as readonly string[]).includes(value);

const isDurationOption = (value: number): value is DurationOption =>
  (DURATION_OPTIONS as readonly number[]).includes(value);

export const ThemeForm = ({
  advancedTopicOpen,
  availableLanguages,
  availableThemes,
  customTheme,
  customThemeEnabled,
  customThemeSaveEnabled,
  errors,
  languageLabelFor,
  onAdvancedTopicOpenChange,
  onCustomThemeChange,
  onRecommendedStart,
  onSaveCustomThemeForReuseChange,
  onSelectedCefrChange,
  onSelectedDurationChange,
  onSelectedLanguageChange,
  onSetupStepChange,
  onSelectedThemeModeChange,
  onSelectedThemeTitleChange,
  onSpeakerIdChange,
  onSubmit,
  primaryActionLabel,
  recommendationHint,
  saveCustomThemeForReuse,
  selectedCefr,
  selectedDuration,
  selectedLanguage,
  setupStep,
  selectedThemeMode,
  selectedThemeTitle,
  speakerId,
  translate,
}: {
  advancedTopicOpen: boolean;
  availableLanguages: string[];
  availableThemes: ThemeOption[];
  customTheme: string;
  customThemeEnabled: boolean;
  customThemeSaveEnabled: boolean;
  errors: string[];
  languageLabelFor: (languageCode: string) => string;
  onAdvancedTopicOpenChange: (open: boolean) => void;
  onCustomThemeChange: (value: string) => void;
  onRecommendedStart: () => void;
  onSaveCustomThemeForReuseChange: (value: boolean) => void;
  onSelectedCefrChange: (value: CefrLevel) => void;
  onSelectedDurationChange: (value: DurationOption) => void;
  onSelectedLanguageChange: (value: string) => void;
  onSetupStepChange: (step: SessionSetupStep) => void;
  onSelectedThemeModeChange: (value: "library" | "custom") => void;
  onSelectedThemeTitleChange: (value: string) => void;
  onSpeakerIdChange: (value: string) => void;
  onSubmit: () => void;
  primaryActionLabel: string;
  recommendationHint: string;
  saveCustomThemeForReuse: boolean;
  selectedCefr: CefrLevel;
  selectedDuration: DurationOption;
  selectedLanguage: string;
  setupStep: SessionSetupStep;
  selectedThemeMode: "library" | "custom";
  selectedThemeTitle: string;
  speakerId: string;
  translate: Translate;
}) => {
  const speakerError = translate("setup.error_speaker_id");
  const themeError = translate("setup.error_theme");
  const hasErrors = errors.length > 0;
  const hasSpeakerError = errors.includes(speakerError);
  const hasThemeError = errors.includes(themeError);
  const describedBy = (fieldErrorId?: string): string | undefined =>
    hasErrors ? [SETUP_ERRORS_ID, fieldErrorId].filter(Boolean).join(" ") : undefined;
  const speakerDescriptionId = "setup-speaker-id-help";
  const cefrDescriptionId = "setup-cefr-help";
  const durationDescriptionId = "setup-duration-help";
  const speakerDescribedBy = [speakerDescriptionId, describedBy(hasSpeakerError ? SETUP_SPEAKER_ERROR_ID : undefined)]
    .filter(Boolean)
    .join(" ");
  const themeDescribedBy = describedBy(hasThemeError ? SETUP_THEME_ERROR_ID : undefined);

  const stepButtonStyle = (active: boolean) =>
    ({
      minHeight: "44px",
      padding: "0.7rem 0.8rem",
      borderRadius: "8px",
      border: active ? "1px solid rgba(15, 118, 110, 0.38)" : "1px solid rgba(18, 61, 55, 0.12)",
      backgroundColor: active ? "#d7ebe5" : "rgba(255, 255, 255, 0.86)",
      color: "#10201c",
      cursor: "pointer",
      font: "inherit",
      fontWeight: 700,
      textAlign: "left",
    }) as const;

  return (
    <section style={sectionStyle}>
      <div style={stepperStyle} {...semanticAttributes("setup.wizard")}>
        <p style={helperTextStyle}>{translate("setup.wizard_intro")}</p>
        <div style={stepListStyle}>
          <button
            type="button"
            onClick={() => onSetupStepChange("learner")}
            aria-current={setupStep === "learner" ? "step" : undefined}
            style={stepButtonStyle(setupStep === "learner")}
            {...semanticAttributes("setup.step_learner")}
          >
            1. {translate("setup.wizard_step_learner")}
          </button>
          <button
            type="button"
            onClick={() => onSetupStepChange("practice")}
            aria-current={setupStep === "practice" ? "step" : undefined}
            style={stepButtonStyle(setupStep === "practice")}
            {...semanticAttributes("setup.step_practice")}
          >
            2. {translate("setup.wizard_step_practice")}
          </button>
        </div>
      </div>

      <h2
        style={{
          margin: 0,
          fontSize: "1.5rem",
          color: "#10201c",
        }}
      >
        {translate("setup.title")}
      </h2>
      <p
        style={{
          margin: 0,
          lineHeight: 1.6,
          color: "#33514b",
        }}
      >
        {translate("setup.body")}
      </p>

      {errors.length > 0 ? (
        <ul
          id={SETUP_ERRORS_ID}
          role="alert"
          aria-live="assertive"
          style={{
            margin: 0,
            paddingLeft: "1.125rem",
            color: "#b42318",
            display: "grid",
            gap: "0.375rem",
          }}
        >
          {errors.map((error, index) => (
            <li
              id={
                error === speakerError
                  ? SETUP_SPEAKER_ERROR_ID
                  : error === themeError
                    ? SETUP_THEME_ERROR_ID
                    : `${SETUP_ERRORS_ID}-${index}`
              }
              key={`${error}-${index}`}
            >
              {error}
            </li>
          ))}
        </ul>
      ) : null}

      <fieldset style={fieldsetStyle}>
        <legend style={legendStyle}>{translate("setup.learner_profile_title")}</legend>
        <p style={groupBodyStyle}>{translate("setup.learner_profile_body")}</p>

        <label htmlFor="setup-speaker-id" style={labelStyle}>
          <span>{translate("setup.speaker_id")}</span>
          <input
            id="setup-speaker-id"
            type="text"
            value={speakerId}
            onChange={(event) => onSpeakerIdChange(event.currentTarget.value)}
            aria-describedby={speakerDescribedBy}
            aria-invalid={hasSpeakerError}
            style={controlStyle}
            {...semanticAttributes("setup.speaker_id")}
          />
        </label>
        <p id={speakerDescriptionId} style={helperTextStyle}>
          {translate("setup.speaker_id_help")}
        </p>
        <p
          style={{
            ...helperTextStyle,
            fontWeight: 600,
          }}
          {...semanticAttributes("setup.recommendation_hint")}
        >
          {recommendationHint}
        </p>
        {setupStep === "learner" ? (
          <div style={actionRowStyle}>
            <button
              type="button"
              onClick={onRecommendedStart}
              style={primaryButtonStyle}
              {...semanticAttributes("setup.recommended_start")}
            >
              {translate("setup.recommended_start")}
            </button>
            <button
              type="button"
              onClick={() => onSetupStepChange("practice")}
              style={secondaryButtonStyle}
              {...semanticAttributes("setup.customize_details")}
            >
              {translate("setup.customize_details")}
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => onSetupStepChange("learner")}
            style={secondaryButtonStyle}
            {...semanticAttributes("setup.change_details")}
          >
            {translate("setup.change_details")}
          </button>
        )}
      </fieldset>

      {setupStep === "practice" ? (
        <fieldset style={fieldsetStyle}>
          <legend style={legendStyle}>{translate("setup.session_goal_title")}</legend>
          <p style={groupBodyStyle}>{translate("setup.session_goal_body")}</p>

          <label htmlFor="setup-learning-language" style={labelStyle}>
            <span>{translate("setup.learning_language")}</span>
            <select
              id="setup-learning-language"
              value={selectedLanguage}
              onChange={(event) => onSelectedLanguageChange(event.currentTarget.value)}
              style={controlStyle}
              {...semanticAttributes("setup.learning_language")}
            >
              {availableLanguages.map((languageCode) => (
                <option key={languageCode} value={languageCode}>
                  {languageLabelFor(languageCode)}
                </option>
              ))}
            </select>
          </label>

          <label htmlFor="setup-cefr-level" style={labelStyle}>
            <span>{translate("setup.cefr")}</span>
            <select
              id="setup-cefr-level"
              value={selectedCefr}
              onChange={(event) => {
                const value = event.currentTarget.value;
                if (isCefrLevel(value)) {
                  onSelectedCefrChange(value);
                }
              }}
              aria-describedby={cefrDescriptionId}
              style={controlStyle}
              {...semanticAttributes("setup.cefr")}
            >
              {CEFR_LEVELS.map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </label>
          <p id={cefrDescriptionId} style={helperTextStyle}>
            {translate("setup.cefr_help")}
          </p>

          <label htmlFor="setup-theme" style={labelStyle}>
            <span>{translate("setup.theme")}</span>
            <select
              id="setup-theme"
              value={selectedThemeTitle}
              onChange={(event) => {
                const value = event.currentTarget.value;
                onSelectedThemeModeChange("library");
                onSelectedThemeTitleChange(value);
                onAdvancedTopicOpenChange(false);
              }}
              aria-describedby={themeDescribedBy}
              aria-invalid={hasThemeError}
              style={controlStyle}
              {...semanticAttributes("setup.theme")}
            >
              {availableThemes.map((theme) => (
                <option key={theme.title} value={theme.title}>
                  {theme.title}
                </option>
              ))}
            </select>
          </label>

          <details
            open={advancedTopicOpen}
            onToggle={(event) => onAdvancedTopicOpenChange(event.currentTarget.open)}
          >
            <summary
              onClick={(event) => {
                event.preventDefault();
                onAdvancedTopicOpenChange(!advancedTopicOpen);
              }}
              style={{
                color: "#10201c",
                cursor: "pointer",
                fontWeight: 700,
              }}
              {...semanticAttributes("setup.advanced_topic")}
            >
              {translate("setup.advanced_topic_summary")}
            </summary>
            {customThemeEnabled ? (
              <div
                style={{
                  display: "grid",
                  gap: "0.875rem",
                  paddingTop: "0.875rem",
                }}
              >
                <label htmlFor="setup-custom-theme" style={labelStyle}>
                  <span>{translate("setup.custom_theme_label")}</span>
                  <input
                    id="setup-custom-theme"
                    type="text"
                    value={customTheme}
                    onChange={(event) => onCustomThemeChange(event.currentTarget.value)}
                    aria-describedby={themeDescribedBy}
                    aria-invalid={hasThemeError && selectedThemeMode === "custom"}
                    style={controlStyle}
                    {...semanticAttributes("setup.custom_theme")}
                  />
                </label>

                <label style={checkboxRowStyle}>
                  <input
                    type="checkbox"
                    checked={saveCustomThemeForReuse}
                    disabled={!customThemeSaveEnabled}
                    onChange={(event) => onSaveCustomThemeForReuseChange(event.currentTarget.checked)}
                    style={{
                      marginTop: "0.25rem",
                    }}
                    {...semanticAttributes("setup.save_custom_theme")}
                  />
                  <span
                    style={{
                      opacity: customThemeSaveEnabled ? 1 : 0.6,
                    }}
                  >
                    {translate("setup.custom_theme_save")}
                  </span>
                </label>
              </div>
            ) : null}
          </details>

          <label htmlFor="setup-duration" style={labelStyle}>
            <span>{translate("setup.duration")}</span>
            <select
              id="setup-duration"
              value={String(selectedDuration)}
              onChange={(event) => {
                const value = Number(event.currentTarget.value);
                if (isDurationOption(value)) {
                  onSelectedDurationChange(value);
                }
              }}
              aria-describedby={durationDescriptionId}
              style={controlStyle}
              {...semanticAttributes("setup.duration")}
            >
              {DURATION_OPTIONS.map((duration) => (
                <option key={duration} value={duration}>
                  {duration}
                </option>
              ))}
            </select>
          </label>
          <p id={durationDescriptionId} style={helperTextStyle}>
            {translate("setup.duration_help")}
          </p>
        </fieldset>
      ) : null}

      {setupStep === "practice" ? (
        <button
          type="button"
          onClick={onSubmit}
          style={primaryButtonStyle}
          {...semanticAttributes("setup.continue")}
        >
          {primaryActionLabel}
        </button>
      ) : null}
    </section>
  );
};
