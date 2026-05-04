import { CEFR_LEVELS, DURATION_OPTIONS, type CefrLevel, type DurationOption } from "@/lib/state/sessionDraft";

export interface ThemeOption {
  title: string;
}

type Translate = (key: string, vars?: Record<string, string | number>) => string;

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
  color: "#10201c",
  fontWeight: 600,
} as const;

const controlStyle = {
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

const semanticAttributes = (id: string): Record<string, string> => ({
  "data-testid": id,
  "data-semantic-id": id,
});

export const ThemeForm = ({
  availableLanguages,
  availableThemes,
  customTheme,
  customThemeEnabled,
  customThemeSaveEnabled,
  errors,
  languageLabelFor,
  onCustomThemeChange,
  onSaveCustomThemeForReuseChange,
  onSelectedCefrChange,
  onSelectedDurationChange,
  onSelectedLanguageChange,
  onSelectedThemeModeChange,
  onSelectedThemeTitleChange,
  onSpeakerIdChange,
  onSubmit,
  saveCustomThemeForReuse,
  selectedCefr,
  selectedDuration,
  selectedLanguage,
  selectedThemeMode,
  selectedThemeTitle,
  speakerId,
  translate,
}: {
  availableLanguages: string[];
  availableThemes: ThemeOption[];
  customTheme: string;
  customThemeEnabled: boolean;
  customThemeSaveEnabled: boolean;
  errors: string[];
  languageLabelFor: (languageCode: string) => string;
  onCustomThemeChange: (value: string) => void;
  onSaveCustomThemeForReuseChange: (value: boolean) => void;
  onSelectedCefrChange: (value: CefrLevel) => void;
  onSelectedDurationChange: (value: DurationOption) => void;
  onSelectedLanguageChange: (value: string) => void;
  onSelectedThemeModeChange: (value: "library" | "custom") => void;
  onSelectedThemeTitleChange: (value: string) => void;
  onSpeakerIdChange: (value: string) => void;
  onSubmit: () => void;
  saveCustomThemeForReuse: boolean;
  selectedCefr: CefrLevel;
  selectedDuration: DurationOption;
  selectedLanguage: string;
  selectedThemeMode: "library" | "custom";
  selectedThemeTitle: string;
  speakerId: string;
  translate: Translate;
}) => (
  <section style={sectionStyle}>
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
        style={{
          margin: 0,
          paddingLeft: "1.125rem",
          color: "#b42318",
          display: "grid",
          gap: "0.375rem",
        }}
      >
        {errors.map((error) => (
          <li key={error}>{error}</li>
        ))}
      </ul>
    ) : null}

    <label htmlFor="setup-speaker-id" style={labelStyle}>
      <span>{translate("setup.speaker_id")}</span>
      <input
        id="setup-speaker-id"
        type="text"
        value={speakerId}
        onChange={(event) => onSpeakerIdChange(event.currentTarget.value)}
        style={controlStyle}
        {...semanticAttributes("setup.speaker_id")}
      />
    </label>

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
        onChange={(event) => onSelectedCefrChange(event.currentTarget.value as CefrLevel)}
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

    <label htmlFor="setup-theme" style={labelStyle}>
      <span>{translate("setup.theme")}</span>
      <select
        id="setup-theme"
        value={selectedThemeMode === "custom" ? "__custom__" : selectedThemeTitle}
        onChange={(event) => {
          const value = event.currentTarget.value;
          if (value === "__custom__") {
            onSelectedThemeModeChange("custom");
            return;
          }
          onSelectedThemeModeChange("library");
          onSelectedThemeTitleChange(value);
        }}
        style={controlStyle}
        {...semanticAttributes("setup.theme")}
      >
        {availableThemes.map((theme) => (
          <option key={theme.title} value={theme.title}>
            {theme.title}
          </option>
        ))}
        <option value="__custom__">{translate("setup.custom_theme")}</option>
      </select>
    </label>

    <label htmlFor="setup-custom-theme" style={labelStyle}>
      <span>{translate("setup.custom_theme_label")}</span>
      <input
        id="setup-custom-theme"
        type="text"
        value={customTheme}
        disabled={!customThemeEnabled}
        onChange={(event) => onCustomThemeChange(event.currentTarget.value)}
        style={{
          ...controlStyle,
          opacity: customThemeEnabled ? 1 : 0.6,
          cursor: customThemeEnabled ? "text" : "not-allowed",
        }}
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

    <label htmlFor="setup-duration" style={labelStyle}>
      <span>{translate("setup.duration")}</span>
      <select
        id="setup-duration"
        value={String(selectedDuration)}
        onChange={(event) => onSelectedDurationChange(Number(event.currentTarget.value) as DurationOption)}
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

    <button
      type="button"
      onClick={onSubmit}
      style={{
        minHeight: "44px",
        padding: "0.875rem 1rem",
        borderRadius: "8px",
        border: "1px solid rgba(15, 118, 110, 0.2)",
        backgroundColor: "#d7ebe5",
        color: "#10201c",
        fontWeight: 700,
        font: "inherit",
        cursor: "pointer",
      }}
      {...semanticAttributes("setup.continue")}
    >
      {translate("setup.continue")}
    </button>
  </section>
);
