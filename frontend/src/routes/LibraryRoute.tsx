import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { SampleTrialGrid, sampleTitleLabel } from "@/components/library/SampleTrialGrid";
import { apiClient } from "@/lib/api/client";
import type { SampleItem } from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import {
  languageCodes,
  languageLabel,
  themeLibraryRepository,
  type ThemeEntry,
  type ThemeLibrary,
} from "@/lib/setup/sessionSetupContent";
import { useAppStore } from "@/lib/state/appStore";
import { useDraftRecovery } from "@/lib/state/DraftRecoveryProvider";
import {
  CEFR_LEVELS,
  buildSessionDraft,
  hasSetupDraft,
  TASK_FAMILY_OPTIONS,
  type CefrLevel,
  type TaskFamily,
} from "@/lib/state/sessionDraft";

const SAMPLE_TASK_FAMILY_BY_SLUG: Record<string, TaskFamily> = {
  travel_story: "travel_narrative",
  remote_work: "opinion_monologue",
  public_debate: "opinion_monologue",
};
const SAMPLE_DURATION_BY_CEFR: Record<string, 90 | 120 | 180> = {
  B1: 90,
  B2: 120,
  C1: 180,
};

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid var(--color-border-decorative)",
  borderRadius: "8px",
  backgroundColor: "var(--color-surface)",
  boxShadow: "0 18px 40px var(--color-shadow-ambient)",
} as const;

const secondaryPanelStyle = {
  ...cardStyle,
  backgroundColor: "var(--color-surface-subtle)",
  boxShadow: "none",
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

const actionButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-selected-border)",
  backgroundColor: "var(--color-selected)",
  color: "var(--color-text)",
  font: "inherit",
  fontWeight: 600,
  cursor: "pointer",
} as const;

const fieldStyle = {
  display: "grid",
  gap: "0.35rem",
} as const;

const taskFamilyLabel = (
  translate: ReturnType<typeof createTranslator>,
  value: string,
): string => {
  const translated = translate(`task_family.${value}`);
  return translated.startsWith("[") ? value.replaceAll("_", " ") : translated;
};

const sampleTaskFamily = (sampleId: string): TaskFamily => {
  const slug = sampleId.split("_").slice(2).join("_");
  return SAMPLE_TASK_FAMILY_BY_SLUG[slug] ?? "free_monologue";
};

const sampleDuration = (cefr: string): 90 | 120 | 180 =>
  SAMPLE_DURATION_BY_CEFR[cefr.toUpperCase()] ?? 90;

const themeRowsForLanguage = (library: ThemeLibrary, languageCode: string): ThemeEntry[] =>
  library[languageCode]?.themes ?? [];

export const LibraryRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const draft = useAppStore((state) => state.draft);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const applySetup = useAppStore((state) => state.applySetup);
  const draftRecovery = useDraftRecovery();

  const translate = createTranslator(locale);
  const [library, setLibrary] = useState<ThemeLibrary>(() => themeLibraryRepository.load());
  const codes = useMemo(() => languageCodes(library), [library]);
  const [selectedLanguage, setSelectedLanguage] = useState(
    codes.includes(draft.learningLanguage) ? draft.learningLanguage : codes[0] ?? "",
  );
  const [manageLanguage, setManageLanguage] = useState(selectedLanguage);
  const [themeTitle, setThemeTitle] = useState("");
  const [themeLevel, setThemeLevel] = useState<CefrLevel>(draft.cefrLevel || "B1");
  const [themeFamily, setThemeFamily] = useState<TaskFamily>("free_monologue");
  const [errors, setErrors] = useState<Record<string, boolean>>({});
  const [successMessage, setSuccessMessage] = useState("");
  const [draftError, setDraftError] = useState("");

  const samplesQuery = useQuery({
    queryKey: queryKeys.samples,
    queryFn: () => apiClient.getSamples(),
  });

  useEffect(() => {
    setCurrentPage("library");
  }, [setCurrentPage]);

  const hasSetup = hasSetupDraft(draft);
  const selectedThemes = themeRowsForLanguage(library, selectedLanguage);
  const filteredSamples = useMemo(
    () =>
      (samplesQuery.data?.items ?? []).filter((sample) => {
        const languageMatches = !selectedLanguage || sample.language.toLowerCase() === selectedLanguage;
        const cefrMatches = !hasSetup || sample.cefr.toUpperCase() === draft.cefrLevel;
        return languageMatches && cefrMatches;
      }),
    [draft.cefrLevel, hasSetup, samplesQuery.data?.items, selectedLanguage],
  );
  const handlePrepareSample = (sample: SampleItem) => {
    const languageCodeValue = sample.language.toLowerCase();
    const title = sampleTitleLabel(sample) || translate("library.none");
    const nextDraft = buildSessionDraft({
      ...draft,
      sessionId: undefined,
      retryOfSessionId: "", coachingGoal: "", retryGoalEligible: false,
      learningLanguage: languageCodeValue || draft.learningLanguage,
      learningLanguageLabel:
        languageCodeValue && library[languageCodeValue]
          ? languageLabel(library, languageCodeValue)
          : languageCodeValue.toUpperCase() || draft.learningLanguageLabel,
      cefrLevel: sample.cefr.toUpperCase() as CefrLevel,
      themeId: sample.sample_id,
      themeLabel: title,
      taskFamily: sampleTaskFamily(sample.sample_id),
      durationSec: sampleDuration(sample.cefr),
      promptId: sample.sample_id,
      promptText: title,
    });
    void (async () => {
      setDraftError("");
      if (!hasSetupDraft(nextDraft)) {
        applySetup(nextDraft);
        navigate("/session-setup");
        return;
      }
      if (!await draftRecovery.prepareNewAttempt(nextDraft.speakerId)) return;
      try { await draftRecovery.saveSetup(nextDraft); } catch { setDraftError(translate("draft.storage_error")); return; }
      applySetup(nextDraft);
      navigate("/speak");
    })();
  };

  const samplesContent = samplesQuery.isLoading ? (
    <p
      style={{ margin: 0, color: "var(--color-text-muted)" }}
      data-testid="library-samples-loading"
      data-semantic-id="library.samples_loading"
    >
      {translate("library.samples_loading")}
    </p>
  ) : samplesQuery.isError ? (
    <p
      role="alert"
      style={{ margin: 0, color: "var(--color-error)" }}
      data-testid="library-samples-error"
      data-semantic-id="library.samples_error"
    >
      {translate("library.samples_error")}
    </p>
  ) : (
    <SampleTrialGrid
      hasSetup={hasSetup}
      onPrepareSample={handlePrepareSample}
      samples={filteredSamples}
      translate={translate}
    />
  );

  const handleSaveTheme = () => {
    const nextErrors = {
      theme_title: !themeTitle.trim(),
    };
    setErrors(nextErrors);
    if (Object.values(nextErrors).some(Boolean)) {
      return;
    }

    const resolvedLanguageCode = manageLanguage;
    const resolvedLanguageLabel = languageLabel(library, resolvedLanguageCode);
    const nextLibrary = themeLibraryRepository.addCustomTheme({
      languageCode: resolvedLanguageCode,
      languageLabel: resolvedLanguageLabel,
      title: themeTitle,
      level: themeLevel,
      taskFamily: themeFamily,
    });
    setLibrary(nextLibrary);
    setSelectedLanguage(resolvedLanguageCode.trim().toLowerCase());
    setManageLanguage(resolvedLanguageCode.trim().toLowerCase());
    setThemeTitle("");
    setErrors({});
    setSuccessMessage(translate("library.saved_success", { theme: themeTitle.trim() }));
  };

  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      {...semanticAttributes(SEMANTIC_IDS.library.screen)}
    >
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.75rem", color: "var(--color-text)" }}>
          {translate("library.title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "var(--color-text-muted)" }}>
          {translate("library.body")}
        </p>
      </section>

      <section style={secondaryPanelStyle}>
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("library.language_filter")}
          </span>
          <select
            value={selectedLanguage}
            onChange={(event) => {
              setSelectedLanguage(event.target.value);
              setManageLanguage(event.target.value);
            }}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.library.languageFilter)}
          >
            {codes.map((code) => (
              <option key={code} value={code}>
                {languageLabel(library, code)}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
          {translate("library.samples_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "var(--color-text-muted)" }}>
          {translate("library.samples_body")}
        </p>
        {draftError && <p role="alert">{draftError}</p>}
        {samplesContent}
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
          {translate("library.existing_title")}
        </h2>
        {selectedThemes.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", minWidth: "100%", color: "var(--color-text)" }}>
              <thead>
                <tr>
                  {[translate("library.table_title"), translate("library.table_level"), translate("library.table_task_family")].map((label) => (
                    <th key={label} style={{ padding: "0.65rem", textAlign: "left", color: "var(--color-text-muted)" }}>
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {selectedThemes.map((theme, index) => (
                  <tr key={`${theme.level}-${theme.title}-${theme.task_family}-${index}`}>
                    <td style={{ borderTop: "1px solid var(--color-border-decorative)", padding: "0.65rem" }}>
                      {theme.title}
                    </td>
                    <td style={{ borderTop: "1px solid var(--color-border-decorative)", padding: "0.65rem" }}>
                      {theme.level}
                    </td>
                    <td style={{ borderTop: "1px solid var(--color-border-decorative)", padding: "0.65rem" }}>
                      {taskFamilyLabel(translate, theme.task_family)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ margin: 0, color: "var(--color-text-muted)" }}>
            {selectedLanguage ? translate("library.empty_language") : translate("library.empty_library")}
          </p>
        )}
      </section>

      <section
        style={secondaryPanelStyle}
        data-testid="library.custom_theme_builder"
        data-semantic-id="library.custom_theme_builder"
      >
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
          {translate("library.add_title")}
        </h2>
        <p style={{ margin: 0, color: "var(--color-text-muted)", lineHeight: 1.55 }}>
          {translate("library.custom_theme_caption")}
        </p>
        {successMessage ? <p style={{ margin: 0, color: "var(--color-success)" }}>{successMessage}</p> : null}
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("setup.learning_language")}
          </span>
          <select
            value={manageLanguage}
            onChange={(event) => setManageLanguage(event.target.value)}
            style={inputStyle}
            {...semanticAttributes(SEMANTIC_IDS.library.manageLanguage)}
          >
            {codes.map((code) => (
              <option key={code} value={code}>
                {languageLabel(library, code)}
              </option>
            ))}
          </select>
        </label>
        <p style={{ margin: 0, color: "var(--color-text-muted)" }}>
          {translate("library.saving_under", {
            code: manageLanguage,
            label: languageLabel(library, manageLanguage),
          })}
        </p>
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("library.theme_title")}
          </span>
          <input value={themeTitle} onChange={(event) => setThemeTitle(event.target.value)} style={inputStyle} />
          {errors.theme_title ? <span style={{ color: "var(--color-error)" }}>{translate("library.error_theme_title")}</span> : null}
        </label>
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("library.theme_level")}
          </span>
          <select value={themeLevel} onChange={(event) => setThemeLevel(event.target.value as CefrLevel)} style={inputStyle}>
            {CEFR_LEVELS.map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
        </label>
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
            {translate("library.theme_family")}
          </span>
          <select value={themeFamily} onChange={(event) => setThemeFamily(event.target.value as TaskFamily)} style={inputStyle}>
            {TASK_FAMILY_OPTIONS.map((family) => (
              <option key={family} value={family}>
                {taskFamilyLabel(translate, family)}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={handleSaveTheme}
          style={{ ...actionButtonStyle, width: "fit-content" }}
          {...semanticAttributes(SEMANTIC_IDS.library.saveTheme)}
        >
          {translate("library.save_theme")}
        </button>
      </section>
    </div>
  );
};
