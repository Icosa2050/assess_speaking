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
import {
  CEFR_LEVELS,
  hasSetupDraft,
  TASK_FAMILY_OPTIONS,
  type CefrLevel,
  type TaskFamily,
} from "@/lib/state/sessionDraft";

const NEW_LANGUAGE_OPTION = "__new_language__";
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
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
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
  border: "1px solid rgba(15, 118, 110, 0.2)",
  backgroundColor: "#d7ebe5",
  color: "#10201c",
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

  const translate = createTranslator(locale);
  const [library, setLibrary] = useState<ThemeLibrary>(() => themeLibraryRepository.load());
  const codes = useMemo(() => languageCodes(library), [library]);
  const [selectedLanguage, setSelectedLanguage] = useState(
    codes.includes(draft.learningLanguage) ? draft.learningLanguage : codes[0] ?? "",
  );
  const [manageLanguage, setManageLanguage] = useState(selectedLanguage || NEW_LANGUAGE_OPTION);
  const [languageCode, setLanguageCode] = useState("");
  const [languageLabelInput, setLanguageLabelInput] = useState("");
  const [themeTitle, setThemeTitle] = useState("");
  const [themeLevel, setThemeLevel] = useState<CefrLevel>(draft.cefrLevel || "B1");
  const [themeFamily, setThemeFamily] = useState<TaskFamily>("free_monologue");
  const [errors, setErrors] = useState<Record<string, boolean>>({});
  const [successMessage, setSuccessMessage] = useState("");

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
    const nextDraft = {
      ...draft,
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
    };
    applySetup(nextDraft);
    navigate(hasSetupDraft(nextDraft) ? "/speak" : "/session-setup");
  };

  const handleSaveTheme = () => {
    const nextErrors = {
      language_code: manageLanguage === NEW_LANGUAGE_OPTION && !languageCode.trim(),
      language_label: manageLanguage === NEW_LANGUAGE_OPTION && !languageLabelInput.trim(),
      theme_title: !themeTitle.trim(),
    };
    setErrors(nextErrors);
    if (Object.values(nextErrors).some(Boolean)) {
      return;
    }

    const resolvedLanguageCode = manageLanguage === NEW_LANGUAGE_OPTION ? languageCode : manageLanguage;
    const resolvedLanguageLabel =
      manageLanguage === NEW_LANGUAGE_OPTION
        ? languageLabelInput
        : languageLabel(library, resolvedLanguageCode);
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
    setLanguageCode("");
    setLanguageLabelInput("");
    setErrors({});
    setSuccessMessage(translate("library.saved_success", { theme: themeTitle.trim() }));
  };

  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      {...semanticAttributes(SEMANTIC_IDS.library.screen)}
    >
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.75rem", color: "#10201c" }}>
          {translate("library.title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("library.body")}
        </p>
      </section>

      <section style={cardStyle}>
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate("library.language_filter")}
          </span>
          <select
            value={selectedLanguage}
            onChange={(event) => {
              setSelectedLanguage(event.target.value);
              setManageLanguage(event.target.value || NEW_LANGUAGE_OPTION);
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
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("library.samples_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("library.samples_body")}
        </p>
        <SampleTrialGrid
          hasSetup={hasSetup}
          onPrepareSample={handlePrepareSample}
          samples={filteredSamples}
          translate={translate}
        />
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("library.existing_title")}
        </h2>
        {selectedThemes.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", minWidth: "100%", color: "#10201c" }}>
              <thead>
                <tr>
                  {[translate("library.table_title"), translate("library.table_level"), translate("library.table_task_family")].map((label) => (
                    <th key={label} style={{ padding: "0.65rem", textAlign: "left", color: "#33514b" }}>
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {selectedThemes.map((theme) => (
                  <tr key={`${theme.level}-${theme.title}-${theme.task_family}`}>
                    <td style={{ borderTop: "1px solid rgba(18, 61, 55, 0.08)", padding: "0.65rem" }}>
                      {theme.title}
                    </td>
                    <td style={{ borderTop: "1px solid rgba(18, 61, 55, 0.08)", padding: "0.65rem" }}>
                      {theme.level}
                    </td>
                    <td style={{ borderTop: "1px solid rgba(18, 61, 55, 0.08)", padding: "0.65rem" }}>
                      {taskFamilyLabel(translate, theme.task_family)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ margin: 0, color: "#33514b" }}>
            {selectedLanguage ? translate("library.empty_language") : translate("library.empty_library")}
          </p>
        )}
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("library.add_title")}
        </h2>
        {successMessage ? <p style={{ margin: 0, color: "#166534" }}>{successMessage}</p> : null}
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate("library.manage_language")}
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
            <option value={NEW_LANGUAGE_OPTION}>{translate("library.new_language")}</option>
          </select>
        </label>
        {manageLanguage === NEW_LANGUAGE_OPTION ? (
          <>
            <label style={fieldStyle}>
              <span style={{ fontWeight: 600, color: "#33514b" }}>
                {translate("library.language_code")}
              </span>
              <input value={languageCode} onChange={(event) => setLanguageCode(event.target.value)} style={inputStyle} />
              {errors.language_code ? <span style={{ color: "#b42318" }}>{translate("library.error_language_code")}</span> : null}
            </label>
            <label style={fieldStyle}>
              <span style={{ fontWeight: 600, color: "#33514b" }}>
                {translate("library.language_label")}
              </span>
              <input value={languageLabelInput} onChange={(event) => setLanguageLabelInput(event.target.value)} style={inputStyle} />
              {errors.language_label ? <span style={{ color: "#b42318" }}>{translate("library.error_language_label")}</span> : null}
            </label>
          </>
        ) : (
          <p style={{ margin: 0, color: "#33514b" }}>
            {translate("library.saving_under", {
              code: manageLanguage,
              label: languageLabel(library, manageLanguage),
            })}
          </p>
        )}
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
            {translate("library.theme_title")}
          </span>
          <input value={themeTitle} onChange={(event) => setThemeTitle(event.target.value)} style={inputStyle} />
          {errors.theme_title ? <span style={{ color: "#b42318" }}>{translate("library.error_theme_title")}</span> : null}
        </label>
        <label style={fieldStyle}>
          <span style={{ fontWeight: 600, color: "#33514b" }}>
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
          <span style={{ fontWeight: 600, color: "#33514b" }}>
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
