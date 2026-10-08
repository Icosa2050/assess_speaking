import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { PracticeBriefCard } from "@/components/setup/PracticeBriefCard";
import { ThemeForm, type SessionSetupStep } from "@/components/setup/ThemeForm";
import { apiClient } from "@/lib/api/client";
import type { HistoryRow } from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import {
  buildPracticeBrief,
  languageCodes,
  languageLabel,
  type ThemeEntry,
  themeEntryId,
  themeLibraryRepository,
  themesForLanguageAndLevel,
  type ThemeLibrary,
} from "@/lib/setup/sessionSetupContent";
import { selectRuntimeReadiness, useAppStore } from "@/lib/state/appStore";
import { type CefrLevel, type DurationOption, type TaskFamily } from "@/lib/state/sessionDraft";

const pageStyle = {
  display: "grid",
  gap: "1rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 22rem), 1fr))",
  alignItems: "start",
} as const;

const RECOMMENDED_CEFR: CefrLevel = "B1";
const RECOMMENDED_DURATION: DurationOption = 90;
const HISTORY_RECOMMENDATION_MAX_AGE_MS = 60 * 24 * 60 * 60 * 1000;

type RecommendedSetupSource = "generic" | "history_exact" | "history_language";

type RecommendedSetup = {
  languageCode: string;
  source: RecommendedSetupSource;
  theme: ThemeEntry | null;
};

const deriveInitialLanguage = (library: ThemeLibrary, requestedLanguage: string): string => {
  const available = languageCodes(library);
  if (available.includes(requestedLanguage)) {
    return requestedLanguage;
  }
  return available[0] ?? "en";
};

const resolveTaskFamilyLabel = (
  translate: ReturnType<typeof createTranslator>,
  taskFamily: TaskFamily,
): string => {
  const translated = translate(`task_family.${taskFamily}`);
  return translated.startsWith("[") ? taskFamily.replaceAll("_", " ") : translated;
};

const normalizeSpeakerId = (value: string): string => value.trim().toLowerCase();

const isRecentHistoryRow = (row: HistoryRow): boolean => {
  const timestamp = new Date(row.timestamp).getTime();
  if (!Number.isFinite(timestamp)) {
    return false;
  }

  return Date.now() - timestamp <= HISTORY_RECOMMENDATION_MAX_AGE_MS;
};

const resolveHistoryRecommendation = ({
  availableLanguages,
  historyRows,
  library,
  speakerId,
}: {
  availableLanguages: string[];
  historyRows: HistoryRow[];
  library: ThemeLibrary;
  speakerId: string;
}): RecommendedSetup | null => {
  const normalizedSpeakerId = normalizeSpeakerId(speakerId);
  if (!normalizedSpeakerId) {
    return null;
  }

  const recentRows = historyRows
    .filter((row) => normalizeSpeakerId(row.speaker_id) === normalizedSpeakerId)
    .filter(isRecentHistoryRow)
    .sort((left, right) => new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime());

  for (const row of recentRows) {
    const languageCode = String(row.learning_language || "").trim().toLowerCase();
    if (!languageCode || !availableLanguages.includes(languageCode)) {
      continue;
    }

    const b1Themes = themesForLanguageAndLevel(library, languageCode, RECOMMENDED_CEFR);
    if (b1Themes.length === 0) {
      continue;
    }

    const exactTheme = b1Themes.find((theme) => theme.title === row.theme);
    return {
      languageCode,
      source: exactTheme ? "history_exact" : "history_language",
      theme: exactTheme ?? b1Themes[0],
    };
  }

  return null;
};

export const SessionSetupRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const draft = useAppStore((state) => state.draft);
  const updateDraft = useAppStore((state) => state.updateDraft);
  const applySetup = useAppStore((state) => state.applySetup);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setReturnTo = useAppStore((state) => state.setReturnTo);

  const translate = createTranslator(locale);

  const [library, setLibrary] = useState<ThemeLibrary>(() => themeLibraryRepository.load());
  const speakerId = draft.speakerId;
  const [selectedLanguage, setSelectedLanguage] = useState(() =>
    deriveInitialLanguage(library, draft.learningLanguage),
  );
  const [selectedCefr, setSelectedCefr] = useState<CefrLevel>(draft.cefrLevel);
  const [selectedThemeMode, setSelectedThemeMode] = useState<"library" | "custom">("library");
  const [selectedThemeTitle, setSelectedThemeTitle] = useState("");
  const [customTheme, setCustomTheme] = useState("");
  const [saveCustomThemeForReuse, setSaveCustomThemeForReuse] = useState(false);
  const [selectedDuration, setSelectedDuration] = useState<DurationOption>(draft.durationSec);
  const [setupStep, setSetupStep] = useState<SessionSetupStep>(() =>
    draft.themeLabel ? "practice" : "learner",
  );
  const [advancedTopicOpen, setAdvancedTopicOpen] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const draftThemeHydratedRef = useRef(false);

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
  });

  const historyQuery = useQuery({
    queryKey: queryKeys.history,
    queryFn: () => apiClient.getHistory(),
  });

  useEffect(() => {
    setCurrentPage("setup");
  }, [setCurrentPage]);

  const availableLanguages = useMemo(() => languageCodes(library), [library]);
  const availableThemes = useMemo(
    () => themesForLanguageAndLevel(library, selectedLanguage, selectedCefr),
    [library, selectedLanguage, selectedCefr],
  );

  useEffect(() => {
    if (!availableLanguages.includes(selectedLanguage)) {
      setSelectedLanguage(availableLanguages[0] ?? "en");
    }
  }, [availableLanguages, selectedLanguage]);

  useEffect(() => {
    if (draftThemeHydratedRef.current) {
      return;
    }
    draftThemeHydratedRef.current = true;

    const themeMatch = availableThemes.find((theme) => theme.title === draft.themeLabel);
    if (!draft.themeLabel) {
      setSelectedThemeTitle(availableThemes[0]?.title ?? "");
      return;
    }

    if (themeMatch) {
      setSelectedThemeMode("library");
      setSelectedThemeTitle(themeMatch.title);
      setAdvancedTopicOpen(false);
      return;
    }

    setSelectedThemeMode("custom");
    setCustomTheme(draft.themeLabel);
    setAdvancedTopicOpen(true);
  }, [availableThemes, draft.themeLabel]);

  useEffect(() => {
    if (
      selectedThemeMode === "library" &&
      selectedThemeTitle &&
      availableThemes.some((theme) => theme.title === selectedThemeTitle)
    ) {
      return;
    }
    if (selectedThemeMode === "library") {
      setSelectedThemeTitle(availableThemes[0]?.title ?? "");
    }
  }, [availableThemes, selectedThemeMode, selectedThemeTitle]);

  const selectedThemeEntry = useMemo(
    () => availableThemes.find((theme) => theme.title === selectedThemeTitle),
    [availableThemes, selectedThemeTitle],
  );

  const resolvedThemeLabel =
    selectedThemeMode === "custom" ? customTheme.trim() : selectedThemeEntry?.title ?? "";
  const resolvedTaskFamily =
    selectedThemeMode === "custom"
      ? (draft.taskFamily ?? ("free_monologue" as TaskFamily))
      : (selectedThemeEntry?.task_family ?? draft.taskFamily ?? ("free_monologue" as TaskFamily));
  const selectedLanguageLabel = languageLabel(library, selectedLanguage);
  const taskFamilyLabel = resolveTaskFamilyLabel(translate, resolvedTaskFamily);

  const brief = resolvedThemeLabel
    ? buildPracticeBrief({
        taskFamily: resolvedTaskFamily,
        theme: resolvedThemeLabel,
        targetDurationSec: selectedDuration,
        languageCode: selectedLanguage,
      })
    : {
        prompt: "",
        successFocus: [],
      };

  const runtimeConfigured = Boolean(runtimeQuery.data?.configured);
  const runtimeReadiness = selectRuntimeReadiness({
    preferences: {
      ...preferences,
      activeConnectionId: runtimeConfigured
        ? preferences.activeConnectionId ||
          `${runtimeQuery.data?.provider || "runtime"}:${runtimeQuery.data?.model || "active"}`
        : preferences.activeConnectionId,
      setupComplete: runtimeConfigured || preferences.setupComplete,
    },
  });
  const primaryActionLabel = runtimeReadiness.ready
    ? translate("setup.start_speaking")
    : translate("setup.save_and_setup_device");
  const runtimeCalloutTitle = runtimeReadiness.ready
    ? translate("setup.runtime_ready_title")
    : translate("setup.runtime_setup_needed_title");
  const runtimeCalloutBody = runtimeReadiness.ready
    ? translate("setup.runtime_ready_body")
    : translate("setup.runtime_setup_needed_body");

  const historyRecommendation = useMemo(
    () =>
      resolveHistoryRecommendation({
        availableLanguages,
        historyRows: historyQuery.data?.items ?? [],
        library,
        speakerId,
      }),
    [availableLanguages, historyQuery.data?.items, library, speakerId],
  );

  const resolveRecommendedSetup = (): RecommendedSetup => {
    if (historyRecommendation) {
      return historyRecommendation;
    }

    const preferredLanguages = [
      selectedLanguage,
      draft.learningLanguage,
      "it",
      ...availableLanguages,
    ].filter((languageCode, index, languages) =>
      Boolean(languageCode) &&
      availableLanguages.includes(languageCode) &&
      languages.indexOf(languageCode) === index,
    );

    for (const languageCode of preferredLanguages) {
      const recommendedThemes = themesForLanguageAndLevel(library, languageCode, RECOMMENDED_CEFR);
      if (recommendedThemes.length > 0) {
        return {
          languageCode,
          source: "generic",
          theme: recommendedThemes[0],
        };
      }
    }

    return {
      languageCode: availableLanguages[0] ?? "en",
      source: "generic",
      theme: availableThemes[0] ?? null,
    };
  };

  const recommendationPreview = resolveRecommendedSetup();
  const recommendationHint =
    recommendationPreview.source === "history_exact"
      ? translate("setup.recommendation_hint_exact", {
          duration: RECOMMENDED_DURATION,
          language: languageLabel(library, recommendationPreview.languageCode),
        })
      : recommendationPreview.source === "history_language"
        ? translate("setup.recommendation_hint_language", {
            duration: RECOMMENDED_DURATION,
            language: languageLabel(library, recommendationPreview.languageCode),
          })
        : translate("setup.recommendation_hint_generic", {
            duration: RECOMMENDED_DURATION,
            language: languageLabel(library, recommendationPreview.languageCode),
          });

  const handleRecommendedStart = () => {
    const recommended = resolveRecommendedSetup();
    const nextErrors: string[] = [];
    if (!speakerId.trim()) {
      nextErrors.push(translate("setup.error_speaker_id"));
    }
    if (!recommended.theme) {
      nextErrors.push(translate("setup.error_theme"));
    }
    setErrors(nextErrors);
    if (nextErrors.length > 0 || !recommended.theme) {
      return;
    }

    setSelectedLanguage(recommended.languageCode);
    setSelectedCefr(RECOMMENDED_CEFR);
    setSelectedThemeMode("library");
    setSelectedThemeTitle(recommended.theme.title);
    setCustomTheme("");
    setSaveCustomThemeForReuse(false);
    setSelectedDuration(RECOMMENDED_DURATION);
    setAdvancedTopicOpen(false);
    setSetupStep("practice");
  };

  const handleAdvancedTopicOpenChange = (open: boolean) => {
    setAdvancedTopicOpen(open);
    setSelectedThemeMode(open ? "custom" : "library");
  };

  const handleSubmit = () => {
    const nextErrors: string[] = [];
    if (!speakerId.trim()) {
      nextErrors.push(translate("setup.error_speaker_id"));
    }
    if (!resolvedThemeLabel) {
      nextErrors.push(translate("setup.error_theme"));
    }
    setErrors(nextErrors);
    if (nextErrors.length > 0) {
      return;
    }

    if (selectedThemeMode === "custom" && saveCustomThemeForReuse) {
      const nextLibrary = themeLibraryRepository.addCustomTheme({
        languageCode: selectedLanguage,
        languageLabel: selectedLanguageLabel,
        title: resolvedThemeLabel,
        level: selectedCefr,
        taskFamily: resolvedTaskFamily,
      });
      setLibrary(nextLibrary);
    }

    applySetup({
      speakerId: speakerId.trim(),
      learningLanguage: selectedLanguage,
      learningLanguageLabel: selectedLanguageLabel,
      cefrLevel: selectedCefr,
      themeId: themeEntryId({
        title: resolvedThemeLabel,
        level: selectedCefr,
      }),
      themeLabel: resolvedThemeLabel,
      taskFamily: resolvedTaskFamily,
      durationSec: selectedDuration,
      promptText: brief.prompt,
    });

    if (runtimeReadiness.ready) {
      navigate("/speak");
      return;
    }

    setReturnTo("setup");
    navigate("/runtime-setup");
  };

  if (availableLanguages.length === 0) {
    return (
      <section
        style={{
          display: "grid",
          gap: "0.75rem",
          padding: "1.25rem",
          borderRadius: "8px",
          border: "1px solid rgba(180, 35, 24, 0.16)",
          backgroundColor: "rgba(255, 245, 245, 0.94)",
        }}
      >
        <h2
          style={{
            margin: 0,
            color: "#10201c",
          }}
        >
          {translate("setup.title")}
        </h2>
        <p
          style={{
            margin: 0,
            color: "#b42318",
          }}
        >
          {translate("setup.no_languages")}
        </p>
      </section>
    );
  }

  return (
    <div style={pageStyle} data-testid="setup.layout" data-semantic-id="setup.layout">
      <ThemeForm
        availableLanguages={availableLanguages}
        availableThemes={availableThemes.map((theme) => ({ title: theme.title }))}
        customTheme={customTheme}
        customThemeEnabled={advancedTopicOpen}
        customThemeSaveEnabled={selectedThemeMode === "custom" && Boolean(customTheme.trim())}
        errors={errors}
        languageLabelFor={(languageCode) => languageLabel(library, languageCode)}
        advancedTopicOpen={advancedTopicOpen}
        onAdvancedTopicOpenChange={handleAdvancedTopicOpenChange}
        onCustomThemeChange={(value) => {
          setSelectedThemeMode("custom");
          setCustomTheme(value);
          if (errors.length > 0) {
            setErrors([]);
          }
        }}
        onRecommendedStart={handleRecommendedStart}
        recommendationHint={recommendationHint}
        onSaveCustomThemeForReuseChange={setSaveCustomThemeForReuse}
        onSelectedCefrChange={(value) => {
          setSelectedCefr(value);
          setErrors([]);
        }}
        onSelectedDurationChange={setSelectedDuration}
        onSelectedLanguageChange={(value) => {
          setSelectedLanguage(value);
          setErrors([]);
        }}
        onSelectedThemeModeChange={(value) => {
          setSelectedThemeMode(value);
          setErrors([]);
        }}
        onSelectedThemeTitleChange={(value) => {
          setSelectedThemeTitle(value);
          setErrors([]);
        }}
        onSetupStepChange={setSetupStep}
        onSpeakerIdChange={(value) => {
          updateDraft({ speakerId: value });
          if (errors.length > 0) {
            setErrors([]);
          }
        }}
        onSubmit={handleSubmit}
        primaryActionLabel={primaryActionLabel}
        saveCustomThemeForReuse={saveCustomThemeForReuse}
        selectedCefr={selectedCefr}
        selectedDuration={selectedDuration}
        selectedLanguage={selectedLanguage}
        setupStep={setupStep}
        selectedThemeMode={selectedThemeMode}
        selectedThemeTitle={selectedThemeTitle}
        speakerId={speakerId}
        translate={translate}
      />
      <PracticeBriefCard
        customThemeSaveHelp={translate("setup.custom_theme_save_help", {
          language: selectedLanguageLabel,
          level: selectedCefr,
          task_family: taskFamilyLabel,
        })}
        promptText={brief.prompt}
        resolvedThemeLabel={resolvedThemeLabel}
        runtimeCalloutBody={runtimeCalloutBody}
        runtimeCalloutTitle={runtimeCalloutTitle}
        selectionDetails={[
          {
            label: translate("setup.speaker_id"),
            value: speakerId.trim() || "-",
          },
          {
            label: translate("setup.learning_language"),
            value: selectedLanguageLabel,
          },
          {
            label: translate("setup.cefr"),
            value: selectedCefr,
          },
          {
            label: translate("setup.theme"),
            value: resolvedThemeLabel || "-",
          },
          {
            label: translate("setup.duration"),
            value: `${selectedDuration} s`,
          },
          {
            label: translate("history.task_family_name"),
            value: taskFamilyLabel,
          },
        ]}
        shouldShowRuntimeCallout={setupStep === "practice"}
        shouldShowCustomThemeSaveHelp={selectedThemeMode === "custom"}
        successFocus={brief.successFocus}
        translate={translate}
      />
    </div>
  );
};
