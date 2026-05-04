import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { PracticeBriefCard } from "@/components/setup/PracticeBriefCard";
import { ThemeForm } from "@/components/setup/ThemeForm";
import { apiClient } from "@/lib/api/client";
import { createTranslator } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import {
  buildPracticeBrief,
  languageCodes,
  languageLabel,
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
  gridTemplateColumns: "minmax(0, 1.1fr) minmax(0, 0.9fr)",
} as const;

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

export const SessionSetupRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const draft = useAppStore((state) => state.draft);
  const applySetup = useAppStore((state) => state.applySetup);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const setReturnTo = useAppStore((state) => state.setReturnTo);

  const translate = createTranslator(locale);

  const [library, setLibrary] = useState<ThemeLibrary>(() => themeLibraryRepository.load());
  const [speakerId, setSpeakerId] = useState(draft.speakerId);
  const [selectedLanguage, setSelectedLanguage] = useState(() =>
    deriveInitialLanguage(library, draft.learningLanguage),
  );
  const [selectedCefr, setSelectedCefr] = useState<CefrLevel>(draft.cefrLevel);
  const [selectedThemeMode, setSelectedThemeMode] = useState<"library" | "custom">("library");
  const [selectedThemeTitle, setSelectedThemeTitle] = useState("");
  const [customTheme, setCustomTheme] = useState("");
  const [saveCustomThemeForReuse, setSaveCustomThemeForReuse] = useState(false);
  const [selectedDuration, setSelectedDuration] = useState<DurationOption>(draft.durationSec);
  const [errors, setErrors] = useState<string[]>([]);

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
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
    const themeMatch = availableThemes.find((theme) => theme.title === draft.themeLabel);
    if (!draft.themeLabel) {
      setSelectedThemeTitle(availableThemes[0]?.title ?? "");
      return;
    }

    if (themeMatch) {
      setSelectedThemeMode("library");
      setSelectedThemeTitle(themeMatch.title);
      return;
    }

    setSelectedThemeMode("custom");
    setCustomTheme(draft.themeLabel);
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
    <div style={pageStyle}>
      <ThemeForm
        availableLanguages={availableLanguages}
        availableThemes={availableThemes.map((theme) => ({ title: theme.title }))}
        customTheme={customTheme}
        customThemeEnabled={selectedThemeMode === "custom"}
        customThemeSaveEnabled={selectedThemeMode === "custom" && Boolean(customTheme.trim())}
        errors={errors}
        languageLabelFor={(languageCode) => languageLabel(library, languageCode)}
        onCustomThemeChange={(value) => {
          setCustomTheme(value);
          if (errors.length > 0) {
            setErrors([]);
          }
        }}
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
        onSpeakerIdChange={(value) => {
          setSpeakerId(value);
          if (errors.length > 0) {
            setErrors([]);
          }
        }}
        onSubmit={handleSubmit}
        saveCustomThemeForReuse={saveCustomThemeForReuse}
        selectedCefr={selectedCefr}
        selectedDuration={selectedDuration}
        selectedLanguage={selectedLanguage}
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
        shouldShowCustomThemeSaveHelp={selectedThemeMode === "custom"}
        successFocus={brief.successFocus}
        translate={translate}
      />
    </div>
  );
};
