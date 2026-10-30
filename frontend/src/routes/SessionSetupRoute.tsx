import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { PracticeBriefCard } from "@/components/setup/PracticeBriefCard";
import { FirstRunPreview } from "@/components/setup/FirstRunPreview";
import { StatusBanner } from "@/components/ui/StatusBanner";
import { ThemeForm, type SessionSetupStep } from "@/components/setup/ThemeForm";
import { apiClient } from "@/lib/api/client";
import type { HistoryRow } from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import { buildSetupReadinessRows, canEnterSpeak, canRecord, isCoreSetupReadinessKey } from "@/lib/setup/readiness";
import { useReadiness } from "@/lib/setup/useReadiness";
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
import { useAppStore } from "@/lib/state/appStore";
import { useDraftRecovery } from "@/lib/state/DraftRecoveryProvider";
import { normalizedLearner, sameDraftAttempt, type DraftStoreError } from "@/lib/state/draftStore";
import { buildSessionDraft, type CefrLevel, type DurationOption, type TaskFamily } from "@/lib/state/sessionDraft";

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
  const retryRestorePending = useAppStore(state => state.retryRestorePending);
  return retryRestorePending ? <Navigate replace to="/speak" /> : <SessionSetupContent />;
};

const SessionSetupContent = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const microphoneStatus = useAppStore((state) => state.microphoneStatus);
  const microphoneSetupPassed = useAppStore((state) => state.microphoneSetupPassed);
  const draft = useAppStore((state) => state.draft);
  const draftRecovery = useDraftRecovery();
  const clearRecording = useAppStore(state => state.clearRecording);
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

  const readinessQuery = useReadiness();

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
    if (!draft.themeLabel || !availableLanguages.includes(draft.learningLanguage)) {
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
  }, [availableLanguages, availableThemes, draft.learningLanguage, draft.themeLabel]);

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

  const requiredChecks = buildSetupReadinessRows({
    readiness: readinessQuery.data,
    readinessError: readinessQuery.isError && !readinessQuery.data,
    readinessPending: readinessQuery.isPending,
    microphoneStatus, microphoneSetupPassed,
  }).filter((row) => isCoreSetupReadinessKey(row.key));
  const readyToRecord = canRecord(requiredChecks);
  const canContinueToSpeak = canEnterSpeak(requiredChecks);
  const primaryActionLabel = canContinueToSpeak
    ? translate("setup.start_speaking")
    : translate("setup.save_and_setup_device");
  const runtimeCalloutTitle = translate(readyToRecord
    ? "setup.runtime_ready_title"
    : canContinueToSpeak ? "first_run.continue_to_speak_title" : "setup.runtime_setup_needed_title");
  const runtimeCalloutBody = translate(readyToRecord
    ? "setup.runtime_ready_body"
    : canContinueToSpeak ? "first_run.continue_to_speak_body" : "setup.runtime_setup_needed_body");

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

  const handleSubmit = async () => {
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

    let nextSetup = buildSessionDraft({
      sessionId: draft.sessionId,
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
      promptId: `${themeEntryId({ title: resolvedThemeLabel, level: selectedCefr })}-${selectedCefr.toLowerCase()}`,
      promptText: brief.prompt,
      retryOfSessionId: draft.retryOfSessionId,
      coachingGoal: draft.coachingGoal,
      retryGoalEligible: draft.retryGoalEligible,
    });
    const taskChanged = normalizedLearner(nextSetup.speakerId) !== normalizedLearner(draft.speakerId) ||
      (["learningLanguage", "cefrLevel", "themeId", "promptId", "promptText", "taskFamily", "durationSec"] as const)
        .some(key => nextSetup[key] !== draft[key]);
    if (taskChanged) {
      nextSetup = buildSessionDraft({ ...nextSetup, retryOfSessionId: "", coachingGoal: "", retryGoalEligible: false });
    }
    const existing = draftRecovery.findLearner(nextSetup.speakerId);
    const replacement = existing && (!sameDraftAttempt(existing, nextSetup) ||
      existing.setup.themeLabel !== nextSetup.themeLabel ||
      existing.setup.learningLanguageLabel !== nextSetup.learningLanguageLabel);
    if (replacement && !window.confirm(translate("draft.replace_setup_detail", {
      learner: existing.setup.speakerId,
      task: existing.setup.themeLabel || existing.setup.promptText,
      size: existing.take ? `${(existing.take.size / 1024 / 1024).toFixed(1)} MiB` : "0 MiB",
    }))) return;
    try {
      if (replacement) await draftRecovery.replaceSetup(nextSetup, existing);
      else await draftRecovery.saveSetup(nextSetup);
    } catch (error) {
      setErrors([translate((error as DraftStoreError).kind === "limit" ? "draft.limit_error" : "draft.storage_error")]);
      return;
    }
    applySetup(nextSetup);

    if (canContinueToSpeak) {
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
          border: "1px solid var(--color-error-border)",
          backgroundColor: "var(--color-error-bg)",
        }}
      >
        <h2
          style={{
            margin: 0,
            color: "var(--color-text)",
          }}
        >
          {translate("setup.title")}
        </h2>
        <p
          style={{
            margin: 0,
            color: "var(--color-error)",
          }}
        >
          {translate("setup.no_languages")}
        </p>
      </section>
    );
  }

  return (
    <div style={pageStyle} data-testid="setup.layout" data-semantic-id="setup.layout">
      <div style={{ gridColumn: "1 / -1" }}>
        <StatusBanner
          title={translate(readyToRecord ? "first_run.ready_for_task" : canContinueToSpeak ? "first_run.continue_to_speak_title" : "first_run.task_setup_title")}
          tone={readyToRecord ? "success" : "info"}
          action={!canContinueToSpeak ? <Link to="/runtime-setup">{translate("first_run.open_local_setup")}</Link> : undefined}
        >
          {translate(readyToRecord ? "first_run.ready_for_task_body" : canContinueToSpeak ? "first_run.continue_to_speak_body" : "first_run.task_setup_body")}
        </StatusBanner>
      </div>
      {draftRecovery.supported && (draftRecovery.drafts.length > 0 || draftRecovery.error) && (
        <button type="button" onClick={draftRecovery.manageDrafts}>{translate("draft.manage_saved")}</button>
      )}
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
          if (draftRecovery.active && normalizedLearner(value) !== draftRecovery.active.speakerId) {
            if (!window.confirm(translate("draft.learner_switch_confirm", { learner: draftRecovery.active.setup.speakerId }))) return;
            draftRecovery.leaveActive();
            clearRecording();
          }
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
      <div style={{ gridColumn: "1 / -1" }}><FirstRunPreview locale={locale} /></div>
    </div>
  );
};
