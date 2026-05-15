import sharedContent from "../../../../assessment_runtime/data/session_setup_content.json";

import { TASK_FAMILY_OPTIONS, type CefrLevel, type DurationOption, type TaskFamily } from "@/lib/state/sessionDraft";

export interface ThemeEntry {
  title: string;
  level: string;
  task_family: TaskFamily;
}

export interface ThemeLibraryLanguage {
  label: string;
  themes: ThemeEntry[];
}

export type ThemeLibrary = Record<string, ThemeLibraryLanguage>;

export interface PracticeBrief {
  prompt: string;
  successFocus: string[];
}

export interface ThemeLibraryRepository {
  load: () => ThemeLibrary;
  save: (library: ThemeLibrary) => void;
  addCustomTheme: (input: {
    languageCode: string;
    languageLabel: string;
    title: string;
    level: CefrLevel;
    taskFamily: TaskFamily;
  }) => ThemeLibrary;
}

interface PracticeBriefTemplateGroup {
  travel_narrative: string;
  personal_experience: string;
  opinion_monologue: string;
  free_monologue: string;
  picture_description: string;
  default_duration_minutes: string;
  default_duration_seconds: string;
  success_focus: string[];
}

export interface SessionSetupContent {
  default_theme_library: ThemeLibrary;
  practice_brief_templates: Record<string, PracticeBriefTemplateGroup>;
}

const STORAGE_KEY = "assess-speaking.session-setup.theme-library";
const REQUIRED_PRACTICE_BRIEF_TEMPLATE_KEYS = [
  ...TASK_FAMILY_OPTIONS,
  "default_duration_minutes",
  "default_duration_seconds",
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

export const isValidTaskFamily = (value: string): value is TaskFamily =>
  TASK_FAMILY_OPTIONS.includes(value as TaskFamily);

const validateEnglishPracticeBriefTemplates = (
  templates: Record<string, unknown>,
): Record<string, PracticeBriefTemplateGroup> => {
  if (!isRecord(templates.en)) {
    throw new Error("Session setup content is missing English practice brief templates.");
  }

  for (const key of REQUIRED_PRACTICE_BRIEF_TEMPLATE_KEYS) {
    if (typeof templates.en[key] !== "string" || !String(templates.en[key]).trim()) {
      throw new Error(`English practice brief template '${key}' is required.`);
    }
  }

  if (
    !Array.isArray(templates.en.success_focus) ||
    !templates.en.success_focus.every((item) => typeof item === "string")
  ) {
    throw new Error("English practice brief success_focus must be a string list.");
  }

  return templates as Record<string, PracticeBriefTemplateGroup>;
};

export const parseSessionSetupContent = (raw: unknown): SessionSetupContent => {
  if (!isRecord(raw)) {
    throw new Error("Session setup content must be an object.");
  }

  if (!isRecord(raw.default_theme_library)) {
    throw new Error("Session setup content is missing default_theme_library.");
  }

  if (!isRecord(raw.practice_brief_templates)) {
    throw new Error("Session setup content is missing practice_brief_templates.");
  }

  const practiceBriefTemplates = validateEnglishPracticeBriefTemplates(raw.practice_brief_templates);

  return {
    default_theme_library: raw.default_theme_library as ThemeLibrary,
    practice_brief_templates: practiceBriefTemplates,
  };
};

const content = parseSessionSetupContent(sharedContent);

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

export const loadDefaultThemeLibrary = (): ThemeLibrary => clone(content.default_theme_library);

export const normalizeThemeLibrary = (library: unknown): ThemeLibrary => {
  const normalized = loadDefaultThemeLibrary();
  if (!library || typeof library !== "object") {
    return normalized;
  }

  for (const [languageCode, payload] of Object.entries(library as Record<string, unknown>)) {
    if (!payload || typeof payload !== "object") {
      continue;
    }

    const languagePayload = payload as Partial<ThemeLibraryLanguage>;
    const label = String(languagePayload.label || languageCode).trim() || languageCode;
    const themes = normalized[languageCode]?.themes ?? [];
    normalized[languageCode] = {
      label,
      themes,
    };

    for (const rawTheme of languagePayload.themes ?? []) {
      if (!rawTheme || typeof rawTheme !== "object") {
        continue;
      }

      const theme = rawTheme as Partial<ThemeEntry>;
      const title = String(theme.title || "").trim();
      if (!title) {
        continue;
      }

      const taskFamily = String(theme.task_family || "free_monologue").trim();
      const entry: ThemeEntry = {
        title,
        level: String(theme.level || "B1").trim().toUpperCase(),
        task_family: isValidTaskFamily(taskFamily) ? taskFamily : "free_monologue",
      };

      if (
        !normalized[languageCode].themes.some(
          (candidate) =>
            candidate.title === entry.title &&
            candidate.level === entry.level &&
            candidate.task_family === entry.task_family,
        )
      ) {
        normalized[languageCode].themes.push(entry);
      }
    }
  }

  return normalized;
};

const readStoredThemeLibrary = (): ThemeLibrary => {
  if (typeof window === "undefined") {
    return loadDefaultThemeLibrary();
  }

  const serialized = window.localStorage.getItem(STORAGE_KEY);
  if (!serialized) {
    return loadDefaultThemeLibrary();
  }

  try {
    return normalizeThemeLibrary(JSON.parse(serialized) as unknown);
  } catch {
    return loadDefaultThemeLibrary();
  }
};

const writeStoredThemeLibrary = (library: ThemeLibrary): void => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalizeThemeLibrary(library)));
};

export const languageCodes = (library: ThemeLibrary): string[] => Object.keys(library).sort();

export const languageLabel = (library: ThemeLibrary, languageCode: string): string =>
  String(library[languageCode]?.label || languageCode);

export const themesForLanguageAndLevel = (
  library: ThemeLibrary,
  languageCode: string,
  level: CefrLevel,
): ThemeEntry[] =>
  (library[languageCode]?.themes ?? []).filter((theme) => theme.level === level);

export const themeEntryId = (theme: Pick<ThemeEntry, "title" | "level">): string => {
  const title =
    theme.title.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "theme";
  const level = theme.level.trim().toLowerCase() || "b1";
  return `${level}-${title}`;
};

export const buildPracticeBrief = (input: {
  taskFamily: TaskFamily;
  theme: string;
  targetDurationSec: DurationOption;
  languageCode: string;
}): PracticeBrief => {
  const templates =
    content.practice_brief_templates[input.languageCode] ?? content.practice_brief_templates.en;
  const minutes = Math.round((Number(input.targetDurationSec) / 60) * 10) / 10;
  const promptTemplate = templates[input.taskFamily] ?? templates.free_monologue;
  const durationTemplate =
    minutes >= 1 ? templates.default_duration_minutes : templates.default_duration_seconds;

  return {
    prompt: promptTemplate.replaceAll("{theme}", input.theme.trim()),
    successFocus: [
      durationTemplate
        .replaceAll("{minutes}", String(minutes))
        .replaceAll("{seconds}", String(input.targetDurationSec)),
      ...templates.success_focus,
    ],
  };
};

export const createThemeLibraryRepository = (): ThemeLibraryRepository => ({
  load: () => readStoredThemeLibrary(),
  save: (library) => {
    writeStoredThemeLibrary(library);
  },
  addCustomTheme: ({ languageCode, languageLabel: label, title, level, taskFamily }) => {
    const normalized = normalizeThemeLibrary(readStoredThemeLibrary());
    const code = languageCode.trim().toLowerCase();
    const resolvedTitle = title.trim();
    if (!code || !resolvedTitle) {
      throw new Error("Language code and theme title are required.");
    }

    const entry: ThemeEntry = {
      title: resolvedTitle,
      level,
      task_family: taskFamily,
    };

    const nextLibrary = normalizeThemeLibrary({
      ...normalized,
      [code]: {
        label: label.trim() || code,
        themes: [...(normalized[code]?.themes ?? []), entry],
      },
    });
    writeStoredThemeLibrary(nextLibrary);
    return nextLibrary;
  },
});

export const themeLibraryRepository = createThemeLibraryRepository();
