import deMessages from "@locales/de.json";
import enMessages from "@locales/en.json";
import esMessages from "@locales/es.json";
import frMessages from "@locales/fr.json";
import itMessages from "@locales/it.json";

import {
  DEFAULT_UI_LOCALE,
  SUPPORTED_UI_LOCALES,
  type UiLocale,
} from "@/lib/state/sessionDraft";

export interface LocaleTree {
  [key: string]: LocaleTree | string;
}

export const SEMANTIC_IDS = {
  home: {
    runtimeSetupButton: "home.runtime_setup_button",
    startNew: "home.start_new",
    resume: "home.resume",
    openHistory: "home.open_history",
    openLibrary: "home.open_library",
    openGuide: "home.open_guide",
    openSettings: "home.open_settings",
  },
  runtimeConnection: {
    form: "runtime_connection.form",
    formStatus: "runtime_connection.form_status",
    formError: "runtime_connection.form_error",
    provider: "runtime_connection.provider",
    connectionLabel: "runtime_connection.connection_label",
    model: "runtime_connection.model",
    baseUrl: "runtime_connection.base_url",
    apiKey: "runtime_connection.api_key",
    openrouterHttpReferer: "runtime_connection.openrouter_http_referer",
    openrouterAppTitle: "runtime_connection.openrouter_app_title",
    secretState: "runtime_connection.secret_state",
    clearSavedKey: "runtime_connection.clear_saved_key",
    clearSavedKeyConfirm: "runtime_connection.clear_saved_key_confirm",
    clearSavedKeyCancel: "runtime_connection.clear_saved_key_cancel",
    clearSavedKeyUndo: "runtime_connection.clear_saved_key_undo",
    replaceSavedKey: "runtime_connection.replace_saved_key",
    testConnection: "runtime_connection.test_connection",
    saveConnection: "runtime_connection.save_connection",
  },
  runtimeSetup: {
    screen: "runtime_setup.screen",
    whisperStatus: "runtime_setup.whisper_status",
    whisperDownloadProgress: "runtime_setup.whisper_download_progress",
    downloadModel: "runtime_setup.download_model",
    detectLocalModels: "runtime_setup.detect_local_models",
    advancedProvidersToggle: "runtime_setup.advanced_providers_toggle",
    testConnection: "runtime_setup.test_connection",
    saveConnection: "runtime_setup.save_connection",
    backHome: "runtime_setup.back_home",
    clearSavedKey: "runtime_setup.clear_saved_key",
    clearSavedKeyConfirm: "runtime_setup.clear_saved_key_confirm",
    clearSavedKeyCancel: "runtime_setup.clear_saved_key_cancel",
    clearSavedKeyUndo: "runtime_setup.clear_saved_key_undo",
  },
  settings: {
    screen: "settings.screen",
    sectionConnections: "settings.section_connections",
    sectionRuntimeDefaults: "settings.section_runtime_defaults",
    sectionMaintenance: "settings.section_maintenance",
    sectionSupportBundle: "settings.section_support_bundle",
    openSetup: "settings.open_setup",
    connectionId: "settings.connection_id",
    connectionRow: "settings.connection_row",
    connectionRowEdit: "settings.connection_row_edit",
    connectionRowDelete: "settings.connection_row_delete",
    connectionRowSetDefault: "settings.connection_row_set_default",
    activeConnectionIndicator: "settings.active_connection_indicator",
    defaultConnectionIndicator: "settings.default_connection_indicator",
    uiLocale: "settings.ui_locale",
    provider: "settings.provider",
    connectionLabel: "settings.connection_label",
    model: "settings.model",
    baseUrl: "settings.base_url",
    apiKey: "settings.api_key",
    whisperModel: "settings.whisper_model",
    clearSavedKey: "settings.clear_saved_key",
    clearSavedKeyConfirm: "settings.clear_saved_key_confirm",
    clearSavedKeyCancel: "settings.clear_saved_key_cancel",
    clearSavedKeyUndo: "settings.clear_saved_key_undo",
    testConnection: "settings.test_connection",
    save: "settings.save",
    supportCleanupPreview: "settings.support_cleanup_preview",
    supportCleanupRun: "settings.support_cleanup_run",
    supportCleanupRunConfirm: "settings.support_cleanup_run_confirm",
    supportCleanupRunCancel: "settings.support_cleanup_run_cancel",
    supportIncludeRuntimeHealth: "settings.support_bundle_include_runtime_health",
    supportCreateBundle: "settings.support_create_bundle",
    return: "settings.return",
  },
  library: {
    screen: "library.screen",
    languageFilter: "library.language_filter",
    sampleGrid: "library.sample_grid",
    samplePrepare: "library.sample_prepare",
    manageLanguage: "library.manage_language",
    saveTheme: "library.save_theme",
  },
  guide: {
    screen: "guide.screen",
  },
  speak: {
    inputModeRecord: "speak.input_mode_record",
    inputModeUpload: "speak.input_mode_upload",
    audioInput: "speak.audio_input",
    recordStart: "speak.record_start",
    recordStop: "speak.record_stop",
    recordingStatus: "speak.recording_status",
    uploadInput: "speak.upload_input",
    removeRecording: "speak.remove_recording",
    label: "speak.label",
    notes: "speak.notes",
    submit: "speak.submit",
    cancelAssessment: "speak.cancel_assessment",
    statusPanel: "speak.status_panel",
    reviewAutoNav: "speak.review_auto_nav",
    openrouterKeyWarning: "speak.openrouter_key_warning",
  },
} as const;

export type SemanticId =
  | (typeof SEMANTIC_IDS.home)[keyof typeof SEMANTIC_IDS.home]
  | (typeof SEMANTIC_IDS.runtimeConnection)[keyof typeof SEMANTIC_IDS.runtimeConnection]
  | (typeof SEMANTIC_IDS.runtimeSetup)[keyof typeof SEMANTIC_IDS.runtimeSetup]
  | (typeof SEMANTIC_IDS.settings)[keyof typeof SEMANTIC_IDS.settings]
  | (typeof SEMANTIC_IDS.library)[keyof typeof SEMANTIC_IDS.library]
  | (typeof SEMANTIC_IDS.guide)[keyof typeof SEMANTIC_IDS.guide]
  | (typeof SEMANTIC_IDS.speak)[keyof typeof SEMANTIC_IDS.speak];

export type TranslateOptions = {
  locale?: string;
  fallback?: string;
  vars?: Record<string, string | number>;
};

const localeMessages: Record<UiLocale, LocaleTree> = {
  de: deMessages as LocaleTree,
  en: enMessages as LocaleTree,
  es: esMessages as LocaleTree,
  fr: frMessages as LocaleTree,
  it: itMessages as LocaleTree,
};

export const resolveUiLocale = (candidate?: string): UiLocale => {
  const normalized = String(candidate || "")
    .trim()
    .toLowerCase()
    .split("-")[0] as UiLocale;

  return SUPPORTED_UI_LOCALES.includes(normalized) ? normalized : DEFAULT_UI_LOCALE;
};

export const detectPreferredUiLocale = (): UiLocale => {
  if (typeof navigator === "undefined") {
    return DEFAULT_UI_LOCALE;
  }

  const candidates = navigator.languages.length > 0 ? navigator.languages : [navigator.language];
  for (const candidate of candidates) {
    const locale = resolveUiLocale(candidate);
    if (candidate && locale) {
      return locale;
    }
  }

  return DEFAULT_UI_LOCALE;
};

export const loadLocaleMessages = (locale?: string): LocaleTree =>
  localeMessages[resolveUiLocale(locale)];

export const t = (key: string, options: TranslateOptions = {}): string => {
  const strings = loadLocaleMessages(options.locale);
  let value: LocaleTree | string | undefined = strings;

  for (const part of key.split(".")) {
    if (typeof value !== "object" || value === null || !(part in value)) {
      return options.fallback ?? `[${key}]`;
    }
    value = value[part] as LocaleTree | string;
  }

  if (typeof value !== "string") {
    return options.fallback ?? `[${key}]`;
  }

  if (!options.vars) {
    return value;
  }

  return value.replace(/\{(\w+)\}/g, (_, token) => {
    const replacement = options.vars?.[token];
    return replacement === undefined ? `{${token}}` : String(replacement);
  });
};

export const createTranslator =
  (locale?: string) =>
  (key: string, vars?: Record<string, string | number>): string =>
    t(key, { locale, vars });

export const flattenLocaleKeys = (mapping: LocaleTree, prefix = ""): Set<string> => {
  const keys = new Set<string>();

  for (const [key, value] of Object.entries(mapping)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "object" && value !== null) {
      for (const nestedKey of flattenLocaleKeys(value as LocaleTree, fullKey)) {
        keys.add(nestedKey);
      }
    } else {
      keys.add(fullKey);
    }
  }

  return keys;
};

export const localeKeyMap = (): Record<UiLocale, Set<string>> =>
  Object.fromEntries(
    SUPPORTED_UI_LOCALES.map((locale) => [locale, flattenLocaleKeys(loadLocaleMessages(locale))]),
  ) as Record<UiLocale, Set<string>>;

export const semanticId = (id: SemanticId): string => id.trim();

export const semanticAttributes = (
  id: SemanticId,
  extras: Record<string, string | number | boolean | undefined> = {},
): Record<string, string> => {
  const normalized = semanticId(id);
  const attributes: Record<string, string> = {
    "data-testid": normalized,
    "data-semantic-id": normalized,
  };

  for (const [key, value] of Object.entries(extras)) {
    if (value !== undefined) {
      const normalizedKey = key.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
      attributes[`data-${normalizedKey}`] = String(value);
    }
  }

  return attributes;
};
