import { buildSessionDraft, CEFR_LEVELS, DURATION_OPTIONS, hasSetupDraft, TASK_FAMILY_OPTIONS, type SessionDraft } from "./sessionDraft";

export const RETRY_DRAFT_STORAGE_KEY = "vostavo.retry-draft.v1";

const isText = (value: unknown): value is string => typeof value === "string";

// Only the setup context and the learner's brief are stored here. Audio, reports,
// runtime credentials, and assessment state remain outside browser storage.
export const readRetryDraft = (): SessionDraft | null => {
  try {
    const raw = window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY);
    if (!raw) return null;
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object") return null;
    const draft = value as Record<string, unknown>;
    const textFields = ["sessionId", "speakerId", "learningLanguage", "learningLanguageLabel", "cefrLevel",
      "themeId", "themeLabel", "taskFamily", "promptId", "promptText", "retryOfSessionId", "coachingGoal"];
    if (!textFields.every(key => isText(draft[key])) ||
      !DURATION_OPTIONS.includes(draft.durationSec as SessionDraft["durationSec"]) ||
      !CEFR_LEVELS.includes(draft.cefrLevel as SessionDraft["cefrLevel"]) ||
      !TASK_FAMILY_OPTIONS.includes(draft.taskFamily as SessionDraft["taskFamily"]) ||
      !["en", "it"].includes(String(draft.learningLanguage))) return null;
    const restored = buildSessionDraft(draft as unknown as SessionDraft);
    return restored.retryOfSessionId && hasSetupDraft(restored) ? restored : null;
  } catch {
    return null;
  }
};

export const persistRetryDraft = (draft: SessionDraft): void => {
  try {
    if (draft.retryOfSessionId && hasSetupDraft(draft)) {
      window.localStorage.setItem(RETRY_DRAFT_STORAGE_KEY, JSON.stringify(draft));
    } else {
      window.localStorage.removeItem(RETRY_DRAFT_STORAGE_KEY);
    }
  } catch {
    // Storage can be disabled or full. The in-memory draft remains usable.
  }
};
