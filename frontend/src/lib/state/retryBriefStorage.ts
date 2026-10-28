import { buildSessionDraft, CEFR_LEVELS, DURATION_OPTIONS, hasSetupDraft, TASK_FAMILY_OPTIONS, type SessionDraft } from "./sessionDraft";

export const RETRY_DRAFT_STORAGE_KEY = "vostavo.retry-draft.v2";
export const RETRY_DRAFT_TTL_MS = 7 * 24 * 60 * 60 * 1000;

type StoredRetry = { version: 2; createdAt: number; context: string; draft: SessionDraft };

const textLimits: Partial<Record<keyof SessionDraft, number>> = {
  sessionId: 128, speakerId: 256, learningLanguage: 2, learningLanguageLabel: 100,
  cefrLevel: 2, themeId: 256, themeLabel: 1000, taskFamily: 64,
  promptId: 256, promptText: 12000, retryOfSessionId: 128, coachingGoal: 12000,
};

const normalizeRetryDraft = (draft: SessionDraft): SessionDraft => {
  const bounded = { ...draft };
  for (const [key, max] of Object.entries(textLimits) as [keyof SessionDraft, number][]) {
    (bounded as unknown as Record<string, unknown>)[key] = String(draft[key]).slice(0, max);
  }
  return buildSessionDraft(bounded);
};

const contextFor = (draft: SessionDraft): string => JSON.stringify([
  draft.speakerId, draft.learningLanguage, draft.themeId, draft.promptId,
  draft.taskFamily, draft.cefrLevel, draft.durationSec, draft.promptText, draft.retryOfSessionId,
]);

export const clearRetryDraft = (): void => {
  try {
    window.localStorage.removeItem(RETRY_DRAFT_STORAGE_KEY);
    window.localStorage.removeItem("vostavo.retry-draft.v1");
  } catch { /* Storage may be disabled. */ }
};

const parseStored = (raw: string | null): StoredRetry | null => {
  if (!raw) return null;
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== "object") return null;
  const stored = value as Record<string, unknown>;
  const data = stored.draft;
  if (stored.version !== 2 || typeof stored.createdAt !== "number" ||
      !data || typeof data !== "object" || Array.isArray(data)) return null;
  const draft = data as Record<string, unknown>;
  if (!Object.entries(textLimits).every(([key, max]) =>
    typeof draft[key] === "string" && (draft[key] as string).length <= max) ||
    draft.retryGoalEligible !== true ||
    !DURATION_OPTIONS.includes(draft.durationSec as SessionDraft["durationSec"]) ||
    !CEFR_LEVELS.includes(draft.cefrLevel as SessionDraft["cefrLevel"]) ||
    !TASK_FAMILY_OPTIONS.includes(draft.taskFamily as SessionDraft["taskFamily"]) ||
    !["en", "it"].includes(String(draft.learningLanguage))) return null;
  const restored = normalizeRetryDraft(draft as unknown as SessionDraft);
  if (!restored.retryOfSessionId || !hasSetupDraft(restored) || stored.context !== contextFor(restored)) return null;
  return { version: 2, createdAt: stored.createdAt, context: stored.context, draft: restored };
};

// Only the setup context and the learner's brief are stored here. Audio, reports,
// runtime credentials, and assessment state remain outside browser storage.
export const readRetryDraft = (): SessionDraft | null => {
  try {
    const raw = window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY);
    if (!raw) return null;
    const stored = parseStored(raw);
    const age = stored ? Date.now() - stored.createdAt : -1;
    if (stored && age >= 0 && age <= RETRY_DRAFT_TTL_MS) return stored.draft;
    clearRetryDraft();
    return null;
  } catch {
    clearRetryDraft();
    return null;
  }
};

export const persistRetryDraft = (draft: SessionDraft): void => {
  try {
    const normalized = normalizeRetryDraft(draft);
    if (!normalized.retryGoalEligible || !normalized.retryOfSessionId || !hasSetupDraft(normalized)) {
      clearRetryDraft();
      return;
    }
    const context = contextFor(normalized);
    let previous: StoredRetry | null = null;
    try { previous = parseStored(window.localStorage.getItem(RETRY_DRAFT_STORAGE_KEY)); }
    catch { clearRetryDraft(); }
    const now = Date.now();
    const createdAt = previous?.context === context && now - previous.createdAt >= 0 &&
      now - previous.createdAt <= RETRY_DRAFT_TTL_MS ? previous.createdAt : now;
    window.localStorage.setItem(RETRY_DRAFT_STORAGE_KEY, JSON.stringify({ version: 2, createdAt, context, draft: normalized } satisfies StoredRetry));
  } catch {
    // Storage can be disabled or full. The in-memory draft remains usable.
  }
};
