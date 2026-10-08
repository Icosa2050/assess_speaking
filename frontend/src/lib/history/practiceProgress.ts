import type { HistoryRow } from "@/lib/api/types";
import { CEFR_LEVELS, DURATION_OPTIONS, TASK_FAMILY_OPTIONS, type SessionDraft } from "@/lib/state/sessionDraft";

export const measurement = (value: unknown): number | null => {
  if (typeof value !== "number" && typeof value !== "string") return null;
  if (typeof value === "string" && !value.trim()) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

export const chronological = (rows: HistoryRow[]): HistoryRow[] => [...rows].sort(
  (a, b) => (Date.parse(a.timestamp) || 0) - (Date.parse(b.timestamp) || 0),
);

// Unknown conditions are not evidence of matching conditions. Legacy reports stay in the journal.
export const comparisonKey = (row: HistoryRow): string | null => {
  const context = row.practice;
  if (row.eligibility && row.eligibility.state !== "assessable") return null;
  if (!context || context.version !== 1 || !context.goal || !context.scoring_version ||
      !context.analysis_signature || !["hybrid", "deterministic_only"].includes(context.scoring_mode ?? "") ||
      !context.provider || !context.model || !context.whisper_model || context.dry_run ||
      !row.session_id || !row.speaker_id || !row.learning_language || !row.task_family ||
      !(context.target_duration_sec > 0)) return null;
  return JSON.stringify([
    row.speaker_id, row.learning_language, context.goal, row.task_family,
    context.target_duration_sec, context.version, context.scoring_version,
    context.scoring_mode, context.analysis_signature,
    context.provider, context.model, context.asr_provider, context.whisper_model,
  ]);
};

export const comparableAttempts = (rows: HistoryRow[], selected: HistoryRow): HistoryRow[] => {
  const key = comparisonKey(selected);
  return key ? chronological(rows.filter((row) => comparisonKey(row) === key)) : [];
};

export const retryDraft = (row: HistoryRow): Partial<SessionDraft> | null => {
  const context = row.practice;
  if (!context?.prompt_text?.trim() || !row.session_id || !row.speaker_id ||
      !CEFR_LEVELS.includes(context.goal as SessionDraft["cefrLevel"]) ||
      !DURATION_OPTIONS.includes(context.target_duration_sec as SessionDraft["durationSec"]) ||
      !TASK_FAMILY_OPTIONS.includes(row.task_family as SessionDraft["taskFamily"]) ||
      !["en", "it"].includes(row.learning_language)) return null;
  return {
    speakerId: row.speaker_id,
    learningLanguage: row.learning_language,
    learningLanguageLabel: row.learning_language === "it" ? "Italiano" : "English",
    cefrLevel: context.goal as SessionDraft["cefrLevel"],
    durationSec: context.target_duration_sec as SessionDraft["durationSec"],
    taskFamily: row.task_family as SessionDraft["taskFamily"],
    themeId: context.prompt_id || `saved-${row.session_id}`,
    themeLabel: row.theme,
    promptId: context.prompt_id,
    promptText: context.prompt_text,
    retryOfSessionId: row.session_id,
  };
};

export const comparisonAttempt = (rows: HistoryRow[], selected: HistoryRow): HistoryRow | null => {
  const comparable = comparableAttempts(rows, selected);
  const previous = comparable.filter((row) => row.session_id !== selected.session_id &&
    Date.parse(row.timestamp) <= Date.parse(selected.timestamp));
  // An explicit retry is compared to its parent. Never silently substitute a different parent.
  if (selected.practice?.retry_of_session_id) {
    const parent = previous.find((row) => row.session_id === selected.practice?.retry_of_session_id);
    return parent && parent.practice?.prompt_text === selected.practice.prompt_text ? parent : null;
  }
  return previous.at(-1) ?? null;
};
