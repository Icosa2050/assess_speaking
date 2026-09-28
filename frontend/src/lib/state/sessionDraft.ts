export const DEFAULT_UI_LOCALE = "en";
export const SUPPORTED_UI_LOCALES = ["en", "de", "es", "fr", "it"] as const;
export const CEFR_LEVELS = ["B1", "B2", "C1"] as const;
export const DURATION_OPTIONS = [60, 90, 120, 180] as const;
export const TASK_FAMILY_OPTIONS = [
  "travel_narrative",
  "personal_experience",
  "opinion_monologue",
  "picture_description",
  "free_monologue",
] as const;

export type UiLocale = (typeof SUPPORTED_UI_LOCALES)[number];
export type CefrLevel = (typeof CEFR_LEVELS)[number];
export type DurationOption = (typeof DURATION_OPTIONS)[number];
export type TaskFamily = (typeof TASK_FAMILY_OPTIONS)[number];
export type RecordingStatus = "idle" | "recording" | "ready" | "assessing" | "submitted";
export type RecordingInputMethod = "record" | "upload" | "";
export type AssessmentLifecycleState =
  | "idle"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";
export type RuntimeReadinessMissing = "connection";

export type RuntimeSetupSecretState =
  | { kind: "absent" }
  | { kind: "present"; secretRef?: string }
  | { kind: "missing"; secretRef: string }
  | { kind: "confirming_clear"; secretRef: string }
  | { kind: "cleared_undoable"; secretRef: string };

export type RuntimeSetupWhisperState =
  | { kind: "unknown" }
  | { kind: "missing" }
  | { kind: "downloading"; progressPercent: number }
  | { kind: "ready"; cachedPath?: string };

export type RuntimeSetupConnectionTestState =
  | { kind: "idle" }
  | { kind: "testing" }
  | { kind: "ok"; message: string }
  | { kind: "failed"; message: string };

export interface AppPreferencesState {
  uiLocale: UiLocale;
  activeConnectionId: string;
  setupComplete: boolean;
}

export interface SessionDraft {
  sessionId: string;
  speakerId: string;
  learningLanguage: string;
  learningLanguageLabel: string;
  cefrLevel: CefrLevel;
  themeId: string;
  themeLabel: string;
  taskFamily: TaskFamily;
  durationSec: DurationOption;
  promptId: string;
  promptText: string;
}

export interface AssessmentJobState {
  assessmentId: string;
  status: string;
  phase: string;
  progress: number;
  error: string;
  reportPath: string;
}

export interface RecordingState {
  status: RecordingStatus;
  assessmentState: AssessmentLifecycleState;
  audioPath: string;
  durationSec: number;
  inputDigest: string;
  inputMethod: RecordingInputMethod;
  error: string;
  labelInput: string;
  notesInput: string;
  job: AssessmentJobState;
}

export interface ReviewState {
  reportId: string;
  transcript: string;
  scoreOverall: number | null;
  band: string;
  summary: string;
  payload: Record<string, unknown>;
}

export interface NavigationState {
  currentPage: string;
  returnTo: string;
}

const randomToken = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 8);
  }

  return Math.random().toString(16).slice(2, 10);
};

export const createDraftSessionId = (): string => `draft-${randomToken()}`;

export const buildAppPreferences = (
  overrides: Partial<AppPreferencesState> = {},
): AppPreferencesState => ({
  uiLocale: overrides.uiLocale ?? DEFAULT_UI_LOCALE,
  activeConnectionId: overrides.activeConnectionId ?? "",
  setupComplete: overrides.setupComplete ?? false,
});

export const buildSessionDraft = (
  overrides: Partial<SessionDraft> = {},
): SessionDraft => ({
  sessionId: overrides.sessionId ?? createDraftSessionId(),
  speakerId: overrides.speakerId ?? "",
  learningLanguage: overrides.learningLanguage ?? "it",
  learningLanguageLabel: overrides.learningLanguageLabel ?? "Italiano",
  cefrLevel: overrides.cefrLevel ?? "B1",
  themeId: overrides.themeId ?? "",
  themeLabel: overrides.themeLabel ?? "",
  taskFamily: overrides.taskFamily ?? "free_monologue",
  durationSec: overrides.durationSec ?? 90,
  promptId: overrides.promptId ?? "",
  promptText: overrides.promptText ?? "",
});

export const buildAssessmentJobState = (
  overrides: Partial<AssessmentJobState> = {},
): AssessmentJobState => ({
  assessmentId: overrides.assessmentId ?? "",
  status: overrides.status ?? "",
  phase: overrides.phase ?? "",
  progress: overrides.progress ?? 0,
  error: overrides.error ?? "",
  reportPath: overrides.reportPath ?? "",
});

export const buildRecordingState = (
  overrides: Partial<RecordingState> = {},
): RecordingState => ({
  status: overrides.status ?? "idle",
  assessmentState: overrides.assessmentState ?? "idle",
  audioPath: overrides.audioPath ?? "",
  durationSec: overrides.durationSec ?? 0,
  inputDigest: overrides.inputDigest ?? "",
  inputMethod: overrides.inputMethod ?? "",
  error: overrides.error ?? "",
  labelInput: overrides.labelInput ?? "",
  notesInput: overrides.notesInput ?? "",
  job: buildAssessmentJobState(overrides.job),
});

export const buildReviewState = (
  overrides: Partial<ReviewState> = {},
): ReviewState => ({
  reportId: overrides.reportId ?? "",
  transcript: overrides.transcript ?? "",
  scoreOverall: overrides.scoreOverall ?? null,
  band: overrides.band ?? "",
  summary: overrides.summary ?? "",
  payload: overrides.payload ?? {},
});

export const buildNavigationState = (
  overrides: Partial<NavigationState> = {},
): NavigationState => ({
  currentPage: overrides.currentPage ?? "home",
  returnTo: overrides.returnTo ?? "home",
});

export const hasSetupDraft = (draft: SessionDraft): boolean =>
  Boolean(draft.speakerId && draft.themeId && draft.promptText && draft.cefrLevel);

export const hasReviewState = (review: ReviewState): boolean => Boolean(review.reportId);

export const hasRecordingAttachment = (recording: RecordingState): boolean =>
  Boolean(recording.audioPath && recording.inputMethod);
