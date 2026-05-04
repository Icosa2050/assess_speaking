import { createContext, createElement, useContext, useRef, type ReactNode } from "react";
import { useStore } from "zustand";
import { createStore, type StoreApi } from "zustand/vanilla";

import { detectPreferredUiLocale, resolveUiLocale } from "@/lib/i18n";

import {
  buildAppPreferences,
  buildNavigationState,
  buildRecordingState,
  buildReviewState,
  buildSessionDraft,
  hasRecordingAttachment,
  hasSetupDraft,
  type AppPreferencesState,
  type AssessmentJobState,
  type NavigationState,
  type RecordingState,
  type ReviewState,
  type RuntimeReadinessMissing,
  type SessionDraft,
} from "./sessionDraft";

export interface AppStoreState {
  preferences: AppPreferencesState;
  draft: SessionDraft;
  recording: RecordingState;
  review: ReviewState;
  navigation: NavigationState;
}

export type AppStoreSeed = Partial<{
  preferences: Partial<AppPreferencesState>;
  draft: Partial<SessionDraft>;
  recording: Partial<RecordingState>;
  review: Partial<ReviewState>;
  navigation: Partial<NavigationState>;
}>;

export interface AppStoreActions {
  setUiLocale: (locale: string) => void;
  setActiveConnectionId: (connectionId: string) => void;
  setSetupComplete: (setupComplete: boolean) => void;
  setCurrentPage: (pageId: string) => void;
  setReturnTo: (pageId: string) => void;
  beginNewSession: (options?: { preservePreferences?: boolean }) => void;
  updateDraft: (updates: Partial<SessionDraft>) => void;
  applySetup: (updates: Partial<SessionDraft>) => void;
  updateRecording: (updates: Partial<RecordingState>) => void;
  updateRecordingInputs: (labelInput: string, notesInput: string) => void;
  setRecordingError: (message: string) => void;
  setRecordingAssessing: (job?: Partial<AssessmentJobState>) => void;
  setRecordingJob: (job: Partial<AssessmentJobState>) => void;
  clearRecording: (options?: { preserveInputs?: boolean }) => void;
  updateReview: (updates: Partial<ReviewState>) => void;
  clearAttempt: (options?: { keepSetup?: boolean }) => void;
}

export type AppStore = AppStoreState & AppStoreActions;
export type AppStoreApi = StoreApi<AppStore>;

export interface RuntimeReadiness {
  ready: boolean;
  missing: RuntimeReadinessMissing[];
}

export type SpeakRouteGuardTarget = "/session-setup" | "/runtime-setup" | null;

const buildAppStoreState = (seed: AppStoreSeed = {}): AppStoreState => {
  const preferredLocale = resolveUiLocale(
    seed.preferences?.uiLocale ?? detectPreferredUiLocale(),
  );

  return {
    preferences: buildAppPreferences({
      ...seed.preferences,
      uiLocale: preferredLocale,
    }),
    draft: buildSessionDraft(seed.draft),
    recording: buildRecordingState(seed.recording),
    review: buildReviewState(seed.review),
    navigation: buildNavigationState(seed.navigation),
  };
};

export const createAppStore = (seed: AppStoreSeed = {}): AppStoreApi => {
  const initialState = buildAppStoreState(seed);

  return createStore<AppStore>()((set) => ({
    ...initialState,
    setUiLocale: (locale) =>
      set((state) => ({
        preferences: {
          ...state.preferences,
          uiLocale: resolveUiLocale(locale),
        },
      })),
    setActiveConnectionId: (connectionId) =>
      set((state) => ({
        preferences: {
          ...state.preferences,
          activeConnectionId: String(connectionId || "").trim(),
        },
      })),
    setSetupComplete: (setupComplete) =>
      set((state) => ({
        preferences: {
          ...state.preferences,
          setupComplete: Boolean(setupComplete),
        },
      })),
    setCurrentPage: (pageId) =>
      set((state) => ({
        navigation: {
          ...state.navigation,
          currentPage: pageId,
        },
      })),
    setReturnTo: (pageId) =>
      set((state) => ({
        navigation: {
          ...state.navigation,
          returnTo: pageId,
        },
      })),
    beginNewSession: ({ preservePreferences = true } = {}) =>
      set((state) => ({
        preferences: preservePreferences
          ? state.preferences
          : buildAppPreferences({
              uiLocale: resolveUiLocale(state.preferences.uiLocale),
            }),
        draft: buildSessionDraft(),
        recording: buildRecordingState(),
        review: buildReviewState(),
        navigation: buildNavigationState(),
      })),
    updateDraft: (updates) =>
      set((state) => ({
        draft: buildSessionDraft({
          ...state.draft,
          ...updates,
        }),
      })),
    applySetup: (updates) =>
      set(() => {
        const nextDraft = buildSessionDraft(updates);
        const promptId =
          updates.promptId ??
          (nextDraft.themeId
            ? `${nextDraft.themeId}-${nextDraft.cefrLevel.toLowerCase()}`
            : nextDraft.promptId);

        return {
          draft: {
            ...nextDraft,
            promptId,
          },
          recording: buildRecordingState(),
          review: buildReviewState(),
        };
      }),
    updateRecording: (updates) =>
      set((state) => ({
        recording: buildRecordingState({
          ...state.recording,
          ...updates,
        }),
      })),
    updateRecordingInputs: (labelInput, notesInput) =>
      set((state) => ({
        recording: buildRecordingState({
          ...state.recording,
          labelInput,
          notesInput,
        }),
      })),
    setRecordingError: (message) =>
      set((state) => ({
        recording: buildRecordingState({
          ...state.recording,
          status: hasRecordingAttachment(state.recording) ? "ready" : "idle",
          assessmentState: state.recording.job.status === "cancelled" ? "cancelled" : "failed",
          error: message,
          job: buildRecordingState().job,
        }),
      })),
    setRecordingAssessing: (job = {}) =>
      set((state) => ({
        recording: buildRecordingState({
          ...state.recording,
          status: "assessing",
          assessmentState: job.status === "queued" ? "queued" : "running",
          error: "",
          job: {
            ...state.recording.job,
            ...job,
          },
        }),
      })),
    setRecordingJob: (job) =>
      set((state) => {
        const nextJob = {
          ...state.recording.job,
          ...job,
        };

        let nextStatus = state.recording.status;
        let nextError = state.recording.error;
        let nextAssessmentState = state.recording.assessmentState;

        if (nextJob.status === "queued") {
          nextStatus = "assessing";
          nextError = "";
          nextAssessmentState = "queued";
        } else if (nextJob.status === "running") {
          nextStatus = "assessing";
          nextError = "";
          nextAssessmentState = "running";
        } else if (nextJob.status === "completed") {
          nextStatus = "submitted";
          nextError = "";
          nextAssessmentState = "completed";
        } else if (nextJob.status === "failed" || nextJob.status === "cancelled") {
          nextStatus = hasRecordingAttachment(state.recording) ? "ready" : "idle";
          nextError = nextJob.error;
          nextAssessmentState = nextJob.status;
        }

        return {
          recording: buildRecordingState({
            ...state.recording,
            status: nextStatus,
            assessmentState: nextAssessmentState,
            error: nextError,
            job: nextJob,
          }),
        };
      }),
    clearRecording: ({ preserveInputs = true } = {}) =>
      set((state) => ({
        recording: buildRecordingState({
          labelInput: preserveInputs ? state.recording.labelInput : "",
          notesInput: preserveInputs ? state.recording.notesInput : "",
        }),
      })),
    updateReview: (updates) =>
      set((state) => ({
        review: buildReviewState({
          ...state.review,
          ...updates,
        }),
      })),
    clearAttempt: ({ keepSetup = true } = {}) =>
      set((state) => ({
        draft: keepSetup
          ? state.draft
          : buildSessionDraft({
              sessionId: state.draft.sessionId,
              speakerId: state.draft.speakerId,
              learningLanguage: state.draft.learningLanguage,
              learningLanguageLabel: state.draft.learningLanguageLabel,
            }),
        recording: buildRecordingState({
          labelInput: keepSetup ? state.recording.labelInput : "",
          notesInput: keepSetup ? state.recording.notesInput : "",
        }),
        review: buildReviewState(),
      })),
  }));
};

const AppStoreContext = createContext<AppStoreApi | null>(null);

export const AppStoreProvider = ({
  children,
  store,
}: {
  children: ReactNode;
  store?: AppStoreApi;
}) => {
  const storeRef = useRef<AppStoreApi | null>(store ?? null);

  if (storeRef.current === null) {
    storeRef.current = createAppStore();
  }

  return createElement(
    AppStoreContext.Provider,
    { value: storeRef.current },
    children,
  );
};

export const useAppStore = <T,>(selector: (state: AppStore) => T): T => {
  const store = useContext(AppStoreContext);

  if (store === null) {
    throw new Error("useAppStore must be used inside an AppStoreProvider.");
  }

  return useStore(store, selector);
};

export const selectRuntimeReadiness = (
  state: Pick<AppStoreState, "preferences">,
): RuntimeReadiness => {
  const hasConnections = Boolean(state.preferences.activeConnectionId);
  const ready = hasConnections || state.preferences.setupComplete;

  return {
    ready,
    missing: ready ? [] : ["connection"],
  };
};

export const resolveSpeakRouteGuardTarget = (
  state: Pick<AppStoreState, "draft" | "preferences">,
): SpeakRouteGuardTarget => {
  if (!hasSetupDraft(state.draft)) {
    return "/session-setup";
  }

  if (!selectRuntimeReadiness({ preferences: state.preferences }).ready) {
    return "/runtime-setup";
  }

  return null;
};

export const selectAssessmentLifecycleState = (
  state: Pick<AppStoreState, "recording">,
): RecordingState["assessmentState"] => state.recording.assessmentState;

export const selectCanSubmitAssessment = (
  state: Pick<AppStoreState, "draft" | "recording">,
): boolean => {
  const hasBlockingError =
    Boolean(state.recording.error) && state.recording.assessmentState !== "cancelled";

  return (
    hasSetupDraft(state.draft) &&
    hasRecordingAttachment(state.recording) &&
    state.recording.assessmentState !== "queued" &&
    state.recording.assessmentState !== "running" &&
    !hasBlockingError
  );
};
