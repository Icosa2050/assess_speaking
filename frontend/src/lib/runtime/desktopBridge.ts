export type DesktopLaunchMode = "repo" | "packaged";
export type DesktopAuthMode = "guest" | "optional" | "required";
import type { SessionDraft } from "@/lib/state/sessionDraft";

export interface SavedTake {
  mediaId: string;
  originalFilename: string;
  mime: string;
  size: number;
  sha256: string;
  durationSec: number;
  captureEndedAt: number;
  inputMethod: "record" | "upload";
}

export interface DraftSubmission {
  uploadRequestId: string;
  audioId: string;
  assessmentRequestId: string;
  assessmentId: string;
  requestFingerprint: string;
  frozenRequestBody: string;
  lastConfirmedPhase: "none" | "upload" | "assessment" | "completed";
}

export interface SavedDraft {
  version: 1;
  draftId: string;
  speakerId: string;
  setup: SessionDraft;
  createdAt: number;
  updatedAt: number;
  expiresAt: number;
  revision: number;
  status: "setup" | "take_ready" | "upload_uncertain" | "uploaded" | "assessment_uncertain" | "assessing";
  take: SavedTake | null;
  submission: DraftSubmission | null;
}

export interface DesktopDraftStoreBridge {
  list(): Promise<SavedDraft[]>;
  recoveryIds(): Promise<string[]>;
  saveSetup(setup: SessionDraft, draftId?: string, expectedRevision?: number): Promise<SavedDraft>;
  beginTake(draftId: string, expectedRevision: number, take: Omit<SavedTake, "mediaId">): Promise<string>;
  writeTakeChunk(transferId: string, bytes: number[]): Promise<number>;
  abortTake(transferId: string): Promise<void>;
  finishTake(transferId: string): Promise<SavedDraft>;
  loadTakeChunk(draftId: string, mediaId: string, index: number): Promise<number[]>;
  markSubmission(draftId: string, expectedRevision: number, status: SavedDraft["status"], submission: DraftSubmission): Promise<SavedDraft>;
  discard(draftId: string): Promise<void>;
  expire(): Promise<number>;
}

export interface DesktopRuntimeBridge {
  apiBaseUrl: string;
  sessionToken?: string;
  mediaToken?: string;
  deploymentMode: "local";
  launchMode: DesktopLaunchMode;
  packagingSafe: boolean;
  authMode: DesktopAuthMode;
  saveLearnerBackup?: (backupId: string) => Promise<{ status: "saved" | "cancelled" }>;
  draftSupportEmail?: (bundleId: string, recipient: string) => Promise<{ status: "drafted" }>;
  saveSupportBundle?: (bundleId: string) => Promise<{ status: "saved" | "cancelled" }>;
  draftStore?: DesktopDraftStoreBridge;
}

type DesktopBridgeWindow = Window &
  typeof globalThis & {
    __TAURI_INTERNALS__?: unknown;
    __VOSTAVO_DESKTOP__?: Partial<DesktopRuntimeBridge>;
  };

const DEFAULT_DESKTOP_RUNTIME: DesktopRuntimeBridge = {
  apiBaseUrl: "http://127.0.0.1:8000",
  deploymentMode: "local",
  launchMode: "repo",
  packagingSafe: false,
  authMode: "guest",
};

const readDesktopWindow = (): DesktopBridgeWindow | null => {
  if (typeof window === "undefined") {
    return null;
  }

  return window as DesktopBridgeWindow;
};

const isLaunchMode = (value: string): value is DesktopLaunchMode =>
  value === "repo" || value === "packaged";

const isAuthMode = (value: string): value is DesktopAuthMode =>
  value === "guest" || value === "optional" || value === "required";

const isDraftStore = (value: unknown): value is DesktopDraftStoreBridge => {
  if (!value || typeof value !== "object") return false;
  return ["list", "recoveryIds", "saveSetup", "beginTake", "writeTakeChunk", "abortTake", "finishTake", "loadTakeChunk", "markSubmission", "discard", "expire"]
    .every((method) => typeof (value as Record<string, unknown>)[method] === "function");
};

export const hasDesktopDraftStore = (): boolean =>
  isDraftStore(readDesktopWindow()?.__VOSTAVO_DESKTOP__?.draftStore);

export const isDesktopRuntime = (): boolean => {
  const desktopWindow = readDesktopWindow();
  return Boolean(desktopWindow?.__TAURI_INTERNALS__ || desktopWindow?.__VOSTAVO_DESKTOP__);
};

export const readDesktopRuntimeBridge = (): DesktopRuntimeBridge => {
  const desktopWindow = readDesktopWindow();
  const candidate = desktopWindow?.__VOSTAVO_DESKTOP__;

  return {
    apiBaseUrl:
      typeof candidate?.apiBaseUrl === "string" && candidate.apiBaseUrl.trim()
        ? candidate.apiBaseUrl.trim()
        : DEFAULT_DESKTOP_RUNTIME.apiBaseUrl,
    sessionToken: typeof candidate?.sessionToken === "string" ? candidate.sessionToken : undefined,
    mediaToken: typeof candidate?.mediaToken === "string" ? candidate.mediaToken : undefined,
    saveLearnerBackup: typeof candidate?.saveLearnerBackup === "function" ? candidate.saveLearnerBackup : undefined,
    draftSupportEmail: typeof candidate?.draftSupportEmail === "function" ? candidate.draftSupportEmail : undefined,
    saveSupportBundle: typeof candidate?.saveSupportBundle === "function" ? candidate.saveSupportBundle : undefined,
    draftStore: isDraftStore(candidate?.draftStore) ? candidate.draftStore : undefined,
    deploymentMode: "local",
    launchMode: isLaunchMode(String(candidate?.launchMode || ""))
      ? candidate!.launchMode!
      : DEFAULT_DESKTOP_RUNTIME.launchMode,
    packagingSafe:
      typeof candidate?.packagingSafe === "boolean"
        ? candidate.packagingSafe
        : DEFAULT_DESKTOP_RUNTIME.packagingSafe,
    authMode: isAuthMode(String(candidate?.authMode || ""))
      ? candidate!.authMode!
      : DEFAULT_DESKTOP_RUNTIME.authMode,
  };
};

export const desktopBridge = {
  defaultRuntime: DEFAULT_DESKTOP_RUNTIME,
  isDesktopRuntime,
  readRuntime: readDesktopRuntimeBridge,
  hasDraftStore: hasDesktopDraftStore,
} as const;
