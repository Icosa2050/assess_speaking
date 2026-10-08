export type DesktopLaunchMode = "repo" | "packaged";
export type DesktopAuthMode = "guest" | "optional" | "required";

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
} as const;
