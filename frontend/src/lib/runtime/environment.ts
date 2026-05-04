const DEFAULT_LOCAL_DESKTOP_API_BASE_URL = "http://127.0.0.1:8000";

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "");

const readEnvBaseUrl = (): string => {
  const envValue =
    import.meta.env.VITE_LOCAL_API_BASE_URL ??
    import.meta.env.VITE_API_BASE_URL ??
    "";

  return String(envValue).trim();
};

const readWindowOrigin = (): string => {
  if (typeof window === "undefined") {
    return "";
  }

  return trimTrailingSlash(String(window.location.origin || ""));
};

export const resolveLocalDesktopApiBaseUrl = (override?: string): string => {
  const explicitOverride = String(override || "").trim();
  if (explicitOverride) {
    return trimTrailingSlash(explicitOverride);
  }

  const envOverride = readEnvBaseUrl();
  if (envOverride) {
    return trimTrailingSlash(envOverride);
  }

  const origin = readWindowOrigin();
  if (origin.endsWith(":8000")) {
    return origin;
  }

  return DEFAULT_LOCAL_DESKTOP_API_BASE_URL;
};

export const buildApiUrl = (path: string, baseUrl?: string): string => {
  const resolvedBaseUrl = resolveLocalDesktopApiBaseUrl(baseUrl);
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${trimTrailingSlash(resolvedBaseUrl)}${normalizedPath}`;
};

export const runtimeEnvironment = {
  defaultLocalDesktopApiBaseUrl: DEFAULT_LOCAL_DESKTOP_API_BASE_URL,
  resolveLocalDesktopApiBaseUrl,
  buildApiUrl,
} as const;
