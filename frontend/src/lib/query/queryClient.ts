import { QueryClient } from "@tanstack/react-query";

export const pollingIntervals = {
  liveMs: 1_500,
  assessmentMs: 4_000,
  backgroundMs: 15_000,
} as const;

export const queryKeys = {
  health: ["health"] as const,
  diagnostics: ["diagnostics"] as const,
  runtime: ["runtime"] as const,
  samples: ["samples"] as const,
  history: ["history"] as const,
  historyDetail: (sessionId: string) => ["history", "detail", sessionId] as const,
  assessment: (assessmentId: string) => ["assessments", assessmentId] as const,
  maintenanceStorage: ["maintenance", "storage"] as const,
} as const;

const shouldRetry = (failureCount: number, error: unknown): boolean => {
  if (failureCount >= 1) {
    return false;
  }

  if (
    typeof error === "object" &&
    error !== null &&
    "responseStatus" in error &&
    typeof error.responseStatus === "number" &&
    error.responseStatus >= 400 &&
    error.responseStatus < 500
  ) {
    return false;
  }

  return true;
};

export const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: shouldRetry,
        staleTime: 30_000,
        gcTime: 5 * 60_000,
        refetchOnWindowFocus: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

export const sharedQueryClient = createQueryClient();
