import { useQuery, type QueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import type { RuntimeResponse } from "@/lib/api/types";

const RECENT_TEST_MS = 5 * 60 * 1000;
type ConnectionTestResult = { testedAt: number };

export const connectionVerificationKey = (runtime?: RuntimeResponse) => [
  "runtime", "connection-verification", runtime?.provider ?? "", runtime?.model ?? "", runtime?.base_url ?? "",
] as const;

export const isRecentConnectionTest = (result?: ConnectionTestResult | null, now = Date.now()) =>
  Boolean(result && result.testedAt <= now && now - result.testedAt < RECENT_TEST_MS);

export const markConnectionTest = (queryClient: QueryClient, runtime: RuntimeResponse, passed: boolean) => {
  queryClient.setQueryData<ConnectionTestResult | null>(
    connectionVerificationKey(runtime),
    passed ? { testedAt: Date.now() } : null,
  );
};

export const useRecentConnectionTest = (runtime?: RuntimeResponse) => {
  const [now, setNow] = useState(() => Date.now());
  const query = useQuery<ConnectionTestResult | null>({
    queryKey: connectionVerificationKey(runtime),
    queryFn: async () => null,
    enabled: false,
  });
  useEffect(() => {
    setNow(Date.now());
    if (!query.data?.testedAt) return;
    const remaining = Math.max(0, query.data.testedAt + RECENT_TEST_MS - Date.now());
    const timer = window.setTimeout(() => setNow(Date.now()), remaining + 1);
    return () => window.clearTimeout(timer);
  }, [query.data?.testedAt]);
  return isRecentConnectionTest(query.data, now);
};
