import { useQuery } from "@tanstack/react-query";

import { apiClient } from "@/lib/api/client";
import { queryKeys } from "@/lib/query/queryClient";

export const useReadiness = () => useQuery({
  queryKey: queryKeys.readiness,
  queryFn: () => apiClient.getReadiness(),
  // The backend bounds local probes and never makes a paid cloud call. Sharing
  // this query also avoids probing again on every route render.
  staleTime: 60_000,
});
