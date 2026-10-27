import { useQuery } from "@tanstack/react-query";

import type { RuntimeCloudSettingsResponse, RuntimeSettingsResponse } from "@/lib/api/types";
import { cloudRequest } from "@/components/setup/CloudSettingsPanel";

export const hasSelectedGroqConnection = (
  runtimeSettings?: RuntimeSettingsResponse,
  cloudSettings?: RuntimeCloudSettingsResponse,
) => {
  const selectedId = cloudSettings?.settings.asr_connection_id;
  return Boolean(
    runtimeSettings?.asr_provider === "groq" &&
    cloudSettings?.settings.asr_provider === "groq" &&
    selectedId &&
    runtimeSettings.connections.some((connection) =>
      connection.connection_id === selectedId && connection.provider_key === "groq" && connection.has_api_key,
    ),
  );
};

export const useCloudSpeechConnection = (runtimeSettings?: RuntimeSettingsResponse) => {
  const selected = runtimeSettings?.asr_provider === "groq";
  const query = useQuery({
    queryKey: ["runtime", "cloud"],
    queryFn: () => cloudRequest<RuntimeCloudSettingsResponse>(),
    enabled: selected,
    retry: false,
  });
  return {
    selected,
    connected: hasSelectedGroqConnection(runtimeSettings, query.data),
    pending: selected && query.isPending,
    error: selected && query.isError && !query.data,
  };
};
