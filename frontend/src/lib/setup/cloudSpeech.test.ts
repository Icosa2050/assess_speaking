import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RuntimeCloudSettingsResponse, RuntimeSettingsResponse } from "@/lib/api/types";
import { hasSelectedGroqConnection, useCloudSpeechConnection } from "./cloudSpeech";

const runtimeSettings: RuntimeSettingsResponse = {
  ui_locale: "en", whisper_model: "small", active_connection_id: "", asr_provider: "groq",
  connections: [{
    connection_id: "groq-1", provider_key: "groq", provider_choice: "groq", provider_label: "Groq",
    label: "Speech", model: "whisper-large-v3", base_url: "https://api.groq.com", is_default: false,
    is_local: false, requires_api_key: true, has_api_key: true, secret_state: "present",
    last_test_status: "", last_tested_at: "", openrouter_http_referer: "", openrouter_app_title: "", provider_metadata: {},
  }],
};

const cloudSettings = (id: string): RuntimeCloudSettingsResponse => ({
  settings: { asr_provider: "groq", asr_connection_id: id },
});

describe("cloud speech connection", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("requires the selected Groq connection and its credential", () => {
    expect(hasSelectedGroqConnection(runtimeSettings, cloudSettings("groq-1"))).toBe(true);
    expect(hasSelectedGroqConnection(runtimeSettings, cloudSettings("deleted"))).toBe(false);
    expect(hasSelectedGroqConnection({ ...runtimeSettings, connections: [{ ...runtimeSettings.connections[0], has_api_key: false }] }, cloudSettings("groq-1"))).toBe(false);
  });

  it("keeps a cached connection usable during a background settings refresh", async () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["runtime", "cloud"], cloudSettings("groq-1"));
    let finishRefresh: (response: Response) => void = () => {};
    const fetchMock = vi.fn(() => new Promise<Response>((resolve) => { finishRefresh = resolve; }));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useCloudSpeechConnection(runtimeSettings), {
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client: queryClient }, children),
    });
    act(() => { void queryClient.invalidateQueries({ queryKey: ["runtime", "cloud"] }); });
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(result.current).toMatchObject({ selected: true, connected: true, pending: false, error: false });

    finishRefresh(new Response(JSON.stringify(cloudSettings("groq-1")), {
      status: 200, headers: { "Content-Type": "application/json" },
    }));
    await waitFor(() => expect(queryClient.getQueryState(["runtime", "cloud"])?.fetchStatus).toBe("idle"));
  });

  it("keeps cached connection checks stable after a background fetch fails", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    queryClient.setQueryData(["runtime", "cloud"], cloudSettings("groq-1"));
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));

    const { result } = renderHook(() => useCloudSpeechConnection(runtimeSettings), {
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client: queryClient }, children),
    });
    act(() => { void queryClient.invalidateQueries({ queryKey: ["runtime", "cloud"] }); });
    await waitFor(() => expect(queryClient.getQueryState(["runtime", "cloud"])?.fetchStatus).toBe("idle"));
    expect(result.current).toMatchObject({ connected: true, pending: false, error: false });
  });
});
