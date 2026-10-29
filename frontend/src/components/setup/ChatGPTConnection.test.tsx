import "@testing-library/jest-dom/vitest";

import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "@/lib/api/client";
import type { RuntimeSettingsConnection } from "@/lib/api/types";
import { createQueryClient, queryKeys } from "@/lib/query/queryClient";
import { renderWithProviders } from "@/test/renderWithProviders";
import { readyReadiness } from "@/test/readinessFixtures";
import { ChatGPTConnection } from "./ChatGPTConnection";

vi.mock("@/lib/api/client", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api/client")>();
  return { ...original, apiClient: { ...original.apiClient, getRuntimeSettings: vi.fn() } };
});

const connection: RuntimeSettingsConnection = {
  connection_id: "chatgpt-1", provider_key: "chatgpt", provider_choice: "chatgpt",
  provider_label: "ChatGPT", label: "ChatGPT", model: "model-one", base_url: "https://example.test",
  is_default: true, is_local: false, requires_api_key: true, has_api_key: true,
  secret_state: "present", last_test_status: "", last_tested_at: "",
  openrouter_http_referer: "", openrouter_app_title: "",
  provider_metadata: { models: [
    { slug: "model-one", display_name: "Model one" },
    { slug: "model-two", display_name: "Model two" },
  ] },
};

const props = { locale: "en" as const, onTest: vi.fn(), onSave: vi.fn(), testMessage: "" };
const json = (value: object) => new Response(JSON.stringify(value), { status: 200 });

describe("ChatGPT connection readiness refresh", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.getRuntimeSettings).mockResolvedValue({
      ui_locale: "en", whisper_model: "small", active_connection_id: "chatgpt-1", connections: [connection],
    });
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("invalidates readiness after changing the model and disconnecting", async () => {
    const fetchMock = vi.fn(async (url: string) => json(String(url).includes("connections/chatgpt-1")
      ? { revocation_confirmed: true } : {}));
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    queryClient.setQueryData(queryKeys.readiness, readyReadiness);
    renderWithProviders(<ChatGPTConnection {...props} connectionId="chatgpt-1" />, { queryClient });

    fireEvent.change(await screen.findByRole("combobox", { name: "Model" }), { target: { value: "model-two" } });
    await waitFor(() => expect(queryClient.getQueryState(queryKeys.readiness)?.isInvalidated).toBe(true));
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("connections/chatgpt-1/model"), expect.objectContaining({ method: "PUT" }));

    queryClient.setQueryData(queryKeys.readiness, readyReadiness);
    expect(queryClient.getQueryState(queryKeys.readiness)?.isInvalidated).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Disconnect ChatGPT" }));
    await waitFor(() => expect(queryClient.getQueryState(queryKeys.readiness)?.isInvalidated).toBe(true));
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("connections/chatgpt-1"), expect.objectContaining({ method: "DELETE" }));
  });

  it("invalidates readiness when a pending sign-in completes", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => json(String(url).endsWith("/pending")
      ? { attempt_id: "attempt-1" }
      : { status: "connected", connection_id: "chatgpt-1", persistent: true })));
    const queryClient = createQueryClient();
    queryClient.setQueryData(queryKeys.readiness, readyReadiness);
    const onConnected = vi.fn();
    renderWithProviders(<ChatGPTConnection {...props} onConnected={onConnected} />, { queryClient });

    await waitFor(() => expect(onConnected).toHaveBeenCalledWith("chatgpt-1"));
    expect(queryClient.getQueryState(queryKeys.readiness)?.isInvalidated).toBe(true);
  });
});
