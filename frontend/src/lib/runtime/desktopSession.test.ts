import { afterEach, expect, it, vi } from "vitest";
import { buildAudioUrl, desktopSessionHeaders, resolveLocalDesktopApiBaseUrl } from "./environment";
import { createApiClient } from "@/lib/api/client";

afterEach(() => { delete (window as any).__VOSTAVO_DESKTOP__; vi.unstubAllGlobals(); vi.unstubAllEnvs(); });
const install = () => { (window as any).__VOSTAVO_DESKTOP__ = { apiBaseUrl: "http://127.0.0.1:8123", launchMode: "packaged", sessionToken: "a".repeat(64), mediaToken: "b".repeat(64) }; };

it("uses the packaged bridge and never sends its token to another server", () => {
  install();
  vi.stubEnv("VITE_LOCAL_API_BASE_URL", "https://different.example");
  expect(resolveLocalDesktopApiBaseUrl()).toBe("http://127.0.0.1:8123");
  expect(desktopSessionHeaders()).toEqual({ "X-Vostavo-Session": "a".repeat(64) });
  expect(desktopSessionHeaders("https://different.example")).toEqual({});
  expect(buildAudioUrl("/v1/history/id/audio")).toBe("http://127.0.0.1:8123/v1/history/id/audio?session=" + "b".repeat(64));
});

it("authenticates JSON requests and upload requests", async () => {
  install();
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(new Response("{}", { headers: { "Content-Type": "application/json" } })));
  vi.stubGlobal("fetch", fetch);
  await createApiClient().getHealth();
  await createApiClient().uploadAudio(new File(["data"], "practice.wav"));
  for (const call of fetch.mock.calls) expect(call[1].headers["X-Vostavo-Session"]).toBe("a".repeat(64));
});
