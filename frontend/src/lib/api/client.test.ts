import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createApiClient } from "./client";

const jsonResponse = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

describe("api client", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn(async () => jsonResponse({ payload: {} }));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    Reflect.deleteProperty(window, "__VOSTAVO_DESKTOP__");
    vi.unstubAllGlobals();
  });

  it("uses the desktop URL even when the bridge arrives after client creation", async () => {
    const client = createApiClient();
    Object.assign(window, {
      __VOSTAVO_DESKTOP__: { apiBaseUrl: "http://127.0.0.1:54321/" },
    });

    await client.getHealth();
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://127.0.0.1:54321/v1/health",
      expect.any(Object),
    );

    Object.assign(window, {
      __VOSTAVO_DESKTOP__: { apiBaseUrl: "http://127.0.0.1:54322" },
    });
    await client.getHistory();
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://127.0.0.1:54322/v1/history",
      expect.any(Object),
    );
  });

  it("preserves an explicit client URL when a desktop bridge exists", async () => {
    Object.assign(window, {
      __VOSTAVO_DESKTOP__: { apiBaseUrl: "http://127.0.0.1:54321" },
    });
    await createApiClient("http://localhost:8771").getHealth();
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://localhost:8771/v1/health",
      expect.any(Object),
    );
  });

  it("encodes dynamic history path segments", async () => {
    const client = createApiClient("http://localhost:8771");

    await client.getHistoryDetail("session/one two");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8771/v1/history/session%2Fone%20two",
      expect.any(Object),
    );
  });

  it("encodes dynamic assessment path segments", async () => {
    const client = createApiClient("http://localhost:8771");

    await client.getAssessmentStatus("asmt/one two");
    await client.cancelAssessment("asmt/one two");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8771/v1/assessments/asmt%2Fone%20two",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8771/v1/assessments/asmt%2Fone%20two/cancel",
      expect.any(Object),
    );
  });

  it("encodes dynamic local-support and runtime path segments", async () => {
    const client = createApiClient("http://localhost:8771");

    await client.downloadSupportBundle("bundle/one two");
    await client.postRuntimeSettingsSetDefault("conn/one two");
    await client.deleteRuntimeSettingsConnection("conn/one two");
    await client.getWhisperModelStatus("large/v3");
    await client.postWhisperModelDownload("large/v3");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8771/v1/support-bundles/bundle%2Fone%20two",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8771/v1/runtime/settings/connections/conn%2Fone%20two/default",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8771/v1/runtime/settings/connections/conn%2Fone%20two",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8771/v1/runtime/whisper-models/large%2Fv3",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://localhost:8771/v1/runtime/whisper-models/large%2Fv3/download",
      expect.any(Object),
    );
  });

  it("clears request timeout when an external signal is already aborted", async () => {
    const client = createApiClient("http://localhost:8771");
    const controller = new AbortController();
    controller.abort();
    const clearTimeoutSpy = vi.spyOn(window, "clearTimeout");

    await client.getHealth({ signal: controller.signal, timeoutMs: 1234 });

    expect(clearTimeoutSpy).toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8771/v1/health",
      expect.objectContaining({
        signal: expect.objectContaining({ aborted: true }),
      }),
    );
  });
});
