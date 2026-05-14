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
    vi.unstubAllGlobals();
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
});
