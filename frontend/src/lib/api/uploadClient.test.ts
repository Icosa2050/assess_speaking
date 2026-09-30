import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient } from "./client";

class UploadXHR {
  static last: UploadXHR;
  upload: { onprogress?: (event: { lengthComputable: boolean; loaded: number; total: number }) => void } = {};
  timeout = 0;
  status = 200;
  statusText = "OK";
  responseText = JSON.stringify({ audio_id: "saved-audio" });
  onload?: () => void;
  onerror?: () => void;
  ontimeout?: () => void;
  onabort?: () => void;
  open = vi.fn();
  setRequestHeader = vi.fn();
  send = vi.fn();
  abort = vi.fn(() => this.onabort?.());
  constructor() { UploadXHR.last = this; }
}
afterEach(() => vi.unstubAllGlobals());
describe("upload transport", () => {
  it("reports progress and resolves only after durable upload acknowledgement", async () => {
    vi.stubGlobal("XMLHttpRequest", UploadXHR);
    const progress = vi.fn();
    const done = vi.fn();
    const pending = createApiClient("http://localhost:12345").uploadAudio(new File(["audio"], "voice.wav"), { onProgress: progress }).then(done);
    const xhr = UploadXHR.last;
    expect(xhr.open).toHaveBeenCalledWith("POST", "http://localhost:12345/v1/uploads");
    expect(xhr.timeout).toBe(180000);
    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 10, total: 10 });
    expect(progress).toHaveBeenCalledWith(100);
    expect(done).not.toHaveBeenCalled();
    xhr.onload?.();
    await pending;
    expect(done).toHaveBeenCalledWith({ audio_id: "saved-audio" });
  });
  it("cancels in-flight transfer and propagates storage errors", async () => {
    vi.stubGlobal("XMLHttpRequest", UploadXHR);
    const controller = new AbortController();
    const pending = createApiClient().uploadAudio(new File(["audio"], "voice.wav"), { signal: controller.signal, onProgress: vi.fn() });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(UploadXHR.last.abort).toHaveBeenCalledOnce();
    const failed = createApiClient().uploadAudio(new File(["audio"], "voice.wav"), { onProgress: vi.fn() });
    UploadXHR.last.status = 507;
    UploadXHR.last.responseText = JSON.stringify({ detail: { code: "storage_error", detail: "Free disk space and retry" } });
    UploadXHR.last.onload?.();
    await expect(failed).rejects.toMatchObject({ code: "storage_error", responseStatus: 507 });
  });
});
