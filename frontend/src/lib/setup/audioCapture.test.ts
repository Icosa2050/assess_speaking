import { afterEach, expect, it, vi } from "vitest";
import { capturedAudioSeconds } from "./audioCapture";
afterEach(() => vi.unstubAllGlobals());
it("returns the decoded duration rather than a wall-clock estimate without starting playback", async () => {
  const decode = vi.fn().mockResolvedValue({ duration: 29.4 });
  vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = decode; });
  vi.stubGlobal("AudioContext", vi.fn(() => { throw new Error("Playback context must not be used"); }));
  const file = { size: 128, arrayBuffer: async () => new ArrayBuffer(128) } as Blob;
  expect(await capturedAudioSeconds(file)).toBe(29.4);
  expect(AudioContext).not.toHaveBeenCalled();
});
it("leaves unsupported or oversized captures to the bounded backend decoder", async () => {
  vi.stubGlobal("OfflineAudioContext", class { decodeAudioData = vi.fn().mockRejectedValue(new Error("Unsupported codec")); });
  expect(await capturedAudioSeconds({ size: 128, arrayBuffer: async () => new ArrayBuffer(128) } as Blob)).toBeUndefined();
  const read = vi.fn();
  expect(await capturedAudioSeconds({ size: 32 * 1024 * 1024, arrayBuffer: read } as unknown as Blob)).toBeUndefined();
  expect(read).not.toHaveBeenCalled();
});
