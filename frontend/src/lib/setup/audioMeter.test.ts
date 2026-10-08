import { expect, it } from "vitest";
import { RecordingSignal, sanitizedInputSettings } from "./audioMeter";
it("allows normal pauses but warns about ten seconds without input", () => {
  const signal = new RecordingSignal();
  signal.update(.04, .1);
  for (let i = 0; i < 99; i++) expect(signal.update(0, 0)).toBe("");
  expect(signal.update(0, 0)).toBe("silent");
  expect(signal.update(.04, .1)).toBe("");
  expect(signal.result).toBe("ready");
});
it("requires sustained clipping, retains damage after its level drops, and rejects silent files", () => {
  const signal = new RecordingSignal();
  expect(signal.update(.04, .99)).toBe("");
  for (let i = 0; i < 20; i++) signal.update(.04, .1);
  expect(signal.result).toBe("ready");
  for (let i = 0; i < 10; i++) signal.update(.5, .99);
  expect(signal.update(.04, .1)).toBe("clipping");
  expect(signal.result).toBe("clipping");
  expect(new RecordingSignal().result).toBe("silent");
});
it("keeps only non-identifying input settings for support diagnostics", () => {
  const stream = { getAudioTracks: () => [{ getSettings: () => ({ deviceId: "private", groupId: "private", sampleRate: 48000, channelCount: 1, autoGainControl: false }) }] } as unknown as MediaStream;
  expect(sanitizedInputSettings(stream)).toEqual({ sampleRate: 48000, channelCount: 1, autoGainControl: false });
});
