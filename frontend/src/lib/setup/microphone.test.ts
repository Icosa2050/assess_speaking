import { expect, it } from "vitest";
import { microphoneConstraints, microphoneSignal } from "./microphone";

it("measures both average speech level and short full-scale peaks", () => {
  const samples = new Float32Array(1000); samples[10] = -1;
  const result = microphoneSignal(samples);
  expect(result.peak).toBe(1);
  expect(result.rms).toBeCloseTo(Math.sqrt(1 / 1000));
  expect(microphoneSignal(new Float32Array())).toEqual({ rms: 0, peak: 0 });
});
it("pins the tested microphone without automatic boost and respects processing choice", () => {
  expect(microphoneConstraints("usb-mic", false)).toEqual({ audio: {
    deviceId: { exact: "usb-mic" }, autoGainControl: false, echoCancellation: false, noiseSuppression: false,
  } });
  expect(microphoneConstraints("", true)).toEqual({ audio: {
    autoGainControl: false, echoCancellation: true, noiseSuppression: true,
  } });
});
