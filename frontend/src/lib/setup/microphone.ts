export type MicrophoneStatus =
  | "unknown" | "ready" | "needs_review" | "clipping" | "quiet" | "distorted" | "denied" | "missing" | "busy"
  | "timeout" | "silent" | "unsupported" | "error";

export const microphoneErrorStatus = (error: unknown): MicrophoneStatus => {
  const name = error instanceof Error || error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "denied";
  if (name === "NotFoundError" || name === "DevicesNotFoundError") return "missing";
  if (name === "NotReadableError" || name === "TrackStartError") return "busy";
  return "error";
};

/** Use the same input and processing for the setup sample and actual recording. */
export const microphoneConstraints = (deviceId: string, voiceProcessing: boolean): MediaStreamConstraints => ({
  audio: {
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    autoGainControl: false,
    echoCancellation: voiceProcessing,
    noiseSuppression: voiceProcessing,
  },
});

export const microphoneSignal = (samples: Float32Array): { rms: number; peak: number } => {
  let sum = 0; let peak = 0;
  for (const sample of samples) { sum += sample * sample; peak = Math.max(peak, Math.abs(sample)); }
  return { rms: samples.length ? Math.sqrt(sum / samples.length) : 0, peak };
};
