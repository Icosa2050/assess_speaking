/** Keep setup playback and practice recording on the same native encoder path. */
const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
export const preferredRecordingMimeType = (): string =>
  typeof MediaRecorder !== "undefined" && typeof MediaRecorder.isTypeSupported === "function"
    ? MIME_TYPES.find(type => MediaRecorder.isTypeSupported(type)) ?? "" : "";

/** Inspect locally captured audio without playing it or opening another mic. */
export async function capturedAudioSeconds(file: Blob): Promise<number | undefined> {
  if (typeof OfflineAudioContext === "undefined" || file.size > 16 * 1024 * 1024) return undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const context = new OfflineAudioContext(1, 1, 16000);
    const decoded = await Promise.race([
      file.arrayBuffer().then(bytes => context.decodeAudioData(bytes)),
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("Audio duration check timed out")), 10000); }),
    ]);
    return Number.isFinite(decoded.duration) && decoded.duration > 0 ? decoded.duration : undefined;
  } catch { return undefined; } // The backend's bounded decoder remains authoritative.
  finally { if (timer) clearTimeout(timer); }
}
