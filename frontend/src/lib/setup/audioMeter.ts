import { microphoneSignal } from "./microphone";

/** A zero-gain output pulls WebKit's graph without playing input to speakers. */
export function connectInputMeter(context: AudioContext, stream: MediaStream) {
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  const source = context.createMediaStreamSource(stream);
  const sink = context.createGain();
  sink.gain.value = 0;
  source.connect(analyser); analyser.connect(sink); sink.connect(context.destination);
  const samples = new Float32Array(analyser.fftSize);
  return { read: () => { analyser.getFloatTimeDomainData(samples); return microphoneSignal(samples); },
    disconnect: () => { source.disconnect(); analyser.disconnect(); sink.disconnect(); } };
}

export function microphoneContext(): AudioContext {
  const options: AudioContextOptions & { sinkId?: { type: "none" } } =
    "sinkId" in AudioContext.prototype ? { sinkId: { type: "none" } } : {};
  return new AudioContext(options);
}

/** Do not retain hardware identifiers or labels in support diagnostics. */
export function sanitizedInputSettings(stream: MediaStream) {
  const settings = stream.getAudioTracks?.()[0]?.getSettings?.() ?? {};
  return Object.fromEntries(["sampleRate", "channelCount", "autoGainControl", "echoCancellation", "noiseSuppression"]
    .flatMap(key => { const value = settings[key as keyof MediaTrackSettings]; return typeof value === "number" || typeof value === "boolean" ? [[key, value]] : []; }));
}
let inputSettings: ReturnType<typeof sanitizedInputSettings> = {};
export const rememberInputSettings = (stream: MediaStream) => { inputSettings = sanitizedInputSettings(stream); };
export const readInputSettings = () => ({ ...inputSettings });

export class RecordingSignal {
  private frames = 0;
  private heard = false;
  private lastSignal = 0;
  private clipping: boolean[] = [];
  private damaged = false;
  update(rms: number, peak: number): "silent" | "clipping" | "" {
    this.frames++;
    if (rms >= 0.001) { this.heard = true; this.lastSignal = this.frames; }
    this.clipping.push(peak >= 0.98);
    if (this.clipping.length > 20) this.clipping.shift();
    if (this.clipping.filter(Boolean).length >= 10) this.damaged = true;
    if (this.damaged) return "clipping";
    return this.frames - this.lastSignal >= 100 ? "silent" : "";
  }
  get result(): "silent" | "clipping" | "ready" { return this.damaged ? "clipping" : this.heard ? "ready" : "silent"; }
}
