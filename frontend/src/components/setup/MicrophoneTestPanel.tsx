import { connectInputMeter, microphoneContext, rememberInputSettings } from "@/lib/setup/audioMeter";
import { useCallback, useEffect, useRef, useState } from "react";

import { useAppStore } from "@/lib/state/appStore";
import { microphoneConstraints, microphoneErrorStatus, type MicrophoneStatus } from "@/lib/setup/microphone";
import { preferredRecordingMimeType } from "@/lib/setup/audioCapture";
import styles from "./SetupReadinessPanel.module.css";

export const useMicrophoneTest = () => {
  const status = useAppStore(state => state.microphoneStatus);
  const passed = useAppStore(state => state.microphoneSetupPassed);
  const deviceId = useAppStore(state => state.microphoneDeviceId);
  const voiceProcessing = useAppStore(state => state.microphoneVoiceProcessing);
  const setDeviceId = useAppStore(state => state.setMicrophoneDeviceId);
  const setVoiceProcessing = useAppStore(state => state.setMicrophoneVoiceProcessing);
  const setStatus = useAppStore(state => state.setMicrophoneStatus);
  const setPassed = useAppStore(state => state.setMicrophoneSetupPassed);
  const [phase, setPhase] = useState<"idle" | "requesting" | "listening" | "finalizing">("idle");
  const [level, setLevel] = useState(0);
  const [peak, setPeak] = useState(0);
  const [preview, setPreview] = useState("");
  const [listened, setListened] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const generation = useRef(0);
  const mounted = useRef(true);
  const previewRef = useRef("");
  const resources = useRef<{
    stream?: MediaStream; context?: AudioContext; recorder?: MediaRecorder;
    deadline?: number; interval?: number;
  }>({});

  const clearPreview = useCallback(() => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = "";
    setPreview("");
    setListened(false);
  }, []);

  const release = useCallback(() => {
    generation.current += 1;
    const { stream, context, recorder, deadline, interval } = resources.current;
    resources.current = {};
    window.clearTimeout(deadline);
    window.clearInterval(interval);
    if (recorder) {
      recorder.onstop = null; recorder.onerror = null; recorder.ondataavailable = null;
      if (recorder.state !== "inactive") { try { recorder.stop(); } catch { /* Already stopped. */ } }
    }
    stream?.getTracks().forEach(track => track.stop());
    if (context && context.state !== "closed") void context.close().catch(() => undefined);
  }, []);

  const refreshDevices = useCallback(async () => {
    try {
      const inputs = await navigator.mediaDevices?.enumerateDevices?.();
      if (mounted.current && inputs) setDevices(inputs.filter(device => device.kind === "audioinput"));
    } catch { /* Permission-free enumeration can be unavailable; default input still works. */ }
  }, []);

  const stop = useCallback(() => {
    release(); clearPreview();
    setPhase("idle"); setLevel(0); setPeak(0);
    setStatus("unknown"); setPassed(false);
  }, [release, clearPreview, setStatus, setPassed]);

  useEffect(() => {
    mounted.current = true;
    void refreshDevices();
    const changed = () => {
      // Permission grants may expose input labels and trigger devicechange while
      // a test is starting. Its live track is checked by the ended listener.
      if (!resources.current.context) stop();
      void refreshDevices();
    };
    navigator.mediaDevices?.addEventListener?.("devicechange", changed);
    return () => {
      mounted.current = false;
      navigator.mediaDevices?.removeEventListener?.("devicechange", changed);
      release();
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
    };
  }, [release, refreshDevices, stop]);

  const start = useCallback(async () => {
    release(); clearPreview();
    const current = generation.current;
    setLevel(0); setPeak(0); setStatus("unknown"); setPassed(false);
    const finish = (result: MicrophoneStatus) => {
      if (current !== generation.current) return;
      release(); setPhase("idle"); setLevel(0); setStatus(result);
    };
    if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === "undefined" || typeof MediaRecorder === "undefined") {
      finish("unsupported"); return;
    }
    setPhase("requesting");
    resources.current.deadline = window.setTimeout(() => finish("timeout"), 15_000);
    try {
      const context = microphoneContext();
      resources.current.context = context;
      const resumed = context.resume();
      void resumed.catch(() => undefined);
      const stream = await navigator.mediaDevices.getUserMedia(microphoneConstraints(deviceId, voiceProcessing));
      if (current !== generation.current) { stream.getTracks().forEach(track => track.stop()); return; }
      resources.current.stream = stream;
      await resumed;
      if (current !== generation.current) return;
      const track = stream.getAudioTracks()[0];
      if (!track) { finish("missing"); return; }
      const actualDeviceId = track.getSettings?.().deviceId;
      if (actualDeviceId && actualDeviceId !== deviceId) setDeviceId(actualDeviceId);
      void refreshDevices();
      rememberInputSettings(stream);
      const meter = connectInputMeter(context, stream);
      let signalSamples = 0; let hotSamples = 0; let maxRms = 0; let maxPeak = 0;
      const startedAt = context.currentTime;
      const mimeType = preferredRecordingMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      resources.current.recorder = recorder;
      const chunks: Blob[] = [];
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => finish("error");
      recorder.onstop = () => {
        if (current !== generation.current) return;
        const result = context.currentTime - startedAt < 0.5 ? "error"
          : hotSamples >= 3 ? "clipping" : signalSamples >= 10 ? "needs_review" : maxRms > 0.001 ? "quiet" : "silent";
        if (!chunks.length) { finish("error"); return; }
        const url = URL.createObjectURL(new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" }));
        previewRef.current = url; setPreview(url);
        setPeak(maxPeak); finish(result);
      };
      track.addEventListener("ended", () => finish("missing"), { once: true });
      window.clearTimeout(resources.current.deadline);
      setPhase("listening"); recorder.start(1000);
      resources.current.interval = window.setInterval(() => {
        const { rms, peak: samplePeak } = meter.read();
        maxRms = Math.max(maxRms, rms); maxPeak = Math.max(maxPeak, samplePeak);
        setLevel(Math.min(100, Math.round(rms * 500))); setPeak(samplePeak);
        if (rms >= 0.01) signalSamples += 1;
        if (samplePeak >= 0.98) hotSamples += 1;
      }, 100);
      resources.current.deadline = window.setTimeout(() => {
        if (current !== generation.current) return;
        window.clearInterval(resources.current.interval);
        setPhase("finalizing");
        // Bound encoders that never dispatch their final data/stop events.
        resources.current.deadline = window.setTimeout(() => finish("error"), 2_000);
        try { recorder.stop(); } catch { finish("error"); }
      }, 5_000);
    } catch (error) { finish(microphoneErrorStatus(error)); }
  }, [release, clearPreview, setStatus, setPassed, deviceId, voiceProcessing, setDeviceId, refreshDevices]);

  const confirm = () => {
    if (status !== "needs_review" || !preview || !listened) return;
    setStatus("ready"); setPassed(true);
  };
  const reject = () => { setPassed(false); setStatus("distorted"); setListened(false); };
  const selectDevice = (id: string) => { stop(); setDeviceId(id); };
  const changeProcessing = (enabled: boolean) => { stop(); setVoiceProcessing(enabled); };
  return { status, passed, phase, level, peak, preview, listened, devices, deviceId, voiceProcessing,
    start, stop, confirm, reject, selectDevice, changeProcessing, markListened: () => setListened(true) };
};

export const MicrophoneTestPanel = ({ test, translate }: {
  test: ReturnType<typeof useMicrophoneTest>;
  translate: (key: string) => string;
}) => {
  const active = test.phase !== "idle";
  return (
    <section id="runtime-setup-microphone" className={styles.panel} aria-labelledby="microphone-test-title">
      <div className={styles.header}>
        <h2 id="microphone-test-title" className={styles.title}>{translate("runtime_setup.microphone_test_title")}</h2>
        <p className={styles.body}>{translate("runtime_setup.microphone_test_body")}</p>
        <p role="status" data-testid="microphone-test-status">
          {translate(`runtime_setup.microphone_test_${active ? test.phase : test.status}`)}
        </p>
      </div>
      <div className={styles.microphoneControls}>
        <label htmlFor="microphone-device">{translate("runtime_setup.microphone_device")}</label>
        <select id="microphone-device" data-testid="microphone-device" value={test.deviceId} disabled={active}
          onChange={event => test.selectDevice(event.target.value)}>
          <option value="">{translate("runtime_setup.microphone_default_device")}</option>
          {test.devices.filter(device => device.deviceId).map((device, index) => (
            <option key={device.deviceId} value={device.deviceId}>{device.label || `${translate("runtime_setup.microphone_device")} ${index + 1}`}</option>
          ))}
        </select>
        <label><input type="checkbox" data-testid="microphone-processing" checked={test.voiceProcessing} disabled={active}
          onChange={event => test.changeProcessing(event.target.checked)} /> {translate("runtime_setup.microphone_processing")}</label>
        <label htmlFor="microphone-input-level">{translate("runtime_setup.microphone_test_level")}</label>
        <meter id="microphone-input-level" min={0} max={100} value={test.level}
          style={{ width: "100%", height: "1.5rem", accentColor: test.peak >= 0.98 ? "#b42318" : "#0f766e" }} />
        <p data-testid="microphone-level-guidance">{translate(test.peak >= 0.98 ? "runtime_setup.microphone_test_clipping" : "runtime_setup.microphone_adjustment")}</p>
        <button className={styles.rowAction} type="button" data-testid="microphone-test-start" onClick={active ? test.stop : test.start}>
          {translate(active ? "runtime_setup.microphone_test_stop" : "runtime_setup.setup_guide_check_microphone")}
        </button>
        {test.preview ? <div>
          <p>{translate("runtime_setup.microphone_listen")}</p>
          <audio data-testid="microphone-test-playback" controls style={{ width: "100%", maxWidth: "100%" }} src={test.preview} onEnded={test.markListened} />
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "0.75rem" }}>
            <button className={styles.rowAction} type="button" data-testid="microphone-test-confirm" disabled={active || test.status !== "needs_review" || !test.listened} onClick={test.confirm}>
              {translate("runtime_setup.microphone_confirm")}
            </button>
            <button className={styles.rowAction} type="button" data-testid="microphone-test-reject" disabled={active} onClick={test.reject}>
              {translate("runtime_setup.microphone_reject")}
            </button>
          </div>
        </div> : null}
      </div>
    </section>
  );
};
