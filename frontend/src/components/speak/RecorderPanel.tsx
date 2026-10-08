import { connectInputMeter, microphoneContext, rememberInputSettings, RecordingSignal } from "@/lib/setup/audioMeter";
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

import { microphoneConstraints, microphoneErrorStatus, type MicrophoneStatus } from "@/lib/setup/microphone";
import { capturedAudioSeconds, preferredRecordingMimeType } from "@/lib/setup/audioCapture";
import { MIN_CAPTURE_SECONDS, MIN_REVIEW_SECONDS } from "@/lib/recordingPolicy";
import { Icon } from "@/components/ui/Icon";
import type { RecordingInputMethod } from "@/lib/state/sessionDraft";

import styles from "./RecorderPanel.module.css";

type Translate = (key: string, vars?: Record<string, string | number>) => string;
type RecorderPhase = "idle" | "requesting" | "recording" | "finalizing" | "error";
type VisualizerState = "idle" | "requesting" | "recording" | "ready" | "upload" | "error";

const MAX_RECORDING_SECONDS = 5 * 60;
const MEDIA_PERMISSION_TIMEOUT_MS = 8_000;

const recordingMimeExtension = (mimeType: string): string => {
  if (mimeType.includes("mp4")) {
    return "m4a";
  }
  if (mimeType.includes("ogg")) {
    return "ogg";
  }
  return "webm";
};


const isLocalRecordingHost = (): boolean => {
  if (typeof window === "undefined") {
    return true;
  }

  return ["localhost", "127.0.0.1", "[::1]", "::1"].includes(window.location.hostname);
};

const recordingErrorMessage = (translate: Translate, error: unknown): string => {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return translate("speak.recording_permission_denied");
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return translate("speak.recording_device_missing");
    }
    if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      return translate("speak.recording_device_busy");
    }
  }
  if (error instanceof Error && error.name === "TimeoutError") {
    return translate("speak.recording_permission_timeout");
  }

  return translate("speak.recording_failed");
};

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.35rem",
  border: "1px solid rgba(15, 118, 110, 0.22)",
  borderRadius: "8px",
  backgroundColor: "rgba(250, 253, 252, 0.98)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const modeButtonStyle = (active: boolean) =>
  ({
    minHeight: "40px",
    padding: "0.625rem 0.875rem",
    borderRadius: "8px",
    border: active
      ? "1px solid rgba(15, 118, 110, 0.28)"
      : "1px solid rgba(18, 61, 55, 0.12)",
    backgroundColor: active ? "#d7ebe5" : "rgba(255, 255, 255, 0.95)",
    color: "#10201c",
    fontWeight: 600,
    font: "inherit",
    cursor: "pointer",
  }) as const;

const inputStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  font: "inherit",
} as const;

const hiddenFileInputStyle = {
  position: "absolute",
  width: "1px",
  height: "1px",
  margin: "-1px",
  padding: 0,
  border: 0,
  overflow: "hidden",
  opacity: 0,
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  whiteSpace: "nowrap",
} as const;

const actionButtonStyle = {
  minHeight: "40px",
  padding: "0.625rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  backgroundColor: "rgba(255, 255, 255, 0.95)",
  color: "#10201c",
  fontWeight: 600,
  font: "inherit",
  cursor: "pointer",
} as const;

const primaryActionButtonStyle = {
  ...actionButtonStyle,
  backgroundColor: "#0f766e",
  border: "1px solid #0f766e",
  color: "#fff",
} as const;

const dangerActionButtonStyle = {
  ...actionButtonStyle,
  backgroundColor: "#b42318",
  border: "1px solid #b42318",
  color: "#fff",
} as const;

const uploadBoxStyle = {
  display: "grid",
  gap: "0.625rem",
  padding: "0.875rem",
  border: "1px dashed rgba(15, 118, 110, 0.34)",
  borderRadius: "8px",
  backgroundColor: "rgba(248, 251, 250, 0.96)",
  color: "#10201c",
  fontWeight: 600,
  cursor: "pointer",
} as const;

const idleLevels = Array<number>(9).fill(0);

export const RecorderPanel = ({
  canRemove,
  maxSeconds = MAX_RECORDING_SECONDS,
  allowUpload = true,
  onRecordingActiveChange,
  onMicrophoneStatusChange,
  microphoneSetupPassed = false,
  microphoneDeviceId = "",
  microphoneVoiceProcessing = true,
  onSetupMicrophone,
  downloadName,
  inputMode,
  onFileSelected,
  onInputModeChange,
  onRemove,
  previewUrl,
  showReadyCheckpoint,
  statusMessage,
  statusTone,
  translate,
}: {
  canRemove: boolean;
  onMicrophoneStatusChange?: (status: MicrophoneStatus) => void;
  microphoneSetupPassed?: boolean;
  microphoneDeviceId?: string;
  microphoneVoiceProcessing?: boolean;
  onSetupMicrophone?: () => void;
  maxSeconds?: number;
  allowUpload?: boolean;
  onRecordingActiveChange?: (active: boolean) => void;
  downloadName?: string;
  inputMode: RecordingInputMethod;
  onFileSelected: (file: File | null, durationSec?: number) => void;
  onInputModeChange: (mode: RecordingInputMethod) => void;
  onRemove: () => void;
  previewUrl: string;
  showReadyCheckpoint: boolean;
  statusMessage: string;
  statusTone: "error" | "info" | "success" | "warning";
  translate: Translate;
}) => {
  const [recorderPhase, setRecorderPhase] = useState<RecorderPhase>("idle");
  const [recorderMessage, setRecorderMessage] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [levels, setLevels] = useState(idleLevels);
  const [signalWarning, setSignalWarning] = useState("");
  const meterContext = useRef<AudioContext | null>(null);
  const meterTimer = useRef<number | null>(null);
  const signal = useRef<RecordingSignal | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const discardStopRef = useRef(false);
  const recorderFailedRef = useRef(false);
  const elapsedSecondsRef = useRef(0);
  const mountedRef = useRef(true);
  const recordingMimeTypeRef = useRef("audio/webm");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const requestIdRef = useRef(0);
  const startedAtRef = useRef(0);
  const stopFallbackRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    onRecordingActiveChange?.(recorderPhase === "requesting" || recorderPhase === "recording" || recorderPhase === "finalizing");
  }, [onRecordingActiveChange, recorderPhase]);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const clearStopFallback = useCallback(() => {
    if (stopFallbackRef.current !== null) {
      window.clearTimeout(stopFallbackRef.current);
      stopFallbackRef.current = null;
    }
  }, []);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const cleanupRecorder = useCallback(() => {
    if (meterTimer.current !== null) window.clearInterval(meterTimer.current);
    meterTimer.current = null;
    void meterContext.current?.close().catch(() => undefined);
    meterContext.current = null;
    clearTimer();
    clearStopFallback();
    stopStream();
    recorderRef.current = null;
    startedAtRef.current = 0;
  }, [clearStopFallback, clearTimer, stopStream]);

  const finalizeRecording = useCallback(
    async (recorder: MediaRecorder) => {
      if (recorderRef.current !== recorder && chunksRef.current.length === 0) {
        return;
      }

      const finalizationId = requestIdRef.current;
      const chunks = [...chunksRef.current];
      const shouldDiscard = discardStopRef.current;
      const recordedSeconds = startedAtRef.current
        ? Math.floor((Date.now() - startedAtRef.current) / 1000)
        : elapsedSecondsRef.current;
      const selectedMimeType = recorder.mimeType || recordingMimeTypeRef.current || "audio/webm";
      chunksRef.current = [];
      cleanupRecorder();

      if (!mountedRef.current) {
        return;
      }

      if (recorderFailedRef.current) {
        setRecorderPhase("error");
        return;
      }
      setRecorderPhase("idle");
      elapsedSecondsRef.current = recordedSeconds;
      setElapsedSeconds(recordedSeconds);

      if (shouldDiscard) {
        return;
      }

      if (chunks.length === 0) {
        setRecorderPhase("error");
        setRecorderMessage(translate("speak.recording_empty"));
        return;
      }

      const extension = recordingMimeExtension(selectedMimeType);
      const timestamp = new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 14);
      const file = new File(chunks, `recording-${timestamp}.${extension}`, {
        type: selectedMimeType,
      });
      // Saving a file cannot grant setup calibration. Preserve the take even if
      // the input needs attention, and require a new setup sample next time.
      if (signal.current) onMicrophoneStatusChange?.(signal.current.result);
      setRecorderPhase("finalizing");
      const decodedSeconds = await capturedAudioSeconds(file);
      if (!mountedRef.current || requestIdRef.current !== finalizationId) return;
      setRecorderPhase("idle");
      const duration = decodedSeconds ?? recordedSeconds;
      setElapsedSeconds(Math.floor(duration));
      onFileSelected(file, duration);
      setRecorderMessage(
        duration < (decodedSeconds === undefined ? MIN_CAPTURE_SECONDS : MIN_REVIEW_SECONDS)
          ? translate("speak.recording_too_short")
          : recordedSeconds >= maxSeconds
          ? translate("speak.recording_auto_stopped")
          : translate("speak.recording_saved"),
      );
    },
    [cleanupRecorder, maxSeconds, onFileSelected, onMicrophoneStatusChange, translate],
  );

  const stopRecording = useCallback(
    ({ discard = false } = {}) => {
      requestIdRef.current += 1;
      discardStopRef.current = discard;
      const recorder = recorderRef.current;

      if (recorder && recorder.state !== "inactive") {
        try {
          recorder.requestData();
        } catch {
          // Some browsers throw if data is not ready yet; stopping still finalizes below.
        }
        recorder.stop();
        stopFallbackRef.current = window.setTimeout(() => {
          if (recorderRef.current === recorder) {
            finalizeRecording(recorder);
          }
        }, 1_500);
        return;
      }

      cleanupRecorder();
      if (mountedRef.current) {
        setRecorderPhase("idle");
      }
    },
    [cleanupRecorder, finalizeRecording],
  );

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      requestIdRef.current += 1;
      discardStopRef.current = true;
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        try {
          recorder.stop();
        } catch {
          // The component is unmounting, so cleanup below is enough.
        }
      }
      cleanupRecorder();
    };
  }, [cleanupRecorder]);

  const handleModeChange = (mode: RecordingInputMethod) => {
    if (mode !== inputMode) {
      stopRecording({ discard: true });
      setRecorderMessage("");
      elapsedSecondsRef.current = 0;
      setElapsedSeconds(0);
    }
    onInputModeChange(mode);
  };

  const handleStartRecording = async () => {
    if (!microphoneSetupPassed) {
      setRecorderMessage(translate("speak.microphone_setup_required"));
      return;
    }
    if (typeof window !== "undefined" && window.isSecureContext === false && !isLocalRecordingHost()) {
      onMicrophoneStatusChange?.("unsupported");
      setRecorderPhase("error");
      setRecorderMessage(translate("speak.recording_secure_context_required"));
      return;
    }

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      onMicrophoneStatusChange?.("unsupported");
      setRecorderPhase("error");
      setRecorderMessage(translate("speak.recording_unsupported"));
      return;
    }

    setRecorderPhase("requesting");
    setRecorderMessage("");
    elapsedSecondsRef.current = 0;
    setElapsedSeconds(0);
    chunksRef.current = [];
    setSignalWarning(""); setLevels(idleLevels); signal.current = null;
    onFileSelected(null);
    // Resume synchronously from the Record gesture, before permission awaits.
    if (typeof AudioContext !== "undefined") {
      try { meterContext.current = microphoneContext(); void meterContext.current.resume().catch(() => undefined); }
      catch { setSignalWarning(translate("speak.meter_unavailable")); }
    }

    let requestTimeoutId: number | null = null;
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const streamPromise = navigator.mediaDevices.getUserMedia(microphoneConstraints(microphoneDeviceId, microphoneVoiceProcessing));
    streamPromise.then(
      (stream) => {
        if (requestIdRef.current !== requestId) {
          stream.getTracks().forEach((track) => track.stop());
        }
      },
      () => undefined,
    );

    try {
      const stream = await Promise.race([
        streamPromise,
        new Promise<never>((_, reject) => {
          requestTimeoutId = window.setTimeout(() => {
            const error = new Error("Microphone permission did not resolve.");
            error.name = "TimeoutError";
            reject(error);
          }, MEDIA_PERMISSION_TIMEOUT_MS);
        }),
      ]);
      if (requestTimeoutId !== null) {
        window.clearTimeout(requestTimeoutId);
      }
      if (requestIdRef.current !== requestId) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      const mimeType = preferredRecordingMimeType();
      streamRef.current = stream;
      rememberInputSettings(stream);
      if (meterContext.current) {
        const meter = connectInputMeter(meterContext.current, stream);
        const monitor = new RecordingSignal(); signal.current = monitor;
        meterTimer.current = window.setInterval(() => {
          const { rms, peak } = meter.read();
          setLevels(previous => [...previous.slice(1), Math.min(1, rms * 5)]);
          const warning = monitor.update(rms, peak);
          setSignalWarning(warning ? translate(`speak.recording_${warning}_warning`) : "");
        }, 100);
      }
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      recordingMimeTypeRef.current = recorder.mimeType || mimeType || "audio/webm";
      discardStopRef.current = false;
      recorderFailedRef.current = false;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onerror = (event) => {
        recorderFailedRef.current = true;
        cleanupRecorder();
        if (!mountedRef.current) {
          return;
        }
        const error = "error" in event ? event.error : undefined;
        setRecorderPhase("error");
        onMicrophoneStatusChange?.(microphoneErrorStatus(error));
        setRecorderMessage(recordingErrorMessage(translate, error));
      };
      recorder.onstop = () => { void finalizeRecording(recorder); };

      recorder.start(1_000);
      startedAtRef.current = Date.now();
      setRecorderPhase("recording");
      timerRef.current = window.setInterval(() => {
        const nextElapsed = Math.floor((Date.now() - startedAtRef.current) / 1000);
        elapsedSecondsRef.current = nextElapsed;
        setElapsedSeconds(nextElapsed);
        if (nextElapsed >= maxSeconds && recorderRef.current?.state === "recording") {
          recorderRef.current.stop();
        }
      }, 1000);
    } catch (error) {
      if (requestTimeoutId !== null) {
        window.clearTimeout(requestTimeoutId);
      }
      if (requestIdRef.current !== requestId) {
        cleanupRecorder();
        return;
      }
      requestIdRef.current += 1;
      cleanupRecorder();
      if (!mountedRef.current) {
        return;
      }
      setRecorderPhase("error");
      onMicrophoneStatusChange?.(error instanceof Error && error.name === "TimeoutError" ? "timeout" : microphoneErrorStatus(error));
      setRecorderMessage(recordingErrorMessage(translate, error));
    }
  };

  const visibleStatusMessage =
    inputMode === "upload"
      ? statusMessage
      : recorderPhase === "finalizing"
      ? translate("speak.recording_checking")
      : recorderPhase === "requesting"
      ? translate("speak.recording_requesting")
      : recorderPhase === "recording"
        ? translate("speak.recording_live", { seconds: elapsedSeconds })
        : recorderMessage || statusMessage;
  const visibleStatusTone =
    inputMode === "upload"
      ? statusTone
      : recorderPhase === "error"
        ? "error"
        : recorderPhase === "recording"
          ? "warning"
          : statusTone;
  const visualizerState: VisualizerState =
    inputMode === "upload"
      ? "upload"
      : recorderPhase === "recording"
        ? "recording"
        : recorderPhase === "requesting"
          ? "requesting"
          : recorderPhase === "error" || visibleStatusTone === "error"
            ? "error"
            : visibleStatusTone === "success"
              ? "ready"
              : "idle";

  return (
    <section
      style={cardStyle}
      data-testid="speak.recording_panel"
      data-semantic-id="speak.recording_panel"
    >
      <h2
        style={{
          margin: 0,
          fontSize: "1.35rem",
          color: "#10201c",
        }}
      >
        {translate("speak.recording_title")}
      </h2>
      <p
        style={{
          margin: 0,
          lineHeight: 1.6,
          color: "#33514b",
        }}
      >
        {translate("speak.recording_body")}
      </p>
      <div
        role="img"
        aria-label={visibleStatusMessage}
        className={styles.visualizer}
        data-recording-state={visualizerState}
        data-testid="speak.recording_visualizer"
        data-semantic-id="speak.recording_visualizer"
      >
        <div className={styles.visualizerBars} aria-hidden="true">
          {levels.map((scale, index) => (
            <span
              key={index}
              className={styles.visualizerBar}
              style={
                {
                  "--bar-delay": `${index * 90}ms`,
                  "--bar-height": `${0.15 + scale * 4}rem`,
                } as CSSProperties
              }
            />
          ))}
        </div>
      </div>
      {signalWarning && <p role="status" data-testid="speak.signal_warning">{signalWarning}</p>}
      <div
        aria-label={translate("speak.input_method")}
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.625rem",
        }}
      >
        <button
          type="button"
          onClick={() => handleModeChange("record")}
          className={styles.controlButton}
          style={modeButtonStyle(inputMode === "record")}
          data-testid="speak.input_mode_record"
          data-semantic-id="speak.input_mode_record"
        >
          <Icon className={styles.controlIcon} name="microphone" size={18} />
          {translate("speak.input_method_record")}
        </button>
        {allowUpload && <button
          type="button"
          onClick={() => handleModeChange("upload")}
          className={styles.controlButton}
          style={modeButtonStyle(inputMode === "upload")}
          data-testid="speak.input_mode_upload"
          data-semantic-id="speak.input_mode_upload"
        >
          <Icon className={styles.controlIcon} name="upload" size={18} />
          {translate("speak.input_method_upload")}
        </button>}
      </div>
      {inputMode === "record" ? (
        <div
          style={{
            display: "grid",
            gap: "0.625rem",
          }}
        >
          {!microphoneSetupPassed ? <div data-testid="speak.microphone_setup_required">
            <p>{translate("speak.microphone_setup_required")}</p>
            {onSetupMicrophone ? <button type="button" data-testid="speak.microphone_setup" onClick={onSetupMicrophone}>
              {translate("runtime_setup.setup_guide_check_microphone")}
            </button> : null}
          </div> : null}
          <button
            type="button"
            onClick={handleStartRecording}
            disabled={!microphoneSetupPassed || recorderPhase === "requesting" || recorderPhase === "recording" || recorderPhase === "finalizing"}
            aria-pressed={recorderPhase === "recording"}
            className={styles.controlButton}
            style={primaryActionButtonStyle}
            data-testid="speak.record_start"
            data-semantic-id="speak.record_start"
          >
            <Icon className={styles.controlIcon} name="microphone" size={18} />
            {translate("speak.recording_start")}
          </button>
          {recorderPhase === "recording" ? (
            <button
              type="button"
              onClick={() => stopRecording()}
              className={styles.controlButton}
              style={dangerActionButtonStyle}
              data-testid="speak.record_stop"
              data-semantic-id="speak.record_stop"
            >
              <Icon className={styles.controlIcon} name="stop" size={18} />
              {translate("speak.recording_stop")}
            </button>
          ) : null}
        </div>
      ) : (
        <label
          htmlFor="speak-upload-input"
          className={styles.uploadControl}
          style={uploadBoxStyle}
        >
          <span>{translate("speak.upload")}</span>
          <span className={styles.controlButton} style={actionButtonStyle}>
            <Icon className={styles.controlIcon} name="upload" size={18} />
            {translate("speak.upload_choose")}
          </span>
          <input
            id="speak-upload-input"
            type="file"
            accept="audio/*"
            aria-label={translate("speak.upload")}
            onChange={(event) => onFileSelected(event.currentTarget.files?.[0] ?? null)}
            style={hiddenFileInputStyle}
            data-testid="speak.upload_input"
            data-semantic-id="speak.upload_input"
          />
        </label>
      )}
      {inputMode === "upload" ? (
        <p
          style={{
            margin: 0,
            color: "#33514b",
            lineHeight: 1.55,
          }}
        >
          {translate("speak.upload_help")}
        </p>
      ) : null}
      <p
        aria-live="polite"
        style={{
          margin: 0,
          color:
            visibleStatusTone === "success"
              ? "#166534"
              : visibleStatusTone === "warning"
                ? "#9a6700"
                : visibleStatusTone === "error"
                  ? "#b42318"
                  : "#33514b",
          lineHeight: 1.55,
        }}
        data-testid="speak.recording_status"
        data-semantic-id="speak.recording_status"
      >
        {visibleStatusMessage}
      </p>
      {previewUrl ? (
        <>
        <audio
          controls
          src={previewUrl}
          style={{
            width: "100%",
          }}
        />
        <a href={previewUrl} download={downloadName || "practice-recording.webm"} data-testid="speak.download_recording">
          {translate("speak.download_recording")}
        </a>
        </>
      ) : null}
      {showReadyCheckpoint ? (
        <div
          style={{
            display: "grid",
            gap: "0.35rem",
            padding: "0.75rem",
            border: "1px solid rgba(15, 118, 110, 0.16)",
            borderRadius: "8px",
            backgroundColor: "rgba(248, 251, 250, 0.96)",
          }}
          data-testid="speak.recording_ready_checkpoint"
          data-semantic-id="speak.recording_ready_checkpoint"
        >
          <strong
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.4rem",
              color: "#10201c",
            }}
          >
            <Icon name="check" size={17} />
            {translate("speak.recording_ready_checkpoint_title")}
          </strong>
          <p
            style={{
              margin: 0,
              color: "#33514b",
              lineHeight: 1.5,
            }}
          >
            {translate("speak.recording_ready_checkpoint_body")}
          </p>
        </div>
      ) : null}
      {canRemove ? (
        <button
          type="button"
          onClick={onRemove}
          style={actionButtonStyle}
          {...{
            "data-testid": "speak.remove_recording",
            "data-semantic-id": "speak.remove_recording",
          }}
        >
          {translate("speak.remove_recording")}
        </button>
      ) : null}
    </section>
  );
};
