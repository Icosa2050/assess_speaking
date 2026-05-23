import { useCallback, useEffect, useRef, useState } from "react";

import type { RecordingInputMethod } from "@/lib/state/sessionDraft";

type Translate = (key: string, vars?: Record<string, string | number>) => string;
type RecorderPhase = "idle" | "requesting" | "recording" | "error";

const MAX_RECORDING_SECONDS = 5 * 60;
const MEDIA_PERMISSION_TIMEOUT_MS = 8_000;
const RECORDING_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg",
] as const;

const recordingMimeExtension = (mimeType: string): string => {
  if (mimeType.includes("mp4")) {
    return "m4a";
  }
  if (mimeType.includes("ogg")) {
    return "ogg";
  }
  return "webm";
};

const preferredRecordingMimeType = (): string => {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return "";
  }

  return RECORDING_MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? "";
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
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
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

export const RecorderPanel = ({
  canRemove,
  inputMode,
  onFileSelected,
  onInputModeChange,
  onRemove,
  previewUrl,
  statusMessage,
  statusTone,
  translate,
}: {
  canRemove: boolean;
  inputMode: RecordingInputMethod;
  onFileSelected: (file: File | null) => void;
  onInputModeChange: (mode: RecordingInputMethod) => void;
  onRemove: () => void;
  previewUrl: string;
  statusMessage: string;
  statusTone: "error" | "info" | "success" | "warning";
  translate: Translate;
}) => {
  const [recorderPhase, setRecorderPhase] = useState<RecorderPhase>("idle");
  const [recorderMessage, setRecorderMessage] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const chunksRef = useRef<Blob[]>([]);
  const discardStopRef = useRef(false);
  const elapsedSecondsRef = useRef(0);
  const mountedRef = useRef(true);
  const recordingMimeTypeRef = useRef("audio/webm");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const requestIdRef = useRef(0);
  const startedAtRef = useRef(0);
  const stopFallbackRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);

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
    clearTimer();
    clearStopFallback();
    stopStream();
    recorderRef.current = null;
    startedAtRef.current = 0;
  }, [clearStopFallback, clearTimer, stopStream]);

  const finalizeRecording = useCallback(
    (recorder: MediaRecorder) => {
      if (recorderRef.current !== recorder && chunksRef.current.length === 0) {
        return;
      }

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
      onFileSelected(file);
      setRecorderMessage(
        recordedSeconds >= MAX_RECORDING_SECONDS
          ? translate("speak.recording_auto_stopped")
          : translate("speak.recording_saved"),
      );
    },
    [cleanupRecorder, onFileSelected, translate],
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
    if (typeof window !== "undefined" && window.isSecureContext === false && !isLocalRecordingHost()) {
      setRecorderPhase("error");
      setRecorderMessage(translate("speak.recording_secure_context_required"));
      return;
    }

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      setRecorderPhase("error");
      setRecorderMessage(translate("speak.recording_unsupported"));
      return;
    }

    setRecorderPhase("requesting");
    setRecorderMessage("");
    elapsedSecondsRef.current = 0;
    setElapsedSeconds(0);
    chunksRef.current = [];
    onFileSelected(null);

    let requestTimeoutId: number | null = null;
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const streamPromise = navigator.mediaDevices.getUserMedia({ audio: true });
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
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      streamRef.current = stream;
      recorderRef.current = recorder;
      recordingMimeTypeRef.current = recorder.mimeType || mimeType || "audio/webm";
      discardStopRef.current = false;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onerror = (event) => {
        cleanupRecorder();
        if (!mountedRef.current) {
          return;
        }
        const error = "error" in event ? event.error : undefined;
        setRecorderPhase("error");
        setRecorderMessage(recordingErrorMessage(translate, error));
      };
      recorder.onstop = () => finalizeRecording(recorder);

      recorder.start(1_000);
      startedAtRef.current = Date.now();
      setRecorderPhase("recording");
      timerRef.current = window.setInterval(() => {
        const nextElapsed = Math.floor((Date.now() - startedAtRef.current) / 1000);
        elapsedSecondsRef.current = nextElapsed;
        setElapsedSeconds(nextElapsed);
        if (nextElapsed >= MAX_RECORDING_SECONDS && recorderRef.current?.state === "recording") {
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
      setRecorderMessage(recordingErrorMessage(translate, error));
    }
  };

  const visibleStatusMessage =
    inputMode === "upload"
      ? statusMessage
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

  return (
    <section style={cardStyle}>
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
          style={modeButtonStyle(inputMode === "record")}
          data-testid="speak.input_mode_record"
          data-semantic-id="speak.input_mode_record"
        >
          {translate("speak.input_method_record")}
        </button>
        <button
          type="button"
          onClick={() => handleModeChange("upload")}
          style={modeButtonStyle(inputMode === "upload")}
          data-testid="speak.input_mode_upload"
          data-semantic-id="speak.input_mode_upload"
        >
          {translate("speak.input_method_upload")}
        </button>
      </div>
      {inputMode === "record" ? (
        <div
          style={{
            display: "grid",
            gap: "0.625rem",
          }}
        >
          <button
            type="button"
            onClick={handleStartRecording}
            disabled={recorderPhase === "requesting" || recorderPhase === "recording"}
            aria-pressed={recorderPhase === "recording"}
            style={primaryActionButtonStyle}
            data-testid="speak.record_start"
            data-semantic-id="speak.record_start"
          >
            {translate("speak.recording_start")}
          </button>
          {recorderPhase === "recording" ? (
            <button
              type="button"
              onClick={() => stopRecording()}
              style={dangerActionButtonStyle}
              data-testid="speak.record_stop"
              data-semantic-id="speak.record_stop"
            >
              {translate("speak.recording_stop")}
            </button>
          ) : null}
        </div>
      ) : (
        <label htmlFor="speak-upload-input" style={uploadBoxStyle}>
          <span>{translate("speak.upload")}</span>
          <span style={actionButtonStyle}>{translate("speak.upload_choose")}</span>
          <input
            id="speak-upload-input"
            type="file"
            accept="audio/*"
            aria-label={translate("speak.upload")}
            onChange={(event) => onFileSelected(event.currentTarget.files?.[0] ?? null)}
            style={inputStyle}
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
        <audio
          controls
          src={previewUrl}
          style={{
            width: "100%",
          }}
        />
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
