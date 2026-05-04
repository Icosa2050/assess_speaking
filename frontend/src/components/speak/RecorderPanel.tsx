import type { RecordingInputMethod } from "@/lib/state/sessionDraft";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

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
}) => (
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
        onClick={() => onInputModeChange("record")}
        style={modeButtonStyle(inputMode === "record")}
        data-testid="speak.input_mode_record"
        data-semantic-id="speak.input_mode_record"
      >
        {translate("speak.input_method_record")}
      </button>
      <button
        type="button"
        onClick={() => onInputModeChange("upload")}
        style={modeButtonStyle(inputMode === "upload")}
        data-testid="speak.input_mode_upload"
        data-semantic-id="speak.input_mode_upload"
      >
        {translate("speak.input_method_upload")}
      </button>
    </div>
    <label
      htmlFor={inputMode === "record" ? "speak-audio-input" : "speak-upload-input"}
      style={{
        display: "grid",
        gap: "0.375rem",
        color: "#10201c",
        fontWeight: 600,
      }}
    >
      <span>{translate(inputMode === "record" ? "speak.audio_input" : "speak.upload")}</span>
      <input
        id={inputMode === "record" ? "speak-audio-input" : "speak-upload-input"}
        type="file"
        accept="audio/*"
        capture={inputMode === "record" ? "user" : undefined}
        onChange={(event) => onFileSelected(event.currentTarget.files?.[0] ?? null)}
        style={inputStyle}
        data-testid={inputMode === "record" ? "speak.audio_input" : "speak.upload_input"}
        data-semantic-id={inputMode === "record" ? "speak.audio_input" : "speak.upload_input"}
      />
    </label>
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
      style={{
        margin: 0,
        color:
          statusTone === "success"
            ? "#166534"
            : statusTone === "warning"
              ? "#9a6700"
              : statusTone === "error"
                ? "#b42318"
                : "#33514b",
        lineHeight: 1.55,
      }}
    >
      {statusMessage}
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
