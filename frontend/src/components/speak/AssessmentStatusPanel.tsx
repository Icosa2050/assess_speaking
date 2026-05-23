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

const fieldStyle = {
  display: "grid",
  gap: "0.375rem",
  color: "#10201c",
  fontWeight: 600,
} as const;

const controlStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  font: "inherit",
} as const;

const detailGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
} as const;

const actionButtonStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  fontWeight: 700,
  font: "inherit",
  cursor: "pointer",
} as const;

export const AssessmentStatusPanel = ({
  canCancel,
  lifecycleState,
  model,
  notesValue,
  onCancel,
  onLabelChange,
  onNotesChange,
  onSubmit,
  phaseMessage,
  provider,
  speakerSummary,
  statusMessage,
  submitDisabled,
  targetDurationSummary,
  translate,
  warningMessage,
  whisperModel,
  labelValue,
  learningLanguageSummary,
  cefrSummary,
}: {
  canCancel: boolean;
  cefrSummary: string;
  labelValue: string;
  learningLanguageSummary: string;
  lifecycleState: string;
  model: string;
  notesValue: string;
  onCancel: () => void;
  onLabelChange: (value: string) => void;
  onNotesChange: (value: string) => void;
  onSubmit: () => void;
  phaseMessage: string;
  provider: string;
  speakerSummary: string;
  statusMessage: string;
  submitDisabled: boolean;
  targetDurationSummary: string;
  translate: Translate;
  warningMessage: string | null;
  whisperModel: string;
}) => (
  <section
    style={cardStyle}
    data-testid="speak.status_panel"
    data-semantic-id="speak.status_panel"
    data-assessment-state={lifecycleState}
  >
    <h2
      style={{
        margin: 0,
        fontSize: "1.35rem",
        color: "#10201c",
      }}
    >
      {translate("speak.assessment_title")}
    </h2>
    <p
      style={{
        margin: 0,
        color: "#33514b",
      }}
    >
      {translate("speak.assessment_caption", {
        provider,
        model,
        whisper: whisperModel,
      })}
    </p>
    <div style={detailGridStyle}>
      {[
        [translate("setup.speaker_id"), speakerSummary],
        [translate("setup.learning_language"), learningLanguageSummary],
        [translate("setup.cefr"), cefrSummary],
        [translate("setup.duration"), targetDurationSummary],
      ].map(([label, value]) => (
        <div key={label} style={cardStyle}>
          <strong
            style={{
              color: "#33514b",
            }}
          >
            {label}
          </strong>
          <span
            style={{
              color: "#10201c",
            }}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
    {warningMessage ? (
      <p
        style={{
          margin: 0,
          color: "#9a6700",
          lineHeight: 1.55,
        }}
        data-testid="speak.openrouter_key_warning"
        data-semantic-id="speak.openrouter_key_warning"
      >
        {warningMessage}
      </p>
    ) : null}
    <label htmlFor="speak-label" style={fieldStyle}>
      <span>{translate("speak.label")}</span>
      <input
        id="speak-label"
        type="text"
        value={labelValue}
        onChange={(event) => onLabelChange(event.currentTarget.value)}
        style={controlStyle}
        data-testid="speak.label"
        data-semantic-id="speak.label"
      />
    </label>
    <label htmlFor="speak-notes" style={fieldStyle}>
      <span>{translate("speak.notes")}</span>
      <textarea
        id="speak-notes"
        value={notesValue}
        onChange={(event) => onNotesChange(event.currentTarget.value)}
        style={{
          ...controlStyle,
          minHeight: "132px",
          resize: "vertical",
        }}
        data-testid="speak.notes"
        data-semantic-id="speak.notes"
      />
    </label>
    <div
      style={{
        display: "grid",
        gap: "0.5rem",
      }}
    >
      <p
        style={{
          margin: 0,
          color:
            lifecycleState === "failed"
              ? "#b42318"
              : lifecycleState === "cancelled"
                ? "#9a6700"
                : lifecycleState === "completed"
                  ? "#166534"
                  : "#33514b",
          lineHeight: 1.55,
        }}
      >
        {statusMessage}
      </p>
      {phaseMessage ? (
        <p
          style={{
            margin: 0,
            color: "#33514b",
            lineHeight: 1.55,
          }}
        >
          {phaseMessage}
        </p>
      ) : null}
      {(lifecycleState === "queued" || lifecycleState === "running") ? (
        <>
          <p
            style={{
              margin: 0,
              color: "#33514b",
              lineHeight: 1.55,
            }}
          >
            {translate("speak.job_auto_refresh")}
          </p>
          <p
            style={{
              margin: 0,
              color: "#33514b",
              lineHeight: 1.55,
            }}
          >
            {translate("speak.job_long_running")}
          </p>
        </>
      ) : null}
    </div>
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "0.75rem",
      }}
    >
      <button
        type="button"
        onClick={onSubmit}
        disabled={submitDisabled}
        style={{
          ...actionButtonStyle,
          border: "1px solid rgba(15, 118, 110, 0.2)",
          backgroundColor: submitDisabled ? "rgba(215, 235, 229, 0.5)" : "#d7ebe5",
          color: "#10201c",
          cursor: submitDisabled ? "not-allowed" : "pointer",
          opacity: submitDisabled ? 0.7 : 1,
        }}
        data-testid="speak.submit"
        data-semantic-id="speak.submit"
      >
        {translate("speak.submit")}
      </button>
      {canCancel ? (
        <button
          type="button"
          onClick={onCancel}
          style={{
            ...actionButtonStyle,
            border: "1px solid rgba(180, 35, 24, 0.18)",
            backgroundColor: "rgba(255, 245, 245, 0.96)",
            color: "#8f1f14",
          }}
          data-testid="speak.cancel_assessment"
          data-semantic-id="speak.cancel_assessment"
        >
          {translate("speak.cancel_assessment")}
        </button>
      ) : null}
    </div>
  </section>
);
