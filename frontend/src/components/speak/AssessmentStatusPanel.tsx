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
  contextMode,
  handoffHint,
  lifecycleState,
  notesValue,
  onCancel,
  onLabelChange,
  onNotesChange,
  onSubmit,
  phaseMessage,
  runtimeDetail,
  statusMessage,
  submitDisabled,
  submitDisabledHelp,
  translate,
  warningMessage,
  labelValue,
}: {
  canCancel: boolean;
  contextMode: "inline" | "optional";
  handoffHint: string;
  labelValue: string;
  lifecycleState: string;
  notesValue: string;
  onCancel: () => void;
  onLabelChange: (value: string) => void;
  onNotesChange: (value: string) => void;
  onSubmit: () => void;
  phaseMessage: string;
  runtimeDetail: string;
  statusMessage: string;
  submitDisabled: boolean;
  submitDisabledHelp: string | null;
  translate: Translate;
  warningMessage: string | null;
}) => {
  const contextFields = (
    <>
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
    </>
  );

  return (
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
      {translate("speak.assessment_caption")}
    </p>
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
    {contextMode === "optional" ? (
      <details
        data-testid="speak.optional_context"
        data-semantic-id="speak.optional_context"
        style={{ display: "grid", gap: "0.75rem" }}
      >
        <summary style={{ cursor: "pointer", fontWeight: 700, color: "#10201c" }}>
          {translate("speak.optional_context_summary")}
        </summary>
        <div style={{ display: "grid", gap: "0.875rem", paddingTop: "0.75rem" }}>
          {contextFields}
        </div>
      </details>
    ) : (
      contextFields
    )}
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
      <p
        style={{
          margin: 0,
          color: "#33514b",
          lineHeight: 1.55,
          fontWeight: 600,
        }}
        data-testid="speak.handoff_hint"
        data-semantic-id="speak.handoff_hint"
      >
        {handoffHint}
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
        <p
          style={{
            margin: 0,
            color: "#33514b",
            lineHeight: 1.55,
          }}
        >
          {translate("speak.job_review_wait")}
        </p>
      ) : null}
    </div>
    {submitDisabledHelp ? (
      <p
        id="speak-submit-help"
        style={{
          margin: 0,
          color: "#33514b",
          fontSize: "0.92rem",
          lineHeight: 1.5,
        }}
        data-testid="speak.submit_disabled_help"
        data-semantic-id="speak.submit_disabled_help"
      >
        {submitDisabledHelp}
      </p>
    ) : null}
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
        aria-describedby={submitDisabledHelp ? "speak-submit-help" : undefined}
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
    <p
      style={{
        margin: 0,
        color: "#6b7a76",
        fontSize: "0.84rem",
        lineHeight: 1.45,
      }}
      data-testid="speak.runtime_detail"
      data-semantic-id="speak.runtime_detail"
    >
      {runtimeDetail}
    </p>
    </section>
  );
};
