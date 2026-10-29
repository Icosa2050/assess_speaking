type Translate = (key: string, vars?: Record<string, string | number>) => string;

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid var(--color-warning-border)",
  borderRadius: "8px",
  backgroundColor: "var(--color-warning-bg)",
  boxShadow: "0 18px 40px var(--color-shadow-ambient)",
} as const;

const listStyle = {
  display: "grid",
  gap: "0.5rem",
  margin: 0,
  paddingInlineStart: "1.25rem",
  color: "var(--color-warning)",
  lineHeight: 1.55,
} as const;

const warningMessage = (value: string, translate: Translate): string => {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return "";
  }

  const mapping: Record<string, string> = {
    transcript_uncertain: "review.transcript_uncertain",
    cloud_asr_preview: "review.warning_codes.cloud_asr_preview",
    llm_skipped_transcript_uncertain: "review.transcript_uncertain",
    coaching_unavailable: "review.warning_codes.coaching_unavailable",
    llm_unavailable: "review.warning_codes.llm_unavailable",
    llm_skipped_low_word_count: "review.warning_codes.llm_skipped_low_word_count",
    llm_invalid_schema: "review.warning_codes.llm_invalid_schema",
    asr_pause_mismatch: "review.warning_codes.asr_pause_mismatch",
    language_detection_uncertain: "review.warning_codes.language_detection_uncertain",
  };

  const localeKey = mapping[normalized];
  return localeKey ? translate(localeKey) : normalized.replaceAll("_", " ");
};

const gateLabel = (gateKey: string, translate: Translate): string => {
  const mapping: Record<string, string> = {
    language_pass: "review.gate_language",
    topic_pass: "review.gate_theme",
    content_validity_pass: "review.gate_content_validity",
    duration_pass: "review.gate_duration",
    min_words_pass: "review.gate_words",
  };

  const localeKey = mapping[gateKey];
  return localeKey ? translate(localeKey) : gateKey.replaceAll("_", " ");
};

export const WarningsPanel = ({
  failedGates,
  requiresHumanReview,
  translate,
  warnings,
}: {
  failedGates: string[];
  requiresHumanReview: boolean;
  translate: Translate;
  warnings: string[];
}) => {
  const warningItems = [...new Set(warnings
    .map((item) => warningMessage(item, translate))
    .filter((item) => item.trim().length > 0))];
  const hasContent = requiresHumanReview || warningItems.length > 0 || failedGates.length > 0;

  if (!hasContent) {
    return null;
  }

  return (
    <section
      style={cardStyle}
      data-testid="review-warnings"
      data-semantic-id="review-warnings"
    >
      {requiresHumanReview ? (
        <p
          style={{
            margin: 0,
            color: "var(--color-error)",
            fontWeight: 700,
            lineHeight: 1.55,
          }}
          data-testid="review-requires-human-review"
          data-semantic-id="review-requires-human-review"
        >
          {translate("review.status_review")}
        </p>
      ) : null}

      {warningItems.length > 0 ? (
        <div style={{ display: "grid", gap: "0.625rem" }}>
          <strong style={{ color: "var(--color-warning)" }}>{translate("review.warning_label")}</strong>
          <ul style={listStyle}>
            {warningItems.map((item, index) => (
              <li
                key={`${item}-${index}`}
                data-testid={`review-warning-item-${index}`}
                data-semantic-id={`review-warning-item-${index}`}
              >
                {item}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {failedGates.length > 0 ? (
        <div style={{ display: "grid", gap: "0.625rem" }}>
          <strong style={{ color: "var(--color-warning)" }} data-testid="review-failed-gates" data-semantic-id="review-failed-gates">
            {translate("review.failed_gates_label")}
          </strong>
          <ul style={listStyle}>
            {failedGates.map((gateKey, index) => (
              <li
                key={`${gateKey}-${index}`}
                data-testid={`review-failed-gate-item-${index}`}
                data-semantic-id={`review-failed-gate-item-${index}`}
              >
                {gateLabel(gateKey, translate)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
};
