import type { ReactNode } from "react";

import type { ReviewState } from "@/lib/state/sessionDraft";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

type GateKey = "language_pass" | "topic_pass" | "duration_pass" | "min_words_pass";

type ProgressItem = {
  kind: string;
  value: unknown;
};

export type ReviewDisplaySummary = {
  band: string;
  baseline: Record<string, unknown> | null;
  coachSummary: string;
  deterministicScore: number | null;
  failedGates: string[];
  gates: Record<GateKey, boolean | null>;
  label: string;
  learningLanguage: string;
  llmScore: number | null;
  mode: string;
  nextExercise: string;
  nextFocus: string;
  notes: string;
  payload: Record<string, unknown>;
  priorities: string[];
  progressItems: ProgressItem[];
  recurringCoherence: string[];
  recurringGrammar: string[];
  reportId: string;
  requiresHumanReview: boolean;
  scoreOverall: number | null;
  strengths: string[];
  transcript: string;
  warnings: string[];
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

const metricGridStyle = {
  display: "grid",
  gap: "0.875rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
} as const;

const detailGridStyle = {
  display: "grid",
  gap: "1rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
} as const;

const readOnlyInputStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  font: "inherit",
} as const;

const toRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};

const asTextList = (value: unknown): string[] => {
  if (typeof value === "string") {
    return value.trim() ? [value] : [];
  }

  if (Array.isArray(value)) {
    return value.map((item) => String(item || "").trim()).filter((item) => item.length > 0);
  }

  return [];
};

const asNumberOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const gateValue = (value: unknown): boolean | null => {
  if (value === true || value === false) {
    return value;
  }

  return null;
};

const humanizeIssueName = (value: string): string => {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return "";
  }

  const spaced = normalized.replaceAll("_", " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
};

const buildProgressItems = (progressDelta: Record<string, unknown>): ProgressItem[] => {
  if (Object.keys(progressDelta).length === 0) {
    return [];
  }

  const items: ProgressItem[] = [];
  const scoreDelta = toRecord(progressDelta.score_delta);

  if (progressDelta.previous_session_id) {
    items.push({
      kind: "previous_session",
      value: String(progressDelta.previous_session_id),
    });
  }

  (["final", "overall", "wpm"] as const).forEach((key) => {
    const value = scoreDelta[key];
    if (typeof value === "number" && Number.isFinite(value) && value !== 0) {
      items.push({
        kind: `delta_${key}`,
        value,
      });
    }
  });

  (["new_priorities", "resolved_priorities"] as const).forEach((key) => {
    const values = asTextList(progressDelta[key]);
    if (values.length > 0) {
      items.push({
        kind: key,
        value: values,
      });
    }
  });

  const repeatingGrammar = asTextList(progressDelta.repeating_grammar_categories).map(humanizeIssueName);
  if (repeatingGrammar.length > 0) {
    items.push({
      kind: "repeating_grammar",
      value: repeatingGrammar,
    });
  }

  const repeatingCoherence = asTextList(progressDelta.repeating_coherence_categories).map(humanizeIssueName);
  if (repeatingCoherence.length > 0) {
    items.push({
      kind: "repeating_coherence",
      value: repeatingCoherence,
    });
  }

  return items;
};

export const selectReviewSummary = (review: Pick<ReviewState, "band" | "payload" | "reportId" | "scoreOverall" | "summary" | "transcript">): ReviewDisplaySummary => {
  const payload = toRecord(review.payload);
  const meta = toRecord(payload.meta);
  const report = toRecord(payload.report);
  const reportInput = toRecord(report.input);
  const scores = toRecord(report.scores);
  const checks = toRecord(report.checks);
  const coaching = toRecord(report.coaching);
  const rubric = toRecord(report.rubric);
  const progressDelta = toRecord(report.progress_delta);
  const baseline = toRecord(payload.baseline_comparison);

  const recurringGrammar = Array.isArray(rubric.recurring_grammar_errors)
    ? rubric.recurring_grammar_errors
        .map((item) => toRecord(item))
        .map((item) => humanizeIssueName(String(item.type || item.category || "")))
        .filter((item) => item.length > 0)
    : [];

  const recurringCoherence = Array.isArray(rubric.coherence_issues)
    ? rubric.coherence_issues
        .map((item) => toRecord(item))
        .map((item) => humanizeIssueName(String(item.type || item.category || "")))
        .filter((item) => item.length > 0)
    : [];

  const failedGates = (["language_pass", "topic_pass", "duration_pass", "min_words_pass"] as const).filter(
    (gateKey) => checks[gateKey] === false,
  );

  const scoreOverall = review.scoreOverall ?? asNumberOrNull(scores.final);

  return {
    band: review.band || String(scores.band || ""),
    baseline: Object.keys(baseline).length > 0 ? baseline : null,
    coachSummary: String(coaching.coach_summary || review.summary || ""),
    deterministicScore: asNumberOrNull(scores.deterministic),
    failedGates,
    gates: {
      language_pass: gateValue(checks.language_pass),
      topic_pass: gateValue(checks.topic_pass),
      duration_pass: gateValue(checks.duration_pass),
      min_words_pass: gateValue(checks.min_words_pass),
    },
    label: String(meta.label || payload.label || ""),
    learningLanguage: String(
      reportInput.expected_language ||
        reportInput.learning_language ||
        meta.learning_language ||
        payload.learning_language ||
        "",
    )
      .trim()
      .toLowerCase(),
    llmScore: asNumberOrNull(scores.llm),
    mode: String(scores.mode || ""),
    nextExercise: String(coaching.next_exercise || ""),
    nextFocus: String(coaching.next_focus || ""),
    notes: String(payload.notes || ""),
    payload,
    priorities: asTextList(coaching.top_3_priorities),
    progressItems: buildProgressItems(progressDelta),
    recurringCoherence,
    recurringGrammar,
    reportId: String(report.session_id || review.reportId || payload.report_path || ""),
    requiresHumanReview: Boolean(report.requires_human_review),
    scoreOverall,
    strengths: asTextList(coaching.strengths),
    transcript: String(
      review.transcript ||
        payload.transcript_full ||
        payload.transcript_preview ||
        report.transcript_preview ||
        "",
    ),
    warnings: asTextList(report.warnings),
  };
};

const statusMessage = (summary: ReviewDisplaySummary, translate: Translate): string => {
  if (summary.requiresHumanReview) {
    return translate("review.status_review");
  }

  if (summary.failedGates.length > 0) {
    const gateLabels = summary.failedGates
      .map((gateKey) => {
        const mapping: Record<string, string> = {
          language_pass: "review.gate_language",
          topic_pass: "review.gate_theme",
          duration_pass: "review.gate_duration",
          min_words_pass: "review.gate_words",
        };

        return translate(mapping[gateKey] ?? gateKey);
      })
      .join(", ");

    return translate("review.status_unstable", { gates: gateLabels });
  }

  return translate("review.status_done");
};

const modeLabel = (mode: string, translate: Translate): string => {
  if (mode === "hybrid") {
    return translate("review.mode_hybrid");
  }

  if (mode === "deterministic_only") {
    return translate("review.mode_deterministic_only");
  }

  return translate("review.mode_unknown");
};

const gateStatus = (value: boolean | null, translate: Translate): string => {
  if (value === true) {
    return translate("review.gate_pass");
  }

  if (value === false) {
    return translate("review.gate_fail");
  }

  return translate("review.gate_unknown");
};

const localizedLanguageLabel = (languageCode: string, translate: Translate): string => {
  if (!languageCode) {
    return "";
  }

  const translated = translate(`locale.${languageCode}`);
  return translated.startsWith("[") ? languageCode.toUpperCase() : translated;
};

const localizedCefrDescriptor = (level: string, translate: Translate): string => {
  if (!level) {
    return "";
  }

  const translated = translate(`review.baseline_descriptors.${level.toLowerCase()}`);
  return translated.startsWith("[") ? "" : translated;
};

const baselineCaption = (summary: ReviewDisplaySummary, translate: Translate): string => {
  if (!summary.baseline) {
    return "";
  }

  const level = String(summary.baseline.level || "").trim().toUpperCase();
  const descriptor = localizedCefrDescriptor(level, translate);
  const languageLabel = localizedLanguageLabel(summary.learningLanguage, translate);

  if (languageLabel && descriptor) {
    return translate("review.baseline_target_caption", {
      comment: descriptor,
      language: languageLabel,
      level,
    });
  }

  if (languageLabel) {
    return translate("review.baseline_target_caption_no_comment", {
      language: languageLabel,
      level,
    });
  }

  if (descriptor) {
    return translate("review.baseline_level_caption", {
      comment: descriptor,
      level,
    });
  }

  return translate("review.baseline_level_caption_no_comment", { level });
};

const progressText = (item: ProgressItem, translate: Translate): string => {
  if (item.kind === "previous_session" && item.value) {
    return translate("review.progress_previous_session", { value: String(item.value) });
  }

  if (item.kind === "delta_final" && typeof item.value === "number") {
    return translate("review.progress_delta_final", { value: item.value.toFixed(2).replace(/^/, item.value >= 0 ? "+" : "") });
  }

  if (item.kind === "delta_overall" && typeof item.value === "number") {
    return translate("review.progress_delta_overall", { value: item.value.toFixed(2).replace(/^/, item.value >= 0 ? "+" : "") });
  }

  if (item.kind === "delta_wpm" && typeof item.value === "number") {
    return translate("review.progress_delta_wpm", { value: item.value.toFixed(2).replace(/^/, item.value >= 0 ? "+" : "") });
  }

  if (item.kind === "new_priorities" && Array.isArray(item.value)) {
    return translate("review.progress_new_priorities", { value: item.value.map(String).join(", ") });
  }

  if (item.kind === "resolved_priorities" && Array.isArray(item.value)) {
    return translate("review.progress_resolved_priorities", { value: item.value.map(String).join(", ") });
  }

  if (item.kind === "repeating_grammar" && Array.isArray(item.value)) {
    return translate("review.progress_repeating_grammar", { value: item.value.map(String).join(", ") });
  }

  if (item.kind === "repeating_coherence" && Array.isArray(item.value)) {
    return translate("review.progress_repeating_coherence", { value: item.value.map(String).join(", ") });
  }

  return "";
};

export const ReviewSummary = ({
  summary,
  translate,
  warningsSlot,
}: {
  summary: ReviewDisplaySummary;
  translate: Translate;
  warningsSlot?: ReactNode;
}) => (
  <div
    style={{ display: "grid", gap: "1rem" }}
    data-testid="review-summary"
    data-semantic-id="review-summary"
  >
    <section style={cardStyle}>
      <p
        style={{
          margin: 0,
          fontSize: "0.875rem",
          fontWeight: 700,
          color: "#0f766e",
        }}
      >
        {translate("review.answers_eyebrow")}
      </p>
      <h2
        style={{
          margin: 0,
          fontSize: "1.45rem",
          color: "#10201c",
        }}
      >
        {translate("review.answers_title")}
      </h2>
      <p
        style={{
          margin: 0,
          color: summary.requiresHumanReview ? "#8f1f14" : "#33514b",
          lineHeight: 1.6,
          fontWeight: 600,
        }}
      >
        {statusMessage(summary, translate)}
      </p>
      {warningsSlot}
      <div
        style={metricGridStyle}
        data-testid="review-metrics"
        data-semantic-id="review-metrics"
      >
        {[
          ["review-metric-score-overall", translate("review.score"), summary.scoreOverall !== null ? summary.scoreOverall.toFixed(1) : "-"],
          ["review-metric-band", translate("review.band"), summary.band || "-"],
          ["review-metric-deterministic-score", translate("review.metric_deterministic"), summary.deterministicScore !== null ? summary.deterministicScore.toFixed(1) : "-"],
          ["review-metric-llm-score", translate("review.metric_llm"), summary.llmScore !== null ? summary.llmScore.toFixed(1) : "-"],
        ].map(([testId, label, value]) => (
          <div key={testId} style={cardStyle} data-testid={testId} data-semantic-id={testId}>
            <strong style={{ color: "#33514b" }}>{label}</strong>
            <span style={{ color: "#10201c", fontSize: "1.25rem", fontWeight: 700 }}>{value}</span>
          </div>
        ))}
      </div>
      <p
        style={{
          margin: 0,
          color: "#33514b",
          lineHeight: 1.55,
        }}
        data-testid="review-metric-mode"
        data-semantic-id="review-metric-mode"
      >
        {translate("review.mode_caption", { value: modeLabel(summary.mode, translate) })}
      </p>
    </section>

    <section style={cardStyle}>
      <p
        style={{
          margin: 0,
          fontSize: "0.875rem",
          fontWeight: 700,
          color: "#0f766e",
        }}
      >
        {translate("review.coaching_tab")}
      </p>
      <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("review.summary_title")}</h2>
      <p
        style={{ margin: 0, color: "#10201c", lineHeight: 1.6 }}
        data-testid="review-coach-summary"
        data-semantic-id="review-coach-summary"
      >
        {summary.coachSummary || translate("review.summary_placeholder")}
      </p>
      <div style={detailGridStyle}>
        <div data-testid="review-strengths" data-semantic-id="review-strengths">
          <strong style={{ color: "#33514b" }}>{translate("review.strengths_title")}</strong>
          {summary.strengths.length > 0 ? (
            <ul style={{ margin: "0.625rem 0 0", paddingInlineStart: "1.25rem", lineHeight: 1.55 }}>
              {summary.strengths.map((item, index) => (
                <li key={`${item}-${index}`} data-testid={`review-strength-item-${index}`} data-semantic-id={`review-strength-item-${index}`}>
                  {item}
                </li>
              ))}
            </ul>
          ) : (
            <p style={{ margin: "0.625rem 0 0", color: "#33514b" }}>{translate("review.no_strengths")}</p>
          )}
        </div>
        <div data-testid="review-priorities" data-semantic-id="review-priorities">
          <strong style={{ color: "#33514b" }}>{translate("review.priorities_title")}</strong>
          {summary.priorities.length > 0 ? (
            <ul style={{ margin: "0.625rem 0 0", paddingInlineStart: "1.25rem", lineHeight: 1.55 }}>
              {summary.priorities.map((item, index) => (
                <li key={`${item}-${index}`} data-testid={`review-priority-item-${index}`} data-semantic-id={`review-priority-item-${index}`}>
                  {item}
                </li>
              ))}
            </ul>
          ) : (
            <p style={{ margin: "0.625rem 0 0", color: "#33514b" }}>{translate("review.no_priorities")}</p>
          )}
        </div>
      </div>
      {summary.nextFocus ? (
        <p style={{ margin: 0, color: "#10201c" }} data-testid="review-next-focus" data-semantic-id="review-next-focus">
          {translate("review.next_focus", { value: summary.nextFocus })}
        </p>
      ) : null}
      {summary.nextExercise ? (
        <p style={{ margin: 0, color: "#10201c" }} data-testid="review-next-exercise" data-semantic-id="review-next-exercise">
          {translate("review.next_exercise", { value: summary.nextExercise })}
        </p>
      ) : null}
      <div style={detailGridStyle}>
        <div data-testid="review-recurring-grammar" data-semantic-id="review-recurring-grammar">
          <strong style={{ color: "#33514b" }}>{translate("review.recurring_grammar_title")}</strong>
          <p style={{ margin: "0.625rem 0 0", color: "#10201c" }}>
            {summary.recurringGrammar.join(", ") || translate("review.none")}
          </p>
        </div>
        <div data-testid="review-recurring-coherence" data-semantic-id="review-recurring-coherence">
          <strong style={{ color: "#33514b" }}>{translate("review.recurring_coherence_title")}</strong>
          <p style={{ margin: "0.625rem 0 0", color: "#10201c" }}>
            {summary.recurringCoherence.join(", ") || translate("review.none")}
          </p>
        </div>
      </div>
      <div data-testid="review-progress" data-semantic-id="review-progress">
        <strong style={{ color: "#33514b" }}>{translate("review.progress_title")}</strong>
        {summary.progressItems.length > 0 ? (
          <ul style={{ margin: "0.625rem 0 0", paddingInlineStart: "1.25rem", lineHeight: 1.55 }}>
            {summary.progressItems
              .map((item) => progressText(item, translate))
              .filter((item) => item.length > 0)
              .map((item, index) => (
                <li key={`${item}-${index}`} data-testid={`review-progress-item-${index}`} data-semantic-id={`review-progress-item-${index}`}>
                  {item}
                </li>
              ))}
          </ul>
        ) : (
          <p style={{ margin: "0.625rem 0 0", color: "#33514b" }}>{translate("review.progress_unavailable")}</p>
        )}
      </div>
      {summary.baseline ? (
        <div data-testid="review-baseline" data-semantic-id="review-baseline" style={{ display: "grid", gap: "0.625rem" }}>
          <strong style={{ color: "#33514b" }}>{translate("review.baseline_title")}</strong>
          <p style={{ margin: 0, color: "#33514b" }}>{baselineCaption(summary, translate)}</p>
          {typeof summary.baseline.targets === "object" && summary.baseline.targets !== null ? (
            <div style={{ display: "grid", gap: "0.5rem" }}>
              {Object.entries(summary.baseline.targets as Record<string, Record<string, unknown>>).map(([metric, entry]) => (
                <div
                  key={metric}
                  style={{
                    display: "grid",
                    gap: "0.5rem",
                    gridTemplateColumns: "minmax(140px, 1.4fr) repeat(3, minmax(0, 1fr))",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    backgroundColor: "rgba(248, 251, 250, 0.96)",
                  }}
                >
                  <span style={{ color: "#10201c", fontWeight: 600 }}>{metric}</span>
                  <span style={{ color: "#33514b" }}>{String(entry.expected ?? "-")}</span>
                  <span style={{ color: "#33514b" }}>{String(entry.actual ?? "-")}</span>
                  <span style={{ color: "#33514b" }}>
                    {translate(entry.ok ? "review.gate_pass" : "review.gate_fail")}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>

    <section style={cardStyle} data-testid="review-gates" data-semantic-id="review-gates">
      <p style={{ margin: 0, fontSize: "0.875rem", fontWeight: 700, color: "#0f766e" }}>{translate("review.gates_eyebrow")}</p>
      <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("review.gates_title")}</h2>
      <div style={metricGridStyle}>
        {([
          ["review-gate-language", translate("review.gate_language"), summary.gates.language_pass],
          ["review-gate-topic", translate("review.gate_theme"), summary.gates.topic_pass],
          ["review-gate-duration", translate("review.gate_duration"), summary.gates.duration_pass],
          ["review-gate-min-words", translate("review.gate_words"), summary.gates.min_words_pass],
        ] as const).map(([testId, label, value]) => (
          <div key={testId} style={cardStyle} data-testid={testId} data-semantic-id={testId}>
            <strong style={{ color: "#33514b" }}>{label}</strong>
            <span style={{ color: "#10201c", fontWeight: 700 }}>{gateStatus(value as boolean | null, translate)}</span>
          </div>
        ))}
      </div>
    </section>

    <section style={cardStyle}>
      <p style={{ margin: 0, fontSize: "0.875rem", fontWeight: 700, color: "#0f766e" }}>{translate("review.details_tab")}</p>
      <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("review.details_title")}</h2>
      <div style={detailGridStyle}>
        <label style={{ display: "grid", gap: "0.375rem", color: "#10201c", fontWeight: 600 }}>
          <span>{translate("review.label_title")}</span>
          <input
            readOnly
            value={summary.label || translate("review.label_placeholder")}
            style={readOnlyInputStyle}
            data-testid="review-label"
            data-semantic-id="review-label"
          />
        </label>
        <div style={{ display: "grid", gap: "0.375rem" }}>
          <strong style={{ color: "#10201c" }}>{translate("review.metric_report_id")}</strong>
          <span data-testid="review-report-id" data-semantic-id="review-report-id" style={{ color: "#33514b" }}>
            {summary.reportId || "-"}
          </span>
          <strong style={{ color: "#10201c" }}>{translate("setup.learning_language")}</strong>
          <span data-testid="review-learning-language" data-semantic-id="review-learning-language" style={{ color: "#33514b" }}>
            {localizedLanguageLabel(summary.learningLanguage, translate) || "-"}
          </span>
        </div>
      </div>
      <label style={{ display: "grid", gap: "0.375rem", color: "#10201c", fontWeight: 600 }}>
        <span>{translate("review.notes_title")}</span>
        <textarea
          readOnly
          value={summary.notes || translate("review.notes_placeholder")}
          style={{ ...readOnlyInputStyle, minHeight: "132px", resize: "vertical" }}
          data-testid="review-notes"
          data-semantic-id="review-notes"
        />
      </label>
      <label style={{ display: "grid", gap: "0.375rem", color: "#10201c", fontWeight: 600 }}>
        <span>{translate("review.transcript_title")}</span>
        <textarea
          readOnly
          value={summary.transcript || translate("review.transcript_placeholder")}
          style={{ ...readOnlyInputStyle, minHeight: "260px", resize: "vertical" }}
          data-testid="review-transcript"
          data-semantic-id="review-transcript"
        />
      </label>
      <details data-testid="review-raw-payload" data-semantic-id="review-raw-payload">
        <summary style={{ cursor: "pointer", fontWeight: 600, color: "#10201c" }}>
          {translate("review.raw_payload")}
        </summary>
        <pre
          style={{
            margin: "0.75rem 0 0",
            padding: "0.875rem",
            borderRadius: "8px",
            backgroundColor: "rgba(248, 251, 250, 0.96)",
            overflowX: "auto",
            whiteSpace: "pre-wrap",
            color: "#10201c",
            fontSize: "0.875rem",
          }}
        >
          {JSON.stringify(summary.payload, null, 2)}
        </pre>
      </details>
    </section>
  </div>
);
