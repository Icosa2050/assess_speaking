import { ReviewSummary, selectReviewSummary, type ReviewDisplaySummary } from "@/components/review/ReviewSummary";
import { WarningsPanel } from "@/components/review/WarningsPanel";

import type { JsonRecord } from "@/lib/api/types";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

type HistoryDetailRecord = {
  bandLabel: string;
  languageLabel: string;
  scoreValue: number | null;
  sessionId: string;
  theme: string;
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

const digestGridStyle = {
  display: "grid",
  gap: "0.875rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
} as const;

const digestItemStyle = {
  display: "grid",
  gap: "0.35rem",
  minWidth: 0,
  padding: "0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.10)",
  backgroundColor: "rgba(248, 251, 250, 0.96)",
} as const;

const digestLabelStyle = {
  color: "#33514b",
  fontSize: "0.8125rem",
  fontWeight: 750,
} as const;

const digestValueStyle = {
  margin: 0,
  color: "#10201c",
  fontWeight: 750,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} as const;

const detailsStyle = {
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.72)",
} as const;

const detailsSummaryStyle = {
  cursor: "pointer",
  padding: "0.875rem 1rem",
  color: "#10201c",
  fontWeight: 750,
} as const;

const detailsContentStyle = {
  display: "grid",
  gap: "1rem",
  padding: "0 1rem 1rem",
} as const;

const firstOrPlaceholder = (values: string[], translate: Translate): string =>
  values.find((value) => value.trim().length > 0) ?? translate("history.none");

const HistoryDetailDigest = ({
  summary,
  translate,
}: {
  summary: ReviewDisplaySummary;
  translate: Translate;
}) => (
  <section
    style={cardStyle}
    data-testid="history-detail-digest"
    data-semantic-id="history-detail-digest"
  >
    <div style={{ display: "grid", gap: "0.35rem" }}>
      <h3 style={{ margin: 0, color: "#10201c", fontSize: "1.2rem" }}>
        {translate("history.details_digest_title")}
      </h3>
      <p style={{ margin: 0, color: "#33514b", lineHeight: 1.55 }}>
        {translate("history.details_digest_body")}
      </p>
    </div>
    <div style={digestGridStyle}>
      {[
        [
          "history-detail-digest-score",
          translate("history.details_digest_score"),
          summary.scoreOverall !== null ? summary.scoreOverall.toFixed(1) : translate("history.none"),
        ],
        ["history-detail-digest-band", translate("history.details_digest_band"), summary.band || translate("history.none")],
        [
          "history-detail-digest-strength",
          translate("history.details_digest_strength"),
          firstOrPlaceholder(summary.strengths, translate),
        ],
        [
          "history-detail-digest-priority",
          translate("history.details_digest_priority"),
          firstOrPlaceholder(summary.priorities, translate),
        ],
        [
          "history-detail-digest-next-focus",
          translate("history.details_digest_next_focus"),
          summary.nextFocus || translate("history.none"),
        ],
        [
          "history-detail-digest-next-exercise",
          translate("history.details_digest_next_exercise"),
          summary.nextExercise || translate("history.none"),
        ],
      ].map(([testId, label, value]) => (
        <div key={testId} style={digestItemStyle} data-testid={testId} data-semantic-id={testId}>
          <strong style={digestLabelStyle}>{label}</strong>
          <p style={digestValueStyle}>{value}</p>
        </div>
      ))}
    </div>
    {summary.coachSummary ? (
      <p
        style={{ margin: 0, color: "#10201c", lineHeight: 1.6 }}
        data-testid="history-detail-digest-coach-summary"
        data-semantic-id="history-detail-digest-coach-summary"
      >
        {summary.coachSummary}
      </p>
    ) : null}
  </section>
);

export const HistoryDetailPanel = ({
  error,
  isLoading,
  payload,
  record,
  translate,
}: {
  error: string | null;
  isLoading: boolean;
  payload: JsonRecord | null;
  record: HistoryDetailRecord | null;
  translate: Translate;
}) => {
  if (!record) {
    return (
      <section style={cardStyle} data-testid="history-detail-unavailable" data-semantic-id="history-detail-unavailable">
        <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.details_title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("history.details_unavailable")}</p>
      </section>
    );
  }

  if (isLoading) {
    return (
      <section style={cardStyle} data-testid="history-detail-loading" data-semantic-id="history-detail-loading">
        <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.details_title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("review.answer_result_pending")}</p>
      </section>
    );
  }

  const hasError = typeof error === "string" && error.length > 0;
  if (hasError || !payload) {
    return (
      <section style={cardStyle} data-testid="history-detail-error" data-semantic-id="history-detail-error">
        <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.details_title")}</h2>
        <p style={{ margin: 0, color: "#b42318", lineHeight: 1.6 }}>{translate("history.details_error")}</p>
      </section>
    );
  }

  const summary = selectReviewSummary({
    band: record.bandLabel,
    payload,
    reportId: record.sessionId,
    scoreOverall: record.scoreValue,
    summary: "",
    transcript: "",
  });

  return (
    <section style={cardStyle} data-testid="history-detail-panel" data-semantic-id="history-detail-panel">
      <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.details_title")}</h2>
      <p
        style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}
        data-testid="history-detail-caption"
        data-semantic-id="history-detail-caption"
      >
        {translate("history.details_caption", {
          language: record.languageLabel,
          session: record.sessionId || "-",
          theme: record.theme || "-",
        })}
      </p>
      <HistoryDetailDigest summary={summary} translate={translate} />
      <WarningsPanel
        failedGates={summary.failedGates}
        requiresHumanReview={summary.requiresHumanReview}
        translate={translate}
        warnings={summary.warnings}
      />
      <details
        style={detailsStyle}
        data-testid="history-detail-full-report"
        data-semantic-id="history-detail-full-report"
      >
        <summary
          style={detailsSummaryStyle}
          data-testid="history-detail-full-report-summary"
          data-semantic-id="history-detail-full-report-summary"
        >
          {translate("history.details_full_report_summary")}
        </summary>
        <div style={detailsContentStyle}>
          <ReviewSummary
            summary={summary}
            translate={translate}
          />
        </div>
      </details>
    </section>
  );
};
