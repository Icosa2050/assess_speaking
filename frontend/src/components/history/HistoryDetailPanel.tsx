import { ReviewSummary, selectReviewSummary } from "@/components/review/ReviewSummary";
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

export const HistoryDetailPanel = ({
  error,
  isLoading,
  payload,
  record,
  translate,
}: {
  error: string;
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

  if (error || !payload) {
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
      <ReviewSummary
        summary={summary}
        translate={translate}
        warningsSlot={
          <WarningsPanel
            failedGates={summary.failedGates}
            requiresHumanReview={summary.requiresHumanReview}
            translate={translate}
            warnings={summary.warnings}
          />
        }
      />
    </section>
  );
};
