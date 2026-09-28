import layoutStyles from "@/components/ui/layout.module.css";

import attemptStyles from "./HistoryList.module.css";

type HistoryListRecord = {
  bandLabel: string;
  languageLabel: string;
  reportPath: string;
  scoreLabel: string;
  sessionId: string;
  statusLabel: string;
  taskFamilyLabel: string;
  theme: string;
  timestampLabel: string;
};

type Translate = (key: string, vars?: Record<string, string | number>) => string;

const cardStyle = {
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const jumpButtonStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  backgroundColor: "rgba(255, 255, 255, 0.95)",
  color: "#10201c",
  font: "inherit",
  textAlign: "left",
  cursor: "pointer",
} as const;

const jumpButtonContentStyle = {
  display: "grid",
  gap: "0.25rem",
  minWidth: 0,
} as const;

const jumpButtonLineStyle = {
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} as const;

const selectStyle = {
  minHeight: "44px",
  width: "100%",
  minWidth: 0,
  maxWidth: "100%",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  font: "inherit",
} as const;

const tableStyle = {
  width: "100%",
  borderCollapse: "separate",
  borderSpacing: 0,
} as const;

const tableCellStyle = {
  padding: "0.75rem 0.875rem",
  borderBottom: "1px solid rgba(18, 61, 55, 0.08)",
  textAlign: "left",
  verticalAlign: "top",
} as const;

const MobileAttemptContent = ({
  record,
  translate,
}: {
  record: HistoryListRecord;
  translate: Translate;
}) => (
  <>
    <span className={attemptStyles.mobileAttemptHeader}>
      <span>{record.timestampLabel}</span>
      <span>{record.languageLabel}</span>
    </span>
    <span className={attemptStyles.mobileAttemptMeta}>
      <span>{record.theme || "-"}</span>
      <span>{record.taskFamilyLabel}</span>
    </span>
    <span className={attemptStyles.mobileAttemptMetrics}>
      <span>{`${translate("history.table_score")} ${record.scoreLabel}`}</span>
      <span>{`${translate("history.table_band")} ${record.bandLabel}`}</span>
      <span>{record.statusLabel}</span>
    </span>
  </>
);

export const HistoryList = ({
  attempts,
  detailAttempts,
  onSelectSession,
  selectedSessionId,
  translate,
}: {
  attempts: HistoryListRecord[];
  detailAttempts: HistoryListRecord[];
  onSelectSession: (sessionId: string) => void;
  selectedSessionId: string;
  translate: Translate;
}) => (
  <section
    className={layoutStyles.stack}
    style={cardStyle}
    data-testid="history-attempts"
    data-semantic-id="history-attempts"
  >
    <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.attempts_title")}</h2>
    <div
      className={attemptStyles.mobileAttemptsList}
      data-testid="history-attempts-mobile-list"
      data-semantic-id="history-attempts-mobile-list"
    >
      <ul className={attemptStyles.mobileAttemptList}>
        {attempts.map((record) => {
          const hasReport = record.reportPath.trim().length > 0;
          const isSelected = record.sessionId === selectedSessionId && hasReport;
          const cardProps = {
            className: attemptStyles.mobileAttemptCard,
            "data-selected": String(isSelected),
            "data-semantic-id": `history-attempt-card-${record.sessionId}`,
            "data-testid": `history-attempt-card-${record.sessionId}`,
          } as const;

          return (
            <li key={record.sessionId}>
              {hasReport ? (
                <button
                  {...cardProps}
                  type="button"
                  aria-pressed={isSelected}
                  onClick={() => onSelectSession(record.sessionId)}
                >
                  <MobileAttemptContent record={record} translate={translate} />
                </button>
              ) : (
                <article {...cardProps}>
                  <MobileAttemptContent record={record} translate={translate} />
                </article>
              )}
            </li>
          );
        })}
      </ul>
    </div>
    <div className={attemptStyles.desktopAttempts}>
      {detailAttempts.length > 0 ? (
        <>
          <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("history.attempts_jump_hint")}</p>
          <div
            style={{
              display: "grid",
              gap: "0.75rem",
              gridTemplateColumns: `repeat(${Math.min(detailAttempts.length, 4)}, minmax(0, 1fr))`,
            }}
          >
            {detailAttempts.slice(0, 4).map((record, index) => (
              <button
                key={record.sessionId}
                type="button"
                onClick={() => onSelectSession(record.sessionId)}
                style={{
                  ...jumpButtonStyle,
                  border:
                    record.sessionId === selectedSessionId
                      ? "1px solid rgba(15, 118, 110, 0.28)"
                      : jumpButtonStyle.border,
                  backgroundColor:
                    record.sessionId === selectedSessionId
                      ? "#d7ebe5"
                      : jumpButtonStyle.backgroundColor,
                }}
                data-testid={`history-jump-${index}`}
                data-semantic-id={`history-jump-${index}`}
              >
                <span style={jumpButtonContentStyle}>
                  <span style={jumpButtonLineStyle}>{`${record.languageLabel} · ${record.timestampLabel}`}</span>
                  <span style={jumpButtonLineStyle}>
                    {`${record.statusLabel} · ${translate("history.table_score")} ${record.scoreLabel} · ${translate("history.table_band")} ${record.bandLabel}`}
                  </span>
                  <span style={jumpButtonLineStyle}>{record.theme || "-"}</span>
                  <span style={jumpButtonLineStyle}>{record.taskFamilyLabel}</span>
                </span>
              </button>
            ))}
          </div>
          <label style={{ display: "grid", gap: "0.375rem", color: "#10201c", fontWeight: 600 }}>
            <span>{translate("history.details_select")}</span>
            <select
              value={selectedSessionId}
              onChange={(event) => onSelectSession(event.currentTarget.value)}
              style={selectStyle}
              data-testid="history-detail-select"
              data-semantic-id="history-detail-select"
            >
              {detailAttempts.map((record) => (
                <option
                  key={record.sessionId}
                  value={record.sessionId}
                  data-testid={`history-detail-select-option-${record.sessionId}`}
                  data-semantic-id={`history-detail-select-option-${record.sessionId}`}
                >
                  {`${record.timestampLabel} · ${record.languageLabel} · ${record.theme || "-"} · ${record.taskFamilyLabel}`}
                </option>
              ))}
            </select>
          </label>
        </>
      ) : null}
      <div style={{ overflowX: "auto" }}>
        <table style={tableStyle} data-testid="history-attempts-table" data-semantic-id="history-attempts-table">
          <thead>
            <tr>
              {[translate("history.table_timestamp"), translate("history.table_language"), translate("history.table_theme"), translate("history.table_score"), translate("history.table_band"), translate("history.table_status")].map((label) => (
                <th
                  key={label}
                  scope="col"
                  style={{
                    ...tableCellStyle,
                    color: "#33514b",
                    fontSize: "0.875rem",
                    fontWeight: 700,
                  }}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {attempts.map((record) => (
              <tr
                key={record.sessionId}
                data-testid={`history-attempts-row-${record.sessionId}`}
                data-semantic-id={`history-attempts-row-${record.sessionId}`}
                style={{
                  backgroundColor:
                    record.sessionId === selectedSessionId && record.reportPath
                      ? "rgba(215, 235, 229, 0.55)"
                      : "transparent",
                }}
              >
                <td style={tableCellStyle}>{record.timestampLabel}</td>
                <td style={tableCellStyle}>{record.languageLabel}</td>
                <td style={tableCellStyle}>{record.theme || "-"}</td>
                <td style={tableCellStyle}>{record.scoreLabel}</td>
                <td style={tableCellStyle}>{record.bandLabel}</td>
                <td style={tableCellStyle}>{record.statusLabel}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </section>
);
