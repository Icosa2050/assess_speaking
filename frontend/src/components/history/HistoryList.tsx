import styles from "./HistoryList.module.css";

type HistoryListRecord = {
  languageLabel: string;
  goal: string;
  reportPath: string;
  scoreLabel: string;
  sessionId: string;
  speakerId: string;
  statusLabel: string;
  theme: string;
  timestampLabel: string;
  topPriorities: string[];
};
type Translate = (key: string, vars?: Record<string, string | number>) => string;

export const HistoryList = ({ attempts, onSelectSession, selectedSessionId, translate }: {
  attempts: HistoryListRecord[];
  onSelectSession: (sessionId: string) => void;
  selectedSessionId: string;
  translate: Translate;
}) => (
  <section className={styles.panel} data-testid="history-attempts">
    <header className={styles.heading}>
      <h2>{translate("history.attempts_title")}</h2>
      <span className={styles.count} aria-hidden="true">{attempts.length}</span>
    </header>
    <p className={styles.hint}>{translate("history.attempts_jump_hint")}</p>
    <ul tabIndex={0} aria-label={translate("history.attempts_title")} className={styles.list} data-testid="history-attempts-list">
      {attempts.map((record, index) => {
        const hasReport = Boolean(record.reportPath.trim() && record.sessionId.trim());
        const isSelected = hasReport && record.sessionId === selectedSessionId;
        const content = <>
          <span className={styles.meta}>{record.languageLabel}{record.goal ? ` · ${record.goal}` : ""} · {record.timestampLabel}</span>
          <span className={styles.topic}>{record.theme || translate("history.none")}</span>
          <span className={styles.learner}>{record.speakerId || translate("history.none")}</span>
          <span className={styles.metrics}>
            <span className={styles.score}>{translate("history.table_score")} {record.scoreLabel}</span>
            <span>{record.statusLabel}</span>
          </span>
          {record.topPriorities[0] && <span className={styles.focus}>{record.topPriorities[0]}</span>}
          <span className={styles.action}>{translate(!hasReport ? "history.review_unavailable" : isSelected ? "history.review_selected" : "history.open_review")} {hasReport && <span aria-hidden="true">→</span>}</span>
        </>;
        const props = {
          className: styles.card,
          "data-selected": String(isSelected),
          "data-testid": `history-attempt-card-${record.sessionId}`,
        };
        return <li key={record.sessionId || `legacy-${index}`}>
          {hasReport ? <button {...props} type="button" aria-pressed={isSelected} onClick={() => onSelectSession(record.sessionId)}>{content}</button>
            : <article {...props}>{content}</article>}
        </li>;
      })}
    </ul>
  </section>
);
