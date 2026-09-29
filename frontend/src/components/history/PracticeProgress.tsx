import { useState } from "react";
import type { HistoryRow } from "@/lib/api/types";
import { comparableAttempts, comparisonAttempt, measurement, retryDraft } from "@/lib/history/practiceProgress";
import { createTranslator } from "@/lib/i18n";
import styles from "./PracticeProgress.module.css";
import { RecordingReplay } from "./RecordingReplay";

type Metric = "final_score" | "elapsed_wpm" | "duration_sec" | "pause_total_sec";
const metrics: Metric[] = ["final_score", "elapsed_wpm", "duration_sec", "pause_total_sec"];

export function PracticeProgress({ rows, selected, locale, onSelect, onRetry }: {
  rows: HistoryRow[];
  selected: HistoryRow | null;
  locale: string;
  onSelect: (id: string) => void;
  onRetry: (row: HistoryRow) => void;
}) {
  const t = createTranslator(locale);
  const [metric, setMetric] = useState<Metric>("final_score");
  const cohort = selected ? comparableAttempts(rows, selected) : [];
  const previous = selected ? comparisonAttempt(rows, selected) : null;
  const valid = cohort.filter((row) => measurement(row[metric]) !== null && Number.isFinite(Date.parse(row.timestamp)));
  const values = valid.map((row) => measurement(row[metric])!);
  const max = metric === "final_score" ? 5 : Math.max(1, ...values);
  const start = Date.parse(valid[0]?.timestamp ?? "") || 0;
  const end = Date.parse(valid.at(-1)?.timestamp ?? "") || start;
  const x = (row: HistoryRow) => end === start ? 300 : 48 + (Date.parse(row.timestamp) - start) / (end - start) * 504;
  const y = (row: HistoryRow) => 174 - measurement(row[metric])! / max * 140;
  const date = (stamp: string) => Number.isFinite(Date.parse(stamp))
    ? new Intl.DateTimeFormat(locale, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(stamp)) : "—";
  const format = (value: unknown) => {
    const number = measurement(value);
    return number === null ? "—" : new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(number);
  };
  const days = new Set(rows.filter((row) => Number.isFinite(Date.parse(row.timestamp))).map((row) =>
    new Date(row.timestamp).toLocaleDateString("en-CA"))).size;
  return <section className={styles.panel} data-testid="practice-progress">
    <header className={styles.header}>
      <div><p className={styles.eyebrow}>{t("practice.journal")}</p><h2>{t("practice.title")}</h2></div>
      <p>{t("practice.activity", { attempts: rows.length, days })}</p>
    </header>
    {selected && <>
      <div className={styles.header}>
        <div><strong>{selected.theme}</strong><p>{t("practice.goal", { goal: selected.practice?.goal || "—" })} · {selected.learning_language.toUpperCase()} · {selected.speaker_id}</p></div>
        {retryDraft(selected) && <button type="button" onClick={() => onRetry(selected)} data-testid="practice-retry">{t("practice.retry")}</button>}
      </div>
      {selected.practice?.prompt_text && <details><summary>{t("practice.saved_prompt")}</summary><p className={styles.prompt}>{selected.practice.prompt_text}</p></details>}
      <label className={styles.metric}>{t("practice.measurement")}
        <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>
          {metrics.map((item) => <option key={item} value={item}>{t(`practice.${item}`)}</option>)}
        </select>
      </label>
      <p className={styles.muted}>{t(`practice.${metric}_help`)}</p>
      {cohort.length > 0 ? <>
        <p>{t("practice.conditions", { count: cohort.length, seconds: selected.practice!.target_duration_sec })}</p>
        {valid.length ? <svg viewBox="0 0 600 220" role="img" aria-label={t(`practice.${metric}`)} className={styles.chart}>
          <title>{t(`practice.${metric}`)}</title>
          <line x1="48" y1="174" x2="552" y2="174" stroke="currentColor" opacity="0.25" />
          <line x1="48" y1="34" x2="552" y2="34" stroke="currentColor" opacity="0.12" />
          <text x="36" y="178" textAnchor="end">0</text><text x="36" y="38" textAnchor="end">{format(max)}</text>
          {valid.map((row) => <circle key={row.session_id} cx={x(row)} cy={y(row)} r={row.session_id === selected.session_id ? 7 : 5} fill="currentColor">
            <title>{date(row.timestamp)} · {row.theme}: {format(row[metric])}</title>
          </circle>)}
          <text x="48" y="205">{date(valid[0].timestamp)}</text>
          {end !== start && <text x="552" y="205" textAnchor="end">{date(valid.at(-1)!.timestamp)}</text>}
        </svg> : <p>{t("practice.no_measurements")}</p>}
        <p className={styles.muted}>{t("practice.select_attempt")}</p>
        <div className={styles.attempts} aria-label={t("practice.select_attempt")}>
          {cohort.map((row) => <button type="button" key={row.session_id} aria-pressed={selected.session_id === row.session_id} onClick={() => onSelect(row.session_id)}>
            {date(row.timestamp)} · {format(row[metric])} · {row.theme}{row.practice?.retry_of_session_id ? ` · ${t("practice.repeat")}` : ""}
          </button>)}
        </div>
      </> : <p data-testid="practice-legacy">{t("practice.legacy")}</p>}
      <section className={styles.comparison} data-testid="practice-comparison">
        <h3>{selected.practice?.retry_of_session_id ? t("practice.retry_comparison") : t("practice.previous_comparison")}</h3>
        {previous ? <>
          <p>{date(previous.timestamp)} · {previous.theme} → {date(selected.timestamp)} · {selected.theme}</p>
          <div className={styles.tableWrap}><table>
            <thead><tr><th scope="col">{t("practice.measurement")}</th><th scope="col">{t("practice.before")}</th><th scope="col">{t("practice.selected")}</th><th scope="col">{t("practice.change")}</th></tr></thead>
            <tbody>{metrics.map((item) => {
              const before = measurement(previous[item]);
              const after = measurement(selected[item]);
              const delta = before !== null && after !== null ? after - before : null;
              return <tr key={item}><th scope="row">{t(`practice.${item}`)}</th><td>{format(before)}</td><td>{format(after)}</td><td>{delta !== null && delta > 0 ? "+" : ""}{format(delta)}</td></tr>;
            })}</tbody>
          </table></div>
          <button type="button" onClick={() => onSelect(previous.session_id)}>{t("practice.open_previous")}</button>
        </> : <p>{t(selected.practice?.retry_of_session_id ? "practice.parent_unavailable" : "practice.first")}</p>}
      </section>
      <div className={styles.replay}>
        {previous && <RecordingReplay key={previous.session_id} sessionId={previous.session_id} label={t("practice.listen_before")} unavailable={t("practice.audio_unavailable")} />}
        <RecordingReplay key={selected.session_id} sessionId={selected.session_id} label={t("practice.listen_selected")} unavailable={t("practice.audio_unavailable")} />
      </div>
      {selected.top_priorities.length > 0 && <aside className={styles.focus}><strong>{t("practice.focus")}</strong><p>{selected.top_priorities[0]}</p><small>{t("practice.focus_help")}</small></aside>}
    </>}
  </section>;
}
