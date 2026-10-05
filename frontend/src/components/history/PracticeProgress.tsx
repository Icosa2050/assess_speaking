import { useEffect, useRef, useState } from "react";
import type { HistoryRow } from "@/lib/api/types";
import { comparableAttempts, comparisonAttempt, measurement, retryDraft } from "@/lib/history/practiceProgress";
import { createTranslator } from "@/lib/i18n";
import styles from "./PracticeProgress.module.css";
import { RecordingReplay } from "./RecordingReplay";

import { WeeklyRhythm } from "./WeeklyRhythm";
import { practiceRhythm } from "@/lib/history/practiceRhythm";

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
  const cohort = selected ? comparableAttempts(rows, selected).filter(row =>
    Date.parse(row.timestamp) <= Date.parse(selected.timestamp)) : [];
  const chartRef = useRef<SVGSVGElement>(null);
  const [chartWidth, setChartWidth] = useState(600);
  useEffect(() => {
    if (!chartRef.current || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(entries => {
      const width = entries[0]?.contentRect.width;
      if (width > 0) setChartWidth(Math.max(220, width));
    });
    observer.observe(chartRef.current);
    return () => observer.disconnect();
  }, [selected?.session_id, metric]);
  const previous = selected ? comparisonAttempt(rows, selected) : null;
  const valid = cohort.filter((row) => measurement(row[metric]) !== null && Number.isFinite(Date.parse(row.timestamp)));
  const values = valid.map((row) => measurement(row[metric])!);
  const max = metric === "final_score" ? 5 : Math.max(1, ...values);
  const start = Date.parse(valid[0]?.timestamp ?? "") || 0;
  const end = Date.parse(valid.at(-1)?.timestamp ?? "") || start;
  const sameDay = valid.length > 1 && new Date(start).toDateString() === new Date(end).toDateString();
  const axisDate = (stamp: string) => new Intl.DateTimeFormat(locale, sameDay
    ? { hour: "2-digit", minute: "2-digit", second: "2-digit" }
    : { month: "short", day: "numeric" }).format(new Date(stamp));
  const x = (row: HistoryRow) => end === start ? chartWidth / 2 : 36 + (Date.parse(row.timestamp) - start) / (end - start) * (chartWidth - 54);
  const y = (row: HistoryRow) => 174 - measurement(row[metric])! / max * 140;
  const date = (stamp: string) => Number.isFinite(Date.parse(stamp))
    ? new Intl.DateTimeFormat(locale, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(stamp)) : "—";
  const format = (value: unknown) => {
    const number = measurement(value);
    return number === null ? "—" : new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(number);
  };
  const rhythm = selected ? practiceRhythm(rows, selected.speaker_id, selected.learning_language) : null;
  const firstScore = measurement(cohort.find(row => measurement(row.final_score) !== null)?.final_score);
  const currentScore = measurement(selected?.final_score);
  const delta = firstScore !== null && currentScore !== null && cohort.length > 1 ? Math.round((currentScore - firstScore) * 10) / 10 : null;
  const priorities = Array.isArray(selected?.top_priorities) ? selected.top_priorities : [];
  return <section className={styles.panel} data-testid="practice-progress">
    <header className={styles.header}>
      <p className={styles.eyebrow}>{t("practice.journal")}</p>
      {selected && <p className={styles.identity}>{selected.speaker_id} · {String(selected.learning_language || "").toUpperCase()} · {t("practice.goal", { goal: selected.practice?.goal || "—" })}</p>}
    </header>
    {selected && <>
      <div className={styles.heroGrid}>
        <section className={styles.performance}>
          <div className={styles.header}>
            <div><p className={styles.eyebrow}>{t("journey.voice_over_time")}</p><h3>{t("journey.small_steps")}</h3></div>
            {delta !== null && metric === "final_score" && <span className={styles.delta} data-positive={delta > 0}>{t("journey.since_first", { delta: `${delta > 0 ? "+" : ""}${format(delta)}` })}</span>}
          </div>
          <p className={styles.score}>{format(selected[metric])}<span>{t(`practice.${metric}`)}</span></p>
          <strong>{selected.theme}</strong>
          <label className={styles.metric}>{t("practice.measurement")}
            <select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>
              {metrics.map((item) => <option key={item} value={item}>{t(`practice.${item}`)}</option>)}
            </select>
          </label>
          {cohort.length > 0 ? <>
            {valid.length ? <svg ref={chartRef} viewBox={`0 0 ${chartWidth} 220`} role="img" aria-label={t(`practice.${metric}`)} className={styles.chart}>
              <title>{t(`practice.${metric}`)}</title>
              <desc>{t("journey.chart_summary", { count: valid.length, first: `${date(valid[0].timestamp)}: ${format(valid[0][metric])}`, last: `${date(valid.at(-1)!.timestamp)}: ${format(valid.at(-1)![metric])}` })}</desc>
              <line x1="36" y1="174" x2={chartWidth - 18} y2="174" stroke="currentColor" opacity="0.2" />
              <line x1="36" y1="34" x2={chartWidth - 18} y2="34" stroke="currentColor" opacity="0.12" />
              <text x="26" y="178" textAnchor="end">0</text><text x="26" y="38" textAnchor="end">{format(max)}</text>
              {valid.length > 1 && <>
                <path d={`M ${x(valid[0])} 174 ${valid.map(row => `L ${x(row)} ${y(row)}`).join(" ")} L ${x(valid.at(-1)!)} 174 Z`} fill="currentColor" opacity="0.09" />
                <polyline points={valid.map(row => `${x(row)},${y(row)}`).join(" ")} fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round" />
              </>}
              {valid.map((row) => <circle key={row.session_id} cx={x(row)} cy={y(row)} r={row.session_id === selected.session_id ? 6 : 4} fill="currentColor">
                <title>{date(row.timestamp)} · {row.theme}: {format(row[metric])}</title>
              </circle>)}
              <text x="36" y="204">{axisDate(valid[0].timestamp)}</text>
              {end !== start && <text x={chartWidth - 18} y="204" textAnchor="end">{axisDate(valid.at(-1)!.timestamp)}</text>}
            </svg> : <p>{t("practice.no_measurements")}</p>}
            <p className={styles.muted}>{t("practice.conditions", { count: cohort.length, seconds: selected.practice!.target_duration_sec })}</p>
          </> : <p data-testid="practice-legacy">{t("practice.legacy")}</p>}
          <p className={styles.muted}>{t(`practice.${metric}_help`)}</p>
        </section>
        <WeeklyRhythm key={JSON.stringify([selected.speaker_id, selected.learning_language])} rows={rows} selected={selected} locale={locale} />
      </div>
      {retryDraft(selected) && <section className={styles.mission}>
        <div><p className={styles.eyebrow}>{t("journey.next_step", { seconds: selected.practice!.target_duration_sec })}</p>
          <h3>{t("journey.retry_title")}</h3>
          <p>{priorities[0] || t("journey.retry_hint")}</p>
        </div>
        <button type="button" onClick={() => onRetry(selected)} data-testid="practice-retry">{t("practice.retry")}</button>
      </section>}
      <div className={styles.evidenceGrid}>
        <section className={styles.audioPanel}>
          <p className={styles.eyebrow}>{t(previous ? "journey.your_difference" : "journey.voice_today")}</p><h3>{t(previous ? "journey.listen_title" : "journey.listen_title_first")}</h3>
          <div className={styles.replay}>
            {previous && <RecordingReplay key={previous.session_id} sessionId={previous.session_id} label={t("practice.listen_before")} unavailable={t("practice.audio_unavailable")} />}
            <RecordingReplay key={selected.session_id} sessionId={selected.session_id} label={t("practice.listen_selected")} unavailable={t("practice.audio_unavailable")} />
          </div>
        </section>
        <aside className={styles.wins}>
          <p className={styles.eyebrow}>{t("journey.worth_noticing")}</p>
          <h3>{t(rhythm && rhythm.days > 1 ? "journey.showed_up" : "journey.first_step")}</h3>
          {rhythm && <p>{t("practice.activity", { attempts: rhythm.attempts, days: rhythm.days })}</p>}
          {rhythm && rhythm.retries > 0 && <><h3>{t("journey.tried_again")}</h3><p>{t("journey.retry_count", { count: rhythm.retries })}</p></>}
        </aside>
      </div>
      {selected.practice?.prompt_text && <details><summary>{t("practice.saved_prompt")}</summary><p className={styles.prompt}>{selected.practice.prompt_text}</p></details>}
      <div className={styles.attempts} role="group" aria-label={t("practice.select_attempt")}>
        {cohort.map((row) => <button type="button" key={row.session_id} disabled={!row.report_path?.trim()} aria-pressed={selected.session_id === row.session_id} onClick={() => onSelect(row.session_id)}>
          {date(row.timestamp)} · {format(row[metric])} · {row.theme}{row.practice?.retry_of_session_id ? ` · ${t("practice.repeat")}` : ""}
        </button>)}
      </div>
      <section className={styles.comparison} data-testid="practice-comparison">
        <h3>{selected.practice?.retry_of_session_id ? t("practice.retry_comparison") : t("practice.previous_comparison")}</h3>
        {previous ? <>
          <p>{date(previous.timestamp)} · {previous.theme} → {date(selected.timestamp)} · {selected.theme}</p>
          <div className={styles.tableWrap}><table>
            <thead><tr><th scope="col">{t("practice.measurement")}</th><th scope="col">{t("practice.before")}</th><th scope="col">{t("practice.selected")}</th><th scope="col">{t("practice.change")}</th></tr></thead>
            <tbody>{metrics.map((item) => {
              const before = measurement(previous[item]);
              const after = measurement(selected[item]);
              const delta = before !== null && after !== null ? Math.round((after - before) * 10) / 10 : null;
              return <tr key={item}><th scope="row">{t(`practice.${item}`)}</th><td>{format(before)}</td><td>{format(after)}</td><td>{delta !== null && delta > 0 ? "+" : ""}{format(delta)}</td></tr>;
            })}</tbody>
          </table></div>
          <button type="button" disabled={!previous.report_path?.trim()} onClick={() => onSelect(previous.session_id)}>{t("practice.open_previous")}</button>
        </> : <p>{t(selected.practice?.retry_of_session_id ? "practice.parent_unavailable" : "practice.first")}</p>}
      </section>
      {priorities.length > 0 && <aside className={styles.focus}><strong>{t("practice.focus")}</strong><p>{priorities[0]}</p><small>{t("practice.focus_help")}</small></aside>}
    </>}
    {!selected && <p>{t("practice.activity", { attempts: rows.length, days: new Set(rows.filter(row => Number.isFinite(Date.parse(row.timestamp))).map(row => new Date(row.timestamp).toDateString())).size })}</p>}
  </section>;
}
