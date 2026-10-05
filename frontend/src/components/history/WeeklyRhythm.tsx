import { useEffect, useState, type CSSProperties } from "react";
import type { HistoryRow } from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { practiceRhythm } from "@/lib/history/practiceRhythm";
import styles from "./PracticeProgress.module.css";

const validGoal = (value: unknown) => {
  const number = Number(value);
  return Number.isInteger(number) && number >= 1 && number <= 7 ? number : 4;
};

export function WeeklyRhythm({ rows, selected, locale }: { rows: HistoryRow[]; selected: HistoryRow; locale: string }) {
  const t = createTranslator(locale);
  const [, refreshCalendar] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => refreshCalendar(value => value + 1), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  const key = `vostavo:weekly-goal:${JSON.stringify([selected.speaker_id, selected.learning_language])}`;
  const [goal, setGoal] = useState(() => {
    try { return validGoal(localStorage.getItem(key)); } catch { return 4; }
  });
  const rhythm = practiceRhythm(rows, selected.speaker_id, selected.learning_language);
  const dayCount = (count: number) => t(count === 1 ? "journey.goal_day" : "journey.goal_days", { count });
  const ratio = Math.min(rhythm.completed / goal, 1);
  return <aside className={styles.rhythm} data-testid="practice-rhythm">
    <p className={styles.eyebrow}>{t("journey.little_often")}</p>
    <h3>{t("journey.weekly_rhythm")}</h3>
    <div className={styles.ring} style={{ "--completion": `${ratio * 360}deg` } as CSSProperties}
      role="img" aria-label={t("journey.weekly_count", { days: dayCount(rhythm.completed), goal: dayCount(goal) })}>
      <div><strong>{rhythm.completed >= goal ? rhythm.completed : t("journey.ring_count", { count: rhythm.completed, goal })}</strong><span>{t(rhythm.completed === 1 && goal === 1 ? "journey.practice_day" : "journey.practice_days")}</span></div>
    </div>
    <strong>{t(rhythm.completed >= goal ? "journey.goal_met" : rhythm.completed === 0 ? "journey.fresh_start" : "journey.keep_rhythm")}</strong>
    <div className={styles.days}>
      {rhythm.week.map(day => <span key={day.date.getTime()} data-completed={day.completed}>
        <span>{new Intl.DateTimeFormat(locale, { weekday: "narrow" }).format(day.date)}</span>
        <span role="img" aria-label={`${new Intl.DateTimeFormat(locale, { dateStyle: "full" }).format(day.date)}: ${t(day.completed ? "journey.practised" : "journey.no_practice")}`}>{day.completed ? "✓" : "·"}</span>
      </span>)}
    </div>
    <label className={styles.goalPicker}>{t("journey.weekly_goal")}
      <select value={goal} onChange={event => {
        const next = validGoal(event.target.value);
        setGoal(next);
        try { localStorage.setItem(key, String(next)); } catch { /* The goal still works for this visit. */ }
      }}>
        {[1, 2, 3, 4, 5, 6, 7].map(value => <option key={value} value={value}>{dayCount(value)}</option>)}
      </select>
    </label>
    <p className={styles.muted}>{t("journey.your_pace")}</p>
  </aside>;
}
