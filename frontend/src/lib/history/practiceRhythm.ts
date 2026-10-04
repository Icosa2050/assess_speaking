import type { HistoryRow } from "@/lib/api/types";

/** Calendar-day keys use the learner's browser timezone, including across DST changes. */
export function localDayKey(date: Date): string {
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

export function practiceRhythm(rows: HistoryRow[], speaker: string, language: string, now = new Date()) {
  const monday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  monday.setDate(monday.getDate() - (monday.getDay() + 6) % 7);
  const personal = rows.filter(row => row.speaker_id === speaker && row.learning_language === language &&
    !row.practice?.dry_run && Number.isFinite(Date.parse(row.timestamp)) && Date.parse(row.timestamp) <= now.getTime());
  const days = new Set(personal.map(row => localDayKey(new Date(row.timestamp))));
  const week = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(monday);
    date.setDate(date.getDate() + index);
    return { date, completed: days.has(localDayKey(date)) };
  });
  return {
    week, completed: week.filter(day => day.completed).length,
    attempts: personal.length, days: days.size,
    retries: personal.filter(row => Boolean(row.practice?.retry_of_session_id)).length,
  };
}
