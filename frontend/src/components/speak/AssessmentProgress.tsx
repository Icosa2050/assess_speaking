import { useEffect, useState } from "react";

import { StatusBanner } from "@/components/ui/StatusBanner";
import type { AssessmentStatusResponse } from "@/lib/api/types";

import styles from "./AssessmentProgress.module.css";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

const STALL_MS = 120_000;
const phases = ["upload", "transcribing", "scoring_rubric", "generating_coaching"] as const;

export function AssessmentProgress({ status, uploadStartedAt, uploadUpdatedAt, uploading, uploaded, translate }: {
  status?: AssessmentStatusResponse;
  uploadStartedAt: number | null;
  uploadUpdatedAt: number | null;
  uploading: boolean;
  uploaded: boolean;
  translate: Translate;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!uploading && status?.status !== "queued" && status?.status !== "running") return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [uploading, status?.status]);

  if (!uploading && !status) return null;
  const current = uploading ? "upload" : status?.phase === "generating_coaching" || status?.phase === "finalizing_report"
    ? "generating_coaching" : status?.phase === "scoring_rubric" ? "scoring_rubric"
      : status?.phase === "transcribing" ? "transcribing" : uploaded ? "transcribing" : "upload";
  const active = uploading || status?.status === "queued" || status?.status === "running";
  const started = uploading ? uploadStartedAt : Date.parse(status?.created_at || "");
  const updated = uploading ? uploadUpdatedAt ?? uploadStartedAt : Date.parse(status?.updated_at || status?.created_at || "");
  const phaseStarted = uploading ? uploadStartedAt : Date.parse(status?.phase_started_at || status?.updated_at || status?.created_at || "");
  const elapsed = Number.isFinite(started) && started !== null ? Math.max(0, Math.floor((now - started) / 1000)) : null;
  const sinceUpdate = Number.isFinite(updated) && updated !== null ? Math.max(0, Math.floor((now - updated) / 1000)) : null;
  const phaseElapsed = Number.isFinite(phaseStarted) && phaseStarted !== null ? Math.max(0, Math.floor((now - phaseStarted) / 1000)) : null;
  const stalled = active && sinceUpdate !== null && sinceUpdate * 1000 >= STALL_MS;
  const format = (seconds: number) => seconds < 60
    ? translate("speak.wait_seconds", { count: seconds })
    : translate("speak.wait_minutes_seconds", { minutes: Math.floor(seconds / 60), seconds: seconds % 60 });

  return <div className={styles.progress} data-testid="speak.assessment_progress" data-stalled={stalled}>
    <ol className={styles.phases} aria-label={translate("speak.wait_phases")}>
      {phases.map((phase, index) => {
        const currentIndex = phases.indexOf(current);
        const done = index < currentIndex || (!active && status?.status === "completed");
        return <li key={phase} className={done ? styles.done : phase === current && active ? styles.current : ""}
          aria-current={phase === current && active ? "step" : undefined}>
          {translate(`speak.wait_${phase}`)}
        </li>;
      })}
    </ol>
    {active && <>
      <p className={styles.announcement} aria-live="polite" aria-atomic="true">
        {translate(stalled ? "speak.wait_stalled_announcement" : "speak.wait_phase_announcement", { phase: translate(`speak.wait_${current}`) })}
      </p>
      <p className={styles.time} aria-hidden="true" data-testid="speak.wait_time">
        {elapsed !== null && translate("speak.wait_elapsed", { duration: format(elapsed) })}
        {sinceUpdate !== null && ` · ${translate("speak.wait_last_update", { duration: format(sinceUpdate) })}`}
        {phaseElapsed !== null && ` · ${translate("speak.wait_phase_elapsed", { duration: format(phaseElapsed) })}`}
      </p>
      {stalled && <StatusBanner title={translate("speak.wait_stalled_title")} tone="warning">
        {translate(uploaded ? "speak.wait_stalled_body" : "speak.wait_stalled_upload_body")}
      </StatusBanner>}
    </>}
  </div>;
}
