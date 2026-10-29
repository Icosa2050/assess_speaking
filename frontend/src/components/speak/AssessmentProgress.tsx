import { useEffect, useState } from "react";

import { StatusBanner } from "@/components/ui/StatusBanner";
import type { AssessmentStatusResponse } from "@/lib/api/types";
import { pollingIntervals } from "@/lib/query/queryClient";

import styles from "./AssessmentProgress.module.css";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

const phases = ["upload", "transcribing", "scoring_rubric", "generating_coaching"] as const;
const UPLOAD_QUIET_MS = 120_000;

export function AssessmentProgress({ status, uploadStartedAt, uploadUpdatedAt, uploading, uploaded, pollFailed = false, translate }: {
  status?: AssessmentStatusResponse;
  uploadStartedAt: number | null;
  uploadUpdatedAt: number | null;
  uploading: boolean;
  uploaded: boolean;
  pollFailed?: boolean;
  translate: Translate;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!uploading && status?.status !== "queued" && status?.status !== "running") return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [uploading, status?.status]);

  if (!uploading && !status) return null;
  const current = uploading ? "upload" : status?.phase === "transcribing" || status?.phase === "scoring_rubric" || status?.phase === "generating_coaching"
    ? status.phase : null;
  const specialPhase = status?.phase === "analyzing_audio" ? "speak.wait_preparing_audio"
    : status?.phase === "finalizing_report" ? "speak.wait_finalizing"
      : "speak.wait_queued";
  const completedThrough = uploading ? 0 : status?.phase === "finalizing_report" ? 3
    : current ? phases.indexOf(current) : uploaded ? 1 : 0;
  const active = uploading || status?.status === "queued" || status?.status === "running";
  const started = uploading ? uploadStartedAt : Date.parse(status?.created_at || "");
  const updated = uploading ? uploadUpdatedAt ?? uploadStartedAt : Date.parse(status?.updated_at || status?.created_at || "");
  const workerSeen = Date.parse(status?.worker_seen_at || "");
  const elapsed = Number.isFinite(started) && started !== null ? Math.max(0, Math.floor((now - started) / 1000)) : null;
  const sinceUpdate = Number.isFinite(updated) && updated !== null ? Math.max(0, Math.floor((now - updated) / 1000)) : null;
  const sinceWorkerSeen = Number.isFinite(workerSeen) ? Math.max(0, Math.floor((now - workerSeen) / 1000)) : null;
  const workerRecentlyConfirmed = sinceWorkerSeen !== null && sinceWorkerSeen * 1000 <= pollingIntervals.assessmentMs * 3;
  const stalled = uploading
    ? sinceUpdate !== null && sinceUpdate * 1000 >= UPLOAD_QUIET_MS
    : active && !pollFailed && sinceUpdate !== null && status?.quiet_warning_after_sec != null &&
      sinceUpdate >= status.quiet_warning_after_sec && workerRecentlyConfirmed;
  const format = (seconds: number) => seconds < 60
    ? translate("speak.wait_seconds", { count: seconds })
    : seconds % 60 === 0 ? translate("speak.wait_minutes", { count: seconds / 60 })
      : translate("speak.wait_minutes_seconds", { minutes: Math.floor(seconds / 60), seconds: seconds % 60 });

  return <div className={styles.progress} data-testid="speak.assessment_progress" data-stalled={stalled}>
    <ol className={styles.phases} aria-label={translate("speak.wait_phases")}>
      {phases.map((phase, index) => {
        const done = index < completedThrough || (!active && status?.status === "completed");
        return <li key={phase} className={done ? styles.done : phase === current && active ? styles.current : ""}
          aria-current={phase === current && active ? "step" : undefined}>
          {done && <span aria-hidden="true">✓ </span>}{translate(`speak.wait_${phase}`)}
          {done && <span className={styles.visuallyHidden}> {translate("speak.wait_completed")}</span>}
        </li>;
      })}
    </ol>
    {active && <>
      <p className={styles.announcement} aria-live="polite" aria-atomic="true">
        {current ? translate("speak.wait_phase_announcement", { phase: translate(`speak.wait_${current}`) }) : translate(specialPhase)}
      </p>
      <p className={styles.time} role="timer" aria-live="off" data-testid="speak.wait_time">
        {elapsed !== null && translate("speak.wait_elapsed", { duration: format(elapsed) })}
        {sinceUpdate !== null && ` · ${translate("speak.wait_last_update", { duration: format(sinceUpdate) })}`}
        {sinceWorkerSeen !== null && ` · ${translate("speak.wait_worker_seen", { duration: format(sinceWorkerSeen) })}`}
      </p>
      {stalled && <StatusBanner title={translate("speak.wait_stalled_title")} tone="warning">
        {translate(uploaded ? "speak.wait_stalled_body" : "speak.wait_stalled_upload_body")}
      </StatusBanner>}
    </>}
  </div>;
}
