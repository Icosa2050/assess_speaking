import type { CSSProperties } from "react";

import styles from "./visualPrimitives.module.css";

type ProgressRingProps = {
  className?: string;
  label: string;
  max?: number;
  showStatus?: boolean;
  status: string;
  value: number;
};

const clampValue = (value: number, max: number): number => {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.min(Math.max(Math.round(value), 0), max);
};

export const ProgressRing = ({
  className,
  label,
  max = 100,
  showStatus = true,
  status,
  value,
}: ProgressRingProps) => {
  const safeMax = Number.isFinite(max) && max > 0 ? Math.round(max) : 100;
  const safeValue = clampValue(value, safeMax);
  const percent = Math.round((safeValue / safeMax) * 100);
  const style = {
    "--progress-ring-degrees": `${(safeValue / safeMax) * 360}deg`,
  } as CSSProperties;

  return (
    <div
      aria-label={label}
      aria-valuemax={safeMax}
      aria-valuemin={0}
      aria-valuenow={safeValue}
      aria-valuetext={status}
      className={[styles.progressRing, className].filter(Boolean).join(" ")}
      role="progressbar"
    >
      <div
        aria-hidden="true"
        className={styles.ringGraphic}
        style={style}
      >
        <span className={styles.ringValue}>{percent}%</span>
      </div>
      <div className={styles.ringCopy}>
        <span className={styles.ringLabel}>{label}</span>
        {showStatus ? <strong className={styles.ringStatus}>{status}</strong> : null}
      </div>
    </div>
  );
};
