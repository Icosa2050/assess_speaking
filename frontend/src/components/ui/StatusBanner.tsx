import type { ReactNode } from "react";

import styles from "./controls.module.css";

export function StatusBanner({ title, children, action, tone = "info", urgent = false, className = "" }: {
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  tone?: "info" | "success" | "warning" | "error";
  urgent?: boolean;
  className?: string;
}) {
  return <div role={urgent ? "alert" : "status"} className={`${styles.banner} ${styles[tone]} ${className}`.trim()}>
    <span aria-hidden="true" className={styles.bannerGlyph} />
    <div className={styles.bannerContent}>
      <strong>{title}</strong>
      {children && <div>{children}</div>}
      {action && <div className={styles.bannerAction}>{action}</div>}
    </div>
  </div>;
}
