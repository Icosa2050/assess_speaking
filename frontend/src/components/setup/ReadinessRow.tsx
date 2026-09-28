import type {
  SetupReadinessKey,
  SetupReadinessRow as SetupReadinessRowModel,
  SetupReadinessStatus,
} from "@/lib/setup/readiness";

import styles from "./SetupReadinessPanel.module.css";

type ReadinessRowProps = {
  row: SetupReadinessRowModel;
  translate: (key: string) => string;
  onAction: (key: SetupReadinessKey) => void;
};

const statusLabelKey = (status: SetupReadinessStatus): string =>
  `runtime_setup.setup_guide_status_${status}`;

const statusClass = (status: SetupReadinessStatus): string => {
  switch (status) {
    case "ready":
      return styles.statusReady;
    case "unavailable":
      return styles.statusUnavailable;
    case "loading":
      return styles.statusLoading;
    case "setup":
      return styles.statusSetup;
  }
};

export const ReadinessRow = ({
  row,
  translate,
  onAction,
}: ReadinessRowProps) => (
  <li
    className={styles.row}
    data-testid={`runtime_setup.setup_guide.${row.key}`}
    data-semantic-id={`runtime_setup.setup_guide.${row.key}`}
    data-status={row.status}
  >
    <div className={`${styles.statusDot} ${statusClass(row.status)}`} />
    <div className={styles.rowText}>
      <div className={styles.rowHeader}>
        <h3 className={styles.rowTitle}>{translate(row.titleKey)}</h3>
        <span className={styles.statusLabel}>{translate(statusLabelKey(row.status))}</span>
      </div>
      <p className={styles.rowDetail}>{translate(row.detailKey)}</p>
    </div>
    <button
      type="button"
      className={styles.rowAction}
      disabled={row.disabled}
      onClick={() => onAction(row.key)}
      data-testid={`runtime_setup.setup_guide.${row.key}.action`}
      data-semantic-id={`runtime_setup.setup_guide.${row.key}.action`}
    >
      {translate(row.actionKey)}
    </button>
  </li>
);
