import type {
  SetupReadinessKey,
  SetupReadinessRow as SetupReadinessRowModel,
} from "@/lib/setup/readiness";
import { isBlockingSetupReadinessRow, isCoreSetupReadinessKey } from "@/lib/setup/readiness";
import { Icon } from "@/components/ui/Icon";
import { ProgressRing } from "@/components/ui/ProgressRing";

import { ReadinessRow } from "./ReadinessRow";
import styles from "./SetupReadinessPanel.module.css";

type SetupReadinessPanelProps = {
  onAction: (key: SetupReadinessKey) => void;
  rows: SetupReadinessRowModel[];
  translate: (key: string, vars?: Record<string, string | number>) => string;
};

export const SetupReadinessPanel = ({
  onAction,
  rows,
  translate,
}: SetupReadinessPanelProps) => {
  const progressRows = rows.filter((row) => isCoreSetupReadinessKey(row.key) && isBlockingSetupReadinessRow(row));
  const hasOptionalSetup = rows.some((row) => !isBlockingSetupReadinessRow(row) && row.status !== "ready");
  const totalCount = progressRows.length;
  const readyCount = progressRows.filter((row) => row.status === "ready").length;
  const hasRows = totalCount > 0;
  const allReady = hasRows && readyCount === totalCount;
  const progressLabel = hasRows
    ? translate("runtime_setup.setup_guide_progress_label", {
        ready: readyCount,
        total: totalCount,
      })
    : translate("runtime_setup.setup_guide_progress_empty");
  const progressStatus = hasRows
    ? translate("runtime_setup.setup_guide_progress_status", {
        ready: readyCount,
        total: totalCount,
      })
    : translate("runtime_setup.setup_guide_progress_empty");
  const unavailableCount = progressRows.filter((row) => row.status === "unavailable").length;

  return (
    <section
      className={styles.panel}
      data-testid="runtime_setup.setup_guide"
      data-semantic-id="runtime_setup.setup_guide"
    >
      <div className={styles.anchor}>
        <div className={`${styles.anchorIcon} ${allReady ? styles.anchorIconReady : ""}`}>
          <Icon
            name={allReady ? "check" : "headphones"}
            size={34}
          />
        </div>
        <div className={styles.header}>
          <h2 className={styles.title}>{translate("runtime_setup.setup_guide_title")}</h2>
          <p className={styles.body}>{translate("runtime_setup.setup_guide_body")}</p>
        </div>
        <ProgressRing
          className={`${styles.progress} ${allReady ? styles.progressReady : ""}`}
          label={progressLabel}
          max={hasRows ? totalCount : 1}
          status={progressStatus}
          value={readyCount}
        />
        <div className={styles.anchorCopy}>
          <p className={styles.anchorTitle}>{translate(allReady ? "runtime_setup.setup_guide_anchor_title" : "first_run.guide_anchor_pending")}</p>
          <p className={styles.anchorBody}>{translate(allReady
            ? hasOptionalSetup ? "first_run.guide_anchor_required_ready_body" : "first_run.guide_anchor_ready_body"
            : "first_run.guide_anchor_body")}</p>
        </div>
      </div>
      {unavailableCount > 0 ? (
        <p role="status">{translate("first_run.checks_need_attention", { count: unavailableCount })}</p>
      ) : null}
      <ol className={styles.rows}>
        {rows.map((row) => (
          <ReadinessRow
            key={row.key}
            row={row}
            translate={translate}
            onAction={onAction}
          />
        ))}
      </ol>
    </section>
  );
};
