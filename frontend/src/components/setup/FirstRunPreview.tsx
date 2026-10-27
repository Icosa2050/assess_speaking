import { Link } from "react-router-dom";

import { Card } from "@/components/ui/Card";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";

import styles from "./FirstRunPreview.module.css";

export function FirstRunPreview({ locale }: { locale: UiLocale }) {
  const translate = createTranslator(locale);

  return (
    <div className={styles.grid} data-testid="first-run.preview">
      <Card as="section" subtle className={styles.preview} aria-labelledby="first-run-prompt-label first-run-prompt-title">
        <p id="first-run-prompt-label" className={styles.eyebrow}>{translate("first_run.prompt_eyebrow")}</p>
        <h2 id="first-run-prompt-title">{translate("first_run.prompt_title")}</h2>
        <p>{translate("first_run.prompt_body")}</p>
        <p className={styles.meta}>{translate("first_run.prompt_meta")}</p>
      </Card>
      <Card as="section" className={styles.preview} aria-labelledby="first-run-sample-label first-run-report-title" aria-describedby="first-run-sample-notice">
        <p id="first-run-sample-label" className={styles.sampleLabel}>{translate("first_run.sample_label")}</p>
        <h2 id="first-run-report-title">{translate("first_run.report_title")}</h2>
        <p>{translate("first_run.report_feedback")}</p>
        <p id="first-run-sample-notice" className={styles.note}>{translate("first_run.sample_notice")}</p>
        <Link to="/guide">{translate("first_run.report_guide")}</Link>
      </Card>
    </div>
  );
}
