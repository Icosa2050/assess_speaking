import { Link } from "react-router-dom";

import { Card, Disclosure } from "@/components/ui/Card";
import type { ReviewDisplaySummary } from "./ReviewSummary";

import styles from "./ReviewResult.module.css";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

export function ReviewResult({ summary, translate }: { summary: ReviewDisplaySummary; translate: Translate }) {
  const hasEstimate = summary.scoreOverall !== null;
  const score = hasEstimate ? summary.scoreOverall!.toFixed(1) : "";
  const speechClarity = summary.dimensions.find((item) => item.key === "pronunciation_intelligibility");

  return <Card as="section" className={styles.result} data-testid="review-next-step-score" data-semantic-id="review-next-step-score">
    <div className={styles.heading}>
      <h2>{translate("review.result.estimate")}</h2>
      {hasEstimate && <span className={styles.provisional}>{translate("review.result.provisional")}</span>}
    </div>
    {hasEstimate ? <div className={styles.estimate}>
      <p className={styles.number} data-testid="review-estimate-text">
        {score}<span>/5</span>
      </p>
      <div className={styles.scaleCopy}>
        <strong>{translate("review.result.scale")}</strong>
        <ol className={styles.scale} aria-hidden="true">
          {[1, 2, 3, 4, 5].map((tick) => <li key={tick} className={tick <= Math.floor(summary.scoreOverall!) ? styles.filled : ""}>{tick}</li>)}
        </ol>
        <p>{translate("review.result.score_meaning")}</p>
      </div>
    </div> : <p className={styles.noEstimate}>{translate("review.result.no_estimate")}</p>}
    <p className={styles.context}>
      {hasEstimate
        ? translate("review.result.estimate_context", { score })
        : translate("review.result.no_estimate_context")}
    </p>
    {summary.cefrLevel && <p className={`${styles.cefr} ${styles[`cefr${summary.cefrLevel}`]}`} data-testid="review-provisional-cefr">
      {translate("review.result.cefr", { level: summary.cefrLevel })}
    </p>}
    <div className={styles.evidence}>
      <div>
        <strong>{translate("review.result.method_title")}</strong>
        <p>{translate(!hasEstimate ? "review.result.no_estimate_context"
          : summary.mode === "hybrid" ? "review.result.method_hybrid" : "review.result.method_rules")}</p>
      </div>
      <div>
        <strong>{translate("review.result.uncertainty_title")}</strong>
        <p>{translate("review.result.uncertainty_body")}</p>
      </div>
      <div>
        <strong>{translate("review.result.clarity_title")}</strong>
        <p>{speechClarity && hasEstimate
          ? translate("review.result.clarity_score", { score: speechClarity.score.toFixed(1) })
          : translate("review.result.clarity_unavailable")}</p>
        <p>{translate("review.result.clarity_limit")}</p>
      </div>
    </div>
    {summary.dimensions.length > 0 && hasEstimate && <Disclosure title={translate("review.result.dimensions_title")}>
      <ul className={styles.dimensions}>
        {summary.dimensions.map((item) => <li key={item.key}>
          <span>{item.key === "pronunciation_intelligibility"
            ? translate("review.result.clarity_title")
            : translate(`guide.dimension_${item.key}`)}</span>
          <strong>{item.score.toFixed(1)}/5</strong>
          <span>{translate("review.result.confidence", {
            value: translate(["low", "medium", "high"].includes(item.confidence)
              ? `review.result.confidence_${item.confidence}` : "review.result.confidence_unavailable"),
          })}</span>
        </li>)}
      </ul>
    </Disclosure>}
    <Disclosure title={translate("review.result.how_calculated")}>
      <p>{translate("review.result.method_detail")}</p>
    </Disclosure>
    <Link to="/guide">{translate("review.result.guide_link")}</Link>
  </Card>;
}
