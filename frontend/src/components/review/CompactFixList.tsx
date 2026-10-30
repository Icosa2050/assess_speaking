import styles from "./CompactFixList.module.css";

type Fix = { youSaid: string; better: string; why: string };
type Translate = (key: string, vars?: Record<string, string | number>) => string;

export function CompactFixList({ fixes, practiceLanguage, feedbackLanguage, translate, testId = "review-compact-fixes" }: {
  fixes: Fix[];
  practiceLanguage: string;
  feedbackLanguage: string;
  translate: Translate;
  testId?: string;
}) {
  if (fixes.length === 0) return null;
  return <section className={styles.fixes} aria-label={translate("review.compact_fixes_title")} data-testid={testId}>
    <h3>{translate("review.compact_fixes_title")}</h3>
    <ol>
      {fixes.slice(0, 3).map((fix, index) => <li key={`${fix.youSaid}-${index}`}>
        <div className={styles.pair}>
          <div><strong>{translate("review.compact_you_said")}</strong><p lang={practiceLanguage}>{fix.youSaid}</p></div>
          <div><strong>{translate("review.compact_better")}</strong><p lang={practiceLanguage}>{fix.better}</p></div>
        </div>
        <p className={styles.why} lang={feedbackLanguage}>{fix.why}</p>
      </li>)}
    </ol>
  </section>;
}
