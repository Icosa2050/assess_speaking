import styles from "./CompactFixList.module.css";

type Fix = { youSaid: string; better: string; why: string };
type Translate = (key: string, vars?: Record<string, string | number>) => string;

export function CompactFixList({ fixes, practiceLanguage, feedbackLanguage, translate, testId = "review-compact-fixes", headingLevel = 3 }: {
  fixes: Fix[];
  practiceLanguage: string;
  feedbackLanguage: string;
  translate: Translate;
  testId?: string;
  headingLevel?: 3 | 4;
}) {
  if (fixes.length === 0) return null;
  const Heading = headingLevel === 4 ? "h4" : "h3";
  return <section className={styles.fixes} aria-label={translate("review.compact_fixes_title")} data-testid={testId}>
    <Heading>{translate("review.compact_fixes_title")}</Heading>
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
