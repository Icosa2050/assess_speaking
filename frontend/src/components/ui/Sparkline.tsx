import styles from "./visualPrimitives.module.css";

type SparklineProps = {
  label: string;
  summary: string;
  testId?: string;
  values: number[];
};

const sparklinePoints = (values: number[]): string => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 280;
      const y = 64 - ((value - min) / range) * 52;
      return `${x},${y}`;
    })
    .join(" ");
};

export const Sparkline = ({
  label,
  summary,
  testId,
  values,
}: SparklineProps) => {
  if (values.length === 0) {
    return null;
  }

  if (values.length === 1) {
    return (
      <div className={styles.sparkline} data-testid={testId}>
        <strong className={styles.sparklineSummary} data-testid={testId ? `${testId}-summary` : undefined}>
          {summary}
        </strong>
      </div>
    );
  }

  return (
    <div className={styles.sparkline} data-testid={testId}>
      <svg
        aria-label={label}
        className={styles.sparklineGraphic}
        data-testid={testId ? `${testId}-graphic` : undefined}
        height="72"
        role="img"
        viewBox="0 0 280 72"
        width="100%"
      >
        <polyline
          className={styles.sparklineLine}
          points={sparklinePoints(values)}
        />
      </svg>
      <strong className={styles.sparklineSummary} data-testid={testId ? `${testId}-summary` : undefined}>
        {summary}
      </strong>
    </div>
  );
};
