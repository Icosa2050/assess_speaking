import { Icon } from "@/components/ui/Icon";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

export type HistoryProgressStoryRecord = {
  finalScore: number | null;
  languageLabel: string;
  scoreLabel: string;
  sessionId: string;
  taskFamily: string;
  theme: string;
  timestamp: string;
  timestampLabel: string;
  topPriorities: string[];
  wpm: number | null;
};

type DeltaState =
  | { kind: "single"; labelValue: null }
  | { kind: "steady"; labelValue: null }
  | { kind: "up"; labelValue: string }
  | { kind: "down"; labelValue: string };

export type HistoryProgressStoryModel = {
  latest: HistoryProgressStoryRecord;
  nextPractice: { focus: string | null; theme: string } | null;
  noLongerFlagged: string | null;
  observedFocus: string | null;
  paceDelta: DeltaState;
  previous: HistoryProgressStoryRecord | null;
  scoreDelta: DeltaState;
};

const SCORE_DELTA_THRESHOLD = 0.05;
const WPM_DELTA_THRESHOLD = 5;

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  background:
    "linear-gradient(135deg, rgba(215, 235, 229, 0.92), rgba(255, 255, 255, 0.96) 58%, rgba(240, 236, 249, 0.88))",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.06)",
} as const;

const storyGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))",
} as const;

const scoreAnchorStyle = {
  display: "grid",
  alignContent: "space-between",
  gap: "0.55rem",
  minHeight: "7.5rem",
  padding: "0.85rem",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.74)",
  border: "1px solid rgba(18, 61, 55, 0.10)",
} as const;

const chipStyle = {
  display: "grid",
  gap: "0.45rem",
  minWidth: 0,
  padding: "0.75rem",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.72)",
  border: "1px solid rgba(18, 61, 55, 0.10)",
} as const;

const chipLabelStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: "0.35rem",
  color: "#33514b",
  fontSize: "0.8125rem",
  fontWeight: 750,
} as const;

const chipValueStyle = {
  margin: 0,
  color: "#10201c",
  fontWeight: 750,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} as const;

const normalizeComparableText = (value: string): string => value.trim().toLowerCase();

const practiceThemeLabel = (record: HistoryProgressStoryRecord): string => {
  const theme = record.theme.trim();
  if (theme) {
    return theme;
  }

  const taskFamily = record.taskFamily.trim();
  if (taskFamily) {
    return taskFamily.replaceAll("_", " ");
  }

  return record.languageLabel;
};

const timestampValue = (value: string): number => {
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
};

const deriveDelta = (
  latest: number | null,
  previous: number | null,
  threshold: number,
): DeltaState => {
  if (latest === null || previous === null) {
    return { kind: "single", labelValue: null };
  }

  const delta = latest - previous;

  if (Math.abs(delta) < threshold) {
    return { kind: "steady", labelValue: null };
  }

  return {
    kind: delta > 0 ? "up" : "down",
    labelValue: Math.abs(delta).toFixed(1),
  };
};

export const deriveHistoryProgressStory = (
  records: HistoryProgressStoryRecord[],
): HistoryProgressStoryModel | null => {
  if (records.length === 0) {
    return null;
  }

  const sortedRecords = records
    .map((record, index) => ({ index, record, sortTime: timestampValue(record.timestamp) }))
    .sort((left, right) => {
      if (right.sortTime !== left.sortTime) {
        return right.sortTime - left.sortTime;
      }

      return left.index - right.index;
    })
    .map(({ record }) => record);

  const latest = sortedRecords[0];
  const previous = sortedRecords[1] ?? null;
  const comparable =
    previous !== null &&
    latest.taskFamily === previous.taskFamily &&
    normalizeComparableText(latest.theme) === normalizeComparableText(previous.theme);
  const noLongerFlagged = comparable
    ? (previous.topPriorities.find((priority) => !latest.topPriorities.includes(priority)) ?? null)
    : null;

  return {
    latest,
    nextPractice: previous
      ? {
          focus: latest.topPriorities[0] ?? null,
          theme: practiceThemeLabel(latest),
        }
      : null,
    noLongerFlagged,
    observedFocus: latest.topPriorities[0] ?? null,
    paceDelta: previous
      ? deriveDelta(latest.wpm, previous.wpm, WPM_DELTA_THRESHOLD)
      : { kind: "single", labelValue: null },
    previous,
    scoreDelta: previous
      ? deriveDelta(latest.finalScore, previous.finalScore, SCORE_DELTA_THRESHOLD)
      : { kind: "single", labelValue: null },
  };
};

const scoreDeltaText = (delta: DeltaState, translate: Translate): string => {
  if (delta.kind === "single") {
    return translate("history.story_score_single");
  }

  if (delta.kind === "steady") {
    return translate("history.story_score_steady");
  }

  return translate(delta.kind === "up" ? "history.story_score_up" : "history.story_score_down", {
    value: delta.labelValue,
  });
};

const paceDeltaText = (delta: DeltaState, translate: Translate): string => {
  if (delta.kind === "single") {
    return translate("history.story_pace_single");
  }

  if (delta.kind === "steady") {
    return translate("history.story_pace_steady");
  }

  return translate(delta.kind === "up" ? "history.story_pace_up" : "history.story_pace_down", {
    value: delta.labelValue,
  });
};

export const HistoryProgressStory = ({
  records,
  translate,
}: {
  records: HistoryProgressStoryRecord[];
  translate: Translate;
}) => {
  const story = deriveHistoryProgressStory(records);

  if (!story) {
    return null;
  }

  const latestContext = translate("history.story_latest_context", {
    language: story.latest.languageLabel,
    theme: story.latest.theme || "-",
    timestamp: story.latest.timestampLabel || "-",
  });

  return (
    <section
      aria-label={translate("history.story_title")}
      data-semantic-id="history-progress-story"
      data-testid="history-progress-story"
      role="region"
      style={cardStyle}
    >
      <div style={{ display: "grid", gap: "0.35rem" }}>
        <h2 style={{ margin: 0, color: "#10201c", fontSize: "1.35rem" }}>
          {translate("history.story_title")}
        </h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.55 }}>
          {translate("history.story_body")}
        </p>
        <p style={{ margin: 0, color: "#33514b", fontSize: "0.875rem", fontWeight: 650 }}>
          {latestContext}
        </p>
      </div>

      <div style={storyGridStyle}>
        <div
          data-semantic-id="history-progress-story-score"
          data-testid="history-progress-story-score"
          style={scoreAnchorStyle}
        >
          <span style={chipLabelStyle}>
            <Icon name="history" size={17} />
            {translate("history.story_score_label")}
          </span>
          <strong style={{ color: "#10201c", fontSize: "2.2rem", lineHeight: 1 }}>
            {story.latest.scoreLabel}
          </strong>
          <p style={{ margin: 0, color: "#33514b", fontWeight: 700 }}>
            {scoreDeltaText(story.scoreDelta, translate)}
          </p>
        </div>

        <div
          data-semantic-id="history-progress-story-pace"
          data-testid="history-progress-story-pace"
          style={chipStyle}
        >
          <span style={chipLabelStyle}>
            <Icon name="play" size={17} />
            {translate("history.story_pace_label")}
          </span>
          <p style={chipValueStyle}>{paceDeltaText(story.paceDelta, translate)}</p>
        </div>

        <div
          data-semantic-id="history-progress-story-observed-focus"
          data-testid="history-progress-story-observed-focus"
          style={chipStyle}
        >
          <span style={chipLabelStyle}>
            <Icon name="target" size={17} />
            {translate("history.story_observed_focus_label")}
          </span>
          <p style={chipValueStyle}>
            {story.observedFocus ?? translate("history.story_no_focus")}
          </p>
        </div>

        {story.noLongerFlagged ? (
          <div
            data-semantic-id="history-progress-story-no-longer-flagged"
            data-testid="history-progress-story-no-longer-flagged"
            style={chipStyle}
          >
            <span style={chipLabelStyle}>
              <Icon name="check" size={17} />
              {translate("history.story_no_longer_flagged_label")}
            </span>
            <p style={chipValueStyle}>{story.noLongerFlagged}</p>
          </div>
        ) : null}

        {story.nextPractice ? (
          <div
            data-semantic-id="history-progress-story-next-practice"
            data-testid="history-progress-story-next-practice"
            style={{ ...chipStyle, gridColumn: "1 / -1" }}
          >
            <span style={chipLabelStyle}>
              <Icon name="play" size={17} />
              {translate("history.story_next_practice_label")}
            </span>
            <p style={chipValueStyle}>
              {story.nextPractice.focus
                ? translate("history.story_next_practice_focus", {
                    focus: story.nextPractice.focus,
                    theme: story.nextPractice.theme,
                  })
                : translate("history.story_next_practice_repeat", {
                    theme: story.nextPractice.theme,
                  })}
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
};
