import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  HistoryProgressStory,
  deriveHistoryProgressStory,
  type HistoryProgressStoryRecord,
} from "@/components/history/HistoryProgressStory";

const translations: Record<string, string> = {
  "history.story_title": "Your progress story",
  "history.story_body": "A quick read on the latest saved attempt.",
  "history.story_latest_context": "{language} · {theme} · {timestamp}",
  "history.story_score_label": "Latest score",
  "history.story_score_up": "+{value} since the previous attempt",
  "history.story_score_down": "{value} since the previous attempt",
  "history.story_score_steady": "Score held steady",
  "history.story_score_single": "First saved attempt for this filter",
  "history.story_pace_label": "Speaking pace",
  "history.story_pace_up": "+{value} WPM since the previous attempt",
  "history.story_pace_down": "{value} WPM since the previous attempt",
  "history.story_pace_steady": "Similar pace to the previous attempt",
  "history.story_pace_single": "Add another attempt to compare pace",
  "history.story_observed_focus_label": "Noticed last time",
  "history.story_no_focus": "No focus area was flagged last time",
  "history.story_no_longer_flagged_label": "No longer flagged",
  "history.story_next_practice_label": "Next practice",
  "history.story_next_practice_focus": "Try {theme} again with this focus: {focus}",
  "history.story_next_practice_repeat": "Repeat {theme} once more and compare the next score.",
};

const translate = (key: string, vars?: Record<string, string | number>) => {
  const template = translations[key] ?? `[${key}]`;

  return Object.entries(vars ?? {}).reduce(
    (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
    template,
  );
};

const record = (overrides: Partial<HistoryProgressStoryRecord>): HistoryProgressStoryRecord => ({
  finalScore: 4,
  languageLabel: "Italian",
  scoreLabel: "4.0",
  sessionId: "session",
  taskFamily: "travel_narrative",
  theme: "Travel story",
  timestamp: "2026-06-06T12:00:00Z",
  timestampLabel: "Jun 6, 2026, 12:00 PM",
  topPriorities: ["Add one clearer closing sentence"],
  wpm: 110,
  ...overrides,
});

describe("HistoryProgressStory", () => {
  it("derives the latest attempt by timestamp instead of incoming row order", () => {
    const story = deriveHistoryProgressStory([
      record({
        finalScore: 3.8,
        scoreLabel: "3.8",
        sessionId: "older",
        timestamp: "2026-06-06T12:00:00Z",
        timestampLabel: "Jun 6, 2026, 12:00 PM",
        wpm: 104,
      }),
      record({
        finalScore: 4.2,
        scoreLabel: "4.2",
        sessionId: "newer",
        timestamp: "2026-06-07T12:00:00Z",
        timestampLabel: "Jun 7, 2026, 12:00 PM",
        topPriorities: ["Tighten the ending"],
        wpm: 116,
      }),
    ]);

    expect(story?.latest.sessionId).toBe("newer");
    expect(story?.previous?.sessionId).toBe("older");
    expect(story?.scoreDelta).toMatchObject({ kind: "up", labelValue: "0.4" });
    expect(story?.paceDelta).toMatchObject({ kind: "up", labelValue: "12.0" });
  });

  it("keeps score and pace neutral below meaningful thresholds", () => {
    const story = deriveHistoryProgressStory([
      record({ finalScore: 4.02, sessionId: "newer", timestamp: "2026-06-07T12:00:00Z", wpm: 113 }),
      record({ finalScore: 4, sessionId: "older", timestamp: "2026-06-06T12:00:00Z", wpm: 110 }),
    ]);

    expect(story?.scoreDelta.kind).toBe("steady");
    expect(story?.paceDelta.kind).toBe("steady");
  });

  it("shows a no-longer-flagged priority only for comparable attempts", () => {
    const comparable = deriveHistoryProgressStory([
      record({
        sessionId: "newer",
        timestamp: "2026-06-07T12:00:00Z",
        topPriorities: ["Tighten the ending"],
      }),
      record({
        sessionId: "older",
        timestamp: "2026-06-06T12:00:00Z",
        topPriorities: ["Add one clearer closing sentence", "Tighten the ending"],
      }),
    ]);

    const themeMismatch = deriveHistoryProgressStory([
      record({
        sessionId: "newer",
        theme: "Work opinion",
        timestamp: "2026-06-07T12:00:00Z",
        topPriorities: ["Tighten the ending"],
      }),
      record({
        sessionId: "older",
        theme: "Travel story",
        timestamp: "2026-06-06T12:00:00Z",
        topPriorities: ["Add one clearer closing sentence", "Tighten the ending"],
      }),
    ]);

    expect(comparable?.noLongerFlagged).toBe("Add one clearer closing sentence");
    expect(themeMismatch?.noLongerFlagged).toBeNull();
  });

  it("renders a stable story for a single attempt without overclaiming improvement", () => {
    render(
      <HistoryProgressStory
        records={[record({ sessionId: "only" })]}
        translate={translate}
      />,
    );

    expect(screen.getByRole("region", { name: "Your progress story" })).toBeVisible();
    expect(screen.getByTestId("history-progress-story-score")).toHaveTextContent("4.0");
    expect(screen.getByTestId("history-progress-story-score")).toHaveTextContent(
      "First saved attempt for this filter",
    );
    expect(screen.getByTestId("history-progress-story-observed-focus")).toHaveTextContent(
      "Add one clearer closing sentence",
    );
    expect(screen.queryByTestId("history-progress-story-no-longer-flagged")).not.toBeInTheDocument();
    expect(screen.queryByTestId("history-progress-story-next-practice")).not.toBeInTheDocument();
  });

  it("turns a comparable latest focus into a next-practice cue", () => {
    render(
      <HistoryProgressStory
        records={[
          record({
            finalScore: 3.8,
            scoreLabel: "3.8",
            sessionId: "older",
            timestamp: "2026-06-06T12:00:00Z",
            timestampLabel: "Jun 6, 2026, 12:00 PM",
            topPriorities: ["Add one clearer closing sentence"],
          }),
          record({
            finalScore: 4.2,
            scoreLabel: "4.2",
            sessionId: "newer",
            timestamp: "2026-06-07T12:00:00Z",
            timestampLabel: "Jun 7, 2026, 12:00 PM",
            topPriorities: ["Tighten the ending"],
          }),
        ]}
        translate={translate}
      />,
    );

    expect(screen.getByTestId("history-progress-story-score")).toBeVisible();
    expect(screen.getByTestId("history-progress-story-pace")).toBeVisible();
    expect(screen.getByTestId("history-progress-story-observed-focus")).toBeVisible();
    expect(screen.getByTestId("history-progress-story-next-practice")).toHaveTextContent(
      "Next practice",
    );
    expect(screen.getByTestId("history-progress-story-next-practice")).toHaveTextContent(
      "Try Travel story again with this focus: Tighten the ending",
    );
    expect(screen.getByTestId("history-progress-story-next-practice")).toHaveStyle({
      gridColumn: "1 / -1",
    });
  });

  it("falls back to repeating the latest theme when no focus was flagged", () => {
    render(
      <HistoryProgressStory
        records={[
          record({
            sessionId: "older",
            timestamp: "2026-06-06T12:00:00Z",
            topPriorities: [],
          }),
          record({
            sessionId: "newer",
            timestamp: "2026-06-07T12:00:00Z",
            topPriorities: [],
          }),
        ]}
        translate={translate}
      />,
    );

    expect(screen.getByTestId("history-progress-story-next-practice")).toHaveTextContent(
      "Repeat Travel story once more and compare the next score.",
    );
  });
});
