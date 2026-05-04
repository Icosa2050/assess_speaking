import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { HistoryDetailPanel } from "@/components/history/HistoryDetailPanel";
import { HistoryList } from "@/components/history/HistoryList";
import { apiClient } from "@/lib/api/client";
import { createTranslator } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import { useAppStore } from "@/lib/state/appStore";

import type { HistoryRow } from "@/lib/api/types";

const ALL_HISTORY_LANGUAGES = "__all__";

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const metricGridStyle = {
  display: "grid",
  gap: "0.875rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
} as const;

const selectStyle = {
  minHeight: "44px",
  padding: "0.75rem 0.875rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.16)",
  backgroundColor: "#fff",
  color: "#10201c",
  font: "inherit",
} as const;

type HistoryViewRecord = {
  bandLabel: string;
  bandValue: number | null;
  coherenceIssueCategories: string[];
  durationPass: boolean | null;
  finalScore: number | null;
  grammarErrorCategories: string[];
  languageCode: string;
  languageLabel: string;
  languagePass: boolean | null;
  minWordsPass: boolean | null;
  overall: number | null;
  reportPath: string;
  requiresHumanReview: boolean;
  scoreLabel: string;
  sessionId: string;
  speakerId: string;
  statusLabel: string;
  taskFamily: string;
  taskFamilyLabel: string;
  theme: string;
  timestamp: string;
  timestampDate: Date | null;
  timestampLabel: string;
  topPriorities: string[];
  topicPass: boolean | null;
  wpm: number | null;
};

const safeFloat = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const safeInt = (value: unknown): number | null => {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const safeBool = (value: unknown): boolean | null => {
  if (value === true || value === false) {
    return value;
  }

  if (typeof value === "string") {
    if (value.toLowerCase() === "true") {
      return true;
    }
    if (value.toLowerCase() === "false") {
      return false;
    }
  }

  return null;
};

const parseTimestamp = (value: string): Date | null => {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const formatTimestamp = (value: string, locale: string): string => {
  const parsed = parseTimestamp(value);
  if (!parsed) {
    return value || "-";
  }

  return new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
};

const formatJumpTimestamp = (value: string, locale: string): string => {
  const parsed = parseTimestamp(value);
  if (!parsed) {
    return value || "-";
  }

  return new Intl.DateTimeFormat(locale, {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "2-digit",
  }).format(parsed);
};

const statusShortLabel = (
  record: Pick<HistoryViewRecord, "durationPass" | "languagePass" | "minWordsPass" | "requiresHumanReview" | "topicPass">,
  translate: ReturnType<typeof createTranslator>,
): string => {
  const failedGates = [record.languagePass, record.topicPass, record.durationPass, record.minWordsPass].filter(
    (value) => value === false,
  ).length;

  if (record.requiresHumanReview) {
    return translate("review.status_short_review");
  }

  if (failedGates > 0) {
    return translate("review.status_short_unstable");
  }

  return translate("review.status_short_done");
};

const taskFamilyLabel = (value: string, translate: ReturnType<typeof createTranslator>): string => {
  const translated = translate(`task_family.${value}`);
  return translated.startsWith("[") ? value.replaceAll("_", " ") : translated;
};

const languageLabel = (value: string, translate: ReturnType<typeof createTranslator>): string => {
  if (!value) {
    return translate("history.none");
  }

  const translated = translate(`locale.${value}`);
  return translated.startsWith("[") ? value.toUpperCase() : translated;
};

const normalizeHistoryRecord = (
  row: HistoryRow,
  locale: string,
  translate: ReturnType<typeof createTranslator>,
): HistoryViewRecord => {
  const languageCode = String(row.learning_language || "").trim().toLowerCase();
  const finalScore = safeFloat(row.final_score);
  const bandValue = safeInt(row.band);

  const baseRecord = {
    bandLabel: String(row.band || "").trim() || translate("history.none"),
    bandValue,
    coherenceIssueCategories: Array.isArray(row.coherence_issue_categories)
      ? row.coherence_issue_categories.map(String).filter(Boolean)
      : [],
    durationPass: safeBool(row.duration_pass),
    finalScore,
    grammarErrorCategories: Array.isArray(row.grammar_error_categories)
      ? row.grammar_error_categories.map(String).filter(Boolean)
      : [],
    languageCode,
    languageLabel: languageLabel(languageCode, translate),
    languagePass: safeBool(row.language_pass),
    minWordsPass: safeBool(row.min_words_pass),
    overall: safeFloat(row.overall),
    reportPath: String(row.report_path || ""),
    requiresHumanReview: safeBool(row.requires_human_review) === true,
    scoreLabel: finalScore !== null ? finalScore.toFixed(1) : translate("history.none"),
    sessionId: String(row.session_id || ""),
    speakerId: String(row.speaker_id || ""),
    taskFamily: String(row.task_family || ""),
    taskFamilyLabel: taskFamilyLabel(String(row.task_family || ""), translate),
    theme: String(row.theme || ""),
    timestamp: String(row.timestamp || ""),
    timestampDate: parseTimestamp(String(row.timestamp || "")),
    timestampLabel: formatTimestamp(String(row.timestamp || ""), locale),
    topPriorities: Array.isArray(row.top_priorities)
      ? row.top_priorities.map(String).filter(Boolean)
      : [],
    topicPass: safeBool(row.topic_pass),
    wpm: safeFloat(row.wpm),
  } satisfies Omit<HistoryViewRecord, "statusLabel">;

  return {
    ...baseRecord,
    statusLabel: statusShortLabel(baseRecord, translate),
  };
};

const scopeCaption = ({
  count,
  selectedLanguage,
  speakerScope,
  translate,
}: {
  count: number;
  selectedLanguage: string;
  speakerScope: string;
  translate: ReturnType<typeof createTranslator>;
}): string => {
  if (speakerScope && selectedLanguage !== ALL_HISTORY_LANGUAGES) {
    return translate("history.scope_current_speaker_language", {
      count,
      language: languageLabel(selectedLanguage, translate),
      speaker: speakerScope,
    });
  }

  if (speakerScope) {
    return translate("history.scope_current_speaker", {
      count,
      speaker: speakerScope,
    });
  }

  if (selectedLanguage !== ALL_HISTORY_LANGUAGES) {
    return translate("history.scope_all_language", {
      count,
      language: languageLabel(selectedLanguage, translate),
    });
  }

  return translate("history.scope_all", { count });
};

const listOrPlaceholder = (values: string[], translate: ReturnType<typeof createTranslator>): string =>
  values.length > 0 ? values.join(", ") : translate("history.none");

const formatTopCounts = (values: string[]): string => {
  if (values.length === 0) {
    return "–";
  }

  const counts = new Map<string, number>();
  values.forEach((value) => {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  });

  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 3)
    .map(([name, count]) => `${name} (${count})`)
    .join(", ");
};

const Sparkline = ({
  values,
}: {
  values: number[];
}) => {
  if (values.length < 2) {
    return null;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 280;
      const y = 64 - ((value - min) / range) * 52;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox="0 0 280 72" width="100%" height="72" aria-hidden="true">
      <polyline
        fill="none"
        points={points}
        stroke="#0f766e"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="3"
      />
    </svg>
  );
};

export const HistoryRoute = () => {
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const draft = useAppStore((state) => state.draft);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const translate = createTranslator(locale);
  const speakerScope = String(draft.speakerId || "").trim();

  const historyQuery = useQuery({
    queryKey: queryKeys.history,
    queryFn: () => apiClient.getHistory(),
  });

  useEffect(() => {
    setCurrentPage("history");
  }, [setCurrentPage]);

  const scopedRecords = useMemo(() => {
    const rows = historyQuery.data?.items ?? [];
    const normalized = rows.map((row) => normalizeHistoryRecord(row, locale, translate));

    return speakerScope
      ? normalized.filter((row) => row.speakerId === speakerScope)
      : normalized;
  }, [historyQuery.data?.items, locale, speakerScope, translate]);

  const availableLanguages = useMemo(
    () => [...new Set(scopedRecords.map((row) => row.languageCode).filter(Boolean))].sort(),
    [scopedRecords],
  );
  const preferredLanguage = String(draft.learningLanguage || "").trim().toLowerCase();
  const [selectedLanguage, setSelectedLanguage] = useState<string>(ALL_HISTORY_LANGUAGES);
  const [hasInitializedLanguage, setHasInitializedLanguage] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState("");

  useEffect(() => {
    if (availableLanguages.length === 0) {
      setSelectedLanguage(ALL_HISTORY_LANGUAGES);
      setHasInitializedLanguage(false);
      return;
    }

    if (!hasInitializedLanguage) {
      setSelectedLanguage(
        preferredLanguage && availableLanguages.includes(preferredLanguage)
          ? preferredLanguage
          : ALL_HISTORY_LANGUAGES,
      );
      setHasInitializedLanguage(true);
      return;
    }

    if (selectedLanguage === ALL_HISTORY_LANGUAGES || availableLanguages.includes(selectedLanguage)) {
      return;
    }

    setSelectedLanguage(
      preferredLanguage && availableLanguages.includes(preferredLanguage)
        ? preferredLanguage
        : ALL_HISTORY_LANGUAGES,
    );
  }, [availableLanguages, hasInitializedLanguage, preferredLanguage, selectedLanguage]);

  const filteredRecords = useMemo(
    () =>
      selectedLanguage === ALL_HISTORY_LANGUAGES
        ? scopedRecords
        : scopedRecords.filter((row) => row.languageCode === selectedLanguage),
    [scopedRecords, selectedLanguage],
  );

  const detailRecords = useMemo(
    () => filteredRecords.filter((row) => row.reportPath.trim().length > 0).slice().reverse(),
    [filteredRecords],
  );

  useEffect(() => {
    if (detailRecords.length === 0) {
      setSelectedSessionId("");
      return;
    }

    if (!detailRecords.some((row) => row.sessionId === selectedSessionId)) {
      setSelectedSessionId(detailRecords[0].sessionId);
    }
  }, [detailRecords, selectedSessionId]);

  const selectedRecord =
    detailRecords.find((row) => row.sessionId === selectedSessionId) ?? null;

  const detailQuery = useQuery({
    queryKey: queryKeys.historyDetail(selectedSessionId || "none"),
    queryFn: () => apiClient.getHistoryDetail(selectedSessionId),
    enabled: Boolean(selectedSessionId),
  });

  const attempts = filteredRecords.slice().reverse();
  const latestRecord = filteredRecords.at(-1) ?? null;
  const finalScores = filteredRecords.map((row) => row.finalScore).filter((value): value is number => value !== null);
  const wpmValues = filteredRecords.map((row) => row.wpm).filter((value): value is number => value !== null);
  const bandValues = filteredRecords.map((row) => row.bandValue).filter((value): value is number => value !== null);
  const priorities = {
    latest: latestRecord?.topPriorities ?? [],
    previous: filteredRecords.length > 1 ? filteredRecords[filteredRecords.length - 2].topPriorities : [],
  };
  const newPriorities = priorities.latest.filter((item) => !priorities.previous.includes(item));
  const resolvedPriorities = priorities.previous.filter((item) => !priorities.latest.includes(item));
  const taskFamilyRows = [...new Set(filteredRecords.map((row) => row.taskFamily).filter(Boolean))]
    .sort()
    .map((family) => {
      const familyRecords = filteredRecords.filter((row) => row.taskFamily === family);
      const familyScores = familyRecords
        .map((row) => row.finalScore)
        .filter((value): value is number => value !== null);
      return {
        avgFinal:
          familyScores.length > 0
            ? (familyScores.reduce((sum, value) => sum + value, 0) / familyScores.length).toFixed(2)
            : null,
        coherence: formatTopCounts(familyRecords.flatMap((row) => row.coherenceIssueCategories)),
        count: familyRecords.length,
        grammar: formatTopCounts(familyRecords.flatMap((row) => row.grammarErrorCategories)),
        latestFinal: familyRecords.at(-1)?.finalScore ?? null,
        taskFamilyLabel: taskFamilyLabel(family, translate),
      };
    });

  if (historyQuery.isError) {
    return (
      <section style={cardStyle} data-testid="history-error" data-semantic-id="history-error">
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("history.title")}</h2>
        <p style={{ margin: 0, color: "#b42318", lineHeight: 1.6 }}>{translate("history.error_loading")}</p>
      </section>
    );
  }

  if (!historyQuery.isPending && scopedRecords.length === 0) {
    return (
      <section style={cardStyle} data-testid="history-empty" data-semantic-id="history-empty">
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("history.title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("history.empty")}</p>
      </section>
    );
  }

  if (!historyQuery.isPending && filteredRecords.length === 0) {
    return (
      <section style={cardStyle} data-testid="history-empty-filtered" data-semantic-id="history-empty-filtered">
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("history.title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("history.empty_filtered")}</p>
      </section>
    );
  }

  const scoreTrendValues = filteredRecords
    .map((row) => row.finalScore)
    .filter((value): value is number => value !== null);
  const paceTrendValues = filteredRecords
    .map((row) => row.wpm)
    .filter((value): value is number => value !== null);

  return (
    <div style={{ display: "grid", gap: "1rem" }} data-testid="history-route" data-semantic-id="history-route">
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("history.title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("history.body")}</p>
      </section>

      {availableLanguages.length > 0 ? (
        <section style={cardStyle}>
          <label style={{ display: "grid", gap: "0.375rem", color: "#10201c", fontWeight: 600 }}>
            <span>{translate("history.language_filter")}</span>
            <select
              value={selectedLanguage}
              onChange={(event) => setSelectedLanguage(event.currentTarget.value)}
              style={selectStyle}
              data-testid="history-language-filter"
              data-semantic-id="history-language-filter"
            >
              <option value={ALL_HISTORY_LANGUAGES}>{translate("history.language_filter_all")}</option>
              {availableLanguages.map((language) => (
                <option key={language} value={language}>
                  {languageLabel(language, translate)}
                </option>
              ))}
            </select>
          </label>
        </section>
      ) : null}

      <section style={cardStyle}>
        <p
          style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}
          data-testid="history-scope-caption"
          data-semantic-id="history-scope-caption"
        >
          {scopeCaption({
            count: filteredRecords.length,
            selectedLanguage,
            speakerScope,
            translate,
          })}
        </p>
        <div style={metricGridStyle} data-testid="history-metrics" data-semantic-id="history-metrics">
          {[
            ["history-metric-runs", translate("history.metric_runs"), String(filteredRecords.length)],
            ["history-metric-avg-final", translate("history.metric_avg_final"), finalScores.length > 0 ? (finalScores.reduce((sum, value) => sum + value, 0) / finalScores.length).toFixed(2) : translate("history.none")],
            ["history-metric-best-final", translate("history.metric_best_final"), finalScores.length > 0 ? Math.max(...finalScores).toFixed(2) : translate("history.none")],
            ["history-metric-avg-wpm", translate("history.metric_avg_wpm"), wpmValues.length > 0 ? (wpmValues.reduce((sum, value) => sum + value, 0) / wpmValues.length).toFixed(1) : translate("history.none")],
            ["history-metric-best-band", translate("history.metric_best_band"), bandValues.length > 0 ? String(Math.max(...bandValues)) : translate("history.none")],
          ].map(([testId, label, value]) => (
            <div key={testId} style={cardStyle} data-testid={testId} data-semantic-id={testId}>
              <strong style={{ color: "#33514b" }}>{label}</strong>
              <span style={{ color: "#10201c", fontSize: "1.2rem", fontWeight: 700 }}>{value}</span>
            </div>
          ))}
        </div>
      </section>

      <section style={cardStyle} data-testid="history-trends" data-semantic-id="history-trends">
        <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.trends_title")}</h2>
        <div style={metricGridStyle}>
          <div data-testid="history-priority-latest" data-semantic-id="history-priority-latest">
            <strong style={{ color: "#33514b" }}>{translate("history.priority_latest")}</strong>
            <p style={{ margin: "0.625rem 0 0", color: "#10201c" }}>{listOrPlaceholder(priorities.latest, translate)}</p>
          </div>
          <div data-testid="history-priority-new" data-semantic-id="history-priority-new">
            <strong style={{ color: "#33514b" }}>{translate("history.priority_new")}</strong>
            <p style={{ margin: "0.625rem 0 0", color: "#10201c" }}>{listOrPlaceholder(newPriorities, translate)}</p>
          </div>
          <div data-testid="history-priority-resolved" data-semantic-id="history-priority-resolved">
            <strong style={{ color: "#33514b" }}>{translate("history.priority_resolved")}</strong>
            <p style={{ margin: "0.625rem 0 0", color: "#10201c" }}>{listOrPlaceholder(resolvedPriorities, translate)}</p>
          </div>
        </div>
        {filteredRecords.length >= 2 ? (
          <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
            {scoreTrendValues.length >= 2 ? (
              <div data-testid="history-chart-score" data-semantic-id="history-chart-score" style={cardStyle}>
                <strong style={{ color: "#33514b" }}>{translate("history.score_chart_title")}</strong>
                <Sparkline values={scoreTrendValues} />
              </div>
            ) : null}
            {paceTrendValues.length >= 2 ? (
              <div data-testid="history-chart-pace" data-semantic-id="history-chart-pace" style={cardStyle}>
                <strong style={{ color: "#33514b" }}>{translate("history.pace_chart_title")}</strong>
                <Sparkline values={paceTrendValues} />
              </div>
            ) : null}
          </div>
        ) : (
          <p
            style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}
            data-testid="history-trends-empty"
            data-semantic-id="history-trends-empty"
          >
            {translate("history.trends_not_enough")}
          </p>
        )}
      </section>

      <section style={cardStyle} data-testid="history-task-family" data-semantic-id="history-task-family">
        <h2 style={{ margin: 0, fontSize: "1.35rem", color: "#10201c" }}>{translate("history.task_family_title")}</h2>
        {taskFamilyRows.length === 0 ? (
          <p style={{ margin: 0, color: "#33514b" }} data-testid="history-task-family-empty" data-semantic-id="history-task-family-empty">
            {translate("history.task_family_empty")}
          </p>
        ) : (
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {taskFamilyRows.map((row, index) => (
              <div
                key={`${row.taskFamilyLabel}-${index}`}
                style={{
                  display: "grid",
                  gap: "0.5rem",
                  gridTemplateColumns: "minmax(150px, 1.2fr) repeat(4, minmax(0, 1fr))",
                  padding: "0.875rem",
                  borderRadius: "8px",
                  backgroundColor: "rgba(248, 251, 250, 0.96)",
                }}
                data-testid={`history-task-family-row-${index}`}
                data-semantic-id={`history-task-family-row-${index}`}
              >
                <strong style={{ color: "#10201c" }}>{row.taskFamilyLabel}</strong>
                <span style={{ color: "#33514b" }}>{row.count}</span>
                <span style={{ color: "#33514b" }}>{row.avgFinal ?? translate("history.none")}</span>
                <span style={{ color: "#33514b" }}>{row.latestFinal !== null ? row.latestFinal.toFixed(2) : translate("history.none")}</span>
                <span style={{ color: "#33514b" }}>{`${row.grammar} · ${row.coherence}`}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <HistoryList
        attempts={attempts.map((row) => ({
          bandLabel: row.bandLabel,
          languageLabel: row.languageLabel,
          reportPath: row.reportPath,
          scoreLabel: row.scoreLabel,
          sessionId: row.sessionId,
          statusLabel: row.statusLabel,
          taskFamilyLabel: row.taskFamilyLabel,
          theme: row.theme,
          timestampLabel: row.timestampLabel,
        }))}
        detailAttempts={detailRecords.map((row) => ({
          bandLabel: row.bandLabel,
          languageLabel: row.languageLabel,
          reportPath: row.reportPath,
          scoreLabel: row.scoreLabel,
          sessionId: row.sessionId,
          statusLabel: row.statusLabel,
          taskFamilyLabel: row.taskFamilyLabel,
          theme: row.theme,
          timestampLabel: formatJumpTimestamp(row.timestamp, locale),
        }))}
        onSelectSession={setSelectedSessionId}
        selectedSessionId={selectedSessionId}
        translate={translate}
      />

      <HistoryDetailPanel
        error={detailQuery.isError ? translate("history.details_error") : ""}
        isLoading={detailQuery.isPending}
        payload={detailQuery.data?.payload ?? null}
        record={
          selectedRecord
            ? {
                bandLabel: selectedRecord.bandLabel,
                languageLabel: selectedRecord.languageLabel,
                scoreValue: selectedRecord.finalScore,
                sessionId: selectedRecord.sessionId,
                theme: selectedRecord.theme,
              }
            : null
        }
        translate={translate}
      />
    </div>
  );
};
