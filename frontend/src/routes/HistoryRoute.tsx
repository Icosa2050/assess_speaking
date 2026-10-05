import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { HistoryDetailPanel } from "@/components/history/HistoryDetailPanel";
import historyStyles from "./HistoryRoute.module.css";
import { HistoryList } from "@/components/history/HistoryList";
import { PracticeProgress } from "@/components/history/PracticeProgress";
import { measurement, retryDraft } from "@/lib/history/practiceProgress";
import layoutStyles from "@/components/ui/layout.module.css";
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

const routeGridStyle = {
  gap: "1rem",
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

const primaryButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid rgba(15, 118, 110, 0.2)",
  backgroundColor: "#d7ebe5",
  color: "#10201c",
  fontWeight: 600,
  font: "inherit",
  cursor: "pointer",
} as const;

type HistoryViewRecord = {
  bandLabel: string;
  bandValue: number | null;
  coherenceIssueCategories: string[];
  durationPass: boolean | null;
  finalScore: number | null;
  grammarErrorCategories: string[];
  goal: string;
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

const safeFloat = measurement;

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

const isMissingTranslation = (key: string, translated: string): boolean => translated === `[${key}]`;

const taskFamilyLabel = (value: string, translate: ReturnType<typeof createTranslator>): string => {
  const key = `task_family.${value}`;
  const translated = translate(key);
  return isMissingTranslation(key, translated) ? value.replaceAll("_", " ") : translated;
};

const languageLabel = (value: string, translate: ReturnType<typeof createTranslator>): string => {
  if (!value) {
    return translate("history.none");
  }

  const key = `locale.${value}`;
  const translated = translate(key);
  return isMissingTranslation(key, translated) ? value.toUpperCase() : translated;
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
    goal: String(row.practice?.goal || ""),
    languageCode,
    languageLabel: languageLabel(languageCode, translate),
    languagePass: safeBool(row.language_pass),
    minWordsPass: safeBool(row.min_words_pass),
    overall: safeFloat(row.overall),
    reportPath: String(row.report_path || ""),
    requiresHumanReview: safeBool(row.requires_human_review) === true,
    scoreLabel: finalScore !== null ? new Intl.NumberFormat(locale, { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(finalScore) : translate("history.none"),
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

export const formatTrendSummary = (
  values: number[],
  locale: string,
  translate: ReturnType<typeof createTranslator>,
): string => {
  if (values.length === 0) {
    return "";
  }

  // Locale formatting also keeps grouping correct for unexpectedly large values.
  const formatter = new Intl.NumberFormat(locale, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  });
  return translate("history.trend_range", {
    start: formatter.format(values[0]),
    end: formatter.format(values[values.length - 1]),
  });
};

export const HistoryRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const applySetup = useAppStore((state) => state.applySetup);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const translate = useMemo(() => createTranslator(locale), [locale]);
  const [speakerScope, setSpeakerScope] = useState("");
  const speakerFilterRef = useRef<HTMLSelectElement>(null);
  const detailRef = useRef<HTMLDivElement>(null);

  const historyQuery = useQuery({
    queryKey: queryKeys.history,
    queryFn: () => apiClient.getHistory(),
  });

  useEffect(() => {
    setCurrentPage("history");
  }, [setCurrentPage]);

  const allRecords = useMemo(() => {
    const rows = historyQuery.data?.items ?? [];
    const normalized = rows.map((row) => normalizeHistoryRecord(row, locale, translate))
      .sort((a, b) => (a.timestampDate?.getTime() ?? 0) - (b.timestampDate?.getTime() ?? 0));

    return normalized;
  }, [historyQuery.data?.items, locale, translate]);

  const availableSpeakers = useMemo(() => [...new Set(allRecords.map((row) => row.speakerId).filter(Boolean))].sort(), [allRecords]);

  const availableLanguages = useMemo(
    () => [...new Set(allRecords.map((row) => row.languageCode).filter(Boolean))].sort(),
    [allRecords],
  );
  const [selectedLanguage, setSelectedLanguage] = useState<string>(ALL_HISTORY_LANGUAGES);
  const [selectedSessionId, setSelectedSessionId] = useState("");

  useEffect(() => {
    if (historyQuery.isFetching || !historyQuery.data) return;
    if (speakerScope && !availableSpeakers.includes(speakerScope)) setSpeakerScope("");
    if (selectedLanguage !== ALL_HISTORY_LANGUAGES && !availableLanguages.includes(selectedLanguage)) setSelectedLanguage(ALL_HISTORY_LANGUAGES);
  }, [availableLanguages, availableSpeakers, speakerScope, selectedLanguage, historyQuery.isFetching, historyQuery.data]);

  const filteredRecords = useMemo(
    () => allRecords.filter((row) =>
      (!speakerScope || row.speakerId === speakerScope) &&
      (selectedLanguage === ALL_HISTORY_LANGUAGES || row.languageCode === selectedLanguage)),
    [allRecords, selectedLanguage, speakerScope],
  );

  const detailRecords = useMemo(
    () => filteredRecords.filter((row) => row.reportPath.trim().length > 0 && row.sessionId.trim().length > 0).slice().reverse(),
    [filteredRecords],
  );

  useEffect(() => {
    // Cached history can predate the attempt that just finished. Choose only after refresh.
    if (historyQuery.isFetching) return;
    if (detailRecords.length === 0) {
      setSelectedSessionId("");
      return;
    }

    if (!detailRecords.some((row) => row.sessionId === selectedSessionId)) {
      setSelectedSessionId(detailRecords[0].sessionId);
    }
  }, [detailRecords, selectedSessionId, historyQuery.isFetching]);

  const selectedRecord =
    detailRecords.find((row) => row.sessionId === selectedSessionId) ?? null;

  const detailQuery = useQuery({
    queryKey: queryKeys.historyDetail(selectedSessionId || "none"),
    queryFn: () => apiClient.getHistoryDetail(selectedSessionId),
    enabled: Boolean(selectedSessionId),
  });

  const attempts = filteredRecords.slice().reverse();
  if (historyQuery.isError) {
    return (
      <section style={cardStyle} data-testid="history-error" data-semantic-id="history-error">
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("history.title")}</h2>
        <p style={{ margin: 0, color: "#b42318", lineHeight: 1.6 }}>{translate("history.error_loading")}</p>
      </section>
    );
  }

  if (!historyQuery.isPending && allRecords.length === 0) {
    return (
      <section style={cardStyle} data-testid="history-empty" data-semantic-id="history-empty">
        <h2 style={{ margin: 0, fontSize: "1.5rem", color: "#10201c" }}>{translate("history.empty_title")}</h2>
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.6 }}>{translate("history.empty_body")}</p>
        <button
          type="button"
          onClick={() => navigate("/session-setup")}
          style={primaryButtonStyle}
          data-testid="history-empty-cta"
          data-semantic-id="history-empty-cta"
        >
          {translate("history.empty_cta")}
        </button>
      </section>
    );
  }

  // Legacy session IDs can be blank: apply the scope to rows, not an ID-based join.
  const practiceRows = (historyQuery.data?.items ?? []).map(row => ({ ...row,
    learning_language: String(row.learning_language || "").trim().toLowerCase(),
    top_priorities: Array.isArray(row.top_priorities) ? row.top_priorities : [],
  })).filter((row) =>
    (!speakerScope || String(row.speaker_id || "") === speakerScope) &&
    (selectedLanguage === ALL_HISTORY_LANGUAGES || String(row.learning_language || "").trim().toLowerCase() === selectedLanguage),
  );
  const selectedPractice = selectedSessionId
    ? practiceRows.find((row) => row.session_id === selectedSessionId) ?? null
    : null;

  return (
    <div
      className={layoutStyles.stack}
      style={routeGridStyle}
      data-testid="history-route"
      data-semantic-id="history-route"
    >
      <section className={historyStyles.header}>
        <div className={historyStyles.heading}>
          <div>
            <h2>{translate("journey.headline")}</h2>
            <p>{translate("history.body")}</p>
          </div>
          <button type="button" style={primaryButtonStyle} onClick={() => navigate("/session-setup")}>
            {translate("history.new_practice")}
          </button>
        </div>
        <div className={historyStyles.filters}>
          <label>
            <span>{translate("history.speaker_filter")}</span>
            <select ref={speakerFilterRef} disabled={historyQuery.isPending} value={speakerScope} onChange={(event) => setSpeakerScope(event.currentTarget.value)} style={selectStyle} data-testid="history-speaker-filter">
              <option value="">{translate("history.speaker_filter_all")}</option>
              {availableSpeakers.map((speaker) => <option key={speaker} value={speaker}>{speaker}</option>)}
            </select>
          </label>
          <label>
            <span>{translate("history.language_filter")}</span>
            <select disabled={historyQuery.isPending} value={selectedLanguage} onChange={(event) => setSelectedLanguage(event.currentTarget.value)} style={selectStyle} data-testid="history-language-filter">
              <option value={ALL_HISTORY_LANGUAGES}>{translate("history.language_filter_all")}</option>
              {availableLanguages.map((language) => <option key={language} value={language}>{languageLabel(language, translate)}</option>)}
            </select>
          </label>
        </div>
        <p data-testid="history-scope-caption" aria-live="polite">
          {scopeCaption({ count: filteredRecords.length, selectedLanguage, speakerScope, translate })}
        </p>
      </section>
      {historyQuery.isPending ? <p role="status">{translate("journey.loading")}</p> :
        filteredRecords.length === 0 ? <section style={cardStyle} data-testid="history-empty-filtered">
          <p>{translate("history.empty_filtered")}</p>
          <button type="button" style={primaryButtonStyle} onClick={() => { setSpeakerScope(""); setSelectedLanguage(ALL_HISTORY_LANGUAGES); speakerFilterRef.current?.focus(); }}>
            {translate("history.clear_filters")}
          </button>
        </section> : <>
      <PracticeProgress
        rows={practiceRows}
        selected={selectedPractice}
        locale={locale}
        onSelect={setSelectedSessionId}
        onRetry={(row) => {
          const savedDraft = retryDraft(row);
          if (savedDraft) {
            applySetup(savedDraft);
            navigate("/speak");
          }
        }}
      />
      <div className={historyStyles.browser}>
        <HistoryList
          attempts={attempts}
          onSelectSession={(sessionId) => {
            setSelectedSessionId(sessionId);
            if (window.matchMedia?.("(max-width: 1000px)").matches) {
              detailRef.current?.focus({ preventScroll: true });
              detailRef.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
            }
          }}
          selectedSessionId={selectedSessionId}
          translate={translate}
        />
        <div ref={detailRef} tabIndex={-1} className={historyStyles.detail} role="region" aria-label={translate("history.details_title")}>
          <HistoryDetailPanel
            locale={locale}
            key={selectedSessionId}
            error={detailQuery.isError ? translate("history.details_error") : null}
            isLoading={detailQuery.isPending}
            payload={detailQuery.data?.payload ?? null}
            record={selectedRecord ? { ...selectedRecord, scoreValue: selectedRecord.finalScore } : null}
            translate={translate}
          />
        </div>
      </div>
      </>}
    </div>
  );
};
