import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { Icon, type IconName } from "@/components/ui/Icon";
import { ProgressRing } from "@/components/ui/ProgressRing";
import { apiClient } from "@/lib/api/client";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import { buildSetupReadinessRows } from "@/lib/setup/readiness";
import { selectRuntimeReadiness, useAppStore } from "@/lib/state/appStore";
import { hasReviewState, hasSetupDraft } from "@/lib/state/sessionDraft";

import styles from "./HomeRoute.module.css";

type DiagnosticsCardState = "checks" | "loading" | "ready" | "setup" | "unavailable";

const isActionableDiagnosticStatus = (status: string): boolean => {
  const normalizedStatus = status.trim().toLowerCase();
  return normalizedStatus !== "ok" && normalizedStatus !== "info";
};

const diagnosticsCopy = (
  state: DiagnosticsCardState,
  itemCount: number,
  translate: ReturnType<typeof createTranslator>,
): { body: string; title: string } => {
  switch (state) {
    case "loading":
      return {
        title: translate("home.diagnostics_title"),
        body: translate("home.diagnostics_loading"),
      };
    case "setup":
      return {
        title: translate("home.diagnostics_next_step_title"),
        body: translate("home.diagnostics_next_step_body"),
      };
    case "ready":
      return {
        title: translate("home.diagnostics_ready_title"),
        body: translate("home.diagnostics_ready_body"),
      };
    case "unavailable":
      return {
        title: translate("home.diagnostics_title"),
        body: translate("home.diagnostics_unavailable_body"),
      };
    case "checks":
      return {
        title: translate("home.diagnostics_issues_title"),
        body: translate("home.diagnostics_issues_body", { count: itemCount }),
      };
  }
};

const diagnosticsIcon = (state: DiagnosticsCardState): IconName => {
  if (state === "ready") {
    return "check";
  }
  if (state === "checks" || state === "unavailable") {
    return "warning";
  }
  return "settings";
};

export const HomeRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const draft = useAppStore((state) => state.draft);
  const microphoneStatus = useAppStore(state => state.microphoneStatus);
  const microphoneSetupPassed = useAppStore(state => state.microphoneSetupPassed);
  const review = useAppStore((state) => state.review);
  const beginNewSession = useAppStore((state) => state.beginNewSession);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);

  const translate = createTranslator(locale);

  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
  });

  const diagnosticsQuery = useQuery({
    queryKey: queryKeys.diagnostics,
    queryFn: () => apiClient.getDiagnostics(),
  });

  const settingsQuery = useQuery({
    queryKey: ["runtime", "settings"],
    queryFn: () => apiClient.getRuntimeSettings(),
  });

  useEffect(() => {
    setCurrentPage("home");
  }, [setCurrentPage]);

  const runtimeConfigured = Boolean(runtimeQuery.data?.configured);
  const runtimeReadiness = selectRuntimeReadiness({
    preferences: {
      ...preferences,
      activeConnectionId: runtimeConfigured
        ? preferences.activeConnectionId || `${runtimeQuery.data?.provider || "runtime"}:${runtimeQuery.data?.model || "active"}`
        : preferences.activeConnectionId,
      setupComplete: runtimeConfigured || preferences.setupComplete,
    },
  });

  const diagnosticsItems = diagnosticsQuery.data?.items ?? [];
  const actionableDiagnostics = diagnosticsItems.filter((item) =>
    isActionableDiagnosticStatus(item.status),
  );
  const diagnosticsCardState: DiagnosticsCardState = (() => {
    if (runtimeQuery.isPending || diagnosticsQuery.isPending) {
      return "loading";
    }
    if (diagnosticsQuery.isError) {
      return "unavailable";
    }
    if (actionableDiagnostics.length > 0) {
      return "checks";
    }
    return runtimeReadiness.ready ? "ready" : "setup";
  })();
  const diagnosticsMessage = diagnosticsCopy(
    diagnosticsCardState,
    actionableDiagnostics.length,
    translate,
  );
  const practiceChecks = buildSetupReadinessRows({
    diagnostics: diagnosticsItems,
    diagnosticsError: diagnosticsQuery.isError,
    diagnosticsPending: diagnosticsQuery.isPending,
    runtime: runtimeQuery.data,
    runtimeError: runtimeQuery.isError,
    runtimePending: runtimeQuery.isPending,
    whisperCached: settingsQuery.data?.asr_provider === "groq",
    microphoneStatus,
    microphoneSetupPassed,
  }).filter(row => row.key !== "sample_check");
  const readyChecks = practiceChecks.filter(row => row.status === "ready").length;
  const practiceReadinessScore = Math.round(100 * readyChecks / practiceChecks.length);
  const practiceReadinessLabel = translate("home.practice_checks", { ready: readyChecks, total: practiceChecks.length });
  const resumeDisabled = !hasSetupDraft(draft);

  const handleStartNew = () => {
    beginNewSession();
    navigate("/session-setup");
  };

  const handleResume = () => {
    navigate(hasReviewState(review) ? "/review" : "/speak");
  };

  return (
    <div className={styles.homeGrid}>
      {!runtimeReadiness.ready ? (
        <section className={`${styles.card} ${styles.primaryPracticeCard}`}>
          <div className={styles.practiceIntro}>
            <div className={styles.titleBlock}>
              <span className={styles.eyebrow}>
                <Icon name="target" />
                {translate("home.practice_meter_label")}
              </span>
              <h2 className={styles.primaryTitle}>{translate("home.runtime_setup_title")}</h2>
              <p className={styles.cardBody}>{translate("home.runtime_setup_body")}</p>
            </div>
            <ProgressRing
              className={styles.readinessMeter}
              label={translate("home.practice_meter_label")}
              showStatus
              status={practiceReadinessLabel}
              value={practiceReadinessScore}
              valueLabel={`${readyChecks}/${practiceChecks.length}`}
            />
          </div>
          <div className={styles.actionRow}>
            <button
              type="button"
              onClick={() => navigate("/runtime-setup")}
              className={`${styles.action} ${styles.primaryAction}`}
              {...semanticAttributes(SEMANTIC_IDS.home.runtimeSetupButton)}
            >
              <Icon name="settings" />
              {translate("home.runtime_setup_button")}
            </button>
          </div>
        </section>
      ) : (
        <section className={`${styles.card} ${styles.primaryPracticeCard}`}>
          <div className={styles.practiceIntro}>
            <div className={styles.titleBlock}>
              <span className={styles.eyebrow}>
                <Icon name="microphone" />
                {translate("home.practice_meter_label")}
              </span>
              <h2 className={styles.primaryTitle}>{translate("home.primary_title")}</h2>
              <p className={styles.cardBody}>{translate("home.primary_body")}</p>
            </div>
            <ProgressRing
              className={styles.readinessMeter}
              label={translate("home.practice_meter_label")}
              showStatus
              status={practiceReadinessLabel}
              value={practiceReadinessScore}
              valueLabel={`${readyChecks}/${practiceChecks.length}`}
            />
          </div>
          <div className={`${styles.actionRow} ${styles.primaryActionRow}`}>
            <button
              type="button"
              onClick={handleStartNew}
              className={`${styles.action} ${styles.primaryAction}`}
              {...semanticAttributes(SEMANTIC_IDS.home.startNew)}
            >
              <Icon name="play" />
              {translate("home.start_new")}
            </button>
            <button
              type="button"
              onClick={handleResume}
              disabled={resumeDisabled}
              className={`${styles.action} ${styles.secondaryAction} ${resumeDisabled ? styles.disabledAction : ""}`}
              {...semanticAttributes(SEMANTIC_IDS.home.resume)}
            >
              <Icon name="arrow-right" />
              {translate("home.resume")}
            </button>
          </div>
        </section>
      )}

      <section
        className={`${styles.card} ${styles.supportCard}`}
        aria-live="polite"
      >
        <div className={styles.supportHeader}>
          <span className={styles.statusIcon}>
            <Icon name={diagnosticsIcon(diagnosticsCardState)} />
          </span>
          <h2 className={styles.supportTitle}>{diagnosticsMessage.title}</h2>
        </div>
        <p className={styles.cardBody}>{diagnosticsMessage.body}</p>
        <div className={styles.actionRow}>
          <button
            type="button"
            onClick={() => navigate("/runtime-setup")}
            className={`${styles.action} ${styles.secondaryAction}`}
            {...semanticAttributes(SEMANTIC_IDS.home.setupGuideButton)}
          >
            <Icon name="guide" />
            {translate("home.diagnostics_open_setup_guide")}
          </button>
        </div>
      </section>
    </div>
  );
};
