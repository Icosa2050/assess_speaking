import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { apiClient } from "@/lib/api/client";
import type { DiagnosticItem } from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import { selectRuntimeReadiness, useAppStore } from "@/lib/state/appStore";
import { hasReviewState, hasSetupDraft } from "@/lib/state/sessionDraft";

import styles from "./HomeRoute.module.css";

const statusColor = (status: string): string => {
  if (status === "ok") {
    return "#166534";
  }
  if (status === "warning" || status === "info") {
    return "#9a6700";
  }
  return "#b42318";
};

const translateDiagnostic = (
  item: DiagnosticItem,
  translate: ReturnType<typeof createTranslator>,
): { detail: string; title: string } => ({
  title: translate(item.title_key),
  detail: translate(item.detail_key, item.detail_args as Record<string, string | number>),
});

export const HomeRoute = () => {
  const navigate = useNavigate();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const preferences = useAppStore((state) => state.preferences);
  const draft = useAppStore((state) => state.draft);
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
          <h2 className={styles.primaryTitle}>{translate("home.runtime_setup_title")}</h2>
          <p className={styles.cardBody}>{translate("home.runtime_setup_body")}</p>
          <div className={styles.actionRow}>
            <button
              type="button"
              onClick={() => navigate("/runtime-setup")}
              className={`${styles.action} ${styles.primaryAction}`}
              {...semanticAttributes(SEMANTIC_IDS.home.runtimeSetupButton)}
            >
              {translate("home.runtime_setup_button")}
            </button>
          </div>
        </section>
      ) : (
        <section className={`${styles.card} ${styles.primaryPracticeCard}`}>
          <h2 className={styles.primaryTitle}>{translate("home.primary_title")}</h2>
          <p className={styles.cardBody}>{translate("home.primary_body")}</p>
          <div className={`${styles.actionRow} ${styles.primaryActionRow}`}>
            <button
              type="button"
              onClick={handleStartNew}
              className={`${styles.action} ${styles.primaryAction}`}
              {...semanticAttributes(SEMANTIC_IDS.home.startNew)}
            >
              {translate("home.start_new")}
            </button>
            <button
              type="button"
              onClick={handleResume}
              disabled={resumeDisabled}
              className={`${styles.action} ${styles.secondaryAction} ${resumeDisabled ? styles.disabledAction : ""}`}
              {...semanticAttributes(SEMANTIC_IDS.home.resume)}
            >
              {translate("home.resume")}
            </button>
          </div>
        </section>
      )}

      <section className={`${styles.card} ${styles.supportCard}`}>
        <h2 className={styles.supportTitle}>{translate("home.diagnostics_title")}</h2>
        <p className={styles.cardBody}>{translate("home.diagnostics_body")}</p>
        <ul className={styles.diagnosticsList}>
          {diagnosticsItems.map((item) => {
            const message = translateDiagnostic(item, translate);
            return (
              <li key={item.key} className={styles.diagnosticItem}>
                <strong
                  style={{
                    color: statusColor(item.status),
                  }}
                >
                  {message.title}
                </strong>
                <span
                  style={{
                    color: "#33514b",
                    lineHeight: 1.5,
                  }}
                >
                  {message.detail}
                </span>
              </li>
            );
          })}
        </ul>
      </section>

      <section className={`${styles.card} ${styles.supportCard}`}>
        <h2 className={styles.supportTitle}>{translate("home.secondary_title")}</h2>
        <p className={styles.cardBody}>{translate("home.secondary_body")}</p>
        <div className={styles.actionRow}>
          <Link
            to="/history"
            className={`${styles.action} ${styles.secondaryAction}`}
            {...semanticAttributes(SEMANTIC_IDS.home.openHistory)}
          >
            {translate("nav.history")}
          </Link>
          <Link
            to="/library"
            className={`${styles.action} ${styles.secondaryAction}`}
            {...semanticAttributes(SEMANTIC_IDS.home.openLibrary)}
          >
            {translate("nav.library")}
          </Link>
          <Link
            to="/guide"
            className={`${styles.action} ${styles.secondaryAction}`}
            {...semanticAttributes(SEMANTIC_IDS.home.openGuide)}
          >
            {translate("nav.guide")}
          </Link>
          <Link
            to="/settings"
            state={{ from: "home" }}
            className={`${styles.action} ${styles.secondaryAction}`}
            {...semanticAttributes(SEMANTIC_IDS.home.openSettings)}
          >
            {translate("nav.settings")}
          </Link>
        </div>
      </section>
    </div>
  );
};
