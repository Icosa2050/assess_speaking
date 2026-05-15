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

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const diagnosticsListStyle = {
  display: "grid",
  gap: "0.75rem",
  padding: 0,
  margin: 0,
  listStyle: "none",
} as const;

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
    <div style={{ display: "grid", gap: "1rem" }}>
      {!runtimeReadiness.ready ? (
        <section style={cardStyle}>
          <h2
            style={{
              margin: 0,
              fontSize: "1.5rem",
              color: "#10201c",
            }}
          >
            {translate("home.runtime_setup_title")}
          </h2>
          <p
            style={{
              margin: 0,
              lineHeight: 1.6,
              color: "#33514b",
            }}
          >
            {translate("home.runtime_setup_body")}
          </p>
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
        <section style={cardStyle}>
          <h2
            style={{
              margin: 0,
              fontSize: "1.5rem",
              color: "#10201c",
            }}
          >
            {translate("home.primary_title")}
          </h2>
          <p
            style={{
              margin: 0,
              lineHeight: 1.6,
              color: "#33514b",
            }}
          >
            {translate("home.primary_body")}
          </p>
          <div className={styles.actionRow}>
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

      <section style={cardStyle}>
        <h2
          style={{
            margin: 0,
            fontSize: "1.35rem",
            color: "#10201c",
          }}
        >
          {translate("home.diagnostics_title")}
        </h2>
        <p
          style={{
            margin: 0,
            lineHeight: 1.6,
            color: "#33514b",
          }}
        >
          {translate("home.diagnostics_body")}
        </p>
        <ul style={diagnosticsListStyle}>
          {diagnosticsItems.map((item) => {
            const message = translateDiagnostic(item, translate);
            return (
              <li
                key={item.key}
                style={{
                  display: "grid",
                  gap: "0.375rem",
                  padding: "0.875rem 1rem",
                  borderRadius: "8px",
                  border: "1px solid rgba(18, 61, 55, 0.08)",
                  backgroundColor: "rgba(248, 251, 250, 0.96)",
                }}
              >
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

      <section style={cardStyle}>
        <h2
          style={{
            margin: 0,
            fontSize: "1.35rem",
            color: "#10201c",
          }}
        >
          {translate("home.secondary_title")}
        </h2>
        <p
          style={{
            margin: 0,
            lineHeight: 1.6,
            color: "#33514b",
          }}
        >
          {translate("home.secondary_body")}
        </p>
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
