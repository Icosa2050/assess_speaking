import { useMemo, useState } from "react";

import { apiClient } from "@/lib/api/client";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import type {
  MaintenanceStorageResponse,
  SupportBundleCreateRequest,
} from "@/lib/api/types";
import type { UiLocale } from "@/lib/state/sessionDraft";

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const actionButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "44px",
  padding: "0.75rem 1rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  backgroundColor: "rgba(255, 255, 255, 0.95)",
  color: "#10201c",
  fontWeight: 600,
  font: "inherit",
} as const;

const storageGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
} as const;

const formatByteCount = (value: number): string => {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  const digits = size >= 10 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(digits)} ${units[unitIndex]}`;
};

const formatExpiresAt = (value: string, locale: UiLocale): string => {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return value;
  }

  return timestamp.toLocaleString(locale);
};

const triggerBundleDownload = async (bundleId: string, filename: string): Promise<void> => {
  const blob = await apiClient.downloadSupportBundle(bundleId);
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => {
    window.URL.revokeObjectURL(objectUrl);
  }, 0);
};

export const SupportPanel = ({
  activeConnectionId,
  locale,
}: {
  activeConnectionId: string;
  locale: UiLocale;
}) => {
  const translate = createTranslator(locale);
  const [storage, setStorage] = useState<MaintenanceStorageResponse | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [isBusy, setIsBusy] = useState(false);
  const [confirmCleanupRun, setConfirmCleanupRun] = useState(false);
  const [includeReports, setIncludeReports] = useState(false);
  const [includeRecordings, setIncludeRecordings] = useState(false);
  const [includeUploads, setIncludeUploads] = useState(false);
  const [includeRuntimeHealth, setIncludeRuntimeHealth] = useState(false);

  const storageRows = useMemo(
    () => Object.entries(storage?.areas ?? {}),
    [storage],
  );

  const resetFeedback = () => {
    setMessage("");
    setError("");
    setWarnings([]);
  };

  const handleRefreshStorage = async () => {
    resetFeedback();
    setIsBusy(true);
    try {
      const nextStorage = await apiClient.getMaintenanceStorage();
      setStorage(nextStorage);
      setMessage(translate("settings.support_storage_refreshed"));
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setError(translate("settings.support_storage_error", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleCleanup = async (dryRun: boolean) => {
    resetFeedback();
    setIsBusy(true);
    try {
      const result = await apiClient.postMaintenanceCleanup({
        target: "all_safe",
        dry_run: dryRun,
      });
      setMessage(
        translate(
          dryRun
            ? "settings.support_cleanup_preview_success"
            : "settings.support_cleanup_run_success",
          {
            file_count: result.deleted_file_count,
            size: formatByteCount(result.freed_bytes),
          },
        ),
      );
      setWarnings(result.warnings ?? []);
      if (!dryRun) {
        setConfirmCleanupRun(false);
        const refreshedStorage = await apiClient.getMaintenanceStorage();
        setStorage(refreshedStorage);
      }
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setError(translate("settings.support_cleanup_error", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  const handleCreateBundle = async () => {
    resetFeedback();
    setIsBusy(true);
    try {
      const request: SupportBundleCreateRequest = {
        include_reports: includeReports,
        include_recordings: includeRecordings,
        include_runtime_health: includeRuntimeHealth,
        include_uploads: includeUploads,
        client_snapshot: {
          active_connection_id: activeConnectionId,
          route: "settings",
          ui_locale: locale,
        },
      };
      const created = await apiClient.createSupportBundle(request);
      await triggerBundleDownload(created.bundle_id, created.filename);
      setMessage(
        translate("settings.support_bundle_success", {
          filename: created.filename,
          size: formatByteCount(created.size_bytes),
          expires_at: formatExpiresAt(created.expires_at, locale),
        }),
      );
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      setError(translate("settings.support_bundle_error", { detail }));
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <section style={cardStyle}>
      <div style={{ display: "grid", gap: "0.35rem" }}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("settings.support_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("settings.support_body")}
        </p>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("settings.support_privacy_note")}
        </p>
      </div>

      {message ? <p style={{ margin: 0, color: "#166534" }}>{message}</p> : null}
      {error ? <p style={{ margin: 0, color: "#b42318" }}>{error}</p> : null}
      {warnings.map((warning) => (
        <p
          key={warning}
          style={{ margin: 0, color: "#9a6700" }}
        >
          {translate("settings.support_backend_warning", { detail: warning })}
        </p>
      ))}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
        <button
          type="button"
          onClick={handleRefreshStorage}
          disabled={isBusy}
          style={actionButtonStyle}
          data-testid="settings.support_refresh_storage"
        >
          {translate("settings.support_refresh_storage")}
        </button>
        <button
          type="button"
          onClick={() => {
            void handleCleanup(true);
          }}
          disabled={isBusy}
          style={actionButtonStyle}
          {...semanticAttributes(SEMANTIC_IDS.settings.supportCleanupPreview)}
        >
          {translate("settings.support_cleanup_preview")}
        </button>
        <button
          type="button"
          onClick={() => setConfirmCleanupRun(true)}
          disabled={isBusy}
          style={actionButtonStyle}
          {...semanticAttributes(SEMANTIC_IDS.settings.supportCleanupRun)}
        >
          {translate("settings.support_cleanup_run")}
        </button>
      </div>

      {confirmCleanupRun ? (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
          <button
            type="button"
            onClick={() => {
              void handleCleanup(false);
            }}
            disabled={isBusy}
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.supportCleanupRunConfirm)}
          >
            {translate("settings.support_cleanup_run")}
          </button>
          <button
            type="button"
            onClick={() => setConfirmCleanupRun(false)}
            disabled={isBusy}
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.supportCleanupRunCancel)}
          >
            {translate("settings.clear_saved_key_cancel")}
          </button>
        </div>
      ) : null}

      <div style={{ display: "grid", gap: "0.75rem" }}>
        <strong style={{ color: "#10201c" }}>{translate("settings.support_storage_title")}</strong>
        {storageRows.length === 0 ? (
          <p style={{ margin: 0, color: "#33514b" }}>{translate("settings.support_storage_empty")}</p>
        ) : (
          <div style={storageGridStyle}>
            {storageRows.map(([areaKey, area]) => (
              <article
                key={areaKey}
                style={cardStyle}
                data-testid="settings.storage_row"
                data-area-key={areaKey}
              >
                <strong style={{ color: "#10201c" }}>
                  {translate(`settings.storage_area_${areaKey}`)}
                </strong>
                <span style={{ color: "#33514b", lineHeight: 1.5 }}>
                  {translate("settings.support_storage_row_detail", {
                    size: formatByteCount(area.size_bytes),
                    file_count: area.file_count,
                  })}
                </span>
                <span style={{ color: "#33514b", lineHeight: 1.5 }}>{area.path}</span>
              </article>
            ))}
          </div>
        )}
      </div>

      <div style={{ display: "grid", gap: "0.75rem" }}>
        <strong style={{ color: "#10201c" }}>{translate("settings.support_bundle_title")}</strong>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("settings.support_bundle_body")}
        </p>
        <label style={{ display: "flex", alignItems: "center", gap: "0.6rem", color: "#33514b" }}>
          <input
            checked={includeReports}
            onChange={(event) => setIncludeReports(event.target.checked)}
            type="checkbox"
          />
          <span>{translate("settings.support_bundle_include_reports")}</span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "0.6rem", color: "#33514b" }}>
          <input
            checked={includeRecordings}
            onChange={(event) => setIncludeRecordings(event.target.checked)}
            type="checkbox"
          />
          <span>{translate("settings.support_bundle_include_recordings")}</span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "0.6rem", color: "#33514b" }}>
          <input
            checked={includeUploads}
            onChange={(event) => setIncludeUploads(event.target.checked)}
            type="checkbox"
          />
          <span>{translate("settings.support_bundle_include_uploads")}</span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "0.6rem", color: "#33514b" }}>
          <input
            checked={includeRuntimeHealth}
            onChange={(event) => setIncludeRuntimeHealth(event.target.checked)}
            type="checkbox"
            {...semanticAttributes(SEMANTIC_IDS.settings.supportIncludeRuntimeHealth)}
          />
          <span>{translate("settings.support_bundle_include_runtime_health")}</span>
        </label>
        <button
          type="button"
          onClick={() => {
            void handleCreateBundle();
          }}
          disabled={isBusy}
          style={actionButtonStyle}
          {...semanticAttributes(SEMANTIC_IDS.settings.supportCreateBundle)}
        >
          {translate("settings.support_bundle_create")}
        </button>
      </div>
    </section>
  );
};
