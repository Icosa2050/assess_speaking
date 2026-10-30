import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import dialogStyles from "@/components/settings/JournalPanel.module.css";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import type { SavedConnectionRecord } from "@/lib/settings/connectionRepository";
import type { UiLocale } from "@/lib/state/sessionDraft";

const cardStyle = {
  display: "grid",
  gap: "0.75rem",
  padding: "1rem",
  border: "1px solid var(--color-border-decorative)",
  borderRadius: "8px",
  backgroundColor: "var(--color-surface)",
} as const;

const fieldStyle = {
  display: "grid",
  gap: "0.35rem",
} as const;

const panelStyle = {
  ...cardStyle,
  gap: "1rem",
} as const;

const resolveSecretStateCopy = (
  connection: SavedConnectionRecord,
  translate: ReturnType<typeof createTranslator>,
): string => {
  if (connection.secretState === "present") {
    return translate(connection.secretPersistent ? "settings.secret_saved_state" : "settings.secret_session_only_state");
  }

  if (connection.secretState === "missing") {
    return translate("settings.secret_missing_state");
  }

  return "";
};

const buildConnectionDetailLine = (
  connection: SavedConnectionRecord,
  translate: ReturnType<typeof createTranslator>,
): string => {
  const testStatus = connection.secretState === "missing"
    ? translate("settings.status_key_missing")
    : ["ok", "passed", "success"].includes(connection.lastTestStatus)
      ? translate("settings.status_test_passed")
      : ["error", "failed"].includes(connection.lastTestStatus)
        ? translate("settings.status_test_failed")
        : translate("settings.status_not_tested");
  return [
    connection.providerLabel,
    connection.model || "-",
    connection.isDefault ? translate("settings.default_badge") : "",
    testStatus,
  ]
    .filter(Boolean)
    .join(" · ");
};

export const SavedConnectionsPanel = ({
  connections,
  locale,
  onDelete,
  onEdit,
  onOpenSetup,
  onSelectConnection,
  onSetDefault,
  selectedConnectionId,
}: {
  connections: SavedConnectionRecord[];
  locale: UiLocale;
  onDelete: (connectionId: string) => void;
  onEdit: (connectionId: string) => void;
  onOpenSetup: () => void;
  onSelectConnection: (connectionId: string) => void;
  onSetDefault: (connectionId: string) => void;
  selectedConnectionId: string;
}) => {
  const translate = createTranslator(locale);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const restoreDeleteFocusId = useRef<string | null>(null);
  const deletedConnectionId = useRef<string | null>(null);
  const deleteDialog = useRef<HTMLDialogElement | null>(null);
  useEffect(() => {
    if (confirmDeleteId && !connections.some((connection) => connection.connectionId === confirmDeleteId)) {
      setConfirmDeleteId(null);
      document.getElementById("settings-connections-heading")?.focus();
    }
  }, [connections, confirmDeleteId]);
  useEffect(() => {
    const targetId = restoreDeleteFocusId.current;
    if (!targetId) return;
    Array.from(document.querySelectorAll<HTMLButtonElement>('[data-testid="settings.connection_row_delete"]'))
      .find((button) => button.dataset.connectionId === targetId)?.focus();
    restoreDeleteFocusId.current = null;
  }, [confirmDeleteId]);
  useEffect(() => {
    const dialog = deleteDialog.current;
    if (!dialog || !confirmDeleteId) return;
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    document.getElementById("connection-delete-cancel")?.focus();
    return () => { if (typeof dialog.close === "function" && dialog.open) dialog.close(); };
  }, [confirmDeleteId]);
  useEffect(() => {
    if (deletedConnectionId.current && !connections.some((connection) => connection.connectionId === deletedConnectionId.current)) {
      deletedConnectionId.current = null;
      document.getElementById("settings-connections-heading")?.focus();
    }
  }, [connections]);
  const selectedConnection =
    connections.find((connection) => connection.connectionId === selectedConnectionId) ?? null;
  const connectionToDelete = connections.find((connection) => connection.connectionId === confirmDeleteId);
  const cancelDelete = () => {
    restoreDeleteFocusId.current = confirmDeleteId;
    setConfirmDeleteId(null);
  };

  return (
    <section
      style={panelStyle}
      {...semanticAttributes(SEMANTIC_IDS.settings.sectionConnections)}
    >
      <div style={{ display: "grid", gap: "0.4rem" }}>
        <h2 id="settings-connections-heading" tabIndex={-1} style={{ margin: 0, fontSize: "1.2rem", color: "var(--color-text)" }}>
          {translate("settings.saved_connections")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "var(--color-text-muted)" }}>
          {connections.length === 0
            ? translate("settings.no_saved_connections")
            : translate("settings.body")}
        </p>
      </div>

      <label style={fieldStyle}>
        <span style={{ fontWeight: 600, color: "var(--color-text-muted)" }}>
          {translate("settings.saved_connection")}
        </span>
        <select
          value={selectedConnectionId}
          onChange={(event) => onSelectConnection(event.target.value)}
          style={{
            minHeight: "44px",
            borderRadius: "8px",
            border: "1px solid var(--color-border-control)",
            padding: "0.7rem 0.8rem",
            font: "inherit",
            color: "var(--color-text)",
            backgroundColor: "var(--color-surface)",
          }}
          {...semanticAttributes(SEMANTIC_IDS.settings.connectionId)}
        >
          {connections.length === 0 ? <option value="">{translate("settings.no_saved_connections")}</option> : null}
          {connections.map((connection) => (
            <option
              key={connection.connectionId}
              value={connection.connectionId}
            >
              {connection.label}
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
        <Button
          type="button"
          onClick={onOpenSetup}
          {...semanticAttributes(SEMANTIC_IDS.settings.openSetup)}
        >
          {translate("settings.setup_provider")}
        </Button>
        {selectedConnection && !selectedConnection.isDefault ? (
          <Button
            type="button"
            onClick={() => onSetDefault(selectedConnection.connectionId)}
            aria-label={translate("settings.make_active_named", { name: selectedConnection.label })}
            {...semanticAttributes(SEMANTIC_IDS.settings.connectionRowSetDefault, {
              connectionId: selectedConnection.connectionId,
            })}
          >
            {translate("settings.make_default")}
          </Button>
        ) : null}
      </div>

      {connections.length > 0 ? (
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {connections.map((connection) => {
            const secretStateCopy = resolveSecretStateCopy(connection, translate);
            const selected = connection.connectionId === selectedConnectionId;

            return (
              <article
                key={connection.connectionId}
                style={{
                  ...cardStyle,
                  borderColor: selected ? "var(--color-selected-border)" : "var(--color-border-decorative)",
                  boxShadow: selected ? "0 0 0 1px var(--color-selected-border)" : "none",
                }}
                {...semanticAttributes(SEMANTIC_IDS.settings.connectionRow, {
                  connectionId: connection.connectionId,
                  selected,
                })}
              >
                <div style={{ display: "grid", gap: "0.25rem" }}>
                  <strong style={{ color: "var(--color-text)" }}>{connection.label}</strong>
                  <span style={{ color: "var(--color-text-muted)", lineHeight: 1.5 }}>
                    {buildConnectionDetailLine(connection, translate)}
                  </span>
                </div>
                {connection.isDefault ? (
                  <span
                    style={{ color: "var(--color-text-muted)", fontWeight: 600 }}
                    {...semanticAttributes(SEMANTIC_IDS.settings.defaultConnectionIndicator, {
                      connectionId: connection.connectionId,
                    })}
                  >
                    {translate("settings.default_badge")}
                  </span>
                ) : null}
                {connection.baseUrl ? (
                  <span style={{ color: "var(--color-text-muted)", lineHeight: 1.5 }}>{connection.baseUrl}</span>
                ) : null}
                {secretStateCopy ? (
                  <span style={{ color: "var(--color-text-muted)", lineHeight: 1.5 }}>{secretStateCopy}</span>
                ) : null}
                {connection.lastTestedAt ? (
                  <span style={{ color: "var(--color-text-muted)", lineHeight: 1.5 }}>
                    {translate("settings.last_tested", { value: connection.lastTestedAt })}
                  </span>
                ) : null}
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
                  <Button aria-label={translate("settings.edit_connection_named", { name: connection.label })} onClick={() => onEdit(connection.connectionId)} {...semanticAttributes(SEMANTIC_IDS.settings.connectionRowEdit, { connectionId: connection.connectionId })}>
                    {translate("settings.edit_connection")}
                  </Button>
                  <Button aria-label={translate("settings.delete_connection_named", { name: connection.label })} onClick={() => setConfirmDeleteId(connection.connectionId)} {...semanticAttributes(SEMANTIC_IDS.settings.connectionRowDelete, { connectionId: connection.connectionId })}>
                    {translate("settings.delete")}
                  </Button>
                </div>
              </article>
            );
          })}
        </div>
      ) : null}
      {connectionToDelete ? <dialog ref={deleteDialog} className={dialogStyles.dialog} aria-labelledby="connection-delete-title" aria-describedby="connection-delete-description" onCancel={(event) => { event.preventDefault(); cancelDelete(); }}>
        <div className={dialogStyles.dialogBody}>
          <h2 id="connection-delete-title">{translate("settings.delete")}</h2>
          <p id="connection-delete-description">{translate("settings.delete_confirm", { name: connectionToDelete.label })}</p>
        </div>
        <div className={dialogStyles.dialogActions}>
          <Button id="connection-delete-cancel" variant="secondary" onClick={cancelDelete}>{translate("settings.cancel")}</Button>
          <Button aria-label={translate("settings.confirm_delete_named", { name: connectionToDelete.label })} variant="danger" onClick={() => { deletedConnectionId.current = connectionToDelete.connectionId; setConfirmDeleteId(null); onDelete(connectionToDelete.connectionId); }} {...semanticAttributes(SEMANTIC_IDS.settings.connectionRowDeleteConfirm, { connectionId: connectionToDelete.connectionId })}>
            {translate("settings.delete")}
          </Button>
        </div>
      </dialog> : null}
    </section>
  );
};
