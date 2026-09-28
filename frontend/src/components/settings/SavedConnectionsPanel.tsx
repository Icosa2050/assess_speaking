import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import type { SavedConnectionRecord } from "@/lib/settings/connectionRepository";
import type { UiLocale } from "@/lib/state/sessionDraft";

const cardStyle = {
  display: "grid",
  gap: "0.75rem",
  padding: "1rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
} as const;

const fieldStyle = {
  display: "grid",
  gap: "0.35rem",
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

const panelStyle = {
  ...cardStyle,
  gap: "1rem",
} as const;

const resolveSecretStateCopy = (
  connection: SavedConnectionRecord,
  translate: ReturnType<typeof createTranslator>,
): string => {
  if (connection.secretState === "present") {
    return translate("settings.secret_saved_state");
  }

  if (connection.secretState === "missing") {
    return translate("settings.secret_missing_state");
  }

  return "";
};

const buildConnectionDetailLine = (
  connection: SavedConnectionRecord,
  translate: ReturnType<typeof createTranslator>,
): string =>
  [
    connection.providerLabel,
    connection.model,
    connection.isDefault ? translate("settings.default_badge") : "",
    connection.lastTestStatus,
  ]
    .filter(Boolean)
    .join(" · ");

export const SavedConnectionsPanel = ({
  connections,
  locale,
  onDelete,
  onOpenSetup,
  onSelectConnection,
  onSetDefault,
  selectedConnectionId,
}: {
  connections: SavedConnectionRecord[];
  locale: UiLocale;
  onDelete: (connectionId: string) => void;
  onOpenSetup: () => void;
  onSelectConnection: (connectionId: string) => void;
  onSetDefault: (connectionId: string) => void;
  selectedConnectionId: string;
}) => {
  const translate = createTranslator(locale);
  const selectedConnection =
    connections.find((connection) => connection.connectionId === selectedConnectionId) ?? null;

  return (
    <section
      style={panelStyle}
      {...semanticAttributes(SEMANTIC_IDS.settings.sectionConnections)}
    >
      <div style={{ display: "grid", gap: "0.4rem" }}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("settings.saved_connections")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {connections.length === 0
            ? translate("settings.no_saved_connections")
            : translate("settings.body")}
        </p>
      </div>

      <label style={fieldStyle}>
        <span style={{ fontWeight: 600, color: "#33514b" }}>
          {translate("settings.saved_connection")}
        </span>
        <select
          value={selectedConnectionId}
          onChange={(event) => onSelectConnection(event.target.value)}
          style={{
            minHeight: "44px",
            borderRadius: "8px",
            border: "1px solid rgba(18, 61, 55, 0.16)",
            padding: "0.7rem 0.8rem",
            font: "inherit",
            color: "#10201c",
            backgroundColor: "rgba(255, 255, 255, 0.96)",
          }}
          {...semanticAttributes(SEMANTIC_IDS.settings.connectionId)}
        >
          <option value="__new__">{translate("settings.create_new_connection")}</option>
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
        <button
          type="button"
          onClick={onOpenSetup}
          style={actionButtonStyle}
          {...semanticAttributes(SEMANTIC_IDS.settings.openSetup)}
        >
          {translate("settings.open_setup")}
        </button>
        {selectedConnection && !selectedConnection.isDefault ? (
          <button
            type="button"
            onClick={() => onSetDefault(selectedConnection.connectionId)}
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.connectionRowSetDefault, {
              connectionId: selectedConnection.connectionId,
            })}
          >
            {translate("settings.make_default")}
          </button>
        ) : null}
        {selectedConnection ? (
          <button
            type="button"
            onClick={() => onDelete(selectedConnection.connectionId)}
            style={actionButtonStyle}
            {...semanticAttributes(SEMANTIC_IDS.settings.connectionRowDelete, {
              connectionId: selectedConnection.connectionId,
            })}
          >
            {translate("settings.delete")}
          </button>
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
                  borderColor: selected ? "rgba(15, 118, 110, 0.4)" : cardStyle.border,
                  boxShadow: selected ? "0 0 0 1px rgba(15, 118, 110, 0.12)" : "none",
                }}
                {...semanticAttributes(SEMANTIC_IDS.settings.connectionRow, {
                  connectionId: connection.connectionId,
                  selected,
                })}
              >
                <div style={{ display: "grid", gap: "0.25rem" }}>
                  <strong style={{ color: "#10201c" }}>{connection.label}</strong>
                  <span style={{ color: "#33514b", lineHeight: 1.5 }}>
                    {buildConnectionDetailLine(connection, translate)}
                  </span>
                </div>
                {connection.isDefault ? (
                  <span
                    style={{ color: "#33514b", fontWeight: 600 }}
                    {...semanticAttributes(SEMANTIC_IDS.settings.defaultConnectionIndicator, {
                      connectionId: connection.connectionId,
                    })}
                  >
                    {translate("settings.default_badge")}
                  </span>
                ) : null}
                {connection.baseUrl ? (
                  <span style={{ color: "#33514b", lineHeight: 1.5 }}>{connection.baseUrl}</span>
                ) : null}
                {secretStateCopy ? (
                  <span style={{ color: "#33514b", lineHeight: 1.5 }}>{secretStateCopy}</span>
                ) : null}
                {connection.lastTestedAt ? (
                  <span style={{ color: "#33514b", lineHeight: 1.5 }}>
                    {translate("settings.last_tested", { value: connection.lastTestedAt })}
                  </span>
                ) : null}
              </article>
            );
          })}
        </div>
      ) : null}
    </section>
  );
};
