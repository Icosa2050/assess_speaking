import type {
  DiagnosticItem,
  RuntimeResponse,
  RuntimeSettingsConnection,
} from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import { useAppStore } from "@/lib/state/appStore";

type ConnectionStatusPanelProps = {
  activeConnection?: RuntimeSettingsConnection | null;
  diagnostics: DiagnosticItem[];
  runtime?: RuntimeResponse;
};

const sectionStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1rem",
  border: "1px solid rgba(18, 61, 55, 0.1)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.92)",
} as const;

const detailStyle = {
  margin: 0,
  lineHeight: 1.55,
  color: "#33514b",
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

const findDiagnostic = (items: DiagnosticItem[], key: string): DiagnosticItem | undefined =>
  items.find((item) => item.key === key);

const renderDiagnosticText = (
  item: DiagnosticItem | undefined,
  translate: ReturnType<typeof createTranslator>,
): string | null => {
  if (!item) {
    return null;
  }

  return translate(item.detail_key, item.detail_args as Record<string, string | number>);
};

export const ConnectionStatusPanel = ({
  activeConnection,
  diagnostics,
  runtime,
}: ConnectionStatusPanelProps) => {
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const translate = createTranslator(locale);

  const runtimeItem = findDiagnostic(diagnostics, "runtime");
  const credentialsItem = findDiagnostic(diagnostics, "runtime_api_key");
  const localHealthItem = findDiagnostic(diagnostics, "runtime_local_health");

  const providerLabel = translate("runtime_setup.provider_label");
  const modelLabel = translate("runtime_setup.model");
  const baseUrlLabel = translate("runtime_setup.base_url");

  const readinessText = runtime?.configured
    ? translate("runtime_setup.current_connection_ready")
    : renderDiagnosticText(runtimeItem, translate) ?? translate("home.runtime_setup_body");

  return (
    <section style={sectionStyle}>
      <p
        style={{
          margin: 0,
          fontWeight: 600,
          color: runtime?.configured ? "#166534" : "#9a6700",
        }}
      >
        {readinessText}
      </p>

      {runtime?.configured ? (
        <dl
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
            gap: "0.875rem",
            margin: 0,
          }}
        >
          <div style={sectionStyle}>
            <dt style={{ fontWeight: 600, color: "#33514b" }}>{providerLabel}</dt>
            <dd style={detailStyle}>
              {activeConnection?.provider_label || runtime?.provider || "-"}
            </dd>
          </div>
          <div style={sectionStyle}>
            <dt style={{ fontWeight: 600, color: "#33514b" }}>{modelLabel}</dt>
            <dd style={detailStyle}>{activeConnection?.model || runtime?.model || "-"}</dd>
          </div>
          <div style={sectionStyle}>
            <dt style={{ fontWeight: 600, color: "#33514b" }}>{baseUrlLabel}</dt>
            <dd style={detailStyle}>{activeConnection?.base_url || runtime?.base_url || "-"}</dd>
          </div>
        </dl>
      ) : null}

      {activeConnection?.secret_state === "present" ? (
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.55 }}>
          {translate("runtime_setup.secret_saved_state")}
        </p>
      ) : null}

      {activeConnection?.secret_state === "missing" ? (
        <p style={{ margin: 0, color: "#9a6700", lineHeight: 1.55 }}>
          {translate("runtime_setup.secret_missing_state")}
        </p>
      ) : null}

      {activeConnection?.last_tested_at ? (
        <p style={{ margin: 0, color: "#33514b", lineHeight: 1.55 }}>
          {translate("settings.last_tested", { value: activeConnection.last_tested_at })}
        </p>
      ) : null}

      {credentialsItem ? (
        <p
          style={{
            margin: 0,
            color: statusColor(credentialsItem.status),
            lineHeight: 1.55,
          }}
        >
          {renderDiagnosticText(credentialsItem, translate)}
        </p>
      ) : null}

      {localHealthItem ? (
        <p
          style={{
            margin: 0,
            color: statusColor(localHealthItem.status),
            lineHeight: 1.55,
          }}
        >
          {renderDiagnosticText(localHealthItem, translate)}
        </p>
      ) : null}
    </section>
  );
};
