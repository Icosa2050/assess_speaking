type Translate = (key: string, vars?: Record<string, string | number>) => string;

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const detailGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
} as const;

const detailItemStyle = {
  display: "grid",
  gap: "0.25rem",
  padding: "0.75rem 0",
  borderTop: "1px solid rgba(18, 61, 55, 0.1)",
} as const;

const briefQuoteStyle = {
  margin: 0,
  padding: "1rem",
  borderLeft: "4px solid rgba(15, 118, 110, 0.4)",
  borderRadius: "0 8px 8px 0",
  backgroundColor: "rgba(239, 248, 245, 0.98)",
  color: "#10201c",
  fontSize: "1.05rem",
  lineHeight: 1.55,
  fontWeight: 600,
} as const;

const receiptCardStyle = {
  ...cardStyle,
  gap: "0.75rem",
  boxShadow: "none",
  backgroundColor: "rgba(255, 255, 255, 0.84)",
} as const;

const semanticAttributes = (id: string): Record<string, string> => ({
  "data-testid": id,
  "data-semantic-id": id,
});

export const PracticeBriefCard = ({
  customThemeSaveHelp,
  promptText,
  resolvedThemeLabel,
  runtimeCalloutBody,
  runtimeCalloutTitle,
  selectionDetails,
  shouldShowRuntimeCallout,
  shouldShowCustomThemeSaveHelp,
  successFocus,
  translate,
}: {
  customThemeSaveHelp: string;
  promptText: string;
  resolvedThemeLabel: string;
  runtimeCalloutBody: string;
  runtimeCalloutTitle: string;
  selectionDetails: Array<{ label: string; value: string }>;
  shouldShowRuntimeCallout: boolean;
  shouldShowCustomThemeSaveHelp: boolean;
  successFocus: string[];
  translate: Translate;
}) => (
  <div style={{ display: "grid", gap: "1rem" }}>
    <section
      aria-label={translate("setup.preview_title")}
      style={cardStyle}
      {...semanticAttributes("setup.preview")}
    >
      <h2
        style={{
          margin: 0,
          fontSize: "1.35rem",
          color: "#10201c",
        }}
      >
        {translate("setup.preview_title")}
      </h2>
      <p
        style={{
          margin: 0,
          color: "#33514b",
          fontWeight: 600,
        }}
      >
        {resolvedThemeLabel || translate("setup.no_theme_selected")}
      </p>
      {promptText ? (
        <>
          <blockquote style={briefQuoteStyle}>{promptText}</blockquote>
          <div
            aria-label={translate("setup.success_focus_title")}
            {...semanticAttributes("setup.success_focus")}
          >
            <strong
              style={{
                color: "#10201c",
              }}
            >
              {translate("setup.success_focus_title")}
            </strong>
            <ul
              style={{
                margin: "0.625rem 0 0",
                paddingLeft: "1.125rem",
                display: "grid",
                gap: "0.375rem",
                color: "#33514b",
              }}
            >
              {successFocus.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        </>
      ) : (
        <p
          style={{
            margin: 0,
            color: "#33514b",
            lineHeight: 1.6,
          }}
        >
          {translate("setup.preview_placeholder")}
        </p>
      )}
    </section>

    <section
      aria-label={translate("setup.selection_title")}
      style={receiptCardStyle}
      {...semanticAttributes("setup.selection")}
    >
      <h2
        style={{
          margin: 0,
          fontSize: "1.35rem",
          color: "#10201c",
        }}
      >
        {translate("setup.selection_title")}
      </h2>
      <div style={detailGridStyle}>
        {selectionDetails.map((detail, index) => (
          <div key={`${detail.label}-${index}`} style={detailItemStyle}>
            <strong
              style={{
                color: "#48645e",
              }}
            >
              {detail.label}
            </strong>
            <span
              style={{
                color: "#10201c",
              }}
            >
              {detail.value}
            </span>
          </div>
        ))}
      </div>
      {shouldShowCustomThemeSaveHelp ? (
        <p
          style={{
            margin: 0,
            color: "#33514b",
            lineHeight: 1.6,
          }}
        >
          {customThemeSaveHelp}
        </p>
      ) : null}
      {shouldShowRuntimeCallout ? (
        <div
          style={{
            display: "grid",
            gap: "0.35rem",
            padding: "0.875rem",
            borderRadius: "8px",
            border: "1px solid rgba(15, 118, 110, 0.14)",
            backgroundColor: "rgba(239, 248, 245, 0.96)",
          }}
          {...semanticAttributes("setup.runtime_callout")}
        >
          <strong
            style={{
              color: "#10201c",
            }}
          >
            {runtimeCalloutTitle}
          </strong>
          <p
            style={{
              margin: 0,
              color: "#33514b",
              lineHeight: 1.5,
            }}
          >
            {runtimeCalloutBody}
          </p>
        </div>
      ) : null}
      <p
        style={{
          margin: 0,
          color: "#33514b",
          lineHeight: 1.6,
        }}
      >
        {translate("setup.flow_hint")}
      </p>
    </section>
  </div>
);
