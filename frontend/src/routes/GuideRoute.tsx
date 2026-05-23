import { useEffect } from "react";

import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";
import { useAppStore } from "@/lib/state/appStore";

const guideData = {
  scoreScale: {
    min: 1,
    max: 5,
  },
  formula: {
    deterministicWeightPct: 40,
    rubricWeightPct: 60,
    topicFailCapScore: 2.5,
  },
  deterministicSignals: [
    { id: "wpm", weightPct: 35, target: 130, tolerance: 90 },
    { id: "pause_ratio", weightPct: 25, ceiling: 0.45 },
    { id: "filler_ratio", weightPct: 20, ceiling: 0.12 },
    { id: "cohesion_markers", weightPct: 10, target: 3 },
    { id: "complexity_index", weightPct: 10, target: 4 },
  ],
  rubricDimensions: ["fluency", "cohesion", "accuracy", "range", "overall"],
  gates: [
    { id: "language_pass" },
    { id: "topic_pass", topicFailCapScore: 2.5 },
    { id: "content_validity_pass" },
    { id: "duration_pass", durationPassRatioPct: 80 },
    { id: "min_words_pass", minWordCount: 5 },
  ],
  cefrDimensions: [
    "fluency",
    "pronunciation_intelligibility",
    "grammar",
    "lexicon",
    "coherence",
    "task_fulfillment",
  ],
  cefrThresholds: [
    { code: "en", B2: 4.05, C1: 4.65, C2: 4.85 },
    { code: "it", B2: 3.45, C1: 4.1, C2: 4.65 },
  ],
};

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1.25rem",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  borderRadius: "8px",
  backgroundColor: "rgba(255, 255, 255, 0.94)",
  boxShadow: "0 18px 40px rgba(16, 32, 28, 0.05)",
} as const;

const summaryGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
} as const;

const tableStyle = {
  borderCollapse: "collapse",
  minWidth: "100%",
  color: "#10201c",
} as const;

const tableCellStyle = {
  borderTop: "1px solid rgba(18, 61, 55, 0.08)",
  padding: "0.65rem",
  textAlign: "left",
} as const;

const gateLabelKey = (gateId: string): string => {
  if (gateId === "topic_pass") {
    return "review.gate_theme";
  }
  if (gateId === "content_validity_pass") {
    return "review.gate_content_validity";
  }
  if (gateId === "min_words_pass") {
    return "review.gate_words";
  }
  if (gateId === "duration_pass") {
    return "review.gate_duration";
  }
  return "review.gate_language";
};

const deterministicTarget = (
  signal: (typeof guideData.deterministicSignals)[number],
  translate: ReturnType<typeof createTranslator>,
): string => {
  if (signal.id === "wpm") {
    return translate("guide.det_target_wpm", {
      target: signal.target ?? "",
      tolerance: signal.tolerance ?? "",
    });
  }
  if (signal.id === "pause_ratio") {
    return translate("guide.det_target_pause", { ceiling: signal.ceiling ?? "" });
  }
  if (signal.id === "filler_ratio") {
    return translate("guide.det_target_filler", { ceiling: signal.ceiling ?? "" });
  }
  if (signal.id === "cohesion_markers") {
    return translate("guide.det_target_cohesion", { target: signal.target ?? "" });
  }
  return translate("guide.det_target_complexity", { target: signal.target ?? "" });
};

const gateRule = (
  gate: (typeof guideData.gates)[number],
  translate: ReturnType<typeof createTranslator>,
): string => {
  if (gate.id === "duration_pass") {
    return translate("guide.gate_rule_duration_pass", {
      ratio: gate.durationPassRatioPct ?? "",
    });
  }
  if (gate.id === "min_words_pass") {
    return translate("guide.gate_rule_min_words_pass", {
      count: gate.minWordCount ?? "",
    });
  }
  if (gate.id === "topic_pass") {
    return translate("guide.gate_rule_topic_pass", {
      cap: gate.topicFailCapScore ?? "",
    });
  }
  if (gate.id === "content_validity_pass") {
    return translate("guide.gate_rule_content_validity_pass");
  }
  return translate("guide.gate_rule_language_pass");
};

export const GuideRoute = () => {
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const setCurrentPage = useAppStore((state) => state.setCurrentPage);
  const translate = createTranslator(locale);
  const languageDisplayNames =
    typeof Intl.DisplayNames === "function"
      ? new Intl.DisplayNames([locale], { type: "language" })
      : null;

  useEffect(() => {
    setCurrentPage("guide");
  }, [setCurrentPage]);

  const summaryCards = [
    ["guide.card_overall_title", "guide.card_overall_body"],
    ["guide.card_band_title", "guide.card_band_body"],
    ["guide.card_deterministic_title", "guide.card_deterministic_body"],
    ["guide.card_rubric_title", "guide.card_rubric_body"],
    ["guide.card_gates_title", "guide.card_gates_body"],
  ];

  return (
    <div
      style={{ display: "grid", gap: "1rem" }}
      {...semanticAttributes(SEMANTIC_IDS.guide.screen)}
    >
      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.75rem", color: "#10201c" }}>
          {translate("guide.title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("guide.body")}
        </p>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("guide.summary_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("guide.summary_body")}
        </p>
        <div style={summaryGridStyle}>
          {summaryCards.map(([titleKey, bodyKey]) => (
            <article key={titleKey} style={{ display: "grid", gap: "0.35rem" }}>
              <strong style={{ color: "#33514b" }}>{translate(titleKey)}</strong>
              <span style={{ lineHeight: 1.5, color: "#10201c" }}>{translate(bodyKey)}</span>
            </article>
          ))}
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("guide.formula_title")}
        </h2>
        <ul style={{ margin: 0, paddingInlineStart: "1.25rem", color: "#33514b", lineHeight: 1.6 }}>
          <li>
            {translate("guide.formula_hybrid", {
              deterministic_weight: guideData.formula.deterministicWeightPct,
              rubric_weight: guideData.formula.rubricWeightPct,
            })}
          </li>
          <li>{translate("guide.formula_deterministic_only")}</li>
          <li>{translate("guide.formula_topic_cap", { cap: guideData.formula.topicFailCapScore })}</li>
          <li>
            {translate("guide.formula_decimal", {
              min_score: guideData.scoreScale.min,
              max_score: guideData.scoreScale.max,
            })}
          </li>
        </ul>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("guide.deterministic_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("guide.deterministic_body")}
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                {[translate("guide.det_column_signal"), translate("guide.det_column_target"), translate("guide.det_column_weight")].map((label) => (
                  <th key={label} style={{ ...tableCellStyle, borderTop: 0, color: "#33514b" }}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {guideData.deterministicSignals.map((signal) => (
                <tr key={signal.id}>
                  <td style={tableCellStyle}>{translate(`guide.det_signal_${signal.id}`)}</td>
                  <td style={tableCellStyle}>{deterministicTarget(signal, translate)}</td>
                  <td style={tableCellStyle}>{signal.weightPct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("guide.rubric_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("guide.rubric_body")}
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                {[translate("guide.rubric_column_dimension"), translate("guide.rubric_column_description")].map((label) => (
                  <th key={label} style={{ ...tableCellStyle, borderTop: 0, color: "#33514b" }}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {guideData.rubricDimensions.map((dimension) => (
                <tr key={dimension}>
                  <td style={tableCellStyle}>{translate(`guide.dimension_${dimension}`)}</td>
                  <td style={tableCellStyle}>{translate(`guide.rubric_desc_${dimension}`)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("guide.gates_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("guide.gates_body")}
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                {[translate("guide.gates_column_gate"), translate("guide.gates_column_rule")].map((label) => (
                  <th key={label} style={{ ...tableCellStyle, borderTop: 0, color: "#33514b" }}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {guideData.gates.map((gate) => (
                <tr key={gate.id}>
                  <td style={tableCellStyle}>{translate(gateLabelKey(gate.id))}</td>
                  <td style={tableCellStyle}>{gateRule(gate, translate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section style={cardStyle}>
        <h2 style={{ margin: 0, fontSize: "1.2rem", color: "#10201c" }}>
          {translate("guide.cefr_title")}
        </h2>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#33514b" }}>
          {translate("guide.cefr_body")}
        </p>
        <p style={{ margin: 0, color: "#33514b" }}>
          {translate("guide.cefr_dimensions_caption", {
            dimensions: guideData.cefrDimensions.map((dimension) => translate(`guide.dimension_${dimension}`)).join(", "),
          })}
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                {[
                  translate("guide.cefr_column_language"),
                  translate("guide.cefr_column_b2"),
                  translate("guide.cefr_column_c1"),
                  translate("guide.cefr_column_c2"),
                ].map((label) => (
                  <th key={label} style={{ ...tableCellStyle, borderTop: 0, color: "#33514b" }}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {guideData.cefrThresholds.map((row) => (
                <tr key={row.code}>
                  <td style={tableCellStyle}>{languageDisplayNames?.of(row.code) ?? row.code.toUpperCase()}</td>
                  <td style={tableCellStyle}>{row.B2}</td>
                  <td style={tableCellStyle}>{row.C1}</td>
                  <td style={tableCellStyle}>{row.C2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ margin: 0, color: "#33514b" }}>{translate("guide.cefr_footer")}</p>
      </section>
    </div>
  );
};
