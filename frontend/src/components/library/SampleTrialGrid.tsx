import type { SampleItem } from "@/lib/api/types";
import { createTranslator, semanticAttributes, SEMANTIC_IDS } from "@/lib/i18n";

const sampleGridStyle = {
  display: "grid",
  gap: "0.75rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
} as const;

const sampleCardStyle = {
  display: "grid",
  gap: "0.6rem",
  padding: "1rem",
  border: "1px solid rgba(18, 61, 55, 0.1)",
  borderRadius: "8px",
  backgroundColor: "rgba(248, 251, 250, 0.96)",
} as const;

const actionButtonStyle = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "40px",
  padding: "0.65rem 0.85rem",
  borderRadius: "8px",
  border: "1px solid rgba(18, 61, 55, 0.12)",
  backgroundColor: "rgba(255, 255, 255, 0.95)",
  color: "#10201c",
  font: "inherit",
  fontWeight: 600,
  cursor: "pointer",
} as const;

export const sampleTitleLabel = (sample: Pick<SampleItem, "title">): string => {
  const title = String(sample.title || "").trim();
  if (!title) {
    return "";
  }
  return title
    .split(/\s+/)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
};

export const SampleTrialGrid = ({
  hasSetup,
  onPrepareSample,
  samples,
  translate,
}: {
  hasSetup: boolean;
  onPrepareSample: (sample: SampleItem) => void;
  samples: SampleItem[];
  translate: ReturnType<typeof createTranslator>;
}) => {
  if (samples.length === 0) {
    return <p style={{ margin: 0, color: "#33514b" }}>{translate("library.samples_empty")}</p>;
  }

  return (
    <div
      style={sampleGridStyle}
      {...semanticAttributes(SEMANTIC_IDS.library.sampleGrid)}
    >
      {samples.map((sample) => {
        const title = sampleTitleLabel(sample) || translate("library.none");
        return (
          <article
            key={sample.sample_id}
            style={sampleCardStyle}
          >
            <span style={{ color: "#33514b", fontSize: "0.9rem", fontWeight: 600 }}>
              {translate("library.sample_caption", {
                language: sample.language.toUpperCase(),
                cefr: sample.cefr.toUpperCase(),
              })}
            </span>
            <strong style={{ color: "#10201c" }}>{title}</strong>
            <button
              type="button"
              onClick={() => onPrepareSample(sample)}
              style={actionButtonStyle}
              {...semanticAttributes(SEMANTIC_IDS.library.samplePrepare, {
                sampleId: sample.sample_id,
              })}
            >
              {translate(hasSetup ? "library.sample_use_now" : "library.sample_prepare")}
            </button>
          </article>
        );
      })}
    </div>
  );
};
