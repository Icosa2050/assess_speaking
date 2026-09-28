import { Icon, type IconName } from "@/components/ui/Icon";

type Translate = (key: string, vars?: Record<string, string | number>) => string;

export type SpeakConfidencePhase = "prepare" | "record" | "submit" | "assess" | "failed";

type StepState = "active" | "attention" | "complete" | "upcoming";

type RailStep = {
  icon: IconName;
  id: "brief" | "record" | "submit" | "review";
  labelKey: string;
};

const cardStyle = {
  display: "grid",
  gap: "0.875rem",
  padding: "1rem",
  border: "1px solid rgba(15, 118, 110, 0.16)",
  borderRadius: "8px",
  backgroundColor: "rgba(248, 251, 250, 0.97)",
  boxShadow: "0 14px 32px rgba(16, 32, 28, 0.045)",
} as const;

const stepGridStyle = {
  display: "grid",
  gap: "0.625rem",
  gridTemplateColumns: "repeat(auto-fit, minmax(8.5rem, 1fr))",
} as const;

const stepStyle = (state: StepState) =>
  ({
    display: "grid",
    gap: "0.45rem",
    minHeight: "92px",
    padding: "0.75rem",
    border: state === "active" ? "1px solid rgba(15, 118, 110, 0.36)" : "1px solid rgba(18, 61, 55, 0.10)",
    borderRadius: "8px",
    backgroundColor:
      state === "active"
        ? "rgba(215, 235, 229, 0.78)"
        : state === "attention"
          ? "rgba(255, 251, 235, 0.94)"
          : state === "complete"
            ? "rgba(255, 255, 255, 0.88)"
            : "rgba(255, 255, 255, 0.66)",
    color: state === "attention" ? "#6b4f00" : "#10201c",
  }) as const;

const iconWrapStyle = (state: StepState) =>
  ({
    alignItems: "center",
    backgroundColor:
      state === "attention"
        ? "rgba(154, 103, 0, 0.12)"
        : state === "active"
          ? "rgba(15, 118, 110, 0.13)"
          : "rgba(18, 61, 55, 0.06)",
    borderRadius: "999px",
    color: state === "attention" ? "#9a6700" : "#0f766e",
    display: "inline-flex",
    height: "2rem",
    justifyContent: "center",
    width: "2rem",
  }) as const;

const stateKey = (state: StepState): string => {
  if (state === "complete") {
    return "speak.status_rail_state_complete";
  }
  if (state === "attention") {
    return "speak.status_rail_state_attention";
  }
  if (state === "active") {
    return "speak.status_rail_state_active";
  }
  return "speak.status_rail_state_upcoming";
};

const bodyKey = (phase: SpeakConfidencePhase): string => {
  if (phase === "submit") {
    return "speak.status_rail_body_submit";
  }
  if (phase === "assess") {
    return "speak.status_rail_body_assess";
  }
  if (phase === "failed") {
    return "speak.status_rail_body_failed";
  }
  if (phase === "prepare") {
    return "speak.status_rail_body_prepare";
  }
  return "speak.status_rail_body_record";
};

const stepState = (phase: SpeakConfidencePhase, stepId: RailStep["id"]): StepState => {
  const order: RailStep["id"][] = ["brief", "record", "submit", "review"];
  const activeStep =
    phase === "prepare"
      ? "brief"
      : phase === "record"
        ? "record"
        : phase === "submit"
          ? "submit"
          : "review";

  if (phase === "failed" && stepId === "review") {
    return "attention";
  }

  const stepIndex = order.indexOf(stepId);
  const activeIndex = order.indexOf(activeStep);
  if (stepIndex < activeIndex) {
    return "complete";
  }
  if (stepIndex === activeIndex) {
    return "active";
  }
  return "upcoming";
};

const steps: RailStep[] = [
  {
    icon: "guide",
    id: "brief",
    labelKey: "speak.status_rail_step_brief",
  },
  {
    icon: "microphone",
    id: "record",
    labelKey: "speak.status_rail_step_record",
  },
  {
    icon: "upload",
    id: "submit",
    labelKey: "speak.status_rail_step_submit",
  },
  {
    icon: "sparkle",
    id: "review",
    labelKey: "speak.status_rail_step_review",
  },
];

export const SpeakStatusRail = ({
  phase,
  translate,
}: {
  phase: SpeakConfidencePhase;
  translate: Translate;
}) => (
  <section
    aria-label={translate("speak.status_rail_title")}
    style={cardStyle}
    data-testid="speak.status_rail"
    data-semantic-id="speak.status_rail"
  >
    <div style={{ display: "grid", gap: "0.25rem" }}>
      <h2 style={{ margin: 0, color: "#10201c", fontSize: "1.05rem" }}>
        {translate("speak.status_rail_title")}
      </h2>
      <p style={{ margin: 0, color: "#33514b", lineHeight: 1.5 }}>
        {translate(bodyKey(phase))}
      </p>
    </div>
    <ol style={{ ...stepGridStyle, listStyle: "none", margin: 0, padding: 0 }}>
      {steps.map((step) => {
        const state = stepState(phase, step.id);

        return (
          <li
            key={step.id}
            style={stepStyle(state)}
            data-step-state={state}
            data-testid={`speak.status_rail_step_${step.id}`}
            data-semantic-id={`speak.status_rail_step_${step.id}`}
          >
            <span style={iconWrapStyle(state)}>
              <Icon name={state === "complete" ? "check" : state === "attention" ? "warning" : step.icon} size={18} />
            </span>
            <span style={{ fontWeight: 750 }}>{translate(step.labelKey)}</span>
            <span style={{ color: state === "attention" ? "#6b4f00" : "#48645e", fontSize: "0.85rem" }}>
              {translate(stateKey(state))}
            </span>
          </li>
        );
      })}
    </ol>
  </section>
);
