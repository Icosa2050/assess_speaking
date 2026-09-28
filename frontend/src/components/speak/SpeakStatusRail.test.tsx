import "@testing-library/jest-dom/vitest";

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createTranslator } from "@/lib/i18n";

import { SpeakStatusRail, type SpeakConfidencePhase } from "./SpeakStatusRail";

const translate = createTranslator("en");

const renderRail = (phase: SpeakConfidencePhase) => {
  render(<SpeakStatusRail phase={phase} translate={translate} />);
};

describe("SpeakStatusRail", () => {
  it("keeps the learner oriented while they are preparing to record", () => {
    renderRail("record");

    expect(screen.getByTestId("speak.status_rail")).toHaveAttribute(
      "data-semantic-id",
      "speak.status_rail",
    );
    expect(screen.getByTestId("speak.status_rail_step_brief")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_record")).toHaveAttribute(
      "data-step-state",
      "active",
    );
    expect(screen.getByTestId("speak.status_rail_step_submit")).toHaveAttribute(
      "data-step-state",
      "upcoming",
    );
    expect(screen.getByTestId("speak.status_rail_step_review")).toHaveAttribute(
      "data-step-state",
      "upcoming",
    );
    expect(screen.getByText("Your path through this attempt")).toBeVisible();
    expect(screen.getByText("Record when you are ready. Nothing is judged until you submit.")).toBeVisible();
  });

  it("moves the active step to submit once audio is attached", () => {
    renderRail("submit");

    expect(screen.getByTestId("speak.status_rail_step_brief")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_record")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_submit")).toHaveAttribute(
      "data-step-state",
      "active",
    );
    expect(screen.getByTestId("speak.status_rail_step_review")).toHaveAttribute(
      "data-step-state",
      "upcoming",
    );
    expect(screen.getByText("Listen once if you want, then send the take for coaching.")).toBeVisible();
  });

  it("shows assessment as the active path toward review while work is running", () => {
    renderRail("assess");

    expect(screen.getByTestId("speak.status_rail_step_brief")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_record")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_submit")).toHaveAttribute(
      "data-step-state",
      "complete",
    );
    expect(screen.getByTestId("speak.status_rail_step_review")).toHaveAttribute(
      "data-step-state",
      "active",
    );
    expect(screen.getByText("Your review will open automatically when coaching is ready.")).toBeVisible();
  });

  it("marks review as needing attention after a failed assessment", () => {
    renderRail("failed");

    const reviewStep = screen.getByTestId("speak.status_rail_step_review");
    expect(reviewStep).toHaveAttribute("data-step-state", "attention");
    expect(within(reviewStep).getByText("Needs attention")).toBeVisible();
    expect(screen.getByText("Something interrupted the assessment. Adjust the take or try again.")).toBeVisible();
  });
});
