import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { AssessmentProgress } from "./AssessmentProgress";
import { createTranslator } from "@/lib/i18n";

afterEach(() => vi.restoreAllMocks());

it("explains a three-minute scoring stall without announcing a changing timer", () => {
  const now = Date.parse("2026-10-29T12:03:00Z");
  vi.spyOn(Date, "now").mockReturnValue(now);
  render(<AssessmentProgress status={{ assessment_id: "asmt-1", status: "running",
    phase: "scoring_rubric", progress: 0, created_at: "2026-10-29T12:00:00Z",
    updated_at: "2026-10-29T12:00:00Z", phase_started_at: "2026-10-29T12:00:00Z" }}
    uploadStartedAt={null} uploadUpdatedAt={null} uploading={false} uploaded translate={createTranslator("en")} />);

  expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "true");
  expect(screen.getByText("Still working")).toBeVisible();
  expect(screen.getByTestId("speak.wait_time")).toHaveTextContent("Last update 3 min 0 sec ago");
  expect(screen.getByTestId("speak.wait_time")).toHaveAttribute("aria-hidden", "true");
  expect(screen.getByText("Score").closest("li")).toHaveAttribute("aria-current", "step");
  expect(screen.getByText("Still working on Score.")).toHaveAttribute("aria-live", "polite");
});

it("describes an unfinished upload as selected locally", () => {
  const now = Date.parse("2026-10-29T12:03:00Z");
  vi.spyOn(Date, "now").mockReturnValue(now);
  render(<AssessmentProgress uploadStartedAt={now - 180_000} uploadUpdatedAt={now - 180_000}
    uploading uploaded={false} translate={createTranslator("en")} />);
  expect(screen.getByText(/recording remains selected on this device/i)).toBeVisible();
  expect(screen.queryByText(/retained audio/i)).not.toBeInTheDocument();
});
