import "@testing-library/jest-dom/vitest";

import { act, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { AssessmentProgress } from "./AssessmentProgress";
import { createTranslator } from "@/lib/i18n";

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

it("warns after the backend deadline only while the worker is recently confirmed", () => {
  const started = Date.parse("2026-10-29T12:00:00Z");
  vi.useFakeTimers();
  vi.setSystemTime(started + 180_000);
  const status = { assessment_id: "asmt-1", status: "running" as const,
    phase: "scoring_rubric", progress: 0, created_at: "2026-10-29T12:00:00Z",
    updated_at: "2026-10-29T12:00:00Z", phase_started_at: "2026-10-29T12:00:00Z",
    worker_seen_at: "2026-10-29T12:02:59Z", quiet_warning_after_sec: 210 };
  const { rerender } = render(<AssessmentProgress status={status}
    uploadStartedAt={null} uploadUpdatedAt={null} uploading={false} uploaded translate={createTranslator("en")} />);

  expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "false");
  const announcement = screen.getByText("Working on Score.");
  act(() => { vi.advanceTimersByTime(31_000); });
  expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "false");
  rerender(<AssessmentProgress status={{ ...status, worker_seen_at: "2026-10-29T12:03:31Z" }}
    uploadStartedAt={null} uploadUpdatedAt={null} uploading={false} uploaded translate={createTranslator("en")} />);
  expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "true");
  expect(screen.getByText("Still working")).toBeVisible();
  expect(screen.getByTestId("speak.wait_time")).toHaveTextContent("Last progress update 3 min");
  expect(screen.getByTestId("speak.wait_time")).toHaveAttribute("role", "timer");
  expect(screen.getByTestId("speak.wait_time")).toHaveAttribute("aria-live", "off");
  expect(screen.getByText("Score").closest("li")).toHaveAttribute("aria-current", "step");
  expect(announcement).toHaveAttribute("aria-live", "polite");
  expect(announcement).toHaveTextContent("Working on Score.");
  rerender(<AssessmentProgress status={{ ...status, worker_seen_at: "2026-10-29T12:03:31Z" }}
    uploadStartedAt={null} uploadUpdatedAt={null} uploading={false} uploaded pollFailed translate={createTranslator("en")} />);
  expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "false");
  rerender(<AssessmentProgress status={{ ...status, worker_seen_at: "2026-10-29T12:00:00Z" }}
    uploadStartedAt={null} uploadUpdatedAt={null} uploading={false} uploaded translate={createTranslator("en")} />);
  expect(screen.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "false");
});

it("does not claim transcription while queued or coaching during finalization", () => {
  const now = Date.parse("2026-10-29T12:03:00Z");
  vi.spyOn(Date, "now").mockReturnValue(now);
  const status = { assessment_id: "asmt-1", status: "running" as const, phase: "queued", progress: 0,
    created_at: "2026-10-29T12:00:00Z", updated_at: "2026-10-29T12:00:00Z" };
  const { rerender } = render(<AssessmentProgress status={status} uploadStartedAt={null} uploadUpdatedAt={null}
    uploading={false} uploaded translate={createTranslator("en")} />);
  expect(screen.getByText("Waiting for the assessment worker.")).toBeVisible();
  expect(screen.getByText("Transcribe").closest("li")).not.toHaveAttribute("aria-current");
  rerender(<AssessmentProgress status={{ ...status, phase: "finalizing_report" }} uploadStartedAt={null} uploadUpdatedAt={null}
    uploading={false} uploaded translate={createTranslator("en")} />);
  expect(screen.getByText("Finalizing the report.")).toBeVisible();
  expect(screen.getByText("Coach").closest("li")).toHaveTextContent("Completed");
});

it("describes an unfinished upload as selected locally", () => {
  const now = Date.parse("2026-10-29T12:03:00Z");
  vi.spyOn(Date, "now").mockReturnValue(now);
  render(<AssessmentProgress uploadStartedAt={now - 180_000} uploadUpdatedAt={now - 180_000}
    uploading uploaded={false} translate={createTranslator("en")} />);
  expect(screen.getByTestId("speak.wait_time")).toHaveTextContent("3 min elapsed");
  expect(screen.getByText("Still working")).toBeVisible();
  expect(screen.getByText(/recording remains selected on this device/i)).toBeVisible();
  expect(screen.queryByText(/retained audio/i)).not.toBeInTheDocument();
});
