import "@testing-library/jest-dom/vitest";

import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { SetupReadinessRow } from "@/lib/setup/readiness";

import { SetupReadinessPanel } from "./SetupReadinessPanel";

const labels: Record<string, string> = {
  "runtime_setup.setup_guide_title": "Setup Guide",
  "runtime_setup.setup_guide_body": "Confirm this device is ready for speaking practice.",
  "runtime_setup.setup_guide_anchor_title": "Ready to speak",
  "runtime_setup.setup_guide_anchor_body": "Speech recognition and the AI tutor power each practice. Microphone access is handled when you start a session.",
  "runtime_setup.setup_guide_progress_label": "{ready} of {total} core services ready",
  "runtime_setup.setup_guide_progress_status": "{ready}/{total} core services ready",
  "runtime_setup.setup_guide_progress_empty": "No core services available yet",
  "runtime_setup.setup_guide_speech_title": "Speech recognition",
  "runtime_setup.setup_guide_speech_setup": "Download a Whisper model before your first assessment.",
  "runtime_setup.setup_guide_ai_title": "AI tutor",
  "runtime_setup.setup_guide_ai_setup": "Connect one local or cloud AI option.",
  "runtime_setup.setup_guide_microphone_title": "Microphone",
  "runtime_setup.setup_guide_microphone_setup": "Microphone access is checked before recording.",
  "runtime_setup.setup_guide_sample_title": "Try a practice run",
  "runtime_setup.setup_guide_sample_blocked": "Finish speech recognition and AI tutor setup first.",
  "runtime_setup.setup_guide_download_model": "Download model",
  "runtime_setup.setup_guide_connect_ai": "Connect AI",
  "runtime_setup.setup_guide_check_microphone": "Set up a session",
  "runtime_setup.setup_guide_run_sample": "Start a session",
  "runtime_setup.setup_guide_status_loading": "Checking",
  "runtime_setup.setup_guide_status_ready": "Ready",
  "runtime_setup.setup_guide_status_setup": "Needs setup",
  "runtime_setup.setup_guide_status_unavailable": "Unavailable",
};

const translate = (key: string, vars?: Record<string, string | number>): string => {
  const value = labels[key] ?? key;

  if (!vars) {
    return value;
  }

  return value.replace(/\{(\w+)\}/g, (_, token) =>
    vars[token] === undefined ? `{${token}}` : String(vars[token]),
  );
};

const setupRows: SetupReadinessRow[] = [
  {
    key: "speech_recognition",
    status: "setup",
    titleKey: "runtime_setup.setup_guide_speech_title",
    detailKey: "runtime_setup.setup_guide_speech_setup",
    actionKey: "runtime_setup.setup_guide_download_model",
    disabled: false,
  },
  {
    key: "ai_tutor",
    status: "setup",
    titleKey: "runtime_setup.setup_guide_ai_title",
    detailKey: "runtime_setup.setup_guide_ai_setup",
    actionKey: "runtime_setup.setup_guide_connect_ai",
    disabled: false,
  },
  {
    key: "microphone",
    status: "setup",
    titleKey: "runtime_setup.setup_guide_microphone_title",
    detailKey: "runtime_setup.setup_guide_microphone_setup",
    actionKey: "runtime_setup.setup_guide_check_microphone",
    disabled: false,
  },
  {
    key: "sample_check",
    status: "setup",
    titleKey: "runtime_setup.setup_guide_sample_title",
    detailKey: "runtime_setup.setup_guide_sample_blocked",
    actionKey: "runtime_setup.setup_guide_run_sample",
    disabled: true,
  },
];

describe("SetupReadinessPanel", () => {
  it("renders readiness rows with status, recovery copy, and one action each", () => {
    render(
      <SetupReadinessPanel
        rows={setupRows}
        translate={translate}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: "Setup Guide" })).toBeVisible();
    expect(screen.getByTestId("runtime_setup.setup_guide")).toHaveAttribute(
      "data-semantic-id",
      "runtime_setup.setup_guide",
    );
    const speechRow = screen.getByTestId("runtime_setup.setup_guide.speech_recognition");
    expect(within(speechRow).getByText("Speech recognition")).toBeVisible();
    expect(within(speechRow).getByText("Needs setup")).toBeVisible();
    expect(within(speechRow).getByText("Download a Whisper model before your first assessment.")).toBeVisible();
    const speechAction = within(speechRow).getByRole("button", { name: "Download model" });
    expect(speechAction).toBeEnabled();
    expect(speechAction).toHaveAttribute(
      "data-semantic-id",
      "runtime_setup.setup_guide.speech_recognition.action",
    );

    expect(screen.getByRole("button", { name: "Start a session" })).toBeDisabled();
  });

  it("shows derived setup progress in the visual anchor without adding a duplicate action", () => {
    const rows: SetupReadinessRow[] = setupRows.map((row, index) =>
      index === 0 ? { ...row, status: "ready" } : row,
    );

    render(
      <SetupReadinessPanel
        rows={rows}
        translate={translate}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByText("Ready to speak")).toBeVisible();
    expect(
      screen.getByText(
        "Speech recognition and the AI tutor power each practice. Microphone access is handled when you start a session.",
      ),
    ).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "1 of 2 core services ready" }),
    ).toHaveAttribute("aria-valuenow", "1");
    expect(screen.getByText("1/2 core services ready")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Ready to speak" })).not.toBeInTheDocument();
  });

  it("marks core setup ready without pretending microphone permission was checked", () => {
    const rows: SetupReadinessRow[] = setupRows.map((row) =>
      row.key === "speech_recognition" || row.key === "ai_tutor"
        ? { ...row, status: "ready" }
        : row,
    );

    render(
      <SetupReadinessPanel
        rows={rows}
        translate={translate}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByRole("progressbar", { name: "2 of 2 core services ready" })).toHaveAttribute(
      "aria-valuemax",
      "2",
    );
    expect(screen.getByText("2/2 core services ready")).toBeVisible();
    expect(screen.getByTestId("runtime_setup.setup_guide.microphone")).toHaveAttribute(
      "data-status",
      "setup",
    );
  });

  it("keeps the readiness rows in the input order", () => {
    render(
      <SetupReadinessPanel
        rows={setupRows}
        translate={translate}
        onAction={vi.fn()}
      />,
    );

    const renderedRows = within(screen.getByTestId("runtime_setup.setup_guide")).getAllByRole(
      "listitem",
    );
    expect(renderedRows.map((row) => row.getAttribute("data-semantic-id"))).toEqual([
      "runtime_setup.setup_guide.speech_recognition",
      "runtime_setup.setup_guide.ai_tutor",
      "runtime_setup.setup_guide.microphone",
      "runtime_setup.setup_guide.sample_check",
    ]);
  });

  it("handles an empty readiness list without a NaN progress state", () => {
    render(
      <SetupReadinessPanel
        rows={[]}
        translate={translate}
        onAction={vi.fn()}
      />,
    );

    expect(screen.getByRole("progressbar", { name: "No core services available yet" })).toHaveAttribute(
      "aria-valuenow",
      "0",
    );
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
  });

  it("reports the selected readiness row when an action is clicked", () => {
    const onAction = vi.fn();
    render(
      <SetupReadinessPanel
        rows={setupRows}
        translate={translate}
        onAction={onAction}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Connect AI" }));

    expect(onAction).toHaveBeenCalledWith("ai_tutor");
  });
});
