import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Icon } from "@/components/ui/Icon";
import { ProgressRing } from "@/components/ui/ProgressRing";
import { Sparkline } from "@/components/ui/Sparkline";

describe("visual primitives", () => {
  it("renders decorative icons as hidden from assistive technology", () => {
    const { container } = render(<Icon name="microphone" />);

    const icon = container.querySelector("svg");

    expect(icon).toHaveAttribute("aria-hidden", "true");
    expect(icon).not.toHaveAttribute("role");
    expect(icon?.querySelector("path, rect, circle")).toBeInTheDocument();
  });

  it.each(["upload", "stop"] as const)("renders the %s icon as decorative by default", (name) => {
    const { container } = render(<Icon name={name} />);

    const icon = container.querySelector("svg");

    expect(icon).toHaveAttribute("aria-hidden", "true");
    expect(icon).not.toHaveAttribute("role");
    expect(icon?.querySelector("path, rect, circle")).toBeInTheDocument();
  });

  it("renders titled icons as named images", () => {
    render(<Icon name="target" title="Practice target" />);

    expect(screen.getByRole("img", { name: "Practice target" })).toBeVisible();
  });

  it("clamps progress values and exposes the status accessibly", () => {
    render(
      <ProgressRing
        label="Practice readiness"
        max={100}
        status="Ready on this device"
        value={125}
      />,
    );

    const meter = screen.getByRole("progressbar", { name: "Practice readiness" });

    expect(meter).toHaveAttribute("aria-valuemin", "0");
    expect(meter).toHaveAttribute("aria-valuemax", "100");
    expect(meter).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText("Ready on this device")).toBeVisible();
  });

  it("renders a named sparkline with a text summary", () => {
    render(
      <Sparkline
        label="Score trend"
        summary="4.0 -> 4.2"
        testId="score-trend"
        values={[4.0, 4.2]}
      />,
    );

    expect(screen.getByRole("img", { name: "Score trend" })).toBeVisible();
    expect(screen.getByTestId("score-trend-summary")).toHaveTextContent("4.0 -> 4.2");
  });
});
