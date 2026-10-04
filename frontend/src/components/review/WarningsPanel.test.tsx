import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createTranslator } from "@/lib/i18n";
import { WarningsPanel } from "./WarningsPanel";

describe("transcript uncertainty warnings", () => {
  it("localizes the skipped-rubric warning without repeating the uncertainty message", () => {
    render(<WarningsPanel warnings={["transcript_uncertain", "llm_skipped_transcript_uncertain"]}
      translate={createTranslator("en")} failedGates={[]} requiresHumanReview={false} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    expect(screen.getByRole("listitem")).toHaveTextContent("Listen to the recording");
    expect(screen.queryByText("llm skipped transcript uncertain")).not.toBeInTheDocument();
  });
});
