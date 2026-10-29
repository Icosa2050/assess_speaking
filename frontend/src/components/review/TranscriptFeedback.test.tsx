import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TranscriptFeedback, transcriptSegments } from "./TranscriptFeedback";
import { createTranslator } from "@/lib/i18n";

const suggestions = [{ original: "I goes", suggestion: "I go", explanation: "Use present tense here." }];

describe("TranscriptFeedback", () => {
  it("preserves original words and skips suggestions that do not match the transcript", () => {
    expect(transcriptSegments("I goes home.", suggestions).map(part => part.text).join("")).toBe("I goes home.");
    expect(transcriptSegments("I went home.", suggestions).some(part => part.suggestion)).toBe(false);
  });
  it("offers a named, described, focusable phrase control", () => {
    render(<TranscriptFeedback transcript="I goes home." suggestions={suggestions} translate={createTranslator("en")} />);
    const phrase = screen.getByRole("button", { name: "Feedback for “I goes”" });
    expect(phrase).toHaveAttribute("aria-expanded", "false");
    const description = document.getElementById(phrase.getAttribute("aria-describedby")!);
    expect(description).toHaveTextContent("Suggestion: I go. Use present tense here.");
    phrase.focus();
    fireEvent.click(phrase);
    expect(phrase).toHaveAttribute("aria-expanded", "true");
    expect(description).toBeVisible();
  });
});
