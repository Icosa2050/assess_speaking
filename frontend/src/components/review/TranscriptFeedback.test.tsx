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
  it("matches complete phrases and finds a later free occurrence after overlap", () => {
    expect(transcriptSegments("This is useful.", [{ original: "is", suggestion: "was", explanation: "Use past tense." }])
      .filter(part => part.suggestion).map(part => part.text)).toEqual(["is"]);
    expect(transcriptSegments("This works.", [{ original: "is", suggestion: "was", explanation: "Use past tense." }])
      .some(part => part.suggestion)).toBe(false);
    const segments = transcriptSegments("I goes and I goes", [
      suggestions[0], { original: "goes", suggestion: "go", explanation: "Use present tense." },
    ]);
    expect(segments.map(part => part.text).join("")).toBe("I goes and I goes");
    expect(segments.filter(part => part.suggestion).map(part => part.text)).toEqual(["I goes", "goes"]);
  });
  it("marks every non-overlapping repeat of a suggestion", () => {
    const transcript = "I goes, and then I goes again.";
    expect(transcriptSegments(transcript, suggestions).filter(part => part.suggestion).map(part => part.text))
      .toEqual(["I goes", "I goes"]);
    render(<TranscriptFeedback transcript={transcript} suggestions={suggestions} translate={createTranslator("en")} />);
    const phrases = screen.getAllByRole("button", { name: "Feedback for “I goes”" });
    expect(phrases).toHaveLength(2);
    expect(phrases[0].getAttribute("aria-controls")).not.toBe(phrases[1].getAttribute("aria-controls"));
    fireEvent.click(phrases[1]);
    expect(phrases[1]).toHaveAttribute("aria-expanded", "true");
    expect(phrases[0]).toHaveAttribute("aria-expanded", "false");
  });
  it("offers a named, described, focusable phrase control", () => {
    render(<TranscriptFeedback transcript="I goes home." suggestions={suggestions} translate={createTranslator("en")} />);
    const phrase = screen.getByRole("button", { name: "Feedback for “I goes”" });
    expect(phrase).toHaveAttribute("aria-expanded", "false");
    const description = document.getElementById(phrase.getAttribute("aria-controls")!);
    expect(description).toHaveTextContent("Suggestion: I go. Use present tense here.");
    expect(description).not.toBeVisible();
    expect(phrase).not.toHaveAttribute("aria-describedby");
    phrase.focus();
    fireEvent.click(phrase);
    expect(phrase).toHaveAttribute("aria-expanded", "true");
    expect(phrase).toHaveAttribute("aria-describedby", description?.id);
    expect(description).toBeVisible();
  });
});
