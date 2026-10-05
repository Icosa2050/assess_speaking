import { describe, expect, it } from "vitest";
import { createRehearsal, retryPart, secondsRemaining } from "./session";
const runtime = { provider: "ollama", model: "test", baseUrl: "http://localhost:11434", whisper: "tiny", feedbackLanguage: "en" };
describe("timed oral rehearsal", () => {
  for (const language of ["en", "it"] as const) for (const goal of ["B1", "B2", "C1"] as const) {
    it(`${language}/${goal} has 15 minutes of preparation and bounded speaking parts`, () => {
      const session = createRehearsal(language, goal, "learner", runtime);
      expect(session.preparationSec + session.parts.reduce((sum, part) => sum + part.durationSec, 0)).toBe(900);
      expect(session.parts.every(part => part.durationSec <= 300 && part.prompt.length > 50)).toBe(true);
      expect(new Set(session.parts.map(part => part.id)).size).toBe(3);
    });
  }
  it("links the chosen retry to its exact prompt, duration and saved parent", () => {
    const session = createRehearsal("it", "C1", "learner", runtime);
    session.parts[2].reportId = "parent";
    const retry = retryPart(session, 2);
    expect(retry.id).not.toBe(session.id);
    expect(retry.parts).toEqual([{ ...session.parts[2], reportId: undefined, retryOf: "parent" }]);
    expect(retry.runtime).toEqual(runtime);
    expect(retry.preparationSec).toBe(60);
    expect(() => retryPart(session, 0)).toThrow("saved attempt");
  });
  it("uses a deadline after a suspended/background tab, not interval tick counts", () => {
    expect(secondsRemaining(10000, 5000)).toBe(5);
    expect(secondsRemaining(10000, 9800)).toBe(1);
    expect(secondsRemaining(10000, 25000)).toBe(0);
  });
});
