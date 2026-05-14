import { describe, expect, it } from "vitest";

import { isValidTaskFamily, normalizeThemeLibrary, parseSessionSetupContent } from "./sessionSetupContent";

describe("session setup content", () => {
  it("rejects shared content without required keys", () => {
    expect(() => parseSessionSetupContent({ default_theme_library: {} })).toThrow(
      /practice_brief_templates/,
    );
  });

  it("guards task families before assigning them", () => {
    expect(isValidTaskFamily("opinion_monologue")).toBe(true);
    expect(isValidTaskFamily("made_up_family")).toBe(false);
  });

  it("normalizes invalid theme task families to free monologue", () => {
    const library = normalizeThemeLibrary({
      en: {
        label: "English",
        themes: [
          {
            title: "A made up prompt",
            level: "B1",
            task_family: "made_up_family",
          },
        ],
      },
    });

    expect(library.en.themes.find((theme) => theme.title === "A made up prompt")?.task_family).toBe(
      "free_monologue",
    );
  });
});
