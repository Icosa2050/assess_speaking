import { describe, expect, it } from "vitest";

import { createThemeLibraryRepository, isValidTaskFamily, languageCodes, normalizeThemeLibrary, parseSessionSetupContent } from "./sessionSetupContent";

describe("session setup content", () => {
  const validEnglishTemplates = {
    default_duration_minutes: "Speak for {minutes} minutes.",
    default_duration_seconds: "Speak for {seconds} seconds.",
    free_monologue: "Speak about {theme}.",
    opinion_monologue: "Give your opinion about {theme}.",
    personal_experience: "Describe your experience with {theme}.",
    picture_description: "Describe {theme}.",
    success_focus: ["Stay on topic."],
    travel_narrative: "Tell a story about {theme}.",
  };

  it("rejects shared content without required keys", () => {
    expect(() => parseSessionSetupContent({ default_theme_library: {} })).toThrow(
      /practice_brief_templates/,
    );
  });

  it("rejects shared content without complete English practice brief templates", () => {
    expect(() =>
      parseSessionSetupContent({
        default_theme_library: {},
        practice_brief_templates: {},
      }),
    ).toThrow(/English/);

    expect(() =>
      parseSessionSetupContent({
        default_theme_library: {},
        practice_brief_templates: {
          en: {
            ...validEnglishTemplates,
            travel_narrative: undefined,
          },
        },
      }),
    ).toThrow(/travel_narrative/);
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

  it("keeps stored German themes out of practice and rejects new themes for disabled languages", () => {
    const library = normalizeThemeLibrary({
      de: { label: "Deutsch", practice_enabled: true, themes: [{ title: "Mein Thema", level: "B1", task_family: "free_monologue" }] },
    });
    expect(library.de.themes.some((theme) => theme.title === "Mein Thema")).toBe(true);
    expect(languageCodes(library)).toEqual(["en", "it"]);
    expect(() => createThemeLibraryRepository().addCustomTheme({
      languageCode: "de", languageLabel: "Deutsch", title: "Noch ein Thema", level: "B1", taskFamily: "free_monologue",
    })).toThrow(/not enabled for practice/);
  });
});
