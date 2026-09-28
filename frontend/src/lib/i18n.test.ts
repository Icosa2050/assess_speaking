import { describe, expect, it } from "vitest";

import { flattenLocaleKeys, loadLocaleMessages, t } from "@/lib/i18n";
import { SUPPORTED_UI_LOCALES } from "@/lib/state/sessionDraft";

const placeholders = (text: string) =>
  [...new Set([...text.matchAll(/\{(\w+)\}/g)].map((match) => match[1]))].sort();

describe("translation catalog integrity", () => {
  const englishKeys = flattenLocaleKeys(loadLocaleMessages("en"));

  for (const locale of SUPPORTED_UI_LOCALES) {
    it(`${locale} preserves keys, interpolation tokens, and valid Unicode`, () => {
      expect(flattenLocaleKeys(loadLocaleMessages(locale))).toEqual(englishKeys);
      for (const key of englishKeys) {
        const text = t(key, { locale });
        expect(text.trim(), `${locale}:${key}`).not.toBe("");
        expect(placeholders(text), `${locale}:${key}`).toEqual(placeholders(t(key, { locale: "en" })));
        expect(text, `${locale}:${key}`).toBe(text.normalize("NFC"));
        expect(text, `${locale}:${key}`).not.toMatch(/\uFFFD|Ã[\u0080-\u00BF]|Â[\u0080-\u00BF]/);
      }
    });
  }

  it("keeps German diacritics instead of ASCII transliterations", () => {
    const text = [...englishKeys].map((key) => t(key, { locale: "de" })).join(" ");
    for (const character of ["ä", "ö", "ü", "ß"]) {
      expect(text).toContain(character);
    }
    expect(text).not.toMatch(/\b(?:fuer|ueber|verfuegbar|Uebung\w*|Pruef\w*|vollstaendig\w*|Oeffne\w*)\b/i);
  });

  it("preserves accented user input when interpolating translated messages", () => {
    for (const locale of SUPPORTED_UI_LOCALES) {
      const translated = t("common.session_summary", {
        locale,
        vars: {
          session_id: "übung-1",
          speaker_id: "Müller · François · José · Niccolò",
          ui_locale: locale,
          learning_language: "Français",
          cefr: "B1",
        },
      });
      expect(translated).toContain("Müller · François · José · Niccolò");
      expect(translated).toContain("Français");
      expect(translated).not.toMatch(/\{\w+\}/);
    }
  });
});
