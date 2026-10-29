import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import tokens from "../../tokens.reference.json";

const tokenSheet = ["src/styles/tokens.css", "frontend/src/styles/tokens.css"]
  .map(path => resolve(process.cwd(), path)).find(existsSync);
if (!tokenSheet) throw new Error("Could not locate frontend/src/styles/tokens.css");
const css = readFileSync(tokenSheet, "utf8");
const themeBlocks = {
  light: css.match(/:root, \[data-theme="light"\]\s*\{([^}]*)\}/)?.[1] ?? "",
  dark: css.match(/\[data-theme="dark"\]\s*\{([^}]*)\}/)?.[1] ?? "",
};

const luminance = (hex: string) => {
  const channel = (start: number) => {
    const value = Number.parseInt(hex.slice(start, start + 2), 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
};

const contrast = (first: string, second: string) => {
  const [lighter, darker] = [luminance(first), luminance(second)].sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
};

const blend = (overlay: string, background: string): string => {
  const channels = overlay.match(/[\d.]+/g)?.map(Number);
  if (!channels || channels.length !== 4) throw new Error(`Expected RGBA, got ${overlay}`);
  const alpha = channels[3];
  const rgb = [0, 1, 2].map(index =>
    Math.round(channels[index] * alpha + Number.parseInt(background.slice(1 + index * 2, 3 + index * 2), 16) * (1 - alpha)));
  return `#${rgb.map(channel => channel.toString(16).padStart(2, "0")).join("")}`;
};

const pairs = [
  ["color-text", "color-surface", 4.5],
  ["color-text-muted", "color-surface", 4.5],
  ["color-brand", "color-surface", 4.5],
  ["color-brand", "color-selected", 4.5],
  ["color-on-brand", "color-brand", 4.5],
  ["color-on-brand", "color-score-fill", 4.5],
  ["color-border-control", "color-surface", 3],
  ["color-border-control", "color-surface-subtle", 3],
  ["color-border-control", "color-canvas", 3],
  ["color-focus", "color-surface", 3],
  ["color-focus", "color-surface-subtle", 3],
  ["color-focus", "color-canvas", 3],
  ["color-focus", "color-selected", 3],
  ["color-focus-on-brand", "color-brand", 3],
  ["color-selected-border", "color-selected", 3],
  ["color-disabled-text", "color-disabled-surface", 4.5],
  ["color-on-recording", "color-recording", 4.5],
  ["color-success", "color-success-bg", 4.5],
  ["color-warning", "color-warning-bg", 4.5],
  ["color-error", "color-error-bg", 4.5],
  ["color-info", "color-info-bg", 4.5],
] as const;

describe("used design token contrast", () => {
  for (const theme of ["light", "dark"] as const) {
    const colors = tokens.semantic[theme];
    it(`${theme}: CSS declarations match JSON semantic colors`, () => {
      for (const [name, value] of Object.entries(colors)) {
        expect(themeBlocks[theme]).toContain(`--${name}: ${value};`);
      }
    });
    it.each(pairs)(`${theme}: %s on %s meets %s:1`, (foreground, background, minimum) => {
      expect(contrast(colors[foreground], colors[background])).toBeGreaterThanOrEqual(minimum);
    });
    it(`${theme}: danger button text meets 4.5:1`, () => {
      const foreground = theme === "light" ? colors["color-surface"] : colors["color-canvas"];
      expect(contrast(foreground, colors["color-error"])).toBeGreaterThanOrEqual(4.5);
    });
    it(`${theme}: the hero secondary action has visible text and both border edges`, () => {
      const brand = colors["color-brand"];
      const tintedFill = blend(colors["color-on-brand-tint"], brand);
      const border = blend(colors["color-on-brand-border"], brand);
      expect(contrast(colors["color-on-brand"], tintedFill)).toBeGreaterThanOrEqual(4.5);
      expect(contrast(border, tintedFill)).toBeGreaterThanOrEqual(3);
      expect(contrast(border, brand)).toBeGreaterThanOrEqual(3);
    });
    it(`${theme}: every CEFR badge label meets normal-text contrast`, () => {
      for (const level of ["a1", "a2", "b1", "b2", "c1", "c2"] as const) {
        expect(contrast(colors[`color-cefr-${level}-text`], colors[`color-cefr-${level}-bg`]), level)
          .toBeGreaterThanOrEqual(4.5);
      }
    });
  }
});
