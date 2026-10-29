import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const colorLiteral = /#[\da-f]{3,8}\b|\b(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch)\s*\(/gi;
const namedColor = /\b(?:color|background(?:Color|-color)?|border(?:Color|-color|-(?:top|right|bottom|left)(?:-color)?)?|outline(?:Color|-color)?|boxShadow|box-shadow|textShadow|text-shadow|fill|stroke)\s*:\s*[^;{}\n]*?(?<![-\w])(?:white|black|red|blue|green|yellow|orange|purple)\b/gi;

const violations = (source: string): string[] =>
  [...source.matchAll(colorLiteral), ...source.matchAll(namedColor)].map(match => match[0]);

const sourceFiles = (directory: string): string[] =>
  readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return /\.(?:css|ts|tsx)$/.test(entry.name) && entry.name !== "tokens.css" ? [path] : [];
  });

describe("semantic color boundary", () => {
  it("detects a newly introduced inline or stylesheet color", () => {
    const hash = String.fromCharCode(35);
    expect(violations(`color: ${hash}abc`)).toHaveLength(1);
    expect(violations(`backgroundColor: "${"rgb"}(1, 2, 3)"`)).toHaveLength(1);
    expect(violations(["color", "white"].join(": "))).toHaveLength(1);
    expect(violations(["border", "1px solid", "red"].join(" ").replace("border ", "border: "))).toHaveLength(1);
    expect(violations("color: var(--color-text)")).toHaveLength(0);
  });

  it("keeps all frontend source colors in tokens.css", () => {
    const src = [resolve(process.cwd(), "src"), resolve(process.cwd(), "frontend/src")]
      .find(path => { try { return readdirSync(path).length > 0; } catch { return false; } });
    if (!src) throw new Error("Could not locate frontend/src");
    const found = sourceFiles(src).flatMap(path => violations(readFileSync(path, "utf8")).map(value => `${path}: ${value}`));
    expect(found).toEqual([]);
  });
});
