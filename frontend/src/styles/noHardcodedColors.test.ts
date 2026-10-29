import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const colorLiteral = /#[\da-f]{3,8}\b|\b(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch)\s*\(/gi;
const cssNamedColors = `aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue
  blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan
  darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen
  darkorange darkorchid darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey
  darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite
  forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink
  indianred indigo ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral
  lightcyan lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen
  lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen magenta
  maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue
  mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin
  navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen
  paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple
  red rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue
  slateblue slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet
  wheat white whitesmoke yellow yellowgreen`.trim().split(/\s+/);
const namedColor = new RegExp(
  String.raw`\b[\w-]*(?:color|background|border|outline|shadow|fill|stroke|decoration|caret)[\w-]*\s*:\s*[^;{}\n]*?(?<![\w-])(?:${cssNamedColors.join("|")})(?![\w-])`,
  "gi",
);

const isNavigationFragment = (source: string, offset: number): boolean => {
  const line = source.slice(source.lastIndexOf("\n", offset - 1) + 1, offset);
  return /\b(?:href|to)\s*(?:=|:)\s*["']$/.test(line);
};

const violations = (source: string): string[] =>
  [
    ...[...source.matchAll(colorLiteral)].filter(match => !isNavigationFragment(source, match.index)),
    ...source.matchAll(namedColor),
  ].map(match => match[0]);

const shouldScan = (path: string, tokenSheet: string): boolean =>
  /\.(?:css|html|js|ts|tsx)$/.test(path) && resolve(path) !== tokenSheet;

const sourceFiles = (directory: string, tokenSheet: string): string[] =>
  readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path, tokenSheet);
    return shouldScan(path, tokenSheet) ? [path] : [];
  });

describe("semantic color boundary", () => {
  it("detects a newly introduced inline or stylesheet color", () => {
    const hash = String.fromCharCode(35);
    expect(violations(`color: ${hash}abc`)).toHaveLength(1);
    expect(violations(`backgroundColor: "${"rgb"}(1, 2, 3)"`)).toHaveLength(1);
    expect(violations(["color", "white"].join(": "))).toHaveLength(1);
    const shorthand = ["border", "1px solid", "white"].join(" ");
    expect(violations(shorthand.replace("border ", "border: "))).toHaveLength(1);
    const shadow = ["boxShadow", "0 0 0 1px", ["rebecca", "purple"].join("")].join(" ");
    expect(violations(shadow.replace("boxShadow ", "boxShadow: "))).toHaveLength(1);
    const hashFragment = ["href=\"", hash, "deadbeef\""].join("");
    expect(violations(hashFragment)).toHaveLength(0);
    expect(violations(["color: \"", hash, "deadbeef\""].join(""))).toHaveLength(1);
    expect(shouldScan(resolve("src/other/tokens.css"), resolve("src/styles/tokens.css"))).toBe(true);
    expect(shouldScan(resolve("src/styles/tokens.css"), resolve("src/styles/tokens.css"))).toBe(false);
    expect(violations("color: var(--color-text)")).toHaveLength(0);
  });

  it("keeps all frontend source colors in tokens.css", () => {
    const src = [resolve(process.cwd(), "src"), resolve(process.cwd(), "frontend/src")]
      .find(path => { try { return readdirSync(path).length > 0; } catch { return false; } });
    if (!src) throw new Error("Could not locate frontend/src");
    const frontend = resolve(src, "..");
    const tokenSheet = resolve(src, "styles/tokens.css");
    const paths = [...sourceFiles(src, tokenSheet), ...sourceFiles(resolve(frontend, "public"), tokenSheet), resolve(frontend, "index.html")];
    const found = paths.flatMap(path => violations(readFileSync(path, "utf8")).map(value => `${path}: ${value}`));
    expect(found).toEqual([]);
  });
});
