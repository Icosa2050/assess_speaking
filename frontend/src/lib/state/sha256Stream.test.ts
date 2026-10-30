import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { Sha256Stream, sha256File } from "./sha256Stream";

describe("streaming SHA-256", () => {
  it.each(["", "abc", "a".repeat(55), "a".repeat(56), "a".repeat(64), "a".repeat(65), "z".repeat(300_001)])(
    "matches native SHA-256 across block boundaries", value => {
      const bytes = new TextEncoder().encode(value);
      const hasher = new Sha256Stream();
      for (let index = 0; index < bytes.length; index += 17) hasher.update(bytes.subarray(index, index + 17));
      expect(hasher.hex()).toBe(createHash("sha256").update(bytes).digest("hex"));
    },
  );

  it("reads a file in bounded slices", async () => {
    const bytes = new Uint8Array(300_001).fill(42);
    const file = new File([bytes], "take.wav");
    const original = file.slice.bind(file);
    const sizes: number[] = [];
    file.slice = (start, end, type) => { sizes.push(Number(end) - Number(start)); return original(start, end, type); };
    expect(await sha256File(file)).toBe(createHash("sha256").update(bytes).digest("hex"));
    expect(sizes).toEqual([262144, 262144]);
  });
});
