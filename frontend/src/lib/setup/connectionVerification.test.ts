import { describe, expect, it } from "vitest";

import { connectionVerificationKey, isRecentConnectionTest } from "./connectionVerification";

describe("connection verification", () => {
  it("expires a user-triggered test after five minutes and scopes it to the selected model", () => {
    const now = Date.now();
    expect(isRecentConnectionTest({ testedAt: now - 60_000 }, now)).toBe(true);
    expect(isRecentConnectionTest({ testedAt: now - 5 * 60_000 }, now)).toBe(false);
    expect(isRecentConnectionTest({ testedAt: now + 1 }, now)).toBe(false);
    expect(connectionVerificationKey({ configured: true, provider: "ollama", model: "one", base_url: "http://localhost", requires_api_key: false, has_api_key: false }))
      .not.toEqual(connectionVerificationKey({ configured: true, provider: "ollama", model: "two", base_url: "http://localhost", requires_api_key: false, has_api_key: false }));
  });
});
