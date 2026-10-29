import { afterEach, expect, it } from "vitest";
import { hasDesktopDraftStore, readDesktopRuntimeBridge } from "./desktopBridge";

afterEach(() => { delete (window as Window & { __VOSTAVO_DESKTOP__?: unknown }).__VOSTAVO_DESKTOP__; });

it("detects only a complete desktop draft bridge", () => {
  expect(hasDesktopDraftStore()).toBe(false);
  (window as any).__VOSTAVO_DESKTOP__ = { draftStore: { list: () => Promise.resolve([]) } };
  expect(hasDesktopDraftStore()).toBe(false);
  expect(readDesktopRuntimeBridge().draftStore).toBeUndefined();
  const methods = ["list", "recoveryIds", "saveSetup", "beginTake", "writeTakeChunk", "abortTake", "finishTake", "loadTakeChunk", "markSubmission", "discard", "expire"];
  const draftStore = Object.fromEntries(methods.map((name) => [name, () => Promise.resolve()]));
  (window as any).__VOSTAVO_DESKTOP__ = { draftStore };
  expect(hasDesktopDraftStore()).toBe(true);
  expect(readDesktopRuntimeBridge().draftStore).toBe(draftStore);
});
