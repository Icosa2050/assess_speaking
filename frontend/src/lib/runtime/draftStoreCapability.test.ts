import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { hasDesktopDraftStore, readDesktopRuntimeBridge } from "./desktopBridge";

afterEach(() => { delete (window as Window & { __VOSTAVO_DESKTOP__?: unknown }).__VOSTAVO_DESKTOP__; });

it("detects only a complete desktop draft bridge", () => {
  expect(hasDesktopDraftStore()).toBe(false);
  (window as any).__VOSTAVO_DESKTOP__ = { draftStore: { list: () => Promise.resolve([]) } };
  expect(hasDesktopDraftStore()).toBe(false);
  expect(readDesktopRuntimeBridge().draftStore).toBeUndefined();
  const methods = ["list", "recoveryIds", "saveSetup", "replaceSetup", "importLegacyBrief", "beginTake", "writeTakeChunk", "abortTake", "abortTakeForDraft", "finishTake", "loadTakeChunk", "markSubmission", "discard", "expire"];
  const draftStore = Object.fromEntries(methods.map((name) => [name, () => Promise.resolve()]));
  (window as any).__VOSTAVO_DESKTOP__ = { draftStore: { ...draftStore, replaceSetup: undefined } };
  expect(hasDesktopDraftStore()).toBe(false);
  (window as any).__VOSTAVO_DESKTOP__ = { draftStore: { ...draftStore, abortTakeForDraft: undefined } };
  expect(hasDesktopDraftStore()).toBe(false);
  (window as any).__VOSTAVO_DESKTOP__ = { draftStore };
  expect(hasDesktopDraftStore()).toBe(true);
  expect(readDesktopRuntimeBridge().draftStore).toBe(draftStore);
});

it("sends only bounded raw take chunks and permits Tauri IPC in the desktop CSP", async () => {
  const tauriRoot = ["src-tauri", "frontend/src-tauri"]
    .map(path => resolve(process.cwd(), path)).find(existsSync);
  if (!tauriRoot) throw new Error("Could not locate Tauri shell sources");
  const source = readFileSync(resolve(tauriRoot, "src/main.rs"), "utf8");
  const script = source.match(/const DRAFT_BRIDGE_JS: &str = r#"([\s\S]*?)"#;/)?.[1];
  expect(script).toBeTruthy();
  const invoke = vi.fn().mockResolvedValue(2);
  const draftStore = (new Function("window", `return ${script}`) as (scope: unknown) => {
    writeTakeChunk: (id: string, bytes: number[]) => Promise<number>;
  })({ __TAURI_INTERNALS__: { invoke } });
  await expect(draftStore.writeTakeChunk("transfer", new Array(262145).fill(1))).rejects.toThrow("Take chunk exceeds limit");
  expect(invoke).not.toHaveBeenCalled();
  await expect(draftStore.writeTakeChunk("transfer", [1, 2])).resolves.toBe(2);
  expect(invoke).toHaveBeenCalledWith("draft_write_take_chunk", new Uint8Array([1, 2]), {
    headers: { "X-Vostavo-Transfer-Id": "transfer" },
  });
  const config = JSON.parse(readFileSync(resolve(tauriRoot, "tauri.conf.json"), "utf8"));
  expect(config.app.security.csp).toContain("connect-src 'self' ipc: http://ipc.localhost");
});
