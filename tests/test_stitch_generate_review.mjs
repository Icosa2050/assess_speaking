import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  generatedScreensFromToolResponse,
  newScreenIds,
  parseArgs,
  runWithClient,
  screenIdsFromListResponse,
  validateArgs,
} from "../scripts/stitch_generate_review.mjs";

test("parseArgs keeps SDK generation defaults", () => {
  const args = parseArgs([
    "--project-id",
    "123",
    "--design-system-asset-id",
    "asset",
    "--manifest",
    "manifest.json",
  ]);

  assert.equal(args.projectId, "123");
  assert.equal(args.designSystemAssetId, "asset");
  assert.equal(args.manifest, "manifest.json");
  assert.equal(args.modelId, "GEMINI_3_FLASH");
  assert.equal(args.deviceType, "DESKTOP");
  assert.equal(args.timeoutMs, 300000);
});

test("validateArgs rejects missing required values", () => {
  assert.throws(
    () => validateArgs(parseArgs(["--project-id", "123"])),
    /--design-system-asset-id/,
  );
});

test("screenIdsFromListResponse extracts bare ids", () => {
  const ids = screenIdsFromListResponse({
    screens: [
      { name: "projects/123/screens/aaa" },
      { name: "projects/123/screens/bbb" },
      { name: "" },
    ],
  });

  assert.deepEqual(ids, ["aaa", "bbb"]);
});

test("generatedScreensFromToolResponse projects design output", () => {
  const screens = generatedScreensFromToolResponse({
    outputComponents: [
      {
        design: {
          screens: [
            {
              id: "screen-1",
              name: "projects/123/screens/screen-1",
              title: "Home",
              width: "1280",
              height: "900",
              screenMetadata: { status: "COMPLETE" },
            },
          ],
        },
      },
    ],
  });

  assert.deepEqual(screens, [
    {
      screenId: "screen-1",
      name: "projects/123/screens/screen-1",
      title: "Home",
      width: "1280",
      height: "900",
      status: "COMPLETE",
    },
  ]);
});

test("newScreenIds compares before and after sets", () => {
  assert.deepEqual(newScreenIds(["a", "b"], ["b", "c", "a", "d"]), ["c", "d"]);
});

test("runWithClient writes a manifest when generation breaks polling", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "stitch-generate-review-"));
  const manifestPath = path.join(tempDir, "manifest.json");
  let listCalls = 0;
  const client = {
    async callTool(name) {
      if (name === "list_screens") {
        listCalls += 1;
        if (listCalls === 1) {
          return { screens: [{ name: "projects/123/screens/existing" }] };
        }
        throw new Error("Already connected to a transport");
      }
      if (name === "generate_screen_from_text") {
        throw new Error("Socket closed before generation completed");
      }
      throw new Error(`unexpected tool: ${name}`);
    },
  };

  const manifest = await runWithClient(client, {
    projectId: "123",
    designSystemAssetId: "asset",
    manifest: manifestPath,
    prompt: "Generate home",
    modelId: "GEMINI_3_FLASH",
    deviceType: "DESKTOP",
    timeoutMs: 300000,
    pollCount: 1,
    pollIntervalMs: 0,
  });
  const written = JSON.parse(await fs.readFile(manifestPath, "utf8"));

  assert.equal(manifest.status, "generation_error_poll_failed");
  assert.equal(written.status, "generation_error_poll_failed");
  assert.deepEqual(written.beforeScreenIds, ["existing"]);
  assert.match(written.error, /Socket closed/);
  assert.match(written.listAfterError, /Already connected/);
});

test("runWithClient can poll with a fresh client after transport failure", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "stitch-generate-review-"));
  const manifestPath = path.join(tempDir, "manifest.json");
  let brokenListCalls = 0;
  const brokenClient = {
    async callTool(name) {
      if (name === "list_screens") {
        brokenListCalls += 1;
        if (brokenListCalls > 1) {
          throw new Error("Already connected to a transport");
        }
        return { screens: [{ name: "projects/123/screens/existing" }] };
      }
      if (name === "generate_screen_from_text") {
        throw new Error("fetch failed");
      }
      throw new Error(`unexpected tool: ${name}`);
    },
  };
  const freshClient = {
    closed: false,
    async callTool(name) {
      assert.equal(name, "list_screens");
      return {
        screens: [
          { name: "projects/123/screens/existing" },
          { name: "projects/123/screens/new-screen" },
        ],
      };
    },
    async close() {
      this.closed = true;
    },
  };

  const manifest = await runWithClient(
    brokenClient,
    {
      projectId: "123",
      designSystemAssetId: "asset",
      manifest: manifestPath,
      prompt: "Generate home",
      modelId: "GEMINI_3_FLASH",
      deviceType: "DESKTOP",
      timeoutMs: 300000,
      pollCount: 1,
      pollIntervalMs: 0,
    },
    { createClient: () => freshClient },
  );

  assert.equal(manifest.status, "generation_error_but_new_screen_detected");
  assert.equal(manifest.usedFreshClientForPolling, true);
  assert.deepEqual(manifest.inferredNewScreenIds, ["new-screen"]);
  assert.equal(freshClient.closed, true);
});
