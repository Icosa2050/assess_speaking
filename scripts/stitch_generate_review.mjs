#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const DEFAULT_PROMPT = [
  "Create one desktop Home screen concept for Vostavo, a speaking-practice app for adult language learners.",
  "Use the configured Vostavo Learner Coach design system.",
  "The screen must be an actual app screen, not a landing page.",
  "Prioritize the learner's next best action: start or continue a speaking practice session.",
  "Include a compact system status row, a primary practice panel, progress or focus summary, and secondary navigation to Review, History, Library, Guide, and Settings.",
  "Keep runtime setup secondary after configuration.",
  "Do not use decorative gradients, orbs, bokeh, nested cards, or oversized marketing hero composition.",
  "Use stable dimensions, 8px or smaller card/control radius, Inter, localized-string-friendly labels, and semantic automation-friendly controls.",
].join(" ");

const DEFAULT_BASE_URL = "https://stitch.googleapis.com/mcp";

export function parseArgs(argv) {
  const args = {
    baseUrl: DEFAULT_BASE_URL,
    deviceType: "DESKTOP",
    modelId: "GEMINI_3_FLASH",
    timeoutMs: 300000,
    pollCount: 2,
    pollIntervalMs: 30000,
    prompt: DEFAULT_PROMPT,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) {
        throw new Error(`Missing value for ${item}`);
      }
      return argv[index];
    };

    if (item === "--project-id") args.projectId = next();
    else if (item === "--design-system-asset-id") args.designSystemAssetId = next();
    else if (item === "--manifest") args.manifest = next();
    else if (item === "--prompt") args.prompt = next();
    else if (item === "--prompt-file") args.promptFile = next();
    else if (item === "--base-url") args.baseUrl = next();
    else if (item === "--device-type") args.deviceType = next();
    else if (item === "--model-id") args.modelId = next();
    else if (item === "--timeout-ms") args.timeoutMs = Number.parseInt(next(), 10);
    else if (item === "--poll-count") args.pollCount = Number.parseInt(next(), 10);
    else if (item === "--poll-interval-ms") args.pollIntervalMs = Number.parseInt(next(), 10);
    else if (item === "--help" || item === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${item}`);
  }

  return args;
}

export function usage() {
  return [
    "Usage:",
    "  npx --package @google/stitch-sdk@0.3.5 node scripts/stitch_generate_review.mjs \\",
    "    --project-id PROJECT_ID \\",
    "    --design-system-asset-id ASSET_ID \\",
    "    --manifest docs/ux-audit-screenshots/2026-05-26/stitch-generation-manifest.json",
    "",
    "Environment:",
    "  STITCH_API_KEY, or STITCH_ACCESS_TOKEN plus GOOGLE_CLOUD_PROJECT",
    "  STITCH_SDK_IMPORT_PATH may point at a temporary SDK index.js outside this repo",
  ].join("\n");
}

export function validateArgs(args) {
  const missing = [];
  if (!args.projectId) missing.push("--project-id");
  if (!args.designSystemAssetId) missing.push("--design-system-asset-id");
  if (!args.manifest) missing.push("--manifest");
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) missing.push("--timeout-ms");
  if (!Number.isFinite(args.pollCount) || args.pollCount < 0) missing.push("--poll-count");
  if (!Number.isFinite(args.pollIntervalMs) || args.pollIntervalMs < 0) {
    missing.push("--poll-interval-ms");
  }
  if (missing.length > 0) {
    throw new Error(`Missing or invalid required argument(s): ${missing.join(", ")}`);
  }
}

export function screenIdFromName(name) {
  if (typeof name !== "string") return "";
  return name.split("/").at(-1) || "";
}

export function screenIdsFromListResponse(response) {
  return (response?.screens || []).map((screen) => screenIdFromName(screen.name)).filter(Boolean);
}

export function generatedScreensFromToolResponse(response) {
  const components = response?.outputComponents || response?.output_components || [];
  return components.flatMap((component) => {
    const screens = component?.design?.screens || [];
    return screens
      .map((screen) => {
        const screenId = screen.id || screen.screenId || screenIdFromName(screen.name);
        return screenId
          ? {
              screenId,
              name: screen.name || `projects/${screen.projectId || ""}/screens/${screenId}`,
              title: screen.title || null,
              width: screen.width || null,
              height: screen.height || null,
              status: screen.screenMetadata?.status || null,
            }
          : null;
      })
      .filter(Boolean);
  });
}

export function newScreenIds(beforeIds, afterIds) {
  const before = new Set(beforeIds);
  return afterIds.filter((screenId) => !before.has(screenId));
}

export function redactError(error) {
  const message = error?.message || String(error);
  return message
    .replaceAll(process.env.STITCH_API_KEY || "\u0000", "[redacted]")
    .replaceAll(process.env.STITCH_ACCESS_TOKEN || "\u0000", "[redacted]");
}

async function sleep(ms) {
  if (ms <= 0) return;
  await new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function loadPrompt(args) {
  if (!args.promptFile) return args.prompt;
  return fs.readFile(args.promptFile, "utf8");
}

async function writeManifest(manifestPath, payload) {
  await fs.mkdir(path.dirname(manifestPath), { recursive: true });
  await fs.writeFile(manifestPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

async function loadStitchToolClient() {
  const importTarget = process.env.STITCH_SDK_IMPORT_PATH
    ? pathToFileURL(process.env.STITCH_SDK_IMPORT_PATH).href
    : "@google/stitch-sdk";
  const { StitchToolClient } = await import(importTarget);
  return StitchToolClient;
}

async function listScreenIds(client, projectId) {
  const response = await client.callTool("list_screens", { projectId });
  return screenIdsFromListResponse(response);
}

async function safeListScreenIds(client, projectId) {
  try {
    return { ids: await listScreenIds(client, projectId), error: null };
  } catch (caught) {
    return { ids: [], error: redactError(caught) };
  }
}

export async function runWithClient(client, args, options = {}) {
  const prompt = await loadPrompt(args);
  const before = await safeListScreenIds(client, args.projectId);
  const beforeScreenIds = before.ids;
  let status = "unknown";
  let error = null;
  let generatedScreens = [];

  try {
    const response = await client.callTool("generate_screen_from_text", {
      projectId: args.projectId,
      designSystem: `assets/${args.designSystemAssetId}`,
      deviceType: args.deviceType,
      modelId: args.modelId,
      prompt,
    });
    generatedScreens = generatedScreensFromToolResponse(response);
    status = generatedScreens.length > 0 ? "generated" : "generated_without_screen_projection";
  } catch (caught) {
    status = "generation_error";
    error = redactError(caught);
  }

  let pollClient = client;
  let pollClientIsFresh = false;
  let after = await safeListScreenIds(pollClient, args.projectId);
  if (after.error && options.createClient) {
    pollClient = await options.createClient();
    pollClientIsFresh = true;
    after = await safeListScreenIds(pollClient, args.projectId);
  }
  let afterScreenIds = after.ids;
  for (
    let attempt = 0;
    !after.error && attempt < args.pollCount && newScreenIds(beforeScreenIds, afterScreenIds).length === 0;
    attempt += 1
  ) {
    await sleep(args.pollIntervalMs);
    after = await safeListScreenIds(pollClient, args.projectId);
    afterScreenIds = after.ids;
  }
  if (pollClientIsFresh && typeof pollClient.close === "function") {
    await pollClient.close();
  }

  const inferredNewScreenIds = newScreenIds(beforeScreenIds, afterScreenIds);
  if (status === "generation_error" && inferredNewScreenIds.length > 0) {
    status = "generation_error_but_new_screen_detected";
  }
  if (status === "generated" && inferredNewScreenIds.length === 0) {
    status = "generated_but_not_listed";
  }
  if (status === "generation_error" && after.error) {
    status = "generation_error_poll_failed";
  }

  const manifest = {
    createdAt: new Date().toISOString(),
    projectId: args.projectId,
    designSystemAssetId: args.designSystemAssetId,
    method: "stitch-sdk:generate_screen_from_text",
    modelId: args.modelId,
    deviceType: args.deviceType,
    timeoutMs: args.timeoutMs,
    pollCount: args.pollCount,
    pollIntervalMs: args.pollIntervalMs,
    status,
    error,
    listBeforeError: before.error,
    listAfterError: after.error,
    usedFreshClientForPolling: pollClientIsFresh,
    beforeScreenIds,
    afterScreenIds,
    inferredNewScreenIds,
    generatedScreens,
  };

  await writeManifest(args.manifest, manifest);
  return manifest;
}

async function main(argv) {
  const args = parseArgs(argv);
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }
  validateArgs(args);

  const StitchToolClient = await loadStitchToolClient();
  const client = new StitchToolClient({
    apiKey: process.env.STITCH_API_KEY,
    accessToken: process.env.STITCH_ACCESS_TOKEN,
    projectId: process.env.GOOGLE_CLOUD_PROJECT,
    baseUrl: args.baseUrl,
    timeout: args.timeoutMs,
  });

  try {
    const createClient = () =>
      new StitchToolClient({
        apiKey: process.env.STITCH_API_KEY,
        accessToken: process.env.STITCH_ACCESS_TOKEN,
        projectId: process.env.GOOGLE_CLOUD_PROJECT,
        baseUrl: args.baseUrl,
        timeout: args.timeoutMs,
      });
    const manifest = await runWithClient(client, args, { createClient });
    process.stdout.write(`Wrote Stitch generation manifest to ${args.manifest}\n`);
    process.stdout.write(`Status: ${manifest.status}\n`);
    return manifest.status === "generated" || manifest.status === "generation_error_but_new_screen_detected"
      ? 0
      : 1;
  } finally {
    await client.close();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (error) => {
      process.stderr.write(`${redactError(error)}\n`);
      process.exitCode = 1;
    },
  );
}
