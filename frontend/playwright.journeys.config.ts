import { mkdtempSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
process.env.VOSTAVO_JOURNEYS_TEST_ROOT ??= mkdtempSync(path.join(os.tmpdir(), "vostavo-journeys-"));
const data = process.env.VOSTAVO_JOURNEYS_TEST_ROOT;
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;
const backend = "http://127.0.0.1:8814";
export default defineConfig({
  testDir: "./tests/journeys",
  outputDir: "./output/playwright/journeys",
  workers: 1, retries: 0,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: "http://127.0.0.1:4177",
    browserName: "chromium",
    headless: true,
    permissions: ["microphone"],
    launchOptions: { args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `${quote(path.join(root, ".venv/bin/python"))} tests/e2e/journey_backend.py --host 127.0.0.1 --port 8814 --app-data-dir ${quote(path.join(data, "app"))} --cache-dir ${quote(path.join(data, "cache"))}`,
      cwd: root, url: `${backend}/v1/health`, reuseExistingServer: false,
      timeout: 120_000, env: { HF_HUB_OFFLINE: "1" },
      gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4177 --strictPort",
      cwd: path.join(root, "frontend"), url: "http://127.0.0.1:4177",
      env: { VITE_LOCAL_API_BASE_URL: backend }, reuseExistingServer: false,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
  ],
});
