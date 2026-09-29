import { mkdtempSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
process.env.VOSTAVO_OLLAMA_TEST_ROOT ??= mkdtempSync(path.join(os.tmpdir(), "vostavo-ollama-"));
const data = process.env.VOSTAVO_OLLAMA_TEST_ROOT;
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;
export default defineConfig({
  testDir: "./tests/live",
  testMatch: "ollamaBilingual.spec.ts",
  outputDir: "./output/playwright/ollama",
  workers: 1, retries: 0, timeout: 900_000,
  expect: { timeout: 20_000 },
  use: {
    baseURL: "http://127.0.0.1:4179", browserName: "chromium", headless: true,
    screenshot: "only-on-failure", trace: "off", video: "off",
  },
  webServer: [
    {
      command: `${quote(path.join(root, ".venv/bin/python"))} scripts/run_backend.py --host 127.0.0.1 --port 8816 --app-data-dir ${quote(path.join(data, "app"))} --cache-dir ${quote(path.join(data, "cache"))}`,
      cwd: root, url: "http://127.0.0.1:8816/v1/health", reuseExistingServer: false,
      env: { LLM_TIMEOUT_SEC: "180", HF_HUB_OFFLINE: "1" },
      timeout: 120_000, gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
    {
      command: `${quote(process.execPath)} node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4179 --strictPort`,
      cwd: path.join(root, "frontend"), url: "http://127.0.0.1:4179", reuseExistingServer: false,
      env: { VITE_LOCAL_API_BASE_URL: "http://127.0.0.1:8816" },
      gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
  ],
});
