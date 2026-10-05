import { mkdirSync, mkdtempSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
const retained = path.join(root, "frontend/output/live-journeys");
mkdirSync(retained, { recursive: true });
process.env.VOSTAVO_OLLAMA_TEST_ROOT ??= mkdtempSync(path.join(retained, "run-"));
const data = process.env.VOSTAVO_OLLAMA_TEST_ROOT;
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;
function port(value: string) {
  if (!/^\d+$/.test(value) || Number(value) < 1024 || Number(value) > 65535) throw new Error(`Invalid live journey port: ${value}`);
  return Number(value);
}
const backendPort = port(process.env.VOSTAVO_LIVE_BACKEND_PORT || "8816");
const frontendPort = port(process.env.VOSTAVO_LIVE_FRONTEND_PORT || "4179");
if (backendPort === frontendPort) throw new Error("Live backend and frontend require distinct ports");
const backend = `http://127.0.0.1:${backendPort}`;
export default defineConfig({
  testDir: "./tests/live",
  testMatch: "ollamaBilingual.spec.ts",
  outputDir: path.join(data, "playwright"),
  reporter: [["list"], ["json", { outputFile: path.join(data, "results.json") }],
    ["html", { outputFolder: path.join(data, "html"), open: "never" }]],
  workers: 1, retries: 0, timeout: 900_000, maxFailures: 1,
  expect: { timeout: 20_000 },
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`, browserName: "chromium", headless: true,
    permissions: ["microphone"], actionTimeout: 30_000, navigationTimeout: 30_000,
    screenshot: "only-on-failure", trace: "on", video: "off",
  },
  webServer: [
    {
      command: `${quote(process.env.VOSTAVO_TEST_PYTHON || path.join(root, ".venv/bin/python"))} scripts/run_backend.py --host 127.0.0.1 --port ${backendPort} --app-data-dir ${quote(path.join(data, "app"))} --cache-dir ${quote(path.join(data, "cache"))}`,
      cwd: root, url: `${backend}/v1/health`, reuseExistingServer: false,
      env: { LLM_TIMEOUT_SEC: "180", HF_HUB_OFFLINE: "1", PYTHON_KEYRING_BACKEND: "scripts.journey_keyring.MemoryKeyring" },
      timeout: 120_000, gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
    {
      command: `${quote(process.execPath)} node_modules/vite/bin/vite.js --host 127.0.0.1 --port ${frontendPort} --strictPort`,
      cwd: path.join(root, "frontend"), url: `http://127.0.0.1:${frontendPort}`, reuseExistingServer: false,
      env: { VITE_LOCAL_API_BASE_URL: backend },
      gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
  ],
});
