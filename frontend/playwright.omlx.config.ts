import { mkdtempSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const frontendRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = path.resolve(frontendRoot, "..");
const backendUrl = "http://127.0.0.1:8812";
const frontendUrl = "http://127.0.0.1:4175";
// Inherited by workers; each invocation gets an empty application workspace.
process.env.OMLX_E2E_APP_DATA ??= mkdtempSync(path.join(os.tmpdir(), "vostavo-omlx-e2e-"));
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "omlxAssessmentLive.spec.ts",
  outputDir: "./test-results/omlx-assessment",
  timeout: 900_000,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: frontendUrl,
    browserName: "chromium",
    headless: true,
    testIdAttribute: "data-testid",
    trace: "off",
    screenshot: "only-on-failure",
    video: "off",
  },
  webServer: [
    {
      command: `${quote(path.join(repoRoot, ".venv/bin/python"))} scripts/run_backend.py --host 127.0.0.1 --port 8812 --app-data-dir ${quote(process.env.OMLX_E2E_APP_DATA)} --cache-dir ${quote(path.join(os.tmpdir(), "vostavo-omlx-e2e-cache"))}`,
      cwd: repoRoot,
      url: `${backendUrl}/v1/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
    },
    {
      command: `${quote(process.execPath)} node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4175 --strictPort`,
      cwd: frontendRoot,
      env: { VITE_LOCAL_API_BASE_URL: backendUrl },
      url: frontendUrl,
      reuseExistingServer: false,
      timeout: 120_000,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
    },
  ],
});
