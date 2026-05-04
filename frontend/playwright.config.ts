import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

const frontendRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = path.resolve(frontendRoot, "..");
const backendPort = 8800;
const frontendPort = 4173;
const backendBaseUrl = `http://127.0.0.1:${backendPort}`;
const frontendBaseUrl = `http://127.0.0.1:${frontendPort}`;
const pythonExecutable = existsSync(path.join(repoRoot, ".venv", "bin", "python"))
  ? path.join(repoRoot, ".venv", "bin", "python")
  : process.platform === "win32"
    ? "python"
    : "python3";

export default defineConfig({
  testDir: "./tests/e2e",
  outputDir: "./test-results",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [["github"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: frontendBaseUrl,
    browserName: "chromium",
    headless: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "off",
    testIdAttribute: "data-testid",
  },
  webServer: [
    {
      command: `${pythonExecutable} scripts/run_backend.py --host 127.0.0.1 --port ${backendPort} --app-data-dir /tmp/vostavo-frontend-smoke-app-data --cache-dir /tmp/vostavo-frontend-smoke-cache`,
      cwd: repoRoot,
      name: "backend",
      url: `${backendBaseUrl}/v1/health`,
      reuseExistingServer: !process.env.CI,
      stdout: "ignore",
      stderr: "pipe",
      timeout: 120 * 1000,
      gracefulShutdown: {
        signal: "SIGTERM",
        timeout: 5_000,
      },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4173 --strictPort",
      cwd: frontendRoot,
      env: {
        ...process.env,
        VITE_LOCAL_API_BASE_URL: backendBaseUrl,
      },
      name: "frontend",
      url: frontendBaseUrl,
      reuseExistingServer: !process.env.CI,
      stdout: "ignore",
      stderr: "pipe",
      timeout: 120 * 1000,
      gracefulShutdown: {
        signal: "SIGTERM",
        timeout: 5_000,
      },
    },
  ],
});
