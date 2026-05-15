import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

const frontendRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = path.resolve(frontendRoot, "..");
const backendPort = 8800;
const frontendPort = 4173;
const backendBaseUrl = `http://127.0.0.1:${backendPort}`;
const frontendBaseUrl = `http://127.0.0.1:${frontendPort}`;
const pythonCandidates = [
  path.join(repoRoot, ".venv", "bin", "python"),
  path.join(repoRoot, ".venv", "Scripts", "python.exe"),
];
const pythonExecutable =
  pythonCandidates.find((candidate) => existsSync(candidate)) ??
  (process.platform === "win32" ? "python" : "python3");
const backendAppDataDir = path.join(os.tmpdir(), "vostavo-frontend-smoke-app-data");
const backendCacheDir = path.join(os.tmpdir(), "vostavo-frontend-smoke-cache");
const shellArg = (value: string) => `"${value.replace(/"/g, '\\"')}"`;
const backendCommand = [
  shellArg(pythonExecutable),
  "scripts/run_backend.py",
  "--host",
  "127.0.0.1",
  "--port",
  String(backendPort),
  "--app-data-dir",
  shellArg(backendAppDataDir),
  "--cache-dir",
  shellArg(backendCacheDir),
].join(" ");

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
      command: backendCommand,
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
