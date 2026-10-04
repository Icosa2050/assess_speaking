import { existsSync, mkdtempSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";
import original from "./playwright.config";

// Never reuse another checkout's dev servers. Only fixture credentials in these tests.
const root = fileURLToPath(new URL("..", import.meta.url));
if (!process.env.TEST_WORKER_INDEX) {
  process.env.VOSTAVO_CONNECTION_TEST_ROOT = mkdtempSync(path.join(os.tmpdir(), "vostavo-connections-"));
}
const data = process.env.VOSTAVO_CONNECTION_TEST_ROOT!;
const quote = (value: string) => `"${value.replaceAll('"', '\\"')}"`;
const python = [path.join(root, ".venv/bin/python"), path.join(root, ".venv/Scripts/python.exe")].find(existsSync)
  || (process.platform === "win32" ? "python" : "python3");
export default defineConfig({ ...original,
  metadata: { connectionTestRoot: data },
  globalTeardown: "./tests/connectionTeardown.ts",
  testMatch: ["cloudConnections.spec.ts", "providerLoginMethods.spec.ts", "runtimeSetupRecovery.spec.ts"],
  use: { ...original.use, baseURL: "http://127.0.0.1:4187" },
  webServer: [
    { command: `${quote(python)} scripts/run_backend.py --host 127.0.0.1 --port 8817 --app-data-dir ${quote(path.join(data, "app"))} --cache-dir ${quote(path.join(data, "cache"))}`,
      cwd: root, url: "http://127.0.0.1:8817/v1/health", reuseExistingServer: false,
      env: { PYTHON_KEYRING_BACKEND: "scripts.journey_keyring.MemoryKeyring" },
      timeout: 120_000, gracefulShutdown: { signal: "SIGTERM", timeout: 5000 } },
    { command: `${quote(process.execPath)} node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4187 --strictPort`,
      cwd: path.join(root, "frontend"), url: "http://127.0.0.1:4187", reuseExistingServer: false,
      env: { VITE_LOCAL_API_BASE_URL: "http://127.0.0.1:8817" },
      timeout: 120_000, gracefulShutdown: { signal: "SIGTERM", timeout: 5000 } },
  ],
});
