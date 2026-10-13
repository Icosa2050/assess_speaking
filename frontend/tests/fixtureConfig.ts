import { existsSync, mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { PlaywrightTestConfig } from "@playwright/test";

const repo = fileURLToPath(new URL("../..", import.meta.url));
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;
export function fixtureConfig(lane: "default" | "connections" | "journeys", backendPort: number, frontendPort: number, outputCategory: string = lane): PlaywrightTestConfig {
  const variable = `VOSTAVO_${lane.toUpperCase()}_TEST_ROOT`;
  if (!process.env.TEST_WORKER_INDEX) process.env[variable] = mkdtempSync(path.join(os.tmpdir(), `vostavo-${lane}-`));
  const folder = process.env[variable]!;
  const python = process.env.VOSTAVO_TEST_PYTHON || [path.join(repo, ".venv/bin/python"), path.join(repo, ".venv/Scripts/python.exe")].find(existsSync) || "python";
  const port = Number(process.env.VOSTAVO_TEST_BACKEND_PORT || backendPort);
  const uiPort = Number(process.env.VOSTAVO_TEST_FRONTEND_PORT || frontendPort);
  if (![port, uiPort].every(value => Number.isInteger(value) && value > 0 && value < 65536) || port === uiPort) throw new Error("Invalid fixture ports");
  const backend = `http://127.0.0.1:${port}`;
  const frontend = `http://127.0.0.1:${uiPort}`;
  process.env.VOSTAVO_FIXTURE_BACKEND_URL = backend;
  if (lane === "journeys") process.env.VOSTAVO_JOURNEY_AUDIO_DIR = path.join(folder, "audio");
  if (lane === "connections" && !process.env.TEST_WORKER_INDEX) writeFileSync(path.join(folder, "scenario.json"), JSON.stringify({ mode: "free" }));
  mkdirSync(path.join(folder, "hf"), { recursive: true });
  const audioPath = path.join(folder, "tone.wav");
  if (!process.env.TEST_WORKER_INDEX) {
    const rate = 48000; const count = rate * 2; const wav = Buffer.alloc(44 + count * 2);
    wav.write("RIFF", 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
    wav.write("data", 36); wav.writeUInt32LE(count * 2, 40);
    for (let i = 0; i < count; i++) wav.writeInt16LE(Math.round(3200 * Math.sin(2 * Math.PI * 440 * i / rate)), 44 + i * 2);
    writeFileSync(audioPath, wav);
  }
  if (!["default", "connections", "journeys", "contracts"].includes(outputCategory)) throw new Error("Invalid fixture output category");
  const output = path.join(repo, "frontend/output/playwright", outputCategory);
  const backendEnv: Record<string, string> = {
    PYTHON_KEYRING_BACKEND: "scripts.journey_keyring.MemoryKeyring",
    PYTHONPATH: [path.join(repo, "tests/helpers/cloud_guard"), repo].join(path.delimiter),
    VOSTAVO_FIXTURE_GUARD_LOG: path.join(folder, "guards.jsonl"),
    VOSTAVO_HOME: path.join(folder, "app"), VOSTAVO_CACHE_HOME: path.join(folder, "cache"),
    ...(lane === "journeys" ? { VOSTAVO_JOURNEY_AUDIO_DIR: path.join(folder, "audio") } : {}),
    HF_HOME: path.join(folder, "hf"), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1",
  };
  for (const key of ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "LLM_API_KEY", "OLLAMA_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]) backendEnv[key] = "";
  const script = lane === "connections" ? "tests/helpers/cloud_backend.py" : lane === "journeys" ? "tests/e2e/journey_backend.py" : "scripts/run_backend.py";
  const args = lane === "connections" ? `--port ${port} --root ${quote(folder)}` : `--host 127.0.0.1 --port ${port} --app-data-dir ${quote(path.join(folder, "app"))} --cache-dir ${quote(path.join(folder, "cache"))}`;
  return {
    metadata: { fixtureRoot: folder, fixtureLane: lane, fixtureOutput: output },
    globalTeardown: "./tests/fixtureTeardown.ts",
    outputDir: path.join(output, "artifacts"),
    reporter: [["list"], ["json", { outputFile: process.env.VOSTAVO_BROWSER_RESULTS || path.join(output, "results.json") }]],
    retries: 0, workers: 1, forbidOnly: !!process.env.CI, globalTimeout: lane === "journeys" ? 1_800_000 : 480_000,
    use: { baseURL: frontend, browserName: "chromium", headless: true, launchOptions: { args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${audioPath}`] }, screenshot: "only-on-failure", trace: "retain-on-failure", video: "off" },
    webServer: [
      { command: `${quote(python)} tests/helpers/browser_fixture.py --root ${quote(folder)} --evidence ${quote(output)} -- ${script} ${args}`, cwd: repo, url: `${backend}/v1/health`, env: backendEnv, reuseExistingServer: false, stdout: "pipe", stderr: "pipe", timeout: 90_000, gracefulShutdown: { signal: "SIGTERM", timeout: 5000 } },
      { command: `${quote(process.execPath)} node_modules/vite/bin/vite.js --host 127.0.0.1 --port ${uiPort} --strictPort`, cwd: path.join(repo, "frontend"), url: frontend, env: { VITE_LOCAL_API_BASE_URL: backend }, reuseExistingServer: false, stdout: "ignore", stderr: "pipe", timeout: 60_000, gracefulShutdown: { signal: "SIGTERM", timeout: 5000 } },
    ],
  };
}
