import { existsSync, mkdirSync, readFileSync, rmSync, copyFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { FullConfig } from "@playwright/test";

export default function teardown(config: FullConfig) {
  const { fixtureRoot: folder, fixtureLane: lane, fixtureOutput: output } = config.metadata;
  if (typeof folder !== "string" || path.dirname(folder) !== os.tmpdir() || !/^vostavo-(default|connections|journeys)-[a-zA-Z0-9]+$/.test(path.basename(folder))) throw new Error("Unexpected fixture root; refusing cleanup");
  // --list does not launch a backend and must not claim execution/guard proof.
  if (process.argv.includes("--list")) { rmSync(folder, { recursive: true, force: true }); return; }
  const guardPath = existsSync(path.join(folder, "guards.jsonl")) ? path.join(folder, "guards.jsonl") : path.join(output, "guards.jsonl");
  mkdirSync(output, { recursive: true });
  for (const file of ["guards.jsonl", "dispatch.jsonl"]) if (existsSync(path.join(folder, file))) copyFileSync(path.join(folder, file), path.join(output, file));
  if (!existsSync(guardPath)) throw new Error(`${lane}: fixture network guard did not run`);
  const guards = readFileSync(guardPath, "utf8").trim().split("\n").map(line => JSON.parse(line));
  if (!guards.length || guards.some(row => row.guard !== true || !Number.isInteger(row.pid))) throw new Error("Invalid fixture guard evidence");
  const guarded = new Set(guards.map(row => row.pid));
  const dispatch = existsSync(path.join(folder, "dispatch.jsonl")) ? path.join(folder, "dispatch.jsonl") : path.join(output, "dispatch.jsonl");
  if (existsSync(dispatch) && readFileSync(dispatch, "utf8").trim().split("\n").filter(Boolean).some(line => !guarded.has(JSON.parse(line).pid))) throw new Error("An inference process lacked the network guard");
  // The launcher removes its root after graceful backend shutdown. Removing it
  // here races the backend's state-file cleanup and hides teardown diagnostics.
}
