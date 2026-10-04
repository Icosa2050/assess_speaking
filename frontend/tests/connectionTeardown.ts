import { rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { FullConfig } from "@playwright/test";

export default function teardown(config: FullConfig) {
  const folder = config.metadata.connectionTestRoot;
  if (typeof folder !== "string" || path.dirname(folder) !== os.tmpdir()
      || !/^vostavo-connections-[a-zA-Z0-9]+$/.test(path.basename(folder))) {
    throw new Error("Refusing to clean an unexpected connection-test folder");
  }
  rmSync(folder, { recursive: true, force: true });
}
