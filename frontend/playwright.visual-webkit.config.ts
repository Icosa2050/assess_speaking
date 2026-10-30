import { defineConfig } from "@playwright/test";
import { fixtureConfig } from "./tests/fixtureConfig";

const config = fixtureConfig("default", 8800, 4173);

export default defineConfig({
  ...config,
  testDir: "./tests/e2e",
  testMatch: "visualRefreshSmoke.spec.ts",
  use: { ...config.use, browserName: "webkit", launchOptions: {} },
  projects: [{ name: "webkit", use: { browserName: "webkit", launchOptions: {} } }],
});
