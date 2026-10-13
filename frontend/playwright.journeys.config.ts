import { defineConfig } from "@playwright/test";
import { fixtureConfig } from "./tests/fixtureConfig";
export default defineConfig({
  ...fixtureConfig("journeys", 8814, 4177),
  testDir: "./tests/journeys", timeout: 360_000, expect: { timeout: 15_000 },
});
