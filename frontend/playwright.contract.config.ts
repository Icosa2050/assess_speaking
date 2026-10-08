import { defineConfig } from "@playwright/test";
import { fixtureConfig } from "./tests/fixtureConfig";
const config = fixtureConfig("journeys", 8834, 4197, "contracts");
export default defineConfig({ ...config, testDir: "./tests/contracts", timeout: 90_000,
  outputDir: "output/playwright/contracts/artifacts",
  reporter: [["list"], ["json", { outputFile: "output/playwright/contracts/results.json" }]],
  metadata: { ...config.metadata, fixtureOutput: "output/playwright/contracts" },
});
