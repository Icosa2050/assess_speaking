import { defineConfig } from "@playwright/test";
import { fixtureConfig } from "./tests/fixtureConfig";
export default defineConfig({
  ...fixtureConfig("default", 8800, 4173),
  testDir: "./tests/e2e",
  testIgnore: ["**/cloudAccountPersistence.spec.ts", "**/cloudConnections.spec.ts", "**/providerLoginMethods.spec.ts", "**/runtimeSetupRecovery.spec.ts"],
});
