import { defineConfig } from "@playwright/test";
import { fixtureConfig } from "./tests/fixtureConfig";
export default defineConfig({
  ...fixtureConfig("connections", 8817, 4187),
  testDir: "./tests/e2e",
  testMatch: ["cloudConnections.spec.ts", "cloudAccountPersistence.spec.ts", "providerLoginMethods.spec.ts", "runtimeSetupRecovery.spec.ts"],
});
