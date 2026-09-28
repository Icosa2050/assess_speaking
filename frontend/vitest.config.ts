import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      "@locales": fileURLToPath(new URL("../locales", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    passWithNoTests: true,
    exclude: ["tests/e2e/**"],
  },
});
