import { describe, expect, it } from "vitest";

import type { DiagnosticItem, RuntimeResponse } from "@/lib/api/types";

import { buildSetupReadinessRows, resolveSetupReadinessAction } from "./readiness";

const diagnostic = (key: string, status: string): DiagnosticItem => ({
  key,
  status,
  title_key: `diagnostics.${key}_title`,
  detail_key: `diagnostics.${key}_${status}_detail`,
  detail_args: {},
});

const configuredRuntime: RuntimeResponse = {
  configured: true,
  provider: "ollama",
  model: "llama3.2:3b",
  base_url: "http://localhost:11434/v1",
  requires_api_key: false,
  has_api_key: false,
};

const unconfiguredRuntime: RuntimeResponse = {
  configured: false,
  provider: "",
  model: "",
  base_url: "",
  requires_api_key: false,
  has_api_key: false,
};

describe("buildSetupReadinessRows", () => {
  it("maps missing runtime dependencies to setup actions and blocks the sample check", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [diagnostic("whisper", "warning"), diagnostic("runtime", "warning")],
      runtime: unconfiguredRuntime,
      whisperCached: false,
    });

    expect(rows.map((row) => [row.key, row.status, row.disabled])).toEqual([
      ["speech_recognition", "setup", false],
      ["ai_tutor", "setup", false],
      ["microphone", "setup", false],
      ["sample_check", "setup", true],
    ]);
    expect(rows[0].actionKey).toBe("runtime_setup.setup_guide_download_model");
    expect(rows[1].actionKey).toBe("runtime_setup.setup_guide_connect_ai");
    expect(rows[3].detailKey).toBe("runtime_setup.setup_guide_sample_blocked");
  });

  it("enables practice once speech and AI are ready without claiming a browser microphone check", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [
        diagnostic("whisper", "ok"),
        diagnostic("runtime", "ok"),
        diagnostic("microphone", "info"),
      ],
      runtime: configuredRuntime,
      whisperCached: true,
    });

    expect(rows.map((row) => [row.key, row.status, row.disabled])).toEqual([
      ["speech_recognition", "ready", false],
      ["ai_tutor", "ready", false],
      ["microphone", "setup", false],
      ["sample_check", "setup", false],
    ]);
    expect(rows[3].actionKey).toBe("runtime_setup.setup_guide_run_sample");
  });

  it("shows loading rows while runtime or diagnostics are still pending", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [],
      diagnosticsPending: true,
      runtimePending: true,
      whisperPending: true,
    });

    expect(rows.map((row) => row.status)).toEqual([
      "loading",
      "loading",
      "loading",
      "loading",
    ]);
    expect(rows.every((row) => row.disabled)).toBe(true);
  });
});

describe("resolveSetupReadinessAction", () => {
  it("keeps provider setup actions local and enters Session Setup for practice checks", () => {
    expect(resolveSetupReadinessAction("speech_recognition")).toEqual({
      kind: "section",
      value: "runtime-setup-whisper",
    });
    expect(resolveSetupReadinessAction("ai_tutor")).toEqual({
      kind: "section",
      value: "runtime-setup-connection",
    });
    expect(resolveSetupReadinessAction("microphone")).toEqual({
      kind: "route",
      value: "/session-setup",
    });
    expect(resolveSetupReadinessAction("sample_check")).toEqual({
      kind: "route",
      value: "/session-setup",
    });
  });
});
