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
  it("uses browser microphone results even when backend diagnostics cannot run", () => {
    expect(buildSetupReadinessRows({ diagnostics: [], diagnosticsError: true, microphoneStatus: "ready", microphoneSetupPassed: true })[2].status).toBe("ready");
    expect(buildSetupReadinessRows({ diagnostics: [diagnostic("microphone", "ok")] })[2].status).toBe("setup");
    expect(buildSetupReadinessRows({ diagnostics: [], microphoneStatus: "denied" })[2].status).toBe("unavailable");
  });
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

  it("requires a confirmed microphone sample even when speech and AI are ready", () => {
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
      ["ai_tutor", "configured", false],
      ["microphone", "setup", false],
      ["sample_check", "setup", true],
    ]);
    expect(rows[3].actionKey).toBe("runtime_setup.setup_guide_run_sample");
  });

  it("does not mark a configured cloud tutor ready without its required credential", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [diagnostic("runtime", "ok"), diagnostic("runtime_api_key", "error")],
      runtime: { ...configuredRuntime, requires_api_key: true, has_api_key: false },
      whisperCached: true,
    });
    expect(rows[1].status).toBe("unavailable");
    expect(rows[1].detailKey).toBe("first_run.ai_missing_key");
    expect(rows[1].rescueKey).toBe("first_run.rescue_ai_credential");
    expect(rows[3].disabled).toBe(true);
  });

  it("allows continuation without calling an untested local tutor ready", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [diagnostic("whisper", "ok"), diagnostic("runtime", "ok")],
      runtime: configuredRuntime, whisperCached: true,
      microphoneStatus: "ready", microphoneSetupPassed: true,
    });
    expect(rows.map((row) => row.status)).toEqual(["ready", "configured", "ready", "configured"]);
    expect(rows[3].disabled).toBe(false);
  });

  it("marks a local tutor ready only after a recent user-triggered connection test", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [diagnostic("whisper", "ok"), diagnostic("runtime", "ok")],
      runtime: configuredRuntime, connectionTestPassed: true,
      microphoneStatus: "ready", microphoneSetupPassed: true,
    });
    expect(rows.slice(0, 3).map((row) => row.status)).toEqual(["ready", "ready", "ready"]);
  });

  it("keeps a missing selected speech model and unselected Groq account out of Ready", () => {
    const local = buildSetupReadinessRows({
      diagnostics: [diagnostic("whisper", "ok")], runtime: configuredRuntime,
      whisperChecked: true, whisperCached: false,
    });
    expect(local[0].status).toBe("setup");

    const cloud = buildSetupReadinessRows({
      diagnostics: [diagnostic("whisper", "ok")], runtime: configuredRuntime,
      cloudSpeechSelected: true, cloudSpeechConnected: false,
    });
    expect(cloud[0].status).toBe("setup");
  });

  it("does not claim speech ready when its settings cannot be checked", () => {
    const rows = buildSetupReadinessRows({
      diagnostics: [diagnostic("whisper", "ok")], runtime: configuredRuntime,
      cloudSpeechError: true,
    });
    expect(rows[0].status).toBe("unavailable");
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
      kind: "section",
      value: "runtime-setup-microphone",
    });
    expect(resolveSetupReadinessAction("sample_check")).toEqual({
      kind: "route",
      value: "/session-setup",
    });
  });
});
