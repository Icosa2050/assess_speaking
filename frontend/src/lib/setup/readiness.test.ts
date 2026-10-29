import { describe, expect, it } from "vitest";

import type { ReadinessCheck, ReadinessResponse } from "@/lib/api/types";
import { buildSetupReadinessRows, canEnterSpeak, canRecord, resolveSetupReadinessAction } from "./readiness";

const available = (key: ReadinessCheck["key"]): ReadinessCheck => ({
  key, state: "available", blocking: true, reason: "available", detail_args: {},
});
const ready: ReadinessResponse = {
  checks: [available("speech_recognition"), available("ai_tutor"), available("media_decoder"), available("storage")],
  ready: true,
};
const withReason = (key: ReadinessCheck["key"], reason: ReadinessCheck["reason"], state: ReadinessCheck["state"] = "needs_setup"): ReadinessResponse => ({
  checks: ready.checks.map((check) => check.key === key ? { ...check, state, reason } : check),
  ready: false,
});

const mapped = [
  ["ai_tutor", "needs_setup", "not_configured", "first_run.readiness_not_configured", "first_run.readiness_configure"],
  ["ai_tutor", "needs_setup", "credential_missing", "first_run.readiness_credential_missing", "first_run.readiness_add_credential"],
  ["ai_tutor", "unavailable", "local_service_unreachable", "first_run.readiness_service_unreachable", "first_run.readiness_start_service"],
  ["ai_tutor", "needs_setup", "model_unavailable", "first_run.readiness_model_unavailable", "first_run.readiness_select_model"],
  ["speech_recognition", "needs_setup", "model_not_cached", "first_run.readiness_model_not_cached", "first_run.readiness_download_model"],
  ["speech_recognition", "needs_setup", "credential_missing", "first_run.readiness_credential_missing", "first_run.readiness_add_credential"],
  ["media_decoder", "unavailable", "media_decoder_missing", "first_run.readiness_media_missing", "first_run.readiness_install_media"],
  ["storage", "unavailable", "storage_not_writable", "first_run.readiness_storage_unwritable", "first_run.readiness_fix_storage"],
] as const;

describe("backend readiness mapping", () => {
  it.each(mapped)("maps %s/%s/%s to Needs setup with specific rescue", (key, state, reason, detailKey, rescueKey) => {
    const rows = buildSetupReadinessRows({ readiness: withReason(key, reason, state) });
    const row = rows.find((item) => item.key === key);
    expect(row).toMatchObject({ status: "setup", detailKey, rescueKey });
    expect(canEnterSpeak(rows)).toBe(false);
    expect(rows.find((item) => item.key === "sample_check")?.disabled).toBe(true);
  });

  it("marks a cached speech model and reachable local tutor ready from backend checks", () => {
    const rows = buildSetupReadinessRows({ readiness: ready, microphoneStatus: "ready", microphoneSetupPassed: true });
    expect(rows.map((row) => row.status)).toEqual(["ready", "ready", "ready", "ready", "ready", "ready"]);
    expect(canEnterSpeak(rows)).toBe(true);
    expect(canRecord(rows)).toBe(true);
    expect(rows.map((row) => row.status)).not.toContain("configured");
  });

  it("routes a missing Groq speech credential to cloud setup copy", () => {
    const response = withReason("speech_recognition", "credential_missing");
    response.checks[0].detail_args = { provider: "groq", model: "whisper-large-v3" };
    expect(buildSetupReadinessRows({ readiness: response })[0]).toMatchObject({
      status: "setup", detailKey: "first_run.cloud_speech_missing",
      rescueKey: "first_run.rescue_cloud_speech",
      actionKey: "first_run.readiness_configure_cloud_speech",
    });
  });

  it("describes ready Groq transcription as cloud processing with a cloud action", () => {
    const response = structuredClone(ready);
    response.checks[0].detail_args = { provider: "groq", model: "whisper-large-v3" };
    expect(buildSetupReadinessRows({ readiness: response })[0]).toMatchObject({
      status: "ready", detailKey: "first_run.cloud_speech_ready",
      actionKey: "first_run.readiness_manage_cloud_speech",
    });
  });

  it("requires the backend aggregate even when every displayed check is available", () => {
    expect(canEnterSpeak(buildSetupReadinessRows({ readiness: { ...ready, ready: false } }))).toBe(false);
  });

  it("allows a nonblocking setup check when the backend says ready", () => {
    const response = withReason("ai_tutor", "not_configured");
    response.ready = true;
    response.checks[1].blocking = false;
    const rows = buildSetupReadinessRows({ readiness: response });
    expect(rows[1]).toMatchObject({ status: "setup", blocking: false });
    expect(canEnterSpeak(rows)).toBe(true);
  });

  it("fails closed for a future blocking check missing from the displayed rows", () => {
    const futureCheck = { ...available("storage"), key: "future_service", state: "unavailable" } as unknown as ReadinessCheck;
    const rows = buildSetupReadinessRows({ readiness: { checks: [...ready.checks, futureCheck], ready: true } });
    expect(canEnterSpeak(rows)).toBe(false);
  });

  it("requires microphone confirmation for recording but allows entry to Speak", () => {
    const rows = buildSetupReadinessRows({ readiness: ready, microphoneStatus: "ready" });
    expect(rows.find((row) => row.key === "microphone")?.status).toBe("setup");
    expect(canEnterSpeak(rows)).toBe(true);
    expect(canRecord(rows)).toBe(false);
    expect(buildSetupReadinessRows({ readiness: ready, microphoneStatus: "denied" }).find((row) => row.key === "microphone")?.status).toBe("unavailable");
  });

  it("does not let missing backend checks or a first load pass the guard", () => {
    expect(canEnterSpeak(buildSetupReadinessRows({ readiness: { checks: [], ready: true } }))).toBe(false);
    const rows = buildSetupReadinessRows({ readinessPending: true });
    expect(rows.filter((row) => row.key !== "microphone").every((row) => row.status === "loading" && row.disabled)).toBe(true);
    expect(canEnterSpeak(rows)).toBe(false);
  });

  it("shows an unavailable backend only without cached readiness", () => {
    expect(buildSetupReadinessRows({ readinessError: true })[0]).toMatchObject({
      status: "unavailable", detailKey: "first_run.readiness_backend_unavailable",
      rescueKey: "first_run.readiness_retry_backend",
    });
    const rows = buildSetupReadinessRows({ readiness: ready, readinessError: true });
    expect(rows[0].status).toBe("ready");
    expect(canEnterSpeak(rows)).toBe(true);
  });

  it("keeps setup actions and retry actions on the right destinations", () => {
    expect(resolveSetupReadinessAction("ai_tutor")).toEqual({ kind: "section", value: "runtime-setup-connection" });
    expect(resolveSetupReadinessAction("speech_recognition")).toEqual({ kind: "section", value: "runtime-setup-whisper" });
    expect(resolveSetupReadinessAction("microphone")).toEqual({ kind: "section", value: "runtime-setup-microphone" });
    expect(resolveSetupReadinessAction("storage")).toEqual({ kind: "refresh" });
    expect(resolveSetupReadinessAction("media_decoder")).toEqual({ kind: "refresh" });
    expect(resolveSetupReadinessAction("sample_check")).toEqual({ kind: "route", value: "/session-setup" });
  });
});
