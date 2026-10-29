import type { ReadinessCheck, ReadinessResponse } from "@/lib/api/types";

const available = (key: ReadinessCheck["key"]): ReadinessCheck => ({
  key, state: "available", blocking: true, reason: "available", detail_args: {},
});

export const readyReadiness: ReadinessResponse = {
  ready: true,
  checks: [available("speech_recognition"), available("ai_tutor"), available("media_decoder"), available("storage")],
};

export const readinessWith = (
  key: ReadinessCheck["key"], reason: ReadinessCheck["reason"],
  state: ReadinessCheck["state"] = "needs_setup",
): ReadinessResponse => ({
  ready: false,
  checks: readyReadiness.checks.map((check) => check.key === key ? { ...check, reason, state } : check),
});
