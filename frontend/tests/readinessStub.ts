import type { Page } from "@playwright/test";

export const readyReadiness = {
  ready: true,
  checks: ["speech_recognition", "ai_tutor", "media_decoder", "storage"].map((key) => ({
    key, state: "available", blocking: true, reason: "available", detail_args: {},
  })),
};

export const stubReadyReadiness = (page: Page) =>
  page.route("**/v1/readiness", (route) => route.fulfill({ json: readyReadiness }));
