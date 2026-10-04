import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import type { AssessmentStatusResponse } from "../../src/lib/api/types";
import { readOmlxConfig } from "./helpers/omlx";

test.skip(
  process.env.RUN_VOSTAVO_OMLX_ASSESSMENT_E2E !== "1",
  "Use playwright.omlx.config.ts with RUN_VOSTAVO_OMLX_ASSESSMENT_E2E=1.",
);
test.use({ trace: "off", video: "off" });

test("transcribes Italian audio, gets oMLX feedback, and reloads the saved history", async ({
  page,
  request,
}, testInfo) => {
  test.setTimeout(900_000);
  expect(path.basename(testInfo.config.configFile || ""), "Use the isolated oMLX configuration")
    .toBe("playwright.omlx.config.ts");
  const backend = "http://127.0.0.1:8812";
  const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
  const audio = path.join(repoRoot, "samples/cefr/it/B1/travel_story.wav");
  expect(existsSync(audio), `Required Italian audio fixture is missing: ${audio}`).toBeTruthy();
  const { baseUrl, model, apiKey } = readOmlxConfig();
  expect(model, "Set OMLX_MODEL to an installed chat model ID").not.toBe("");
  const whisper = process.env.OMLX_E2E_WHISPER || "small";

  const modelsResponse = await request.get(`${baseUrl}/models`, {
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    timeout: 15_000,
  });
  expect(modelsResponse.ok(), "oMLX must be running and accept the configured key").toBeTruthy();
  expect((await modelsResponse.json()).data)
    .toEqual(expect.arrayContaining([expect.objectContaining({ id: model })]));

  const whisperUrl = `${backend}/v1/runtime/whisper-models/${encodeURIComponent(whisper)}`;
  const whisperResponse = await request.get(whisperUrl);
  expect(whisperResponse.ok()).toBeTruthy();
  let whisperStatus = await whisperResponse.json();
  if (!whisperStatus.cached && process.env.OMLX_E2E_DOWNLOAD_WHISPER === "1") {
    const download = await request.post(`${whisperUrl}/download`, { timeout: 600_000 });
    expect(download.ok(), "Whisper model download must complete").toBeTruthy();
    whisperStatus = await download.json();
  }
  expect(whisperStatus.cached,
    "Whisper is missing; set OMLX_E2E_DOWNLOAD_WHISPER=1 for the first run").toBe(true);

  let connectionId = "";
  try {
    const saved = await request.put(`${backend}/v1/runtime/settings`, {
      data: {
        ui_locale: "en",
        whisper_model: whisper,
        connection: {
          provider_choice: "openai_compatible",
          label: "oMLX automated assessment",
          model,
          base_url: baseUrl,
          api_key: apiKey,
        },
      },
    });
    expect(saved.ok(), "The isolated backend must save its test connection").toBeTruthy();
    const settings = await saved.json();
    connectionId = settings.active_connection_id;
    expect(connectionId).not.toBe("");
    expect(settings.connections).toEqual(expect.arrayContaining([
      expect.objectContaining({ connection_id: connectionId, model, base_url: baseUrl }),
    ]));

    await page.goto("/session-setup");
    await page.getByTestId("setup.speaker_id").fill("omlx-real-audio-test");
    await page.getByTestId("setup.customize_details").click();
    await page.getByTestId("setup.learning_language").selectOption("it");
    await page.getByTestId("setup.cefr").selectOption("B1");
    await page.getByTestId("setup.advanced_topic").click();
    await page.getByTestId("setup.custom_theme").fill("Il mio ultimo viaggio");
    await page.getByTestId("setup.duration").selectOption("90");
    await page.getByTestId("setup.continue").click();
    await expect(page).toHaveURL(/\/speak$/);
    await page.getByTestId("speak.input_mode_upload").click();
    await page.getByTestId("speak.upload_input").setInputFiles(audio);
    await expect(page.getByTestId("speak.submit")).toBeEnabled();

    const createdPromise = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/v1/assessments" &&
      response.request().method() === "POST");
    await page.getByTestId("speak.submit").click();
    const created = await createdPromise;
    expect(created.ok()).toBeTruthy();
    const submitted = created.request().postDataJSON();
    expect(submitted.dry_run).not.toBe(true);
    expect(submitted).toMatchObject({
      provider: "openai_compatible", llm_model: model, expected_language: "it", whisper,
    });
    const { assessment_id: assessmentId } = await created.json();
    let result: AssessmentStatusResponse | undefined;
    await expect.poll(async () => {
      const response = await request.get(`${backend}/v1/assessments/${assessmentId}`);
      expect(response.ok()).toBeTruthy();
      result = await response.json() as AssessmentStatusResponse;
      return result.status;
    }, { timeout: 600_000, intervals: [1000, 2000, 5000] }).not.toMatch(/^(queued|running)$/);
    expect(result?.status, "Real transcription and assessment must complete").toBe("completed");
    const payload = result!.payload!;
    expect(payload).toBeTruthy();
    const transcript = String(payload.transcript_full || "");
    expect(transcript.split(/\s+/).length).toBeGreaterThan(50);
    expect(transcript.toLowerCase()).toContain("mare");
    const report = payload.report as Record<string, any>;
    expect(report.scores.mode, "Deterministic fallback must not pass this test").toBe("hybrid");
    for (const score of [report.scores.final, report.scores.llm]) {
      expect(typeof score).toBe("number");
      expect(score).toBeGreaterThanOrEqual(1);
      expect(score).toBeLessThanOrEqual(5);
    }
    expect(report.rubric).toBeTruthy();
    for (const field of ["fluency", "cohesion", "accuracy", "range", "overall"]) {
      expect(Number.isInteger(report.rubric[field])).toBe(true);
      expect(report.rubric[field]).toBeGreaterThanOrEqual(1);
      expect(report.rubric[field]).toBeLessThanOrEqual(5);
    }
    for (const field of ["coach_summary", "next_focus", "next_exercise"]) {
      expect(typeof report.coaching[field]).toBe("string");
      expect(report.coaching[field].trim()).not.toBe("");
    }
    expect(report.coaching.top_3_priorities).toHaveLength(3);
    for (const priority of report.coaching.top_3_priorities) {
      expect(typeof priority).toBe("string");
      expect(priority.trim()).not.toBe("");
    }
    expect(report.warnings || []).not.toContain("coaching_unavailable");
    await expect(page).toHaveURL(/\/review$/, { timeout: 30_000 });
    await expect(page.getByTestId("review-transcript")).toHaveValue(transcript);
    await page.getByTestId("review-action-view-history").click();
    await expect(page).toHaveURL(/\/history$/);
    await page.reload();
    await expect(page.getByTestId("history-detail-caption")).toContainText(report.session_id);
    const history = await request.get(`${backend}/v1/history/${encodeURIComponent(report.session_id)}`);
    expect(history.ok()).toBeTruthy();
    const persisted = (await history.json()).payload;
    expect(persisted.transcript_full).toBe(transcript);
    expect(persisted.report.scores).toEqual(report.scores);
    expect(persisted.report.coaching).toEqual(report.coaching);
  } finally {
    if (connectionId) {
      const removed = await request.delete(`${backend}/v1/runtime/settings/connections/${encodeURIComponent(connectionId)}`);
      expect(removed.ok(), "Remove the temporary connection and its saved key").toBeTruthy();
    }
  }
});
