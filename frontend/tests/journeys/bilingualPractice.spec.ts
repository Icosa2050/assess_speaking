import { expect, test, type Page, type APIRequestContext } from "@playwright/test";
import { fileURLToPath } from "node:url";
import path from "node:path";

const backend = "http://127.0.0.1:8814";
const root = fileURLToPath(new URL("../../..", import.meta.url));
const samples = { B1: "travel_story.wav", B2: "remote_work.wav", C1: "public_debate.wav" };
type Language = "en" | "it";
type Goal = keyof typeof samples;

async function configure(request: APIRequestContext, language: Language) {
  const response = await request.put(`${backend}/v1/runtime/settings`, { data: {
    ui_locale: language, whisper_model: "small",
    connection: { provider_choice: "ollama_local", label: "Journey fixture inference", model: "journey-fixture", base_url: "http://127.0.0.1:11434/v1" },
  } });
  expect(response.ok()).toBeTruthy();
}

async function setup(page: Page, language: Language, goal: Goal, speaker: string, theme?: string) {
  await page.goto("/session-setup");
  await page.getByTestId("setup.speaker_id").fill(speaker);
  await page.getByTestId("setup.customize_details").click();
  await page.getByTestId("setup.learning_language").selectOption(language);
  await page.getByTestId("setup.cefr").selectOption(goal);
  if (theme) {
    await page.getByTestId("setup.advanced_topic").click();
    await page.getByTestId("setup.custom_theme").fill(theme);
  }
  await page.getByTestId("setup.duration").selectOption("90");
  await page.getByTestId("setup.continue").click();
  await expect(page).toHaveURL(/\/speak$/);
  await expect(page.getByTestId("speak.session_summary")).toContainText(goal);
}

async function upload(page: Page, language: Language, goal: Goal) {
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles(path.join(root, "samples/cefr", language, goal, samples[goal]));
  await expect(page.getByTestId("speak.submit")).toBeEnabled();
}

async function submit(page: Page) {
  const responsePromise = page.waitForResponse(response =>
    new URL(response.url()).pathname === "/v1/assessments" && response.request().method() === "POST");
  await page.getByTestId("speak.submit").click();
  const response = await responsePromise;
  expect(response.ok()).toBeTruthy();
  return { body: response.request().postDataJSON(), id: (await response.json()).assessment_id as string };
}

async function review(page: Page, language: Language) {
  await expect(page).toHaveURL(/\/review$/, { timeout: 45_000 });
  await expect(page.getByTestId("review-summary")).toBeVisible();
  await expect(page.getByTestId("review-transcript")).toHaveValue(language === "en" ? /I visited a library/ : /Ieri ho visitato una biblioteca/);
  await expect(page.getByTestId("review-next-step-card")).toBeVisible();
}

for (const language of ["en", "it"] as const) {
  test.describe(language, () => {
  test.use({ locale: language });
  for (const goal of ["B1", "B2", "C1"] as const) {
    test(`${language} ${goal}: upload → review → microphone retry → saved history → playback → retry`, async ({ page, request }, testInfo) => {
      await configure(request, language);
      const speaker = `journey-${language}-${goal}`;
      await setup(page, language, goal, speaker);
      await upload(page, language, goal);
      const first = await submit(page);
      expect(first.body).toMatchObject({ expected_language: language, feedback_language: language, target_cefr: goal, retry_of_session_id: "" });
      expect(first.body.prompt_text.length).toBeGreaterThan(10);
      await review(page, language);
      const firstStatus = await (await request.get(`${backend}/v1/assessments/${first.id}`)).json();
      const firstSession = firstStatus.payload.report.session_id;
      const optionalStyle = firstStatus.payload.report.rubric.style_suggestions;
      expect(optionalStyle).toHaveLength(1);
      expect(firstStatus.payload.report.rubric.recurring_grammar_errors).toEqual([]);
      await expect(page.getByTestId("review-style-suggestions")).toContainText(optionalStyle[0].original);
      await expect(page.getByTestId("review-priorities")).not.toContainText(optionalStyle[0].original);
      await page.screenshot({ path: testInfo.outputPath(`${language}-${goal}-optional-style-review.png`), fullPage: true });
      expect(firstSession).not.toBe(first.id);
      await page.getByTestId("review-action-try-again").click();
      await expect(page.getByTestId("speak.submit")).toBeDisabled();
      await page.getByTestId("speak.input_mode_record").click();
      await page.getByTestId("speak.record_start").click();
      await expect(page.getByTestId("speak.record_stop")).toBeVisible();
      // MediaRecorder needs real elapsed audio, not a fake timer.
      await page.waitForTimeout(2200);
      await page.getByTestId("speak.record_stop").click();
      await expect(page.getByTestId("speak.submit")).toBeEnabled();
      const second = await submit(page);
      expect(second.body).toMatchObject({ expected_language: language, target_cefr: goal, retry_of_session_id: firstSession, feedback_language: language, prompt_text: first.body.prompt_text, target_duration_sec: 90 });
      await review(page, language);
      const secondStatus = await (await request.get(`${backend}/v1/assessments/${second.id}`)).json();
      const secondSession = secondStatus.payload.report.session_id;
      expect(secondStatus.payload.report.checks.duration_pass).toBe(false);
      const duration = secondStatus.payload.report.metrics.duration_sec;
      expect(duration).toBeGreaterThan(1);
      expect(duration).toBeLessThan(10);
      const rows = (await (await request.get(`${backend}/v1/history`)).json()).items;
      const secondRow = rows.find((row: { session_id: string }) => row.session_id === secondSession);
      expect(secondRow.elapsed_wpm).toBeCloseTo(secondRow.word_count * 60 / duration, 1);
      const stored = await (await request.get(`${backend}/v1/history/${secondSession}`)).json();
      expect(stored.payload.report.rubric.style_suggestions).toEqual(optionalStyle);
      expect(stored.payload.report.rubric.recurring_grammar_errors).toEqual([]);
      expect(stored.payload.report.input.prompt_version).toBe("rubric_multilingual_v5");
      expect(stored.payload.meta.practice).toMatchObject({ goal, prompt_text: first.body.prompt_text, retry_of_session_id: firstSession });
      await page.getByTestId("review-action-view-history").click();
      await expect(page.getByTestId("history-detail-caption")).toContainText(secondSession);
      await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(2);
      await expect(page.getByTestId("practice-comparison").locator("tbody tr")).toHaveCount(4);
      const earlierAudio = page.locator(`audio[src$="/${firstSession}/audio"]`);
      await earlierAudio.evaluate(async (element: HTMLAudioElement) => { await element.play(); });
      await expect.poll(() => earlierAudio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeGreaterThan(0);
      await earlierAudio.evaluate((element: HTMLAudioElement) => { element.pause(); element.currentTime = 1; });
      await expect.poll(() => earlierAudio.evaluate((element: HTMLAudioElement) => element.seeking)).toBe(false);
      expect(await earlierAudio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeCloseTo(1, 1);
      const recordedAudio = page.locator(`audio[src$="/${secondSession}/audio"]`);
      await recordedAudio.evaluate(async (element: HTMLAudioElement) => { await element.play(); });
      await expect.poll(() => recordedAudio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeGreaterThan(0);
      await recordedAudio.evaluate((element: HTMLAudioElement) => element.pause());
      const range = await request.get(`${backend}/v1/history/${firstSession}/audio`, { headers: { Range: "bytes=0-63" } });
      expect(range.status()).toBe(206);
      expect((await range.body()).length).toBe(64);
      await page.reload();
      await page.getByTestId("history-language-filter").selectOption(language);
      // Reload deliberately drops the in-memory draft; disk-backed reports must survive.
      await expect(page.getByTestId("history-detail-caption")).toContainText(secondSession);
      await expect(page.getByTestId("practice-progress")).toContainText(goal);
      await expect(page.getByTestId("review-style-suggestions")).toContainText(optionalStyle[0].suggestion);
      await expect(page.getByTestId("practice-comparison").locator("tbody tr")).toHaveCount(4);
      await page.screenshot({ path: testInfo.outputPath(`${language}-${goal}-desktop.png`), fullPage: true });
      await page.setViewportSize({ width: 360, height: 800 });
      await expect(page.getByTestId("practice-retry")).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`${language}-${goal}-mobile.png`), fullPage: true });
      await page.getByTestId("practice-retry").click();
      await expect(page).toHaveURL(/\/speak$/);
      await expect(page.getByTestId("speak.session_summary")).toContainText(goal);
      await expect(page.getByTestId("speak.submit")).toBeDisabled();
      await upload(page, language, goal);
      const third = await submit(page);
      expect(third.body).toMatchObject({ expected_language: language, target_cefr: goal, prompt_text: first.body.prompt_text, retry_of_session_id: secondSession, target_duration_sec: 90, feedback_language: language });
      await review(page, language);
    });
  }

  test(`${language}: upload error, assessment failure, cancellation and recovery`, async ({ page, request }) => {
    await configure(request, language);
    await setup(page, language, "B1", `recovery-${language}`, "journey-fixture-failure");
    const empty = await request.post(`${backend}/v1/uploads`, { multipart: { file: { name: "empty.wav", mimeType: "audio/wav", buffer: Buffer.alloc(0) } } });
    expect(empty.status()).toBe(400);
    await upload(page, language, "B1");
    await page.route("**/v1/uploads", route => route.fulfill({ status: 503, json: { detail: { code: "runtime_error", detail: "Temporary upload failure" } } }), { times: 1 });
    await page.getByTestId("speak.submit").click();
    await expect(page.getByTestId("speak.status_panel")).toContainText("Temporary upload failure");
    await expect(page.getByTestId("speak.submit")).toBeEnabled();
    const failed = await submit(page);
    await expect.poll(async () => (await (await request.get(`${backend}/v1/assessments/${failed.id}`)).json()).status).toBe("failed");
    await expect(page.getByTestId("speak.status_panel")).toContainText("Journey fixture: assessment failed");
    await setup(page, language, "B1", `recovery-${language}`, "journey-fixture-cancel");
    await upload(page, language, "B1");
    const cancelled = await submit(page);
    await expect.poll(async () => (await (await request.get(`${backend}/v1/assessments/${cancelled.id}`)).json()).phase).toBe("scoring_rubric");
    await page.getByTestId("speak.cancel_assessment").click();
    await expect.poll(async () => (await (await request.get(`${backend}/v1/assessments/${cancelled.id}`)).json()).status).toBe("cancelled");
    await expect(page.getByTestId("speak.submit")).toBeEnabled();
    // Outlast the fixture's six-second work: a surviving worker must not save a late report.
    await page.waitForTimeout(7000);
    expect((await (await request.get(`${backend}/v1/assessments/${cancelled.id}`)).json()).status).toBe("cancelled");
    await setup(page, language, "B1", `recovery-${language}`, "My library / La biblioteca");
    await upload(page, language, "B1");
    await submit(page);
    await review(page, language);
    const history = await (await request.get(`${backend}/v1/history`)).json();
    expect(history.items.filter((row: { speaker_id: string }) => row.speaker_id === `recovery-${language}`)).toHaveLength(1);
  });
  test(`${language}: changed goals and languages stay separate; missing audio is recoverable`, async ({ page, request }) => {
    await configure(request, language);
    const speaker = `isolation-${language}`;
    await setup(page, language, "B1", speaker);
    await upload(page, language, "B1");
    await submit(page);
    await review(page, language);
    await page.getByTestId("review-action-new-setup").click();
    await expect(page.getByTestId("setup.speaker_id")).toHaveValue(speaker);
    await page.getByTestId("setup.customize_details").click();
    await page.getByTestId("setup.cefr").selectOption("B2");
    await page.getByTestId("setup.continue").click();
    await upload(page, language, "B2");
    const changed = await submit(page);
    expect(changed.body).toMatchObject({ target_cefr: "B2", retry_of_session_id: "" });
    await review(page, language);
    await page.getByTestId("review-action-view-history").click();
    await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(1);
    await expect(page.getByTestId("practice-comparison").locator("table")).toHaveCount(0);
    const otherLanguage = language === "en" ? "it" : "en";
    await setup(page, otherLanguage, "B2", speaker);
    await upload(page, otherLanguage, "B2");
    await submit(page);
    await review(page, otherLanguage);
    await page.getByTestId("review-action-view-history").click();
    await expect(page.getByTestId("history-language-filter")).toHaveValue("__all__");
    await page.getByTestId("history-language-filter").selectOption(otherLanguage);
    await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(1);
    await page.getByTestId("history-language-filter").selectOption(language);
    await expect(page.getByTestId("practice-progress")).toContainText("B2");
    await expect(page.getByTestId("practice-progress").locator("svg circle")).toHaveCount(1);
    await page.route("**/v1/history/*/audio", route => route.fulfill({ status: 404 }));
    await page.getByTestId("practice-progress").locator("audio").evaluate((el: HTMLAudioElement) => el.load());
    await expect(page.getByTestId("practice-progress").locator("audio")).toHaveCount(0);
    await expect(page.getByTestId("practice-progress").getByRole("status")).toBeVisible();
    // Missing media must not prevent another attempt from the saved prompt.
    await page.getByTestId("practice-retry").click();
    await expect(page).toHaveURL(/\/speak$/);
    await expect(page.getByTestId("speak.session_summary")).toContainText("B2");
  });
  });
}
