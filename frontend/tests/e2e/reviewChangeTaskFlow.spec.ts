import { expect, test, type Page, type Route } from "@playwright/test";

const runtimeSettings = {
  ui_locale: "en",
  whisper_model: "large-v3",
  active_connection_id: "conn-review-change-task",
  connections: [
    {
      connection_id: "conn-review-change-task",
      provider_key: "ollama",
      provider_choice: "ollama_local",
      provider_label: "Ollama local",
      label: "Review change-task local",
      model: "llama3.2:3b",
      base_url: "http://127.0.0.1:11434/v1",
      is_default: true,
      is_local: true,
      requires_api_key: false,
      has_api_key: false,
      secret_state: "absent",
      last_test_status: "passed",
      last_tested_at: "2026-07-21T09:30:00Z",
      openrouter_http_referer: "",
      openrouter_app_title: "",
      provider_metadata: {},
    },
  ],
};

const reviewPayload = {
  meta: {
    label: "review change task attempt",
    learning_language: "it",
  },
  notes: "Review change task notes",
  transcript_full: "Sono andato a Roma e poi ho visitato il centro con amici.",
  report: {
    session_id: "review-change-task-session",
    transcript_preview: "Sono andato a Roma e poi ho visitato il centro.",
    input: {
      expected_language: "it",
    },
    scores: {
      final: 3.8,
      band: "4",
      mode: "hybrid",
      llm: 3.9,
      deterministic: 3.7,
    },
    checks: {
      language_pass: true,
      topic_pass: true,
      content_validity_pass: true,
      duration_pass: true,
      min_words_pass: true,
    },
    coaching: {
      coach_summary: "Keep the travel story and add one clearer closing sentence.",
      strengths: ["Clear sequencing"],
      top_3_priorities: ["Add one clearer closing sentence"],
      next_focus: "Close the story with one personal reflection.",
      next_exercise: "Repeat the story and end with a clear takeaway.",
    },
    warnings: [],
    requires_human_review: false,
    rubric: {
      recurring_grammar_errors: [],
      coherence_issues: [],
    },
    progress_delta: null,
  },
};

const fulfillJson = (route: Route, json: unknown) =>
  route.fulfill({
    contentType: "application/json",
    json,
  });

const installReviewBackend = async (page: Page) => {
  await page.route("**/v1/diagnostics", (route) => fulfillJson(route, { items: [] }));
  await page.route("**/v1/runtime", (route) =>
    fulfillJson(route, {
      configured: true,
      provider: "ollama",
      model: "llama3.2:3b",
      base_url: "http://127.0.0.1:11434/v1",
      requires_api_key: false,
      has_api_key: false,
    }),
  );
  await page.route("**/v1/runtime/settings", (route) => fulfillJson(route, runtimeSettings));
  await page.route("**/v1/history", (route) => fulfillJson(route, { items: [] }));
  await page.route("**/v1/uploads", (route) =>
    fulfillJson(route, {
      audio_id: "review-change-task-audio",
      stored_path: "/tmp/review-change-task-audio.wav",
      sha1: "sha1-review-change-task-audio",
      original_name: "review-change-task-audio.wav",
    }),
  );
  await page.route("**/v1/assessments", (route) =>
    fulfillJson(route, {
      assessment_id: "review-change-task-assessment",
      status: "queued",
    }),
  );
  await page.route("**/v1/assessments/*", (route) =>
    fulfillJson(route, {
      assessment_id: "review-change-task-assessment",
      status: "completed",
      phase: "finalizing_report",
      progress: 1,
      error: null,
      report_path: "/tmp/review-change-task-report.json",
      payload: reviewPayload,
      summary: {
        score_overall: 3.8,
        band: "4",
        next_focus: "Close the story with one personal reflection.",
      },
    }),
  );
};

const attachAudio = async (page: Page) => {
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles({
    name: "review-change-task.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("fake review change task audio"),
  });
  await expect(page.getByTestId("speak.recording_status")).toHaveText(
    "A recording is attached and ready for assessment.",
  );
};

test.describe("review change-task browser flow", () => {
  test("returns to Session Setup with learner identity preserved and task details cleared", async ({
    page,
  }) => {
    await installReviewBackend(page);

    await page.goto("/");
    await page.getByTestId("home.start_new").click();
    await expect(page).toHaveURL(/\/session-setup$/);

    await page.getByTestId("setup.speaker_id").fill("playwright-review-change-task");
    await page.getByTestId("setup.recommended_start").click();
    await page.getByTestId("setup.continue").click();

    await expect(page).toHaveURL(/\/speak$/);
    await attachAudio(page);
    await page.getByTestId("speak.submit").click();

    await expect(page).toHaveURL(/\/review$/);
    await expect(page.getByTestId("review-next-step-card")).toBeVisible();

    await page.getByTestId("review-action-new-setup").click();

    await expect(page).toHaveURL(/\/session-setup$/);
    await expect(page.getByTestId("setup.layout")).toBeVisible();
    await expect(page.getByTestId("setup.speaker_id")).toHaveValue(
      "playwright-review-change-task",
    );
    await expect(page.getByTestId("setup.recommended_start")).toBeVisible();
    await expect(page.getByTestId("setup.continue")).toHaveCount(0);
    await expect(page.getByText("Travel Story")).toHaveCount(0);
  });
});
