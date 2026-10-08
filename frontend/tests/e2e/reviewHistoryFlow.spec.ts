import { expect, test, type Page, type Route } from "../fixtures";

type Attempt = {
  assessmentId: string;
  audioId: string;
  label: string;
  notes: string;
  score: number;
  sessionId: string;
  transcript: string;
};

const attempts: Attempt[] = [
  {
    assessmentId: "asmt-first",
    audioId: "audio-first",
    label: "first deterministic attempt",
    notes: "First attempt notes",
    score: 3.1,
    sessionId: "sess-first",
    transcript: "Sono andato a Roma e ho visitato il centro.",
  },
  {
    assessmentId: "asmt-second",
    audioId: "audio-second",
    label: "second deterministic attempt",
    notes: "Second attempt notes",
    score: 4.0,
    sessionId: "sess-second",
    transcript: "Sono andato a Roma, poi ho visitato il centro e infine ho cenato con amici.",
  },
];

const runtimeSettings = {
  ui_locale: "en",
  whisper_model: "small",
  active_connection_id: "conn-dry-run",
  connections: [
    {
      connection_id: "conn-dry-run",
      provider_key: "ollama",
      provider_choice: "ollama_local",
      provider_label: "Ollama local",
      label: "Dry-run local",
      model: "llama3.2:3b",
      base_url: "http://127.0.0.1:11434/v1",
      is_default: true,
      is_local: true,
      requires_api_key: false,
      has_api_key: false,
      secret_state: "absent",
      last_test_status: "passed",
      last_tested_at: "2026-05-19T12:00:00Z",
      openrouter_http_referer: "",
      openrouter_app_title: "",
      provider_metadata: {},
    },
  ],
};

const reportPayload = (attempt: Attempt, previousSessionId = "") => ({
  meta: {
    label: attempt.label,
    learning_language: "it",
  },
  notes: attempt.notes,
  transcript_full: attempt.transcript,
  baseline_comparison: {
    level: "B1",
    targets: {
      wpm: {
        expected: "80-130",
        actual: 104,
        ok: true,
      },
      cohesion_markers: {
        expected: "observed",
        actual: 2,
        ok: null,
        status: "observed",
      },
    },
  },
  report: {
    session_id: attempt.sessionId,
    transcript_preview: attempt.transcript,
    input: {
      expected_language: "it",
    },
    scores: {
      final: attempt.score,
      band: attempt.score >= 4 ? "4" : "3",
      mode: "hybrid",
      llm: attempt.score + 0.1,
      deterministic: attempt.score - 0.1,
    },
    checks: {
      language_pass: true,
      topic_pass: true,
      content_validity_pass: true,
      duration_pass: true,
      min_words_pass: true,
    },
    coaching: {
      coach_summary: `Coach summary for ${attempt.sessionId}`,
      strengths: ["Clearer sequencing"],
      top_3_priorities: ["Add one more concrete detail"],
      next_focus: "Keep the story close to the travel theme.",
      next_exercise: "Repeat the trip story with a clear closing sentence.",
    },
    warnings: [],
    requires_human_review: false,
    rubric: {
      recurring_grammar_errors: [],
      coherence_issues: [],
    },
    progress_delta: previousSessionId
      ? {
          comparison_verified: true,
          previous_session_id: previousSessionId,
          score_delta: {
            final: 0.9,
            wpm: 6.5,
          },
          new_priorities: ["Add one more concrete detail"],
          resolved_priorities: ["Stay closer to the travel theme"],
        }
      : null,
  },
});

const historyRows = attempts.map((attempt, index) => ({
  timestamp: `2026-05-19T12:0${index}:00Z`,
  session_id: attempt.sessionId,
  speaker_id: "playwright-review-history",
  learning_language: "it",
  theme: "Il mio ultimo viaggio all'estero",
  task_family: "travel_narrative",
  overall: attempt.score,
  wpm: index === 0 ? 96 : 103,
  report_path: `/tmp/${attempt.sessionId}.json`,
  requires_human_review: false,
  duration_pass: true,
  topic_pass: true,
  language_pass: true,
  min_words_pass: true,
  top_priorities: ["Add one more concrete detail"],
  grammar_error_categories: [],
  coherence_issue_categories: [],
  final_score: attempt.score,
  band: attempt.score >= 4 ? "4" : "3",
}));

const fulfillJson = (route: Route, json: unknown) =>
  route.fulfill({
    contentType: "application/json",
    json,
  });

const installDeterministicBackend = async (page: Page) => {
  const submissions: Record<string, unknown>[] = [];
  let uploadCount = 0;
  let assessmentCount = 0;

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
  await page.route("**/v1/uploads", (route) => {
    const attempt = attempts[Math.min(uploadCount, attempts.length - 1)];
    uploadCount += 1;
    return fulfillJson(route, {
      audio_id: attempt.audioId,
      stored_path: `/tmp/${attempt.audioId}.wav`,
      sha1: `sha1-${attempt.audioId}`,
      original_name: `${attempt.audioId}.wav`,
    });
  });
  await page.route("**/v1/assessments", (route) => {
    const submitted = route.request().postDataJSON();
    expect(submitted.prompt_text).toEqual(expect.any(String));
    expect(submitted.prompt_text.length).toBeGreaterThan(0);
    expect(submitted.retry_of_session_id).toBe(submissions.length ? "sess-first" : "");
    if (submissions.length) {
      expect(submitted.prompt_text).toBe(submissions[0].prompt_text);
      expect(submitted.target_cefr).toBe(submissions[0].target_cefr);
    }
    submissions.push(submitted);
    const attempt = attempts[Math.min(assessmentCount, attempts.length - 1)];
    assessmentCount += 1;
    return fulfillJson(route, {
      assessment_id: attempt.assessmentId,
      status: "queued",
    });
  });
  await page.route("**/v1/assessments/*", (route) => {
    const url = new URL(route.request().url());
    const assessmentId = decodeURIComponent(url.pathname.split("/").pop() ?? "");
    const attempt = attempts.find((item) => item.assessmentId === assessmentId) ?? attempts[0];
    const previousSessionId = attempt.sessionId === "sess-second" ? "sess-first" : "";
    return fulfillJson(route, {
      assessment_id: attempt.assessmentId,
      status: "completed",
      phase: "finalizing_report",
      progress: 1,
      error: null,
      report_path: `/tmp/${attempt.sessionId}.json`,
      payload: reportPayload(attempt, previousSessionId),
      summary: {
        score_overall: attempt.score,
        band: attempt.score >= 4 ? "4" : "3",
        next_focus: "Keep the story close to the travel theme.",
      },
    });
  });
  await page.route("**/v1/history", (route) => fulfillJson(route, { items: historyRows.slice(0, submissions.length).map((row, index) => ({
    ...row,
    elapsed_wpm: index ? 103 : 96, duration_sec: 90, pause_total_sec: index ? 8 : 12,
    practice: {
      version: 1, goal: submissions[index]?.target_cefr || "B1",
      prompt_id: submissions[index]?.prompt_id || "travel-b1",
      prompt_text: submissions[index]?.prompt_text || "Describe your trip.",
      retry_of_session_id: index ? "sess-first" : "", target_duration_sec: submissions[index]?.target_duration_sec || 90,
      scoring_version: "scorer-v1", scoring_mode: "hybrid", analysis_signature: "settings-v1", provider: "ollama", model: "llama3.2:3b",
      asr_provider: "faster_whisper", whisper_model: "small", dry_run: false,
    },
  })) }));
  await page.route("**/v1/history/*", (route) => {
    const url = new URL(route.request().url());
    const sessionId = decodeURIComponent(url.pathname.split("/").pop() ?? "");
    const attempt = attempts.find((item) => item.sessionId === sessionId) ?? attempts[1];
    const previousSessionId = attempt.sessionId === "sess-second" ? "sess-first" : "";
    return fulfillJson(route, {
      payload: reportPayload(attempt, previousSessionId),
    });
  });
};

const installReadyEmptyBackend = async (page: Page) => {
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
};

const attachAudio = async (page: Page, name: string) => {
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles({
    name,
    mimeType: "audio/wav",
    buffer: Buffer.from(`fake audio ${name}`),
  });
  await expect(page.getByTestId("speak.recording_status")).toHaveText(
    "A recording is attached and ready for assessment.",
  );
};

const fillOptionalContext = async (page: Page, attempt: Attempt) => {
  const disclosure = page.getByTestId("speak.optional_context");
  await disclosure.locator("summary").click();
  await expect(disclosure).toHaveAttribute("open", "");
  await page.getByTestId("speak.label").fill(attempt.label);
  await page.getByTestId("speak.notes").fill(attempt.notes);
};

test.describe("review and history replacement flow", () => {
  test("keeps empty Review and History states actionable", async ({ page }) => {
    await installReadyEmptyBackend(page);

    await page.goto("/review");
    await expect(page.getByTestId("review-guard-missing-review")).toBeVisible();
    await page.getByTestId("review-guard-cta").click();
    await expect(page).toHaveURL(/\/session-setup$/);

    await page.goto("/history");
    await expect(page.getByTestId("history-empty")).toBeVisible();
    await page.getByTestId("history-empty-cta").click();
    await expect(page).toHaveURL(/\/session-setup$/);
  });

  test("runs two deterministic attempts, shows progress, and opens the latest history detail", async ({
    page,
  }, testInfo) => {
    await installDeterministicBackend(page);

    await page.goto("/");
    await page.getByTestId("home.start_new").click();
    await expect(page).toHaveURL(/\/session-setup$/);
    await page.getByTestId("setup.speaker_id").fill("playwright-review-history");
    await page.getByTestId("setup.recommended_start").click();
    await page.getByTestId("setup.continue").click();
    await expect(page).toHaveURL(/\/speak$/);
    await expect(page.getByTestId("speak.session_summary")).toContainText(
      "Speaker playwright-review-history",
    );

    await attachAudio(page, "first-attempt.wav");
    await fillOptionalContext(page, attempts[0]);
    await page.getByTestId("speak.submit").click();

    await expect(page).toHaveURL(/\/review$/);
    await expect(page.getByTestId("review-next-step-card")).toBeVisible();
    await expect(page.getByTestId("review-next-step-focus")).toContainText(
      "Keep the story close to the travel theme.",
    );
    await expect(page.getByTestId("review-summary")).toBeVisible();
    await expect(page.getByTestId("review-transcript")).toHaveValue(attempts[0].transcript);
    await expect(page.getByTestId("review-notes")).toHaveValue(attempts[0].notes);
    await expect(page.getByTestId("review-gates")).toContainText("Pass");
    await expect(page.getByTestId("review-metric-mode")).toContainText("Full assessment");

    await page.getByTestId("review-action-try-again").click();
    await expect(page).toHaveURL(/\/speak$/);

    await attachAudio(page, "second-attempt.wav");
    await fillOptionalContext(page, attempts[1]);
    await page.getByTestId("speak.submit").click();

    await expect(page).toHaveURL(/\/review$/);
    await expect(page.getByTestId("review-next-step-card")).toBeVisible();
    await expect(page.getByTestId("review-summary")).toBeVisible();
    await expect(page.getByTestId("review-progress")).toContainText(
      "Change in overall score: +0.90.",
    );
    await expect(page.getByTestId("review-transcript")).toHaveValue(attempts[1].transcript);

    await page.getByTestId("review-action-view-history").click();
    await expect(page).toHaveURL(/\/history$/);
    await expect(page.getByTestId("history-detail-panel")).toBeVisible();
    await expect(page.getByTestId("history-detail-caption")).toContainText("sess-second");
    await expect(page.getByTestId("review-notes")).toHaveValue(attempts[1].notes);
    await expect(page.getByTestId("practice-comparison")).toContainText("Your retry, compared with its original");
    await expect(page.getByTestId("practice-comparison")).toContainText("+0.9");
    await expect(page.getByRole("img", { name: "Overall performance" })).toBeVisible();
    await expect(page.locator('audio[aria-label="Listen: earlier attempt"]')).toHaveAttribute("src", /sess-first\/audio$/);
    await page.screenshot({ path: testInfo.outputPath("practice-desktop.png"), fullPage: true });
    await page.setViewportSize({ width: 320, height: 860 });
    await expect(page.getByTestId("practice-retry")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath("practice-mobile.png"), fullPage: true });
    await page.getByTestId("practice-progress").screenshot({ path: testInfo.outputPath("practice-panel-mobile.png") });
    await page.getByTestId("practice-retry").click();
    await expect(page).toHaveURL(/\/speak$/);
  });
});
