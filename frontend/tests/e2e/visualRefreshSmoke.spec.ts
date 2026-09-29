import { mkdirSync } from "node:fs";
import path from "node:path";

import { expect, test, type Page, type Route } from "@playwright/test";

const speakerId = "visual-refresh-smoke";
const screenshotDir = process.env.VISUAL_REFRESH_SCREENSHOT_DIR ?? "";
const desktopViewport = { width: 1280, height: 720 };
const mobileViewport = { width: 390, height: 844 };
const narrowMobileViewport = { width: 360, height: 780 };

const runtimeSettings = {
  ui_locale: "en",
  whisper_model: "small",
  active_connection_id: "conn-visual-smoke",
  connections: [
    {
      connection_id: "conn-visual-smoke",
      provider_key: "ollama",
      provider_choice: "ollama_local",
      provider_label: "Ollama local",
      label: "Visual smoke local",
      model: "llama3.2:3b",
      base_url: "http://127.0.0.1:11434/v1",
      is_default: true,
      is_local: true,
      requires_api_key: false,
      has_api_key: false,
      secret_state: "absent",
      last_test_status: "passed",
      last_tested_at: "2026-06-06T12:00:00Z",
      openrouter_http_referer: "",
      openrouter_app_title: "",
      provider_metadata: {},
    },
  ],
};

const historyRows = [
  {
    timestamp: "2026-06-06T12:00:00Z",
    session_id: "visual-first",
    speaker_id: speakerId,
    learning_language: "it",
    theme: "Il mio ultimo viaggio all'estero",
    task_family: "travel_narrative",
    overall: 3.8,
    wpm: 104,
    report_path: "/tmp/visual-first.json",
    requires_human_review: false,
    duration_pass: true,
    topic_pass: true,
    language_pass: true,
    min_words_pass: true,
    top_priorities: ["Stay closer to the travel theme", "Add one clearer closing sentence"],
    grammar_error_categories: [],
    coherence_issue_categories: [],
    final_score: 3.8,
    band: "4",
  },
  {
    timestamp: "2026-06-06T12:01:00Z",
    session_id: "visual-second",
    speaker_id: speakerId,
    learning_language: "it",
    theme: "Il mio ultimo viaggio all'estero",
    task_family: "travel_narrative",
    overall: 4.1,
    wpm: 118,
    report_path: "/tmp/visual-second.json",
    requires_human_review: false,
    duration_pass: true,
    topic_pass: true,
    language_pass: true,
    min_words_pass: true,
    top_priorities: ["Add one clearer closing sentence"],
    grammar_error_categories: [],
    coherence_issue_categories: [],
    final_score: 4.1,
    band: "4",
  },
];

const reportPayload = {
  meta: {
    label: "visual smoke attempt",
    learning_language: "it",
  },
  notes: "Visual smoke note",
  transcript_full: "Sono andato a Roma, poi ho visitato il centro e infine ho cenato con amici.",
  report: {
    session_id: "visual-second",
    transcript_preview: "Sono andato a Roma, poi ho visitato il centro.",
    input: {
      expected_language: "it",
    },
    scores: {
      final: 4.1,
      band: "4",
      mode: "hybrid",
      llm: 4.2,
      deterministic: 4.0,
    },
    checks: {
      language_pass: true,
      topic_pass: true,
      content_validity_pass: true,
      duration_pass: true,
      min_words_pass: true,
    },
    coaching: {
      coach_summary: "Keep the structure and close with one personal reflection.",
      strengths: ["Clearer sequencing"],
      top_3_priorities: ["Add one clearer closing sentence"],
      next_focus: "Close with one personal reflection.",
      next_exercise: "Repeat the story with a stronger final sentence.",
    },
    warnings: [],
    requires_human_review: false,
    rubric: {
      recurring_grammar_errors: [],
      coherence_issues: [],
    },
    progress_delta: {
      previous_session_id: "visual-first",
      score_delta: {
        final: 0.3,
        wpm: 14,
      },
      new_priorities: [],
      resolved_priorities: ["Stay closer to the travel theme"],
    },
  },
};

const failedReportPayload = {
  ...reportPayload,
  meta: {
    label: "visual smoke failed gate attempt",
    learning_language: "it",
  },
  notes: "Visual smoke failed-gate note",
  report: {
    ...reportPayload.report,
    session_id: "visual-failed",
    scores: {
      final: 2.4,
      band: "2",
      mode: "hybrid",
      llm: 2.2,
      deterministic: 2.6,
    },
    checks: {
      language_pass: true,
      topic_pass: false,
      content_validity_pass: false,
      duration_pass: true,
      min_words_pass: true,
    },
    coaching: {
      coach_summary: "Stay closer to the travel prompt before adding extra detail.",
      strengths: ["You kept speaking through the full take"],
      top_3_priorities: ["Answer the prompt more directly"],
      next_focus: "Name the travel moment first, then add one supporting detail.",
      next_exercise: "Repeat the story with one clear travel detail in the first sentence.",
    },
    warnings: ["llm_skipped_low_word_count"],
    requires_human_review: true,
    rubric: {
      recurring_grammar_errors: [{ type: "verb_tense" }],
      coherence_issues: [{ type: "off_topic_detail" }],
    },
    progress_delta: {},
  },
};

const assessmentSummary = (payload: typeof reportPayload | typeof failedReportPayload) => ({
  band: payload.report.scores.band,
  next_focus: payload.report.coaching.next_focus,
  score_overall: payload.report.scores.final,
});

const fulfillJson = (route: Route, json: unknown) =>
  route.fulfill({
    contentType: "application/json",
    json,
  });

const installVisualBackend = async (page: Page) => {
  let assessmentCount = 0;
  let activeAssessment: {
    assessmentId: string;
    payload: typeof reportPayload | typeof failedReportPayload;
    reportPath: string;
  } = {
    assessmentId: "visual-assessment-clean",
    payload: reportPayload,
    reportPath: "/tmp/visual-second.json",
  };

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
  await page.route("**/v1/uploads", (route) =>
    fulfillJson(route, {
      audio_id: "visual-audio",
      stored_path: "/tmp/visual-audio.wav",
      sha1: "sha1-visual-audio",
      original_name: "visual-audio.wav",
    }),
  );
  await page.route("**/v1/assessments", (route) => {
    assessmentCount += 1;
    activeAssessment =
      assessmentCount > 1
        ? {
            assessmentId: "visual-assessment-failed",
            payload: failedReportPayload,
            reportPath: "/tmp/visual-failed.json",
          }
        : {
            assessmentId: "visual-assessment-clean",
            payload: reportPayload,
            reportPath: "/tmp/visual-second.json",
          };

    return fulfillJson(route, {
      assessment_id: activeAssessment.assessmentId,
      status: "queued",
    });
  });
  await page.route("**/v1/assessments/*", (route) =>
    fulfillJson(route, {
      assessment_id: activeAssessment.assessmentId,
      status: "completed",
      phase: "finalizing_report",
      progress: 1,
      error: null,
      report_path: activeAssessment.reportPath,
      payload: activeAssessment.payload,
      summary: assessmentSummary(activeAssessment.payload),
    }),
  );
  await page.route("**/v1/history", (route) => fulfillJson(route, { items: historyRows }));
  await page.route("**/v1/history/*", (route) =>
    fulfillJson(route, {
      payload: reportPayload,
    }),
  );
};

const attachAudio = async (page: Page) => {
  await page.getByTestId("speak.input_mode_upload").click();
  await page.getByTestId("speak.upload_input").setInputFiles({
    name: "visual-smoke.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("visual smoke audio"),
  });
  await expect(page.getByTestId("speak.recording_status")).toHaveText(
    "A recording is attached and ready for assessment.",
  );
};

const visibleFontFamilies = async (page: Page) =>
  page.evaluate(() => {
    const families = new Set<string>();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
    let node = walker.nextNode();

    while (node) {
      const element = node as HTMLElement;
      const style = window.getComputedStyle(element);
      const hasVisibleText = Boolean(element.textContent?.trim());
      const isVisible =
        style.visibility !== "hidden" &&
        style.display !== "none" &&
        element.getClientRects().length > 0;

      if (hasVisibleText && isVisible) {
        families.add(style.fontFamily);
      }

      node = walker.nextNode();
    }

    return [...families].sort();
  });

const expectNoVisibleSerifTypography = async (page: Page) => {
  const families = await visibleFontFamilies(page);
  const serifFamilies = families.filter((family) =>
    family
      .split(",")
      .map((part) => part.trim().replace(/^["']|["']$/g, "").toLowerCase())
      .some((part) => part === "times" || part === "times new roman" || part === "serif"),
  );

  expect(serifFamilies).toEqual([]);
};

const expectNoPageHorizontalOverflow = async (page: Page) => {
  const overflow = await page.evaluate(() => {
    const viewportWidth = window.innerWidth;
    const documentWidth = document.scrollingElement?.scrollWidth ?? document.documentElement.scrollWidth;
    const overflowingElements = [...document.body.querySelectorAll<HTMLElement>("*")]
      .map((element) => {
        const rect = element.getBoundingClientRect();

        return {
          className: element.className ? String(element.className) : "",
          tagName: element.tagName.toLowerCase(),
          testId: element.dataset.testid ?? "",
          width: Math.ceil(rect.width),
          right: Math.ceil(rect.right),
        };
      })
      .filter((element) => element.width > viewportWidth || element.right > viewportWidth)
      .sort((left, right) => Math.max(right.width, right.right) - Math.max(left.width, left.right))
      .slice(0, 8);

    return {
      documentWidth,
      overflowingElements,
      viewportWidth,
    };
  });

  expect(overflow, JSON.stringify(overflow, null, 2)).toMatchObject({
    documentWidth: overflow.viewportWidth,
  });
};

const captureScreenshot = async (page: Page, filename: string) => {
  if (!screenshotDir) {
    return;
  }

  mkdirSync(screenshotDir, { recursive: true });
  await page.screenshot({
    fullPage: true,
    path: path.join(screenshotDir, filename),
  });
};

const expectMobileRouteAudit = async (page: Page, filename: string) => {
  await page.setViewportSize(mobileViewport);
  await expectNoVisibleSerifTypography(page);
  await expectNoPageHorizontalOverflow(page);
  await captureScreenshot(page, filename);
  await page.setViewportSize(desktopViewport);
};

test("renders primary visual-refresh artifacts without serif typography", async ({ page }) => {
  await installVisualBackend(page);

  await page.setViewportSize(desktopViewport);
  await page.goto("/");
  await expect(page.getByRole("progressbar", { name: "Practice readiness" })).toBeVisible();
  await expectNoVisibleSerifTypography(page);
  await captureScreenshot(page, "visual-refresh-smoke-home-desktop.png");
  await expectMobileRouteAudit(page, "visual-refresh-smoke-home-mobile.png");

  await page.getByTestId("home.start_new").click();
  await expect(page).toHaveURL(/\/session-setup$/);
  await expect(page.getByRole("heading", { name: "Set up today's speaking practice" })).toBeVisible();
  await expect(page.getByTestId("setup.wizard")).toBeVisible();
  await expect(page.getByTestId("setup.recommended_start")).toBeVisible();
  await expectMobileRouteAudit(page, "visual-refresh-smoke-session-setup-mobile.png");
  await page.getByTestId("setup.speaker_id").fill(speakerId);
  await page.getByTestId("setup.recommended_start").click();
  await expect(page.getByTestId("setup.runtime_callout")).toContainText("Ready to record");
  await expect(page.getByTestId("setup.continue")).toHaveText("Start speaking");
  await page.getByTestId("setup.continue").click();

  await expect(page).toHaveURL(/\/speak$/);
  await expect(page.getByTestId("speak.recording_visualizer")).toHaveAttribute(
    "data-recording-state",
    "idle",
  );
  await expectNoVisibleSerifTypography(page);
  await captureScreenshot(page, "visual-refresh-smoke-speak-desktop.png");
  await expectMobileRouteAudit(page, "visual-refresh-smoke-speak-mobile.png");

  await attachAudio(page);
  await expect(page.getByTestId("speak.optional_context")).toContainText("Add optional context");
  await expect(page.getByTestId("speak.label")).toHaveCount(1);
  await expect(page.getByTestId("speak.notes")).toHaveCount(1);
  await expectMobileRouteAudit(page, "visual-refresh-smoke-speak-ready-mobile.png");
  await page.getByTestId("speak.submit").click();

  await expect(page).toHaveURL(/\/review$/);
  await expect(page.getByTestId("review-next-step-card")).toContainText("Coach note");
  await expect(page.getByTestId("review-coach-takeaway")).toContainText(
    "Keep the structure and close with one personal reflection.",
  );
  const scoreRing = page.getByRole("progressbar", { name: "Current result" });
  await expect(scoreRing).toBeVisible();
  await expect(scoreRing).toHaveAttribute("aria-valuenow", "82");
  await expect(scoreRing).toHaveAttribute("aria-valuetext", "4 · 4.1");
  await expect(page.getByTestId("review-next-step-focus")).toContainText(
    "Close with one personal reflection.",
  );
  await expect(page.getByTestId("review-coach-summary")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Score details" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Feedback summary" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Quality checks" })).toBeVisible();
  await expect(page.getByTestId("review-quality-summary")).toHaveText("5 of 5 checks passed");
  await expect(page.getByTestId("review-gates-disclosure")).not.toHaveAttribute("open", "");
  await expect(page.getByTestId("review-evidence-disclosure")).not.toHaveAttribute("open", "");
  await expectNoVisibleSerifTypography(page);
  await captureScreenshot(page, "visual-refresh-smoke-review-desktop.png");
  await expectMobileRouteAudit(page, "visual-refresh-smoke-review-mobile.png");

  await page.getByTestId("review-action-try-again").click();
  await expect(page).toHaveURL(/\/speak$/);
  await attachAudio(page);
  await page.getByTestId("speak.submit").click();

  await expect(page).toHaveURL(/\/review$/);
  await expect(page.getByTestId("review-next-step-card")).toBeVisible();
  await expect(page.getByTestId("review-quality-summary")).toContainText("3 of 5 checks passed");
  await expect(page.getByTestId("review-gates-disclosure")).toHaveAttribute("open", "");
  await expect(page.getByTestId("review-failed-gates")).toBeVisible();
  await expect(page.getByTestId("review-failed-gate-item-0")).toBeVisible();
  await expect(page.getByTestId("review-failed-gate-item-1")).toBeVisible();
  await expect(page.getByTestId("review-gate-topic")).toBeVisible();
  await expect(page.getByTestId("review-gate-content-validity")).toBeVisible();
  await expect(page.getByTestId("review-requires-human-review")).toBeVisible();
  await expect(page.getByTestId("review-human-review-guidance")).toBeVisible();
  await expectNoVisibleSerifTypography(page);
  await captureScreenshot(page, "visual-refresh-smoke-review-failed-desktop.png");
  await expectMobileRouteAudit(page, "visual-refresh-smoke-review-failed-mobile.png");

  await page.getByTestId("review-action-view-history").click();
  await expect(page).toHaveURL(/\/history$/);
  await expect(page.getByTestId("practice-progress")).toBeVisible();
  // These legacy reports have no saved analysis context; don't chart unlike attempts.
  await expect(page.getByTestId("practice-legacy")).toBeVisible();
  await expect(page.getByTestId("history-priority-resolved")).toHaveCount(0);
  await expect(page.getByTestId("history-attempts-table")).toBeVisible();
  await expect(page.getByTestId("history-attempts-mobile-list")).not.toBeVisible();
  await expect(page.getByTestId("history-detail-digest")).toBeVisible();
  await expect(page.getByTestId("history-detail-digest")).toContainText(
    "Keep the structure and close with one personal reflection.",
  );
  await expect(page.getByTestId("history-detail-digest")).toContainText("Clearer sequencing");
  await expect(page.getByTestId("history-detail-digest")).toContainText("Add one clearer closing sentence");
  await expect(page.getByTestId("history-detail-digest")).toContainText(
    "Close with one personal reflection.",
  );
  await expect(page.getByTestId("history-detail-full-report")).not.toHaveAttribute("open", "");
  await expect(page.getByTestId("review-summary")).not.toBeVisible();
  await expectNoVisibleSerifTypography(page);
  await captureScreenshot(page, "visual-refresh-smoke-history-desktop.png");
  await page.getByTestId("history-detail-full-report-summary").click();
  await expect(page.getByTestId("history-detail-full-report")).toHaveAttribute("open", "");
  await expect(page.getByTestId("review-summary")).toBeVisible();
  await captureScreenshot(page, "visual-refresh-smoke-history-expanded-desktop.png");
  await page.getByTestId("history-detail-full-report-summary").click();
  await expect(page.getByTestId("history-detail-full-report")).not.toHaveAttribute("open", "");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId("practice-progress")).toBeVisible();
  await expect(page.getByTestId("history-attempt-card-visual-second")).toBeVisible();
  await expect(page.getByTestId("history-attempt-card-visual-second")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("history-attempt-card-visual-second")).toContainText("Score 4.1");
  await expect(page.getByTestId("history-attempts-table")).not.toBeVisible();
  await expect(page.getByTestId("history-detail-select")).not.toBeVisible();
  await expect(page.getByTestId("history-detail-digest")).toBeVisible();
  const storyBox = await page.getByTestId("practice-progress").boundingBox();
  expect(storyBox?.y ?? Number.POSITIVE_INFINITY).toBeLessThan(844);
  await expectMobileRouteAudit(page, "visual-refresh-smoke-history-mobile.png");
  await page.setViewportSize(narrowMobileViewport);
  await expect(page.getByTestId("practice-progress")).toBeVisible();
  await expectNoPageHorizontalOverflow(page);
  await captureScreenshot(page, "visual-refresh-smoke-history-narrow-mobile.png");
  await page.setViewportSize(desktopViewport);
  await page.getByTestId("history-detail-full-report-summary").click();
  await expect(page.getByTestId("history-detail-full-report")).toHaveAttribute("open", "");
  await expect(page.getByTestId("review-summary")).toBeVisible();
  await expectMobileRouteAudit(page, "visual-refresh-smoke-history-expanded-mobile.png");
});
