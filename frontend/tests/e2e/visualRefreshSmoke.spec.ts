import { stubReadyReadiness } from "../readinessStub";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { expect, test, type Page, type Route } from "../fixtures";

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
  metrics: { duration_sec: 120 },
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
      cefr_estimate: { level: "B2", calibrated: false, confidence: "low" },
      dimensions: { pronunciation_intelligibility: { score: 3.6, confidence: "low", method: "asr_proxy_v1" } },
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

const installVisualBackend = async (page: Page, locale = "en") => {
  await stubReadyReadiness(page);
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
  await page.route("**/v1/runtime/settings", (route) => fulfillJson(route, { ...runtimeSettings, ui_locale: locale }));
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
  await expect(page.getByTestId("speak.recording_status")).toHaveAttribute("data-tone", "success");
};

test("report finalization marks Coach complete and Finalizing current", async ({ page }) => {
  await installVisualBackend(page);
  await page.unroute("**/v1/assessments/*");
  await page.route("**/v1/assessments/*", route => fulfillJson(route, {
    assessment_id: "visual-assessment-clean", status: "running", phase: "finalizing_report", progress: 0.95,
    created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
    worker_seen_at: new Date().toISOString(), error: null,
  }));
  await page.goto("/");
  await page.getByTestId("home.start_new").click();
  await page.getByTestId("setup.speaker_id").fill("finalize-test");
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  await attachAudio(page);
  await page.getByTestId("speak.submit").click();
  const progress = page.getByTestId("speak.assessment_progress");
  await expect(progress.locator('li[aria-current="step"]')).toHaveText("Finalizing the report.");
  await expect(progress.getByRole("listitem").filter({ hasText: "Coach" })).toContainText("Completed");
  await expect(page.getByTestId("speak.retry_assessment")).toHaveCount(0);
});

test("assessment wait explains a stall and recovers without a second upload", async ({ page }) => {
  await installVisualBackend(page);
  const shots = process.env.ASSESSMENT_WAIT_SCREENSHOT_DIR;
  let uploads = 0;
  let cancels = 0;
  let workerFresh = true;
  let pollNetworkError = false;
  let phaseAgeMs = 180_000;
  let currentId = "";
  let state: "stalled" | "failed" | "running" | "cancelled" = "stalled";
  const requests: Record<string, unknown>[] = [];
  await page.route("**/v1/uploads", route => {
    uploads += 1;
    return fulfillJson(route, { audio_id: "wait-audio", stored_path: "/tmp/wait-audio.wav", sha1: "wait-sha1", original_name: "wait.wav" });
  });
  await page.route("**/v1/assessments", route => {
    requests.push(route.request().postDataJSON());
    currentId = `wait-${requests.length}`;
    state = requests.length === 1 ? "stalled" : requests.length === 2 ? "failed" : "running";
    return fulfillJson(route, { assessment_id: currentId, status: "queued" });
  });
  await page.route("**/v1/assessments/*/cancel", route => {
    cancels += 1;
    state = "cancelled";
    return fulfillJson(route, { assessment_id: currentId, status: "cancelled", phase: "cancelled", last_phase: "scoring_rubric", progress: 1 });
  });
  await page.route("**/v1/assessments/*", route => {
    if (pollNetworkError) return route.abort("failed");
    const old = new Date(Date.now() - phaseAgeMs).toISOString();
    const recent = new Date().toISOString();
    const stopped = state === "failed" || state === "cancelled";
    return fulfillJson(route, { assessment_id: currentId, status: state === "stalled" ? "running" : state,
      phase: stopped ? state : "scoring_rubric", progress: 0,
      created_at: state === "stalled" ? old : recent,
      updated_at: state === "stalled" ? old : recent,
      worker_seen_at: stopped ? null : state === "stalled" && !workerFresh ? old : recent,
      quiet_warning_after_sec: 210,
      phase_started_at: state === "stalled" ? old : recent,
      last_phase: stopped ? "scoring_rubric" : null,
      error: state === "failed" ? { code: "runtime_error", detail: "Model request failed" } : null });
  });

  await page.setViewportSize({ width: 1360, height: 900 });
  await page.goto("/");
  await page.getByTestId("home.start_new").click();
  await page.getByTestId("setup.speaker_id").fill("wait-test");
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  await attachAudio(page);
  await page.getByTestId("speak.submit").click();
  await expect(page.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "false");
  await expect(page.getByTestId("speak.wait_time")).toContainText("Last progress update 3 min");
  await expect(page.getByTestId("speak.cancel_assessment")).toBeVisible();
  pollNetworkError = true;
  await expect(page.getByTestId("speak.status_panel")).toContainText("Could not refresh progress", { timeout: 15_000 });
  await expect(page.getByTestId("speak.cancel_assessment")).toBeVisible();
  pollNetworkError = false;
  await expect(page.getByTestId("speak.status_panel")).not.toContainText("Could not refresh progress", { timeout: 15_000 });
  phaseAgeMs = 240_000;
  await expect(page.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "true");
  await expect(page.getByTestId("speak.media_state")).toContainText("Audio uploaded and retained");
  await expect(page.getByTestId("speak.wait_time")).toContainText("Last progress update 4 min");
  await expect(page.getByTestId("speak.cancel_assessment")).toBeVisible();
  await expect(page.getByTestId("speak.retry_assessment")).toHaveCount(0);
  await page.getByTestId("speak.status_panel").scrollIntoViewIfNeeded();
  await expect(page.getByTestId("speak.cancel_assessment")).toBeInViewport();
  if (shots) {
    mkdirSync(shots, { recursive: true });
    await page.screenshot({ path: path.join(shots, "after-assessment-stall-1360x900.png") });
  }
  await page.setViewportSize({ width: 1100, height: 760 });
  await expect(page.getByTestId("speak.cancel_assessment")).toBeVisible();
  await page.getByTestId("speak.status_panel").scrollIntoViewIfNeeded();
  await expect(page.getByTestId("speak.cancel_assessment")).toBeInViewport();
  if (shots) await page.screenshot({ path: path.join(shots, "after-assessment-stall-1100x760.png") });

  workerFresh = false;
  await expect(page.getByTestId("speak.assessment_progress")).toHaveAttribute("data-stalled", "false");

  await page.getByTestId("speak.cancel_assessment").click();
  await expect(page.getByTestId("speak.assessment_progress")).toHaveCount(0);
  await expect(page.getByTestId("speak.submit")).toHaveCount(0);
  await page.getByTestId("speak.retry_assessment").click();
  await expect(page.getByTestId("speak.status_panel")).toContainText("Model request failed");
  for (const theme of ["light", "dark"] as const) {
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    await expectStatusRailIconContrast(page, "attention");
  }
  await page.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  await expect(page.getByTestId("speak.submit")).toHaveCount(0);
  await page.getByTestId("speak.retry_assessment").click();
  await expect(page.getByTestId("speak.cancel_assessment")).toBeVisible();
  await page.getByTestId("speak.cancel_assessment").click();
  await expect(page.getByTestId("speak.status_panel")).toContainText("cancelled");
  await expect(page.getByTestId("speak.assessment_progress")).toHaveCount(0);
  expect(uploads).toBe(1);
  expect(cancels).toBe(2);
  expect(requests).toHaveLength(3);
  expect(requests[1]).toMatchObject({ audio_id: "wait-audio", resume_assessment_id: "wait-1" });
  expect(requests[2]).toMatchObject({ audio_id: "wait-audio", resume_assessment_id: "wait-2" });
  expect(new Set(requests.map(request => request.request_id)).size).toBe(3);
});

test("cancelled upload keeps local media without claiming a lost confirmation", async ({ page }) => {
  await installVisualBackend(page);
  let uploadStarted!: () => void;
  const started = new Promise<void>(resolve => { uploadStarted = resolve; });
  await page.route("**/v1/uploads", async route => {
    uploadStarted();
    await new Promise(resolve => setTimeout(resolve, 1500));
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      audio_id: "should-not-be-used", stored_path: "/tmp/should-not-be-used.wav", sha1: "unused", original_name: "wait.wav",
    }) }).catch(() => undefined);
  });
  await page.goto("/");
  await page.getByTestId("home.start_new").click();
  await page.getByTestId("setup.speaker_id").fill("upload-abort-test");
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  await attachAudio(page);
  await page.getByTestId("speak.submit").click();
  await started;
  await page.getByTestId("speak.cancel_assessment").click();
  await expect(page.getByTestId("speak.status_panel")).toContainText("Upload cancelled");
  await expect(page.getByTestId("speak.media_state")).not.toContainText("Upload confirmation was lost");
  await expect(page.getByTestId("speak.media_state")).toContainText("selected on this device");
});

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
        const style = getComputedStyle(element);
        if (style.display === "none" || style.visibility !== "visible" || rect.width === 0) return null;
        for (let parent = element.parentElement; parent; parent = parent.parentElement) {
          if (/(auto|scroll)/.test(getComputedStyle(parent).overflowX) && parent.scrollWidth > parent.clientWidth) return null;
        }

        return {
          className: element.className ? String(element.className) : "",
          tagName: element.tagName.toLowerCase(),
          testId: element.dataset.testid ?? "",
          width: Math.ceil(rect.width),
          left: Math.floor(rect.left),
          right: Math.ceil(rect.right),
        };
      })
      .filter((element): element is NonNullable<typeof element> => element !== null)
      .filter((element) => element.width > viewportWidth + 1 || element.right > viewportWidth + 1 || element.left < -1)
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
  expect(overflow.overflowingElements, JSON.stringify(overflow, null, 2)).toEqual([]);
};

const contrastRatio = (foreground: string, background: string) => {
  const luminance = (color: string) => {
    const channels = color.match(/[\d.]+/g)?.slice(0, 3).map(Number);
    if (!channels || channels.length !== 3) throw new Error(`Expected an RGB color, got ${color}`);
    const [red, green, blue] = channels.map(value => {
      const normalized = value / 255;
      return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  };
  const [lighter, darker] = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
};

const expectStatusRailIconContrast = async (page: Page, state: "active" | "attention") => {
  const icon = page.locator(`[data-testid="speak.status_rail"] [data-step-state="${state}"] > span`).first();
  await expect(icon).toBeVisible();
  const colors = await icon.evaluate(element => ({
    foreground: getComputedStyle(element).color,
    background: getComputedStyle(element).backgroundColor,
  }));
  expect(contrastRatio(colors.foreground, colors.background), `${state} status icon`).toBeGreaterThanOrEqual(3);
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
  await expect(page.getByTestId("setup.runtime_callout")).toContainText("Continue to the recording step");
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
  const estimate = page.getByTestId("review-result-estimate");
  await expect(estimate.getByTestId("review-estimate-text")).toHaveText("4.1/5");
  await expect(estimate).toContainText("Position on a 1–5 practice scale");
  await expect(estimate).toContainText("Provisional · not an exam result");
  await expect(estimate.getByTestId("review-provisional-cefr")).toContainText("Provisional CEFR B2");
  await expect(estimate).toContainText("3.6/5 · limited proxy");
  await expect(estimate).toContainText("does not assess individual sounds or accent quality");
  await expect(estimate.getByRole("progressbar")).toHaveCount(0);
  await expect(estimate.getByRole("link", { name: "Read the scoring Guide" })).toHaveAttribute("href", "/guide");
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
  for (const viewport of [{ width: 1360, height: 900 }, { width: 1100, height: 760 }]) {
    await page.setViewportSize(viewport);
    await expectNoPageHorizontalOverflow(page);
    await captureScreenshot(page, `review-result-${viewport.width}x${viewport.height}.png`);
    await page.locator("body").click({ position: { x: 1, y: 1 } });
    await page.keyboard.press("ControlOrMeta+Home");
    let guideFocused = false;
    for (let step = 0; step < 60 && !guideFocused; step += 1) {
      await page.keyboard.press("Tab");
      guideFocused = await page.evaluate(() =>
        document.activeElement?.getAttribute("href") === "/guide" &&
        Boolean(document.activeElement?.closest('[data-testid="review-result-estimate"]')));
    }
    expect(guideFocused).toBe(true);
    await expect(estimate.getByRole("link", { name: "Read the scoring Guide" })).toBeFocused();
  }
  await page.setViewportSize(desktopViewport);
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
  await expect(page.getByTestId("history-attempts-list")).toBeVisible();
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
  await page.evaluate(() => window.scrollTo(0, 0));
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

test("stored appearance is applied before loading and main DOM content is ready", async ({ page }) => {
  await installVisualBackend(page);
  await page.addInitScript(() => {
    document.addEventListener("DOMContentLoaded", () => {
      document.documentElement.setAttribute("data-theme-at-dom-ready", document.documentElement.dataset.theme ?? "");
    });
  });
  await page.goto("/");
  for (const [appearance, system, theme] of [
    ["", "light", "light"], ["", "dark", "dark"],
    ["dark", "light", "dark"], ["system", "dark", "dark"], ["light", "dark", "light"],
  ] as const) {
    await page.evaluate(value => value
      ? localStorage.setItem("vostavo.appearance.v1", value)
      : localStorage.removeItem("vostavo.appearance.v1"), appearance);
    await page.emulateMedia({ colorScheme: system });
    for (const url of ["/loading.html", "/"]) {
      await page.goto(url);
      await expect(page.locator("html")).toHaveAttribute("data-theme-at-dom-ready", theme);
      await expect(page.locator("html")).toHaveCSS("color-scheme", theme);
      const colors = await page.evaluate(() => ({
        text: getComputedStyle(document.body).color,
        background: getComputedStyle(document.body).backgroundColor,
      }));
      expect(contrastRatio(colors.text, colors.background), `${appearance} ${url} first surface`).toBeGreaterThanOrEqual(4.5);
      const opposite = theme === "dark" ? "rgb(255, 255, 255)" : "rgb(0, 0, 0)";
      expect(contrastRatio(opposite, colors.background), `${appearance} ${url} surface polarity`).toBeGreaterThanOrEqual(4.5);
    }
  }
});

test("system default and explicit themes keep the journal card usable at desktop sizes", async ({ page }) => {
  test.setTimeout(120_000);
  await installVisualBackend(page);
  await page.goto("/");
  await page.emulateMedia({ colorScheme: "dark" });
  await expect(page.locator("body")).toHaveCSS("background-color", "rgb(17, 27, 25)");
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/runtime-setup");
  const otherRoutes = page.getByText("Choose another route: cloud or expert setup");
  await otherRoutes.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("region", { name: "Cloud services" })).toBeVisible();
  await page.goto("/");
  await page.getByTestId("home.start_new").click();
  await page.getByTestId("setup.speaker_id").fill(speakerId);
  await page.getByTestId("setup.recommended_start").click();
  await page.getByTestId("setup.continue").click();
  await expect(page).toHaveURL(/\/speak$/);

  for (const viewport of [{ width: 1360, height: 900 }, { width: 1100, height: 760 }]) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => { document.documentElement.dataset.theme = "light"; });
    const suffix = `light-${viewport.width}x${viewport.height}.png`;

    await page.getByRole("link", { name: "Practice", exact: true }).click();
    const homeLink = page.getByRole("link", { name: "Practice", exact: true });
    await homeLink.focus();
    await expect(homeLink).toHaveCSS("outline-width", "3px");
    await page.keyboard.press("Tab");
    expect(await page.evaluate(() => document.activeElement?.tagName)).toBe("A");
    await expectNoPageHorizontalOverflow(page);
    await captureScreenshot(page, `migration-shell-${suffix}`);

    await page.getByTestId("home.start_new").click();
    const speaker = page.getByTestId("setup.speaker_id");
    await speaker.focus();
    await expect(speaker).toHaveCSS("outline-width", "3px");
    await expectNoPageHorizontalOverflow(page);
    await captureScreenshot(page, `migration-form-${suffix}`);

    await speaker.fill(speakerId);
    await page.getByTestId("setup.recommended_start").click();
    await page.getByTestId("setup.continue").click();
    const recordMode = page.getByTestId("speak.input_mode_record");
    await expect(recordMode).toHaveAttribute("aria-pressed", "true");
    await recordMode.focus();
    await expect(recordMode).toHaveCSS("outline-width", "3px");
    await recordMode.hover();
    await expect(recordMode).toHaveCSS("background-color", "rgb(223, 241, 236)");
    await expect(page.getByTestId("speak.record_start")).toHaveCSS("min-height", "48px");
    await expectNoPageHorizontalOverflow(page);
    await captureScreenshot(page, `migration-speak-${suffix}`);

    await page.getByRole("link", { name: "Settings", exact: true }).click();
    const journal = page.getByTestId("journal-panel");
    await expect(journal).toBeVisible();
    await expect(journal.getByRole("button", { name: "Create and save backup" })).toHaveCSS("min-height", "44px");
    const restore = journal.getByRole("button", { name: "Choose a backup ZIP" });
    await restore.focus();
    await expect(restore).toHaveCSS("outline-width", "3px");
    await expectNoPageHorizontalOverflow(page);
    await captureScreenshot(page, `migration-journal-${suffix}`);

    await page.getByRole("link", { name: "Progress", exact: true }).click();
    const archive = page.getByRole("button", { name: "Archive attempt" });
    await expect(archive).toBeVisible();
    await archive.focus();
    await expect(archive).toHaveCSS("outline-width", "3px");
    await captureScreenshot(page, `migration-archive-${suffix}`);
    await expectNoPageHorizontalOverflow(page);

    // Exercise an explicit theme override on the journal, form and focus state.
    await page.evaluate(() => { document.documentElement.dataset.theme = "dark"; });
    await page.goBack();
    await page.goBack();
    await page.getByRole("link", { name: "Session Setup" }).click();
    const darkSpeaker = page.getByTestId("setup.speaker_id");
    await expect(darkSpeaker).toHaveCSS("background-color", "rgb(28, 41, 37)");
    await expect(darkSpeaker).toHaveCSS("color", "rgb(242, 248, 244)");
    await page.getByRole("link", { name: "Settings", exact: true }).click();
    const darkJournal = page.getByTestId("journal-panel");
    const panelBackground = await darkJournal.evaluate(element => getComputedStyle(element).backgroundColor);
    const headingColor = await darkJournal.locator("h2").evaluate(element => getComputedStyle(element).color);
    expect(panelBackground).toBe("rgb(28, 41, 37)");
    expect(contrastRatio(headingColor, panelBackground)).toBeGreaterThanOrEqual(4.5);
    const backup = darkJournal.getByRole("button", { name: "Create and save backup" });
    await backup.focus();
    await page.keyboard.press("Tab");
    await page.keyboard.press("Shift+Tab");
    await expect(backup).toBeFocused();
    await expect(backup).toHaveCSS("outline-width", "3px");
    const focusColor = await backup.evaluate(element => getComputedStyle(element).outlineColor);
    expect(focusColor).toBe("rgb(206, 181, 255)");
    expect(contrastRatio(focusColor, panelBackground)).toBeGreaterThanOrEqual(3);
    await captureScreenshot(page, `migration-journal-dark-${viewport.width}x${viewport.height}.png`);
    await page.evaluate(() => { document.documentElement.dataset.theme = "light"; });
  }

  await page.route("**/v1/journal/status", route => fulfillJson(route, {
    transaction: null, archived: [], completed: [], recovery_error: "Recovery files are unavailable.",
  }));
  await page.goto("/settings");
  await page.getByRole("button", { name: "Start a fresh journal" }).click();
  const danger = page.getByRole("button", { name: "Confirm: start fresh" });
  await danger.hover();
  await page.mouse.down();
  await expect(danger).toHaveCSS("background-color", "rgb(155, 37, 48)");
  const dangerColors = await danger.evaluate(element => ({
    foreground: getComputedStyle(element).color,
    background: getComputedStyle(element).backgroundColor,
  }));
  expect(contrastRatio(dangerColors.foreground, dangerColors.background)).toBeGreaterThanOrEqual(4.5);
  await page.mouse.move(0, 0);
  await page.mouse.up();
  await page.evaluate(() => { document.documentElement.dataset.theme = "dark"; });
  await danger.hover();
  await page.mouse.down();
  await expect(danger).toHaveCSS("background-color", "rgb(255, 180, 180)");
  const darkDangerColors = await danger.evaluate(element => ({
    foreground: getComputedStyle(element).color,
    background: getComputedStyle(element).backgroundColor,
  }));
  expect(contrastRatio(darkDangerColors.foreground, darkDangerColors.background)).toBeGreaterThanOrEqual(4.5);
  await page.mouse.move(0, 0);
  await page.mouse.up();
});

for (const locale of ["de", "en", "es", "fr", "it"] as const) {
test(`semantic light and dark surfaces keep primary routes readable at both desktop sizes (${locale})`, async ({ page }) => {
  test.setTimeout(180_000);
  await installVisualBackend(page, locale);
  const healthyJournal = (route: Route) => fulfillJson(route, {
    transaction: null, archived: [], completed: [], recovery_error: null,
  });
  await page.route("**/v1/journal/status", healthyJournal);
  await page.emulateMedia({ reducedMotion: "reduce" });
  const output = process.env.THEME_MIGRATION_SCREENSHOT_DIR;
  const contrastChecks: Array<{ locale: string; route: string; theme: string; viewport: string; text: number; border: number; focus: number; textCount: number; borderCount: number }> = [];

  const audit = async (route: string, theme: string) => {
    for (const viewport of [{ width: 1360, height: 900 }, { width: 1100, height: 760 }]) {
      await page.setViewportSize(viewport);
      await expect(page.locator("main").getByRole("heading").first()).toBeVisible();
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await expectNoPageHorizontalOverflow(page);
      // Sample the first keyboard-reachable control in each main or dialog state.
      const focusScope = await page.locator("dialog[open]").count() ? "dialog[open]" : "main";
      const focusTarget = page.locator(`${focusScope} button:visible:not(:disabled):not([aria-disabled='true']):not([tabindex='-1']), ${focusScope} input:visible:not(:disabled):not([tabindex='-1']), ${focusScope} select:visible:not(:disabled):not([tabindex='-1']), ${focusScope} a[href]:visible:not([aria-disabled='true']):not([tabindex='-1']), ${focusScope} summary:visible:not([tabindex='-1'])`).first();
      await expect(focusTarget).toBeVisible();
      await focusTarget.focus();
      await page.keyboard.press("Tab");
      await page.keyboard.press("Shift+Tab");
      await expect(focusTarget).toBeFocused();
      await expect(focusTarget).toHaveCSS("outline-offset", "3px");
      const computed = await page.evaluate(() => {
        const scope = document.querySelector("dialog[open]") ?? document.body;
        const parse = (value: string): [number, number, number, number] => {
          const values = value.match(/[\d.]+/g)?.map(Number) ?? [];
          return values.length >= 3 ? [values[0], values[1], values[2], values[3] ?? 1] : [0, 0, 0, 0];
        };
        const blend = (top: number[], bottom: number[]) => {
          const alpha = top[3] + bottom[3] * (1 - top[3]);
          return [0, 1, 2].map(index => (top[index] * top[3] + bottom[index] * bottom[3] * (1 - top[3])) / alpha).concat(alpha);
        };
        const background = (element: Element) => {
          const chain: Element[] = [];
          for (let node: Element | null = element; node; node = node.parentElement) chain.unshift(node);
          return chain.reduce<number[]>((result, node) => blend(parse(getComputedStyle(node).backgroundColor), result), [255, 255, 255, 1]);
        };
        const luminance = (rgb: number[]) => {
          const channels = rgb.slice(0, 3).map(value => {
            const fraction = value / 255;
            return fraction <= 0.04045 ? fraction / 12.92 : ((fraction + 0.055) / 1.055) ** 2.4;
          });
          return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
        };
        const ratio = (first: number[], second: number[]) => {
          const values = [luminance(first), luminance(second)].sort((a, b) => b - a);
          return (values[0] + .05) / (values[1] + .05);
        };
        const visible = (element: Element) => {
          const style = getComputedStyle(element);
          return style.display !== "none" && style.visibility === "visible" && style.opacity !== "0" &&
            element.getClientRects().length > 0;
        };
        const text: Array<{ label: string; ratio: number; foreground: string; background: number[] }> = [];
        const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
          const node = walker.currentNode;
          const element = node.parentElement;
          const label = node.textContent?.trim();
          if (!element || !label || !visible(element) ||
              element.closest("button:disabled, [aria-disabled='true'], script, style, noscript")) continue;
          const range = document.createRange();
          range.selectNodeContents(node);
          if (!range.getClientRects().length) continue;
          const style = getComputedStyle(element);
          const foreground = element instanceof SVGElement ? style.fill : style.color;
          if (foreground === "none") continue;
          const surface = background(element);
          text.push({ label: label.slice(0, 55), ratio: ratio(blend(parse(foreground), surface), surface), foreground, background: surface });
        }
        for (const element of scope.querySelectorAll("input, select, textarea")) {
          if (!visible(element) || (element as HTMLInputElement).disabled) continue;
          const surface = background(element);
          const foreground = getComputedStyle(element).color;
          text.push({ label: element.getAttribute("aria-label") ?? element.tagName,
            ratio: ratio(parse(foreground), surface), foreground, background: surface });
        }
        const border = [...scope.querySelectorAll("button, input, select, textarea, summary, [role='button'], [role='tab']")]
          .filter(element => visible(element) && !(element as HTMLButtonElement).disabled && element.getAttribute("aria-disabled") !== "true")
          .flatMap(element => {
            const style = getComputedStyle(element);
            const fill = background(element);
            const outside = background(element.parentElement ?? element);
            if (ratio(fill, outside) >= 3) return [];
            const edges = ["Top", "Right", "Bottom", "Left"] as const;
            const ratios = edges.flatMap(edge => {
              const width = Number.parseFloat(style[`border${edge}Width`]);
              if (!width || style[`border${edge}Style`] === "none") return [];
              const borderColor = parse(style[`border${edge}Color`]);
              if (borderColor[3] === 0) return [];
              const ink = blend(borderColor, fill);
              return [Math.min(ratio(ink, fill), ratio(ink, outside))];
            });
            return ratios.length ? [{ label: (element.textContent?.trim() || element.getAttribute("aria-label") || element.tagName).slice(0, 55),
              ratio: Math.min(...ratios) }] : [];
          });
        const clippedText = [...scope.querySelectorAll<HTMLElement>("button, a, summary, label, h1, h2, h3, p, span")]
          .filter(element => visible(element) && element.textContent?.trim())
          .flatMap(element => {
            const style = getComputedStyle(element);
            if (element.clientWidth <= 1 || style.clipPath === "inset(50%)" || style.clip === "rect(0px, 0px, 0px, 0px)") return [];
            if (!/hidden|clip/.test(style.overflowX)) return [];
            return element.scrollWidth > element.clientWidth + 1
              ? [{ label: element.textContent!.trim().slice(0, 55), width: element.clientWidth, scrollWidth: element.scrollWidth }]
              : [];
          });
        const focused = document.activeElement;
        if (!focused) throw new Error("No focused control to audit");
        const focusStyle = getComputedStyle(focused);
        const outside = background(focused.parentElement ?? focused);
        const focusContrast = (ink: string) => ratio(parse(ink), outside);
        // A ring matching its surrounding surface must fail this same calculation.
        const invisibleInk = `rgb(${outside.slice(0, 3).join(", ")})`;
        const invisibleRatio = focusContrast(invisibleInk);
        const focus = { width: Number.parseFloat(focusStyle.outlineWidth),
          ratio: focusContrast(getComputedStyle(focused).outlineColor), invisibleRatio,
          label: focused.textContent?.trim().slice(0, 50) ?? focused.tagName,
          ink: focusStyle.outlineColor,
          outside };
        return { text, border, clippedText, focus };
      });
      expect(computed.text.length, `${theme} ${route} ${viewport.width} text coverage`).toBeGreaterThan(0);
      if (route === "review") {
        expect(computed.text.some(item => item.label === "1"), "score-scale tick coverage").toBe(true);
        await expect(page.getByTestId("review-provisional-cefr")).toBeVisible();
        expect(computed.text.some(item => item.label.includes("B2")), "CEFR badge coverage").toBe(true);
      }
      if (route === "settings-error") {
        expect(computed.text.some(item => item.label.includes("Recovery files are unavailable")), "error-banner coverage").toBe(true);
      }
      for (const item of computed.text) expect(item.ratio, `${theme} ${route} ${viewport.width}: ${JSON.stringify(item)}`).toBeGreaterThanOrEqual(4.5);
      for (const item of computed.border) expect(item.ratio, `${theme} ${route} ${viewport.width}: ${item.label} border`).toBeGreaterThanOrEqual(3);
      expect(computed.focus.width, `${locale} ${theme} ${route} ${viewport.width} focus width`).toBeGreaterThanOrEqual(3);
      expect(computed.focus.ratio, `${locale} ${theme} ${route} ${viewport.width} focus contrast ${JSON.stringify(computed.focus)}`).toBeGreaterThanOrEqual(3);
      expect(computed.focus.invisibleRatio, `${locale} ${theme} ${route} invisible focus regression`).toBeLessThan(3);
      expect(computed.clippedText, `${locale} ${theme} ${route} ${viewport.width} clipped text`).toEqual([]);
      if (route === "speak" || route === "speak-wait") await expectStatusRailIconContrast(page, "active");
      contrastChecks.push({ locale, route, theme, viewport: `${viewport.width}x${viewport.height}`,
        text: Math.min(...computed.text.map(item => item.ratio)),
        border: computed.border.length ? Math.min(...computed.border.map(item => item.ratio)) : -1,
        focus: computed.focus.ratio,
        textCount: computed.text.length, borderCount: computed.border.length });
      if (output) {
        mkdirSync(output, { recursive: true });
        if (route !== "your-data" && route !== "speak-wait") await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({ path: path.join(output, `${locale}-${theme}-${route}-${viewport.width}x${viewport.height}.png`) });
        if (route === "settings") {
          await page.getByTestId("settings.appearance").scrollIntoViewIfNeeded();
          await page.screenshot({ path: path.join(output, `${locale}-${theme}-settings-appearance-${viewport.width}x${viewport.height}.png`) });
        }
      }
    }
    // A 1100px window at 200% browser zoom has roughly a 550px CSS viewport.
    await page.setViewportSize({ width: 550, height: 380 });
    await expectNoPageHorizontalOverflow(page);
    const controlRoot = await page.locator("dialog[open]").count() ? "dialog[open]" : "main";
    const controls = page.locator(`${controlRoot} button:visible, ${controlRoot} a:visible, ${controlRoot} input:visible, ${controlRoot} select:visible, ${controlRoot} summary:visible`);
    for (let index = 0; index < Math.min(await controls.count(), 10); index++) {
      const control = controls.nth(index);
      await control.scrollIntoViewIfNeeded();
      const reachability = await control.evaluate(element => {
        const box = element.getBoundingClientRect();
        const center = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
        const hitPoints = [0.2, 0.5, 0.8].map(fraction =>
          document.elementFromPoint(box.left + box.width * fraction, box.top + box.height / 2));
        return { left: box.left, right: box.right, top: box.top, width: box.width, viewport: innerWidth,
          label: element.getAttribute("data-testid") || element.textContent?.trim().slice(0, 40),
          covering: center?.getAttribute("data-testid") || center?.tagName,
          element: element.outerHTML.slice(0, 180), cover: center?.outerHTML.slice(0, 180),
          hidden: (() => {
            for (let node: Element | null = element; node; node = node.parentElement) {
              const style = getComputedStyle(node);
              if (style.opacity === "0" || style.clipPath === "inset(50%)" || style.clip === "rect(0px, 0px, 0px, 0px)") return true;
            }
            return false;
          })(),
          hittable: hitPoints.some(target => Boolean(target && (element === target || element.contains(target)))) };
      });
      if (reachability.hidden) continue;
      expect(reachability.left, `${theme} ${route} 200% left edge`).toBeGreaterThanOrEqual(-1);
      expect(reachability.right, `${theme} ${route} 200% right edge`).toBeLessThanOrEqual(reachability.viewport + 1);
      expect(reachability.hittable, `${theme} ${route} 200% control hit target ${JSON.stringify(reachability)}`).toBe(true);
    }
  };

  for (const theme of ["light", "dark"] as const) {
    await page.goto("/settings");
    await expect(page.locator("html")).toHaveAttribute("lang", locale);
    await page.getByTestId("settings.appearance").selectOption(theme);
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await expect(page.locator("html")).toHaveCSS("color-scheme", theme);
    for (const [route, url] of [
      ["home", "/"], ["setup", "/runtime-setup"], ["history", "/history"],
      ["library", "/library"], ["guide", "/guide"], ["settings", "/settings"],
      ["your-data", "/settings#your-data"], ["rehearsal", "/rehearsal"],
    ]) {
      await page.goto(url);
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      await audit(route, theme);
      if (route === "your-data") {
        await page.unroute("**/v1/journal/status");
        await page.route("**/v1/journal/status", response => fulfillJson(response, {
          transaction: null, archived: [], completed: [], recovery_error: "Recovery files are unavailable.",
        }));
        await page.reload();
        await audit("settings-error", theme);
        await page.locator("#journal-fresh").click();
        await expect(page.locator("dialog[open]")).toBeVisible();
        await audit("settings-dialog", theme);
        await page.locator("#fresh-cancel").click();
        await page.unroute("**/v1/journal/status");
        await page.route("**/v1/journal/status", healthyJournal);
      }
    }

    await page.unroute("**/v1/history");
    await page.route("**/v1/history", response => fulfillJson(response, { items: [] }));
    await page.goto("/history");
    await expect(page.getByTestId("history-empty")).toBeVisible();
    await audit("history-empty", theme);
    await page.unroute("**/v1/history");
    await page.route("**/v1/history", response => fulfillJson(response, { items: historyRows }));

    await page.route("**/v1/samples", response => response.fulfill({ status: 503, json: { detail: "Samples unavailable" } }));
    await page.goto("/library");
    await expect(page.getByTestId("library-samples-error")).toBeVisible({ timeout: 20_000 });
    await audit("library-error", theme);
    await page.unroute("**/v1/samples");

    await page.goto("/review");
    await expect(page.getByTestId("review-guard-missing-review")).toBeVisible();
    await audit("review-empty", theme);

    await page.goto("/");
    await page.getByTestId("home.start_new").click();
    await audit("session-setup", theme);
    await page.getByTestId("setup.speaker_id").fill(speakerId);
    await page.getByTestId("setup.recommended_start").click();
    await page.getByTestId("setup.continue").click();
    await audit("speak", theme);
    const meter = page.getByTestId("speak.recording_visualizer");
    await meter.evaluate(element => element.setAttribute("data-recording-state", "recording"));
    const recordingColors = await meter.evaluate(element => {
      const bar = element.querySelector("span");
      if (!bar) throw new Error("Recording meter bar is missing");
      const background = getComputedStyle(element).backgroundColor;
      const ink = getComputedStyle(bar);
      const base = background.match(/[\d.]+/g)?.slice(0, 3).map(Number) ?? [];
      const top = ink.backgroundColor.match(/[\d.]+/g)?.slice(0, 3).map(Number) ?? [];
      const alpha = Number(ink.opacity);
      return { background, bar: `rgb(${top.map((value, index) => Math.round(value * alpha + base[index] * (1 - alpha))).join(", ")})` };
    });
    expect(contrastRatio(recordingColors.bar, recordingColors.background), `${theme} recording meter`).toBeGreaterThanOrEqual(3);
    await meter.evaluate(element => element.setAttribute("data-recording-state", "idle"));
    await attachAudio(page);
    await page.unroute("**/v1/assessments");
    await page.unroute("**/v1/assessments/*");
    let submission = 0;
    let outcome: "running" | "completed" | "failed" | "cancelled" = "running";
    await page.route("**/v1/assessments", route => {
      submission += 1;
      return fulfillJson(route, { assessment_id: `visual-${theme}-${submission}`, status: "queued" });
    });
    await page.route("**/v1/assessments/*", route => {
      const id = route.request().url().split("/").pop();
      return fulfillJson(route, outcome === "completed"
      ? { assessment_id: id, status: "completed", phase: "finalizing_report", progress: 1,
          report_path: "/tmp/visual-second.json", payload: reportPayload, summary: assessmentSummary(reportPayload) }
      : outcome === "failed" || outcome === "cancelled"
      ? { assessment_id: id, status: outcome, phase: outcome, last_phase: "scoring_rubric", progress: 1,
          error: outcome === "failed" ? { code: "runtime_error", detail: "Model request failed" } : null }
      : { assessment_id: id, status: "running", phase: "scoring_rubric", progress: .6,
          created_at: new Date().toISOString(), updated_at: new Date().toISOString(), worker_seen_at: new Date().toISOString() });
    });
    await page.getByTestId("speak.submit").click();
    await expect(page.getByTestId("speak.assessment_progress")).toBeVisible();
    await audit("speak-wait", theme);
    outcome = "completed";
    await expect(page).toHaveURL(/\/review$/, { timeout: 15000 });
    await audit("review", theme);
    outcome = "running";
    await page.getByTestId("review-action-try-again").click();
    await attachAudio(page);
    await page.getByTestId("speak.submit").click();
    const retryGoal = page.getByTestId("speak-retry-goal");
    await expect(retryGoal).toBeDisabled();
    await expect(retryGoal).toHaveCSS("background-color", theme === "light" ? "rgb(227, 233, 229)" : "rgb(51, 69, 61)");
    await expect(retryGoal).toHaveCSS("color", theme === "light" ? "rgb(65, 88, 80)" : "rgb(198, 215, 206)");
    outcome = "failed";
    await expect(page.getByTestId("speak.retry_assessment")).toBeVisible({ timeout: 15_000 });
    await audit("speak-failed", theme);
    outcome = "cancelled";
    await page.getByTestId("speak.retry_assessment").click();
    await expect(page.getByTestId("speak.status_panel")).toHaveAttribute("data-assessment-state", "cancelled", { timeout: 15_000 });
    await audit("speak-cancelled", theme);

    await page.goto("/");
    await page.setViewportSize({ width: 1100, height: 760 });
    await page.keyboard.press("Tab");
    const skip = page.locator('a[href="#main-content"]');
    await expect(skip).toBeFocused();
    await expect(skip).toHaveCSS("outline-width", "3px");
    const focus = await skip.evaluate(element => ({
      outline: getComputedStyle(element).outlineColor,
      surface: getComputedStyle(element).backgroundColor,
    }));
    expect(contrastRatio(focus.outline, focus.surface)).toBeGreaterThanOrEqual(3);
    await page.keyboard.press("Enter");
    await expect(page.locator("main")).toBeFocused();

    await page.emulateMedia({ reducedMotion: "reduce" });
    expect(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--motion-fast").trim())).toBe("0ms");
    const transitionMs = await page.getByTestId("home.start_new").evaluate(element =>
      getComputedStyle(element).transitionDuration.split(",").map(value => {
        const duration = Number.parseFloat(value);
        return value.trim().endsWith("ms") ? duration : duration * 1000;
      }));
    expect(Math.max(...transitionMs), `${theme} reduced-motion action transition`).toBeLessThanOrEqual(0.01);
    await page.setViewportSize({ width: 550, height: 380 });
    await expectNoPageHorizontalOverflow(page);
  }

  if (output) writeFileSync(path.join(output, `computed-contrast-${locale}.json`), JSON.stringify(contrastChecks, null, 2));
});
}
