import { expect, test } from "../fixtures";

for (const eligible of [false, true]) {
  test(`saved review ${eligible ? "retains eligible guidance" : "withholds passes for three words"}`, async ({ page }) => {
    const checks = { language_pass: true, topic_pass: eligible ? true : null,
      content_validity_pass: eligible ? true : null, duration_pass: eligible, min_words_pass: eligible };
    const payload = {
      meta: { learning_language: "en" }, transcript_full: eligible ? "A synthetic connected response." : "bla bla bla",
      // Simulates an old saved report with incorrectly awarded passes.
      baseline_comparison: { level: "B1", valid: true, passed: true, targets: {
        wpm: { expected: "≥80", actual: 80, ok: true, status: "pass" },
        fillers: { expected: "≤6", actual: 0, ok: true, status: "pass" },
      } },
      report: { session_id: "synthetic-speech", input: { expected_language: "en" }, checks,
        scores: { final: 3.5, deterministic: 3.5, llm: null, band: 4, mode: "deterministic_only" },
        warnings: eligible ? [] : ["llm_skipped_low_word_count"], requires_human_review: !eligible,
        coaching: { coach_summary: "You spoke in the required language.", strengths: ["Verified language"], top_3_priorities: [] },
        progress_delta: { comparison_verified: true, previous_session_id: "older", score_delta: { final: 1 } } },
    };
    await page.route("**/v1/runtime/settings", (route) => route.fulfill({ json: {
      ui_locale: "en", whisper_model: "small", active_connection_id: "", connections: [],
    } }));
    await page.route("**/v1/history", (route) => route.fulfill({ json: { items: [{
      session_id: "synthetic-speech", timestamp: "2026-10-08T07:00:00Z", speaker_id: "synthetic",
      learning_language: "en", theme: "Synthetic topic", report_path: "/tmp/synthetic-speech.json",
      overall: 3.5, final_score: 3.5, band: 4, wpm: 80, ...checks,
    }] } }));
    await page.route("**/v1/history/synthetic-speech", (route) => route.fulfill({ json: { payload } }));
    await page.goto("/history");
    await expect(page.getByTestId("history-detail-panel")).toBeVisible();
    await page.getByTestId("history-detail-full-report-summary").click();
    const baseline = page.getByTestId("review-baseline");
    await expect(baseline).toContainText("Expected");
    await expect(baseline).toContainText("Actual");
    if (eligible) {
      await expect(baseline.getByText("Pass", { exact: true })).toHaveCount(2);
      await expect(page.getByTestId("history-detail-digest-score")).toContainText("3.5");
      await expect(page.getByTestId("review-coach-summary")).toContainText("You spoke in the required language.");
    } else {
      await expect(baseline.getByText("Not assessed", { exact: true })).toHaveCount(2);
      await expect(baseline).not.toContainText("Pass");
      await expect(page.getByText(/Not enough speech to assess/)).toBeVisible();
      await expect(page.getByTestId("history-detail-digest-score")).not.toContainText("3.5");
      await expect(page.getByTestId("review-metric-score-overall")).not.toContainText("3.5");
      await expect(page.getByText("You spoke in the required language.", { exact: true })).toHaveCount(0);
      await expect(page.getByText("Verified language", { exact: true })).toHaveCount(0);
    }
  });
}
