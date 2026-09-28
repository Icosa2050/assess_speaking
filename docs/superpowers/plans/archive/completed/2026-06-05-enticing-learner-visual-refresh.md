# Enticing Learner Visual Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Vostavo feel visually enticing for adult language learners by fixing the global typography drift, adding speaking-specific visual primitives, and polishing the primary learner loop without turning the product into a gamified toy.

**Architecture:** Keep the current React Router route structure, CSS Modules, localization JSON, semantic IDs, and zero-casual-dependency policy. Build a small token and visual primitive layer first, then apply it route-by-route so each implementation slice stays file-bounded and testable.

**Tech Stack:** React 19, React Router 7, Zustand, React Query, Vite, CSS Modules, inline SVG, Vitest, Playwright or Browser probes.

**Completion status, 2026-06-06:** Completed and archived. The implementation
landed the global typography/token foundation, zero-dependency visual
primitives, Home practice-studio polish, Speak recording visual, Review score
and next-step visuals, History trend visuals, localized/no-new-locale-key
verification, and focused Playwright screenshot evidence. The remaining work is
product-design refinement, not this foundation plan.

---

## Current Evidence

- Browser-computed typography shows shell and most non-Home route text inheriting `Times`.
- `frontend/src/routes/HomeRoute.module.css` sets Inter locally, so Home looks unrelated to the shell and other routes.
- `DESIGN.md` and the Stitch design systems specify Inter for all UI text.
- `frontend/src` contains no image assets and no icon/chart dependency. The only meaningful custom graphic found in the route layer is the inline History sparkline.
- The existing UI is calm, but the visual vocabulary is mostly white cards, pale borders, text, and forms. That is not enough for the user's "NEED enticing" bar.

## External Inspiration

- Khan Academy's 2026 design-system write-up is the strongest foundation reference: their visually distinct learner UI required refreshed code components, Figma libraries, and foundational color, typography, sizing, and spacing settings. They also describe semantic color roles such as instructive actions and accessibility stress testing.
- Babbel's rebrand is the strongest adult-language-learning reference: its visual identity uses a flexible design system with product cards, photography, and cut-out imagery/silhouettes. Vostavo should borrow the product-card confidence, not the marketing photo pipeline.
- Duolingo's app-method paper is useful for motivation mechanics: progress paths, immediate feedback, and positive reinforcement are compelling. Vostavo should borrow progress visibility and celebration timing, not mascot-driven loops, XP economies, streak pressure, or childlike characters.
- Memrise is useful because its visuals attach directly to learning value: visual memory tricks, level-matched videos, and AI speaking practice make the learner's target language feel tangible. Vostavo should borrow the "visual artifact tied to practice" idea.
- Brilliant is useful for quantified learning: visual learning and clear progress tracking make practice feel concrete. Vostavo should make speaking progress visible without adding a heavy charting system.

## PAL And Stitch Notes

- Stitch confirms the current design-system intent: calm, motivating, adult, Inter-based, 8px cards, no broad decorative gradients or nested cards.
- PAL agreed with the three-part direction but tightened the ordering: token/base-style foundation first, then visual primitives, then screen polish. The font bug is a symptom of a missing global token layer.
- PAL's strongest warning: a token fix plus icons can still feel like a clean dashboard. Each primary learner screen needs one memorable speaking or progress artifact.
- PAL's second warning: color-blocked cards can fail contrast if existing muted text colors are reused. Contrast checks are part of acceptance, not a follow-up.

## Visual Concept: Focused Practice Studio

The app should feel like an inviting practice studio: professional, human, and focused, with visible evidence that speaking practice is happening. The screen should not look like a settings dashboard, but it should also avoid mascots, broad gradients, stock photography, confetti, streak flames, and heavyweight gamification.

Core visual moves:

- One global Inter/system-sans typography stack and a reliable type scale.
- Stronger color blocking for the primary learner action, especially Home.
- Zero-dependency iconography for practice, recording, readiness, progress, guide, and settings states.
- Speaking-specific artifacts: waveform or level meter in Speak, score ring in Review, trend sparkline or segmented progress in History, and a compact weekly practice/progress module on Home.
- Small CSS motion only for interaction feedback and state transitions. No animation library.

## Non-Goals

- Do not add npm dependencies unless the npm dependency policy is followed in a separate approved task.
- Do not introduce mascots, streak flames, badge collections, broad gradients, decorative orbs, stock photography, or marketing-page hero composition.
- Do not change runtime-provider contracts, assessment scoring, recorder behavior, backend APIs, route guards, localization architecture, or saved-connection behavior.
- Do not hard-code user-visible strings or icon aria labels.
- Do not remove or rename semantic IDs without a focused automation update.

## File-Bounded Tasks

### Task 1: Global Typography And Token Foundation

**Files:**
- Create: `frontend/src/styles/base.css`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/components/shell/AppShell.module.css`
- Modify: `frontend/src/routes/HomeRoute.module.css`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`

- [x] Add `base.css` with `:root` design tokens for font stack, type scale, color roles, radii, spacing, and motion timings.
- [x] Import `base.css` once from `main.tsx`.
- [x] Set `html`, `body`, `button`, `input`, `select`, and `textarea` to the shared font stack.
- [x] Remove the Home-only `font-family` declaration so Home no longer hides global drift.
- [x] Convert AppShell's highest-impact hardcoded colors to token variables.
- [x] Add or extend a route-render test that confirms the shell and Home both render under the shared app frame.
- [x] Verify with a browser probe that `/`, `/runtime-setup`, `/session-setup`, `/speak`, `/review`, `/history`, `/library`, `/guide`, and `/settings` do not resolve visible UI text to `Times`.

Acceptance:

- No visible route inherits browser-default serif typography.
- `DESIGN.md` and Stitch's Inter intent are implemented globally.
- `npm --prefix frontend test` and `npm --prefix frontend run typecheck` pass.

### Task 2: Zero-Dependency Visual Primitive Layer

**Files:**
- Create: `frontend/src/components/ui/Icon.tsx`
- Create: `frontend/src/components/ui/ProgressRing.tsx`
- Create: `frontend/src/components/ui/Sparkline.tsx`
- Create: `frontend/src/components/ui/visualPrimitives.module.css`
- Create: `frontend/src/components/ui/visualPrimitives.test.tsx`

- [x] Create a typed icon registry with only the first required icons: microphone, play, check, arrow-right, warning, target, history, guide, settings, language, headphones, and sparkle.
- [x] Use inline SVG with `currentColor`, stable `viewBox`, 24px default size, and `aria-hidden` by default.
- [x] Add `ProgressRing` for score/readiness/progress summaries with text fallback content.
- [x] Extract the existing History sparkline behavior into `Sparkline` without adding a chart library.
- [x] Add CSS classes for icon buttons, visual badges, progress rings, and chart frames using tokens from Task 1.
- [x] Test accessible naming for standalone icons and numeric output for progress visuals.

Acceptance:

- `frontend/package.json` is unchanged.
- Visual primitives are reusable by Home, Speak, Review, and History.
- Progress visuals do not communicate state by color alone.

### Task 3: Home Practice Studio Polish

**Files:**
- Modify: `frontend/src/routes/HomeRoute.tsx`
- Modify: `frontend/src/routes/HomeRoute.module.css`
- Modify: `frontend/src/routes/tests/HomeSetupRoutes.test.tsx`
- Modify: `frontend/src/components/setup/PracticeBriefCard.tsx`
- Modify: `frontend/src/components/setup/ConnectionStatusPanel.tsx`

- [x] Make the primary practice card visually dominant with token-backed color blocking and clear knockout text.
- [x] Add icon+text actions for the primary learner path and runtime recovery path.
- [x] Add one compact learner-progress artifact using `Sparkline` or `ProgressRing`; if no real attempts exist, show a localized empty-state progress placeholder that routes to setup.
- [x] Keep runtime diagnostics compact and secondary when healthy.
- [x] Preserve route guards, existing semantic IDs, and setup-complete navigation behavior.
- [x] Update tests for the primary CTA, compact runtime status, and progress artifact presence.

Acceptance:

- Home no longer reads like a settings dashboard.
- The primary learner action is the first visual anchor on desktop and mobile.
- The color-blocked card passes contrast for normal text and buttons.

### Task 4: Speak Screen Signature Recording Visual

**Files:**
- Modify: `frontend/src/routes/SpeakRoute.tsx`
- Modify: `frontend/src/components/speak/RecorderPanel.tsx`
- Modify: `frontend/src/components/speak/AssessmentStatusPanel.tsx`
- Modify: `frontend/src/routes/tests/SpeakRoute.test.tsx`

- [x] Add a stable speaking visual around the recorder: waveform, level bars, or recording-state ring using CSS and the visual primitives.
- [x] Keep the prompt and recording action dominant; assessment status remains supportive.
- [x] Add icons for record/upload/retry states without changing recorder behavior.
- [x] Ensure recording and upload states do not cause layout shift.
- [x] Preserve semantic labels for automation and add localized accessible labels for new non-decorative visuals.
- [x] Update tests for recording-state UI and semantic continuity.

Acceptance:

- Speak has one memorable speaking artifact.
- The visual state changes with recording/upload/assessment state but does not alter API behavior.
- Keyboard and screen-reader paths remain intact.

### Task 5: Review And History Progress Visuals

**Files:**
- Modify: `frontend/src/routes/ReviewRoute.tsx`
- Modify: `frontend/src/routes/HistoryRoute.tsx`
- Modify: `frontend/src/components/history/HistoryList.tsx`
- Modify: `frontend/src/routes/tests/ReviewRoute.test.tsx`
- Modify: `frontend/src/routes/tests/HistoryRoute.test.tsx`

- [x] Replace the Review wall-of-text feel with a score ring, next-focus chips, and a retry/new-session action hierarchy.
- [x] Replace the route-local History sparkline implementation with the shared `Sparkline`.
- [x] Add compact trend summaries that pair numbers with text labels.
- [x] Keep empty states action-oriented and non-dead-end.
- [x] Preserve existing test IDs and semantic IDs unless a focused automation update justifies a rename.
- [x] Update tests for visual progress artifacts and accessible summaries.

Acceptance:

- Review gives learners a visually clear sense of result and next focus.
- History gives learners a visually clear sense of progress over time.
- Empty Review/History states remain useful before the first attempt.

### Task 6: Localized Visual Microcopy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add only the strings required for new visual artifacts, accessible labels, and concise progress summaries.
- [x] Keep wording adult and coach-like: concrete, calm, and action-oriented.
- [x] Avoid terms that imply streak pressure, failure, shame, or game currency.
- [x] Run existing frontend route tests that cover each changed string key.
- [x] Verified on 2026-06-06 that the visual slices reused the existing settled locale keys: `scripts/repo_quality_audit.py --coverage-mode skip`, an explicit five-locale JSON parity/placeholder check, and `/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py -q` passed with no locale patch needed.

Acceptance:

- No new hardcoded user-visible strings.
- All five locale files contain the same new keys.

### Task 7: Visual Verification And Evidence

**Files:**
- Modify: `docs/ux-audit-screenshots/2026-06-05/README.md`
- Modify: `docs/PLAYWRIGHT_FLOW_EXPANSION_PLAN.md`
- Modify: `docs/PLAN_STATUS.md`
- Create: `frontend/tests/e2e/visualRefreshSmoke.spec.ts`

- [x] Add a focused visual smoke that checks no route uses visible serif typography, primary route landmarks render, and key visual artifacts exist.
- [x] Capture desktop and mobile screenshots for Home, Speak, Review, and History through Browser or Playwright when available.
- [x] Record exact command and date; no browser/screenshot blocker remained, so no error entry was needed.
- [x] Update plan status with what passed and what remains blocked.
- [x] 2026-06-06 focused visual smoke passed from `frontend/`:
  `env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts`.
  Evidence exists for Home, Speak, Review, and History desktop/mobile under
  `docs/ux-audit-screenshots/2026-06-05/`; the smoke also asserts no mobile
  page-level horizontal overflow at 390px.

Acceptance:

- `npm --prefix frontend test` passes.
- `npm --prefix frontend run typecheck` passes.
- Focused visual smoke passes or has a fresh recorded blocker.
- Screenshot evidence exists or the blocker is classified with exact command/date/error.

## Execution Order

1. Task 1: typography and tokens. This fixes the reported font problem and prevents restyling churn.
2. Task 2: visual primitives. This creates the shared vocabulary before screen work.
3. Task 3: Home. This is the first screen and the strongest visual-impact slice.
4. Task 4: Speak. This makes the core practice action feel alive.
5. Task 5: Review and History. This makes progress visible and rewarding.
6. Task 6: localization. Run when the screen slices settle on the final labels.
7. Task 7: visual verification. Run after each screen slice as practical, then close at the end.

## Verification Bundle

Use zsh from the repo root:

```zsh
npm --prefix frontend test
npm --prefix frontend run typecheck
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
git diff --check
```

When browser behavior or screenshots are in scope:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python scripts/run_backend.py --host 127.0.0.1 --port 8800
env NODE_ENV=development VITE_LOCAL_API_BASE_URL=http://127.0.0.1:8800 npm --prefix frontend run dev -- --host 127.0.0.1 --port 4173 --strictPort
curl -I http://127.0.0.1:4173/
curl -sS http://127.0.0.1:8800/v1/health
cd frontend
env NODE_ENV=development npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

## Design Quality Bar

- The app must look intentionally designed before the learner reads copy.
- Every primary learner route must have one domain-specific visual artifact.
- The primary action must be obvious in under two seconds on desktop and mobile.
- The visual system must use tokens rather than route-local color drift.
- The app must remain adult, calm, local-first, accessible, localized, and automation-safe.
