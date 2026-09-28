# 2026-06-05 Learner UX Screenshot Inventory

Captured from the local React/Vite frontend at `http://127.0.0.1:4173/`
against an isolated localhost backend at `http://127.0.0.1:8800/`.

Backend state used for configured-flow screenshots:

- app data: `/tmp/vostavo-codex-screenshot-app-data`
- cache: `/tmp/vostavo-codex-screenshot-cache`
- saved connection: local Ollama, model `llama3.2:3b`, no API key

Additional visual-refresh verification used:

- app data: `/tmp/vostavo-codex-visual-refresh-app-data`
- cache: `/tmp/vostavo-codex-visual-refresh-cache`
- saved connection: local Ollama, model `llama3.2:3b`, no API key
- 2026-06-06 focused smoke: from `frontend/`,
  `env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-05 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts`
  passed.
- 2026-06-09 focused smoke: reran the same command after the visual-smoke
  hardening and Speak confidence slice. It passed and refreshed the focused
  screenshots.
- focused smoke coverage: Home, Session Setup, Speak, clean Review, failed-gate
  Review, and History visual artifacts; visible non-serif typography; Review
  coach-takeaway assertions; Review failed-gate structural assertions; History
  progress-story assertions; Review score-details/practice-signals headings;
  visible Review quality-check summary; collapsed and failed-open Review
  validation states; collapsed Review transcript/reference disclosure;
  collapsed History detail digest assertions; expanded full-report disclosure
  assertions; desktop table assertions; mobile saved-attempt card assertions;
  desktop/mobile screenshots; and no mobile page-level horizontal overflow at
  390px for every focused mobile route state.
- follow-up assessment: `visual-polish-follow-up-assessment.md` compares the
  current route screenshots and identifies History, then Review, as the next
  taste/product-design refinement targets.

Runtime setup onboarding verification used:

- app data: `/tmp/vostavo-codex-runtime-setup-onboarding-app-data`
- cache: `/tmp/vostavo-codex-runtime-setup-onboarding-cache`
- route: `/runtime-setup`
- browser evidence: in-app Browser reached the route; system Chrome/Playwright captured screenshots and verified no Times font, visible `1 of 4 checks ready` progress, desktop split layout, and no mobile horizontal overflow at 390px.

Screenshots:

| File | Surface |
|---|---|
| `01-home-configured-desktop.png` | Home with saved runtime connection and readiness summary |
| `02-runtime-setup-guide-desktop.png` | Runtime Setup / Setup Guide |
| `03-session-setup-goal-desktop.png` | Session Setup goal-oriented form |
| `04-speak-ready-desktop.png` | Speak route after applying a setup draft |
| `05-review-empty-desktop.png` | Review empty state with start-session CTA |
| `06-history-empty-desktop.png` | History empty state with start-session CTA |
| `07-library-practice-support-desktop.png` | Library reframed as practice support |
| `08-guide-feedback-support-desktop.png` | Scoring Guide reframed as feedback support |
| `09-settings-language-runtime-desktop.png` | Settings with UI language and saved runtime connection |
| `10-home-configured-mobile.png` | Home at 390px mobile viewport |
| `home-visual-refresh-desktop.png` | Home after the visual refresh, configured desktop |
| `home-visual-refresh-mobile.png` | Home after the visual refresh, configured mobile |
| `runtime-setup-onboarding-desktop.png` | Runtime Setup after onboarding readiness visual refresh, desktop |
| `runtime-setup-onboarding-mobile.png` | Runtime Setup after onboarding readiness visual refresh, mobile |
| `visual-refresh-smoke-home-desktop.png` | Focused visual smoke Home, desktop |
| `visual-refresh-smoke-home-mobile.png` | Focused visual smoke Home, 390px mobile |
| `visual-refresh-smoke-session-setup-mobile.png` | Focused visual smoke Session Setup, 390px mobile |
| `visual-refresh-smoke-speak-desktop.png` | Focused visual smoke Speak with recording visualizer, desktop |
| `visual-refresh-smoke-speak-mobile.png` | Focused visual smoke Speak with recording visualizer, 390px mobile |
| `visual-refresh-smoke-review-desktop.png` | Focused visual smoke Review with coach takeaway and score ring, desktop |
| `visual-refresh-smoke-review-mobile.png` | Focused visual smoke Review with coach takeaway and score ring, 390px mobile |
| `visual-refresh-smoke-review-failed-desktop.png` | Focused visual smoke Review with failed quality gates, desktop |
| `visual-refresh-smoke-review-failed-mobile.png` | Focused visual smoke Review with failed quality gates, 390px mobile |
| `visual-refresh-smoke-history-desktop.png` | Focused visual smoke History with progress story and trend sparklines, desktop |
| `visual-refresh-smoke-history-expanded-desktop.png` | Focused visual smoke History with the full saved report disclosure opened, desktop |
| `visual-refresh-smoke-history-mobile.png` | Focused visual smoke History with progress story, trend sparklines, and saved-attempt cards, 390px mobile |
| `visual-refresh-smoke-history-expanded-mobile.png` | Focused visual smoke History with the full saved report disclosure opened, 390px mobile |
| `visual-polish-follow-up-assessment.md` | Route-by-route assessment after the visual-refresh foundation |
