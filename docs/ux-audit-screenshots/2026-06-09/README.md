# 2026-06-09 UX Screenshot Inventory

Focused visual-smoke command from `frontend/`:

```zsh
env NODE_ENV=development VISUAL_REFRESH_SCREENSHOT_DIR=/Users/bernhard/Development/assess_speaking-codex-v6/docs/ux-audit-screenshots/2026-06-09 npx playwright test -c playwright.config.ts tests/e2e/visualRefreshSmoke.spec.ts
```

Result on 2026-06-09: passed locally.

The focused gate covers:

- Home desktop and 390px mobile.
- Session Setup 390px mobile.
- Speak desktop, 390px mobile, and attached-audio 390px mobile.
- Review clean desktop and 390px mobile.
- Review failed-gate desktop and 390px mobile.
- History desktop, 390px mobile, expanded desktop, expanded 390px mobile, and 360px narrow mobile.

Screenshot files:

- `visual-refresh-smoke-home-desktop.png`
- `visual-refresh-smoke-home-mobile.png`
- `visual-refresh-smoke-session-setup-mobile.png`
- `visual-refresh-smoke-speak-desktop.png`
- `visual-refresh-smoke-speak-mobile.png`
- `visual-refresh-smoke-speak-ready-mobile.png`
- `visual-refresh-smoke-review-desktop.png`
- `visual-refresh-smoke-review-mobile.png`
- `visual-refresh-smoke-review-failed-desktop.png`
- `visual-refresh-smoke-review-failed-mobile.png`
- `visual-refresh-smoke-history-desktop.png`
- `visual-refresh-smoke-history-mobile.png`
- `visual-refresh-smoke-history-narrow-mobile.png`
- `visual-refresh-smoke-history-expanded-desktop.png`
- `visual-refresh-smoke-history-expanded-mobile.png`

Learner-confidence simplification evidence:

- History no longer renders duplicate `history-priority-latest`, `history-priority-new`, or `history-priority-resolved` cards below the progress story.
- `visual-refresh-smoke-history-narrow-mobile.png` proves the top History story at 360px with no horizontal page overflow.
- Speak keeps label and notes available through `speak.optional_context` after audio is attached, reducing the submit moment's visual weight.
- `visual-refresh-smoke-speak-ready-mobile.png` proves the attached-audio state has no horizontal overflow after the native file input is visually hidden.
