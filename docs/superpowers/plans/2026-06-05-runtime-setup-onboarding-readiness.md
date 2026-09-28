# Runtime Setup Onboarding Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Runtime Setup readiness checklist into an enticing, learner-facing onboarding module while preserving the existing setup data model, actions, localization, and semantic IDs.

**Architecture:** Keep `SetupReadinessPanel` as a pure presentation component fed by the existing readiness rows. Derive readiness progress from the verifiable speech-recognition and AI-tutor rows, keep microphone setup and practice handoff visible outside that denominator, render a split visual anchor plus sequential checklist, and keep the technical runtime forms below the panel in `SetupRoute`.

**Tech Stack:** React 19, TypeScript, CSS Modules, existing `Icon` and `ProgressRing` primitives, Vitest, Testing Library, Vite/browser smoke.

---

## Design Inputs

- The Locally AI comparison screens show a useful onboarding structure: one clear setup purpose, a large status visual, concise benefit copy, progressive setup state, and clear action hierarchy.
- Stitch generated and recommended a card-based split readiness layout using Vostavo's current system: Inter, light surfaces, teal primary, progress green, focus purple, and 8px radius.
- PAL reviewed the approach and recommended keeping the component presentation-only, deriving progress from verifiable core rows, preserving existing test IDs/semantic IDs, avoiding a duplicate synthetic CTA, and testing progress/accessibility/order stability.

## File-Bounded Tasks

### Task 1: Plan And Test Contract

**Files:**
- Create: `docs/superpowers/plans/2026-06-05-runtime-setup-onboarding-readiness.md`
- Modify: `frontend/src/components/setup/SetupReadinessPanel.test.tsx`

- [x] Add this focused implementation plan.
- [x] Add failing tests for the derived readiness anchor: progress label, semantic ID stability, input row order, no duplicate anchor button, and empty-row safety.
- [x] Run `npm --prefix frontend test -- src/components/setup/SetupReadinessPanel.test.tsx` and confirm the new assertions fail before implementation.

### Task 2: Split Readiness Module

**Files:**
- Modify: `frontend/src/components/setup/SetupReadinessPanel.tsx`
- Modify: `frontend/src/components/setup/SetupReadinessPanel.module.css`

- [x] Derive `readyCount`, `totalCount`, `progressValue`, `progressStatus`, and completion state from the verifiable speech-recognition and AI-tutor rows.
- [x] Render a left visual anchor with `Icon`, `ProgressRing`, localized benefit copy, and a visible readiness count.
- [x] Keep the existing checklist rows in their input order and preserve the outer `runtime_setup.setup_guide` test/semantic IDs.
- [x] Use CSS Grid for two columns on desktop and one column on mobile; keep 8px radii, stable dimensions, and token-backed colors.
- [x] Do not add a new primary CTA inside the anchor.

### Task 3: Localized Visual Microcopy

**Files:**
- Modify: `locales/en.json`
- Modify: `locales/de.json`
- Modify: `locales/es.json`
- Modify: `locales/fr.json`
- Modify: `locales/it.json`

- [x] Add matching keys for the readiness anchor headline, benefit copy, progress label, and progress status.
- [x] Keep copy concise, adult, and action-oriented.
- [x] Do not hard-code user-visible strings in React or CSS.

### Task 4: Verification And Evidence

**Files:**
- Modify: `docs/PLAN_STATUS.md`
- Modify: `docs/superpowers/plans/README.md`
- Modify: `docs/ux-audit-screenshots/2026-06-05/README.md`

- [x] Run focused setup tests, full frontend tests, frontend typecheck, backend i18n test, and `git diff --check`.
- [x] Run backend `/v1/health` and frontend Vite localhost smoke.
- [x] Capture or record browser visual evidence for `/runtime-setup`.
- [x] Update status docs with the completed slice and any fresh browser/screenshot blocker.

## Verification Commands

Use zsh from the repo root:

```zsh
npm --prefix frontend test -- src/components/setup/SetupReadinessPanel.test.tsx
npm --prefix frontend test
npm --prefix frontend run typecheck
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python -m pytest tests/test_app_core_i18n.py
git diff --check
```

Runtime smoke:

```zsh
/Users/bernhard/Development/assess_speaking-codex-v6/.venv/bin/python scripts/run_backend.py --host 127.0.0.1 --port 8800
env NODE_ENV=development VITE_LOCAL_API_BASE_URL=http://127.0.0.1:8800 npm --prefix frontend run dev -- --host 127.0.0.1 --port 4173 --strictPort
curl -sS http://127.0.0.1:8800/v1/health
curl -I http://127.0.0.1:4173/
```

## Acceptance

- The setup guide reads visually as onboarding, not a plain technical checklist.
- Progress is visible, localized, and screen-reader accessible.
- The progress denominator contains only checks the app can verify; microphone setup remains visible without being presented as complete.
- Existing row actions, disabled states, semantic IDs, and order are preserved.
- Technical runtime forms remain available below the learner-facing readiness module.
