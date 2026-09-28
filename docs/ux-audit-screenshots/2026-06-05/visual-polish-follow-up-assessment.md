# Visual Polish Follow-Up Assessment

Date: 2026-06-06

## Verdict

The visual-refresh foundation plan is fulfilled. Vostavo now has a coherent
adult learner-facing baseline: shared typography, stable tokens, iconography,
score/progress primitives, a visually stronger Home, a speaking visual on Speak,
score and next-step hierarchy on Review, and trend visuals on History.

The app is not yet a fully polished or emotionally rich final product
experience. The next stage should not be another foundation pass. It should be
taste and product-design refinement: compare the shipped screenshots
route-by-route, identify the surfaces that still feel sterile or report-like,
and polish only the weakest one or two surfaces in file-bounded slices.

## Evidence Reviewed

- `visual-refresh-smoke-home-desktop.png`
- `visual-refresh-smoke-home-mobile.png`
- `visual-refresh-smoke-speak-desktop.png`
- `visual-refresh-smoke-speak-mobile.png`
- `visual-refresh-smoke-review-desktop.png`
- `visual-refresh-smoke-review-mobile.png`
- `visual-refresh-smoke-history-desktop.png`
- `visual-refresh-smoke-history-mobile.png`
- `runtime-setup-onboarding-desktop.png`
- `runtime-setup-onboarding-mobile.png`
- Stitch project `12105866389178941937`, which confirms the design-system
  direction: Inter, teal/green/purple accents, 8px radius, calm surfaces, and no
  broad decorative gradients.

## Route-By-Route Assessment

### Home

Status: strong enough for the current product baseline.

Home now has the clearest product signal. The teal practice card, readiness ring,
primary CTA, and compact device-readiness card make the route feel intentional
before the learner reads much copy. It has moved furthest away from the original
developer-dashboard feel.

Residual issue: it still feels more like a polished control surface than a
personal learning space. That is acceptable for now because the primary action is
obvious and the route does not block the learner with technical detail.

### Speak

Status: improved, but still restrained.

Speak has a real signature artifact now: the waveform/level visual gives the
recording area shape and makes the route more recognizably about speaking. The
prompt remains prominent, and the record/upload controls are clear.

Residual issue: the screen still feels static. The waveform is useful as a
visual anchor, but it does not yet create much emotional momentum. This is a
second-tier polish candidate after Review/History, especially if recording-state
motion and a more coach-like prompt treatment become important.

### Review

Status: functionally improved, visually still report-heavy.

The top panel is the best part of Review: score ring, focus chip, next exercise,
and retry/change/open-history actions communicate the result and next step
quickly. That satisfies the foundation plan.

Residual issue: below the top panel, Review reverts to a long assessment report.
On mobile, the learner scrolls through many stacked cards and technical sections
before the route feels complete. The result is useful but not emotionally rich.
Review should become a coach-first result screen with evidence behind
progressive disclosure, not a report with a better header.

### History

Status: visually improved, still the weakest product surface.

History has trend sparklines, summary metrics, selectable attempts, and the
mobile overflow bug is fixed. It now communicates progress better than before.

Residual issue: it remains a dense audit/report page. The trend area is promising
but visually modest, and the opened attempt embeds almost the full Review report,
which makes mobile History extremely long. It does not yet tell a learner a
clear progress story such as "you improved this, keep practicing that."

### Runtime Setup

Status: strong for a technical setup surface.

The split readiness card, progress ring, action rows, and sectioned form make a
technical task feel guided. It still exposes technical setup details, but that is
appropriate for this route. It should not be the next emotional-polish target.

## Weakest Surfaces

1. History
2. Review

History is the weakest because it should be the learner's progress memory, but
it still reads like a data/report browser. Review is second because it starts
well, then falls back into assessment-output density.

Speak is third. It is visually better than before, and its main issue is
animation/energy rather than information architecture.

## Recommended Next Slices

### Slice 1: History Progress Story

Goal: turn History from a report browser into a progress story.

File boundary:

- `frontend/src/routes/HistoryRoute.tsx`
- `frontend/src/components/history/HistoryList.tsx`
- `frontend/src/routes/tests/HistoryRoute.test.tsx`
- `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
- `docs/ux-audit-screenshots/2026-06-05/README.md`

Design direction:

- Add a compact "progress story" band above attempts: latest score direction,
  pace direction, resolved focus, and next practice focus.
- Keep sparklines, but give them more learner-facing framing and less dashboard
  feel.
- Make opened attempt details secondary. Prefer a digest first and collapse raw
  report/evidence details on mobile.
- Preserve History's existing semantic IDs and table/detail access.

Acceptance:

- Desktop History still supports selecting attempts and reading detail.
- Mobile History's first viewport communicates progress without requiring a long
  scroll.
- Focused visual smoke still passes and mobile screenshot width remains 390px.

### Slice 2: Review Coach-First Result

Goal: make Review feel like a coaching moment rather than a formatted report.

File boundary:

- `frontend/src/routes/ReviewRoute.tsx`
- `frontend/src/components/review/ReviewSummary.tsx`
- `frontend/src/routes/tests/ReviewRoute.test.tsx`
- `frontend/tests/e2e/visualRefreshSmoke.spec.ts`
- `docs/ux-audit-screenshots/2026-06-05/README.md`

Design direction:

- Keep the score ring and next-step panel as the lead.
- Promote the coach summary, strength, top priority, and next exercise into one
  concise coaching card.
- Demote validation gates, evidence, transcript, and raw payload behind clearer
  secondary sections or disclosure on mobile.
- Preserve the current review data contract and semantic/test IDs.

Acceptance:

- The first desktop and mobile viewport answers: "How did I do?" and "What do I
  practice next?"
- Detailed evidence remains available but no longer dominates the route.
- Existing Review tests and focused visual smoke pass.

## Non-Goals For The Next Stage

- No new global token/theme pass.
- No new npm dependencies.
- No mascots, streaks, XP, badges, confetti, or marketing-style hero pages.
- No backend scoring, runtime setup, recorder behavior, route guard, or
  localization architecture changes.
- No broad redesign of navigation.

## Recommended Order

Do History first, then Review. History has the largest gap between product value
and current visual feel. Review already has a strong top panel, so it can follow
as a focused density/progressive-disclosure pass.
