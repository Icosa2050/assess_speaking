# OpenRouter Vision Second Pass

Date: 2026-05-24

Inputs:

- `01-home-default.jpg`
- `02-runtime-setup-default.jpg`
- `03-session-setup-default.jpg`
- `04-speak-ready.jpg`
- `05-review-empty.jpg`
- `06-history.jpg`
- `07-library.jpg`
- `08-guide.jpg`
- `09-settings.jpg`
- `10-home-configured.jpg`

Models:

- `anthropic/claude-opus-4.7`
- `google/gemini-3.1-pro-preview`
- `x-ai/grok-4.3`

Raw output:

- `openrouter-vision-second-pass.json`

## Consensus

All three models inspected the screenshot set successfully through direct OpenRouter image input.

The strongest shared finding is that Vostavo exposes infrastructure before practice. The first-run Home screen foregrounds runtime setup and startup checks, while the actual learner goal, speaking practice, is visually secondary.

All three models agreed that the UI-language control should move out of the global banner and into Settings. The current banner can be confused with the learning-language selector and consumes high-value header space on every screen.

All three models agreed Runtime Setup should stop competing with Speak and Review as a primary navigation item. The route should remain reachable for broken or first-run runtime states, but daily practice navigation should emphasize the learner loop.

## Model-Specific Notes

### Opus

Opus recommended demoting, not fully hiding, Runtime Setup. It should remain available through Settings, Home warning states, and deep links. Opus also recommended deferring Scoring Guide changes because it is a reference page rather than a primary flow blocker.

### Gemini

Gemini emphasized technical intimidation: file paths, ffmpeg, Hugging Face cache paths, provider names, and model IDs make the app feel like a developer diagnostic tool rather than a learning environment. It recommended summarizing system status and moving technical detail behind progressive disclosure.

### Grok

Grok emphasized conditional navigation: Runtime Setup should disappear from the primary sidebar after a successful connection, but remain available as a reconfigure path. It also called out Review and History empty states as the weakest post-navigation screens.

## Plan Changes Applied

The implementation plan was amended to:

- Add Home simplification as an early slice after moving UI language to Settings.
- Make startup checks collapsed or summarized by default.
- Demote Runtime Setup rather than treating it as a primary practice route.
- Defer Scoring Guide content changes.
- Add non-goals for runtime/provider contract changes and broad visual-token redesign.
