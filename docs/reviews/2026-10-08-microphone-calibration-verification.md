# Required microphone calibration — October 8, 2026

The owner requested required Setup testing after hearing crackling. Live Speak and Rehearsal capture now require a five-second sample with detectable input, playback completion and explicit clear-sound confirmation. Uploads remain usable without microphone access. Silence, very quiet input, repeated near-full-scale peaks and user-reported distortion cannot pass. Changing the input or processing, revoked permission, device changes and capture failures invalidate calibration; it is not persisted across app reloads.

Setup and practice share native MediaRecorder MIME selection and getUserMedia constraints: tested input device, autoGainControl false, selectable echoCancellation/noiseSuppression. Browsers may ignore unsupported settings. Hardware gain is adjusted in system Sound/Input settings; software attenuation cannot restore already clipped input. No dependency changes were needed.

## Verification

- `npm --prefix frontend test`: 32 files, 227 tests passed.
- `npm --prefix frontend run typecheck`: passed, including browser specs.
- `npm --prefix frontend run build`: passed.
- `VOSTAVO_HOME=/tmp/vostavo-mic-unit-20261008 VOSTAVO_CACHE_HOME=/tmp/vostavo-mic-unit-cache-20261008 .venv/bin/python -m pytest -q tests/test_app_core_i18n.py tests/test_app_core_diagnostics.py`: 16 passed. The prescribed sibling environment is absent; this checkout's existing Python 3.12 virtual environment was used.
- `frontend/node_modules/.bin/playwright test --config /tmp/vostavo-microphone-calibration.playwright.config.mts`: nine Chromium cases passed with localhost Vite on 4229 and backend health startup probe on 8920. App/cache directories were isolated; provider credentials/proxies were cleared and Hugging Face offline mode enabled. Tests cover Setup/playback/Home status, Home/drawer live-capture gating, denial, rehearsal choices across Setup, near-clipping rejection, real-worker rejection of a synthetic 12-second upload and saved-review eligibility.
- Unit coverage includes cancellation/unmount/late permission, permission timeout, stalled audio engine/encoder, silence/quiet/clipping, manual distortion rejection, processing changes, device/permission invalidation and explicit calibration distinct from capture status.
- The Setup panel was inspected at desktop and 390-pixel width; the browser asserts no horizontal page overflow. Sample streams ended after testing. Blob URLs are revoked on leaving/retesting.
- `git diff --check`: passed.

## Browser fixture adjustments and limits

Initial October 8 probe command: `frontend/node_modules/.bin/playwright test --config /tmp/vostavo-microphone-calibration.playwright.config.mts --grep 'setup tests real browser audio'`.

A pure generated tone with noise/echo suppression was rejected as quiet. Processing was disabled for the synthetic fixture; production processing remains selectable. The next probe encoded the sample successfully but timed out waiting for playback completion on this headless macOS host. Native media playback was routed through Chromium's silent AudioContext sink in the **test only**, allowing its output clock to advance. The final tests use real encoding, decoding and playback events, with generated audio and a native Web Audio full-scale input for clipping. They do not simulate successful completion events or listen to learner recordings. A separate rerun encountered `OSError: [Errno 48] Address already in use` during prior server shutdown; fresh isolated ports resolved it. A Playwright launchOptions-in-describe fixture error was corrected before the final run.

These checks establish source/browser behavior, not audibility on physical speakers or the cause of the user's crackling. The DMG has not been rebuilt for this change. Physical input, installed WKWebView/TCC, playback and hardware input-volume acceptance remain pending. Ongoing live-practice signal/clipping metering is a separate planned follow-up. No private source or learner audio was transferred to Ubuntu or external review services for this work.
