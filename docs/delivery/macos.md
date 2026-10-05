# macOS delivery

The build produces an Apple Silicon DMG with the frontend, Rust desktop shell,
CPython helper, media decoder and ASR runtime. Whisper model weights are downloaded
into the user's cache, not included in the DMG. Python, Node, Homebrew and system
FFmpeg are not required on the destination Mac.

## Build an internal artifact

On an Apple Silicon Mac with Xcode command-line tools, Node 24, Rust and uv:

```sh
UV_CACHE_DIR=/tmp/vostavo-uv-cache .venv/bin/python scripts/build_macos.py --mode internal
```

The controller uses pinned CPython 3.12.11, a separate hashed production lock,
`npm ci --ignore-scripts`, a locked Rust build and embedded production frontend.
No npm dependency changes or install lifecycle scripts are required. Build inputs
must be reviewed before installing dependencies; the build excludes .env and
learner state. Python native dependencies are checked for arm64, external library
paths and deployment targets. The signed frozen helper must pass its self-test.

Output is under `.build-macos/`:

- `Vostavo-0.1.0-arm64-internal-adhoc.dmg`
- adjacent SHA-256 checksum
- `build-report.json` with native linkage and build evidence

`--reuse-helper` is an internal iteration option, never a release option. Use the
full build after Python changes. Internal signing enables Hardened Runtime; the
ad-hoc helper needs a library-validation exception because it has no Team ID.
Developer ID release mode does not include that exception. See Apple's
[library validation documentation](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.disable-library-validation).

An internal artifact is not a public Gatekeeper-approved release. Do not tell
learners to disable Gatekeeper or remove quarantine to install it.

## Automated artifact acceptance

With the tiny Whisper model already cached:

```sh
.venv/bin/python scripts/test_macos_delivery.py \
  .build-macos/Vostavo-0.1.0-arm64-internal-adhoc.dmg \
  --report .build-macos/acceptance-report.json
VOSTAVO_TEST_LISTENERS=1 .venv/bin/python -m pytest tests/test_desktop_shutdown.py
```

The artifact test mounts read-only, copies the app, ejects, and runs the frozen
helper from `/` with repository, Homebrew and global-cache reads denied. It copies
tiny weights into an isolated cache, runs real English/Italian sample ASR and
spawned assessment workers, checks upload duration decoding, reports, ranged
recording playback, persistence across restart and ownership cleanup. No paid
provider calls are made: local deterministic feedback is tested with an
unavailable local LLM. This is a backend artifact test; it does not prove native
microphone permissions or the Rust webview UI. Existing Chromium journeys cover
upload, synthetic microphone retries and history/progress for both languages.

## Developer ID release

First install a valid **Developer ID Application** identity in Keychain and store
notarization credentials in a named Keychain profile. Then:

```sh
.venv/bin/python scripts/build_macos.py --mode release \
  --identity 'Developer ID Application: YOUR TEAM (TEAMID)' \
  --notary-profile YOUR_PROFILE
```

Release mode fails before building when these prerequisites are absent. It signs
nested native code inside-out, notarizes/staples the app, creates/signs/notarizes/
staples the DMG and verifies codesign, stapler and spctl. Run the artifact suite
against that actual release DMG too. The release path requires testing with a real
certificate; internal acceptance cannot establish release library-validation
compatibility.

## Interactive release acceptance

On a second Mac or clean macOS user:

1. Download the release DMG in a browser, preserving quarantine. Open it, copy
   Vostavo to Applications, eject, and launch without developer tools installed.
2. Confirm the app loads without a localhost frontend dev server. Configure a
   provider through Settings; verify sign-in/browser return if using ChatGPT.
3. Download the recommended ASR model with an empty cache. Reopen offline and
   verify its cached availability.
4. Grant microphone access when prompted. Record/save an English attempt and an
   Italian attempt; replay each, retry and verify History progress.
5. Deny microphone access in a fresh test user and verify recovery guidance;
   do not reset the main user's privacy database.
6. Quit during processing, reopen, confirm no orphan backend/worker remains and
   saved history is intact. Launch a second copy and verify single-instance
   handling. Check the startup log under the selected app-data root if it fails.

State and caches remain outside the application bundle. Per-launch independent
API and media tokens pass through held stdin; readiness files contain port/PID,
not credentials. Media URL credentials authorize only read-only history audio.
Public signing, actual quarantine/App Translocation, physical microphone/TCC and
cold-cache download must be recorded independently of source test results.
