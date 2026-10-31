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
- `THIRD_PARTY_NOTICES.txt`, also embedded in `Vostavo.app/Contents/Resources`
  and available from Settings → About → Third-party licenses

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

First install a valid **Developer ID Application** identity in Keychain. In the
Apple Developer portal, select the signing team and register an explicit App ID
for `com.vostavo.desktop` with **Keychain Sharing**. Create a **Developer ID**
distribution provisioning profile for that App ID and the selected Developer ID
Application certificate. Save the downloaded `.provisionprofile` outside the
repository and set `VOSTAVO_PROVISIONING_PROFILE` to its absolute path. The
build checks the profile's expiration, team, App ID, Keychain group and exact
signing certificate, embeds it at `Vostavo.app/Contents/embedded.provisionprofile`,
and verifies the signed entitlements. The expected entitlements are
`com.apple.application-identifier=TEAMID.com.vostavo.desktop`,
`com.apple.developer.team-identifier=TEAMID`, and
`keychain-access-groups=[TEAMID.com.vostavo.desktop]`. Use a profile obtained
directly from the Apple Developer portal; the build decodes its CMS container
and validates claims but does not independently validate the CMS certificate
chain.

Store notarization credentials in a named Keychain profile. Then:

```sh
export VOSTAVO_PROVISIONING_PROFILE='/absolute/path/outside/repo/Vostavo-Developer-ID.provisionprofile'
.venv/bin/python scripts/build_macos.py --mode release \
  --identity 'Developer ID Application: YOUR TEAM (TEAMID)' \
  --notary-profile YOUR_PROFILE
```

Release mode fails before building when these prerequisites are absent. Packaged
release drafts require the data-protection, `ThisDeviceOnly`, non-synchronizable
Keychain item; a missing entitlement fails closed and the UI reports drafts
unavailable. Development and internal builds retain a logged login-Keychain
fallback.

For a local Developer ID signing check without notarization or Apple upload,
run the same command with `--local-sign-only` instead of `--notary-profile`.
The profile is still needed for draft storage. Without a profile, explicitly
add `--allow-drafts-disabled`; the build report records
`"draft_storage": "unavailable"`, and the signed DMG is only a signing/runtime
diagnostic. The DMG name includes `signed-unnotarized`.

The notarized release path signs
nested native code inside-out, notarizes/staples the app, creates/signs/notarizes/
staples the DMG and verifies codesign, stapler and spctl. Run the artifact suite
against that actual release DMG too. The release path requires testing with a real
certificate; internal acceptance cannot establish release library-validation
compatibility.

For a replacement LGPL FFmpeg build, also pass `--ffmpeg-source-version` and
`--ffmpeg-source-archive` with the exact corresponding-source release and a
durable archive path or URL. The generator records those fields alongside PyAV's
reported version, license and configure flags. A release build fails if either
source field is missing for an LGPL build. Verify the archive is available to
recipients and matches the FFmpeg libraries in the frozen helper. The generated
notice includes the full LGPL 2.1 text, or LGPL 3 plus its incorporated GPL 3
text, according to PyAV's reported library license. This source reference and
license text do not replace the owner's review of LGPL relinking obligations.

## Release checklist

Before distribution, the owner must review the license audit attached to the
AO release-readiness session and the generated notices. The currently pinned PyAV arm64 wheel reports
GPLv3-or-later FFmpeg libraries; resolve that distribution blocker before
publishing a DMG. The repository's own source license is also an owner decision.

1. Confirm the Developer ID Application identity, explicit
   `com.vostavo.desktop` App ID and Developer ID provisioning profile. Verify
   its team, certificate, expiry and device-only Keychain entitlement. Keep the
   profile and notary Keychain credential outside the repository.
2. Run the local `tests/test_build_macos.py` dry-run. It mocks `notarytool` and
   `stapler` and checks that the app archive is submitted and stapled before
   DMG creation, then the signed DMG is submitted and stapled. It contacts no
   Apple service.
3. Once owner credentials are available, run the release build command above.
   It signs the app, submits `Vostavo.zip`, staples and validates `Vostavo.app`,
   then creates/signs/submits/staples and validates the DMG. Record both Apple
   acceptance IDs from `build-report.json`.
4. Verify `codesign --verify --deep --strict` on the app; run `xcrun stapler
   validate` on both app and DMG; run `spctl -a -t exec -vvv` on the app and
   `spctl -a -t open --context context:primary-signature -vvv` on the DMG.
   Verify `Info.plist` version `0.1.0`, identifier `com.vostavo.desktop`,
   copyright and minimum macOS `14.0` against the build report.
5. Verify the adjacent SHA-256 checksum with `(cd .build-macos && shasum -a
   256 -c Vostavo-0.1.0-arm64.dmg.sha256)`. Archive the DMG, checksum,
   build report, notices and the human acceptance record together.
6. On a clean Mac or user, download the DMG with quarantine preserved. Launch
   from Applications, save and reopen a draft in the **real UI**, and confirm
   the microphone permission prompt appears on first recording. After stapling,
   disconnect the network and confirm first launch succeeds offline. Check
   playback, a completed review, history persistence and no orphan helper on
   quit. Record OS version and outcomes.

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
