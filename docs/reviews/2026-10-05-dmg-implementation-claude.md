I found two likely blockers and several smaller bugs. Some depend on files that weren't included, and I've marked those.

## Implementation bugs

**1. The release desktop binary probably loads the dev server, not the bundled frontend.** `scripts/build_macos.py` (`cargo build --release --locked …`)
In Tauri v2, the `tauri` crate turns on its `dev` cfg unless the `custom-protocol` feature is enabled, and `tauri build` normally enables that feature for you. Plain `cargo build --release` doesn't. Unless `Cargo.toml` turns on `tauri/custom-protocol` itself, `generate_context!` won't embed the frontend and both `loading.html` and the main window will load `devUrl`. Nothing catches this, because `test_macos_delivery.py` never launches `vostavo-desktop`. **Fix:** pass `--features tauri/custom-protocol`, or build with `tauri build --no-bundle`.

**2. The audio URL token is the full session token, not a media-only capability.** `app_backend/desktop_session.py` (`session_guard` query fallback), `frontend/src/lib/runtime/environment.ts` (`buildAudioUrl`)
The server only limits *where* `?session=` is accepted. The value is the same 256-bit token that the `X-Vostavo-Session` header accepts on every endpoint. So any captured audio URL (devtools, copied link, crash log, a future access log) works as a full API credential. That breaks your stated rule that query tokens are only for read-only history audio. **Fix:** issue a separate token for media, for example `HMAC(token, "media")` or a per-session HMAC, and accept only that in the query string.

**3. `validate_audio_duration` only works from a process's main thread.** `app_backend/uploads.py`
- `signal.signal()` raises `ValueError` from any other thread. The `finally` block calls it too and raises again. Callers that map `ValueError` to "could not decode" would then show a false decode error to the user.
- If it's called directly on the event loop thread instead, `decoder.wait(timeout=90)` blocks the whole server, health checks included, for up to 90 s.
- It's only correct inside a worker process's main thread. Please check the caller.

**4. The ready file is written non-atomically, and the test harness doesn't handle a partial read.**
- `scripts/run_backend.py` writes it with `args.ready_file.write_text(...)`.
- In `scripts/test_macos_delivery.py`, `launch()` calls `json.loads(ready.read_text())` without catching `ValueError`, so a partial read crashes the test intermittently. The Rust side (`desktop.rs`) already retries on parse errors.
- **Fix:** write to a temp file and `os.replace` it into place.

**5. Part of the token is visible to other local users.** `frontend/src-tauri/src/desktop.rs`
`launch_dir` contains `&token[..8]` and is passed in `--ready-file` on the command line, where `ps` shows it to every local user. That drops the token from 256 to 224 secret bits. It's low severity and easy to fix: use separate random bytes for the directory name.

**6. The frontend isn't actually built from locked inputs.** `scripts/build_macos.py`
It runs `npm … run build` without `npm ci`, so release builds use whatever is already in `node_modules`. The existing venv is also reused regardless of `--python`; the version check only runs after PyInstaller has finished.

**7. The native-code audit has gaps.** `scripts/build_macos.py` (`audit`, `macho_files`)
- It never checks `LC_RPATH`, so an `@rpath/...` dependency can still resolve to `/opt/homebrew`.
- The fat64 magic number (`cafebabf`) is missing.
- Old `LC_VERSION_MIN_MACOSX` binaries print `version`, not `minos`, so they're skipped when computing the minimum macOS version.

**8. Repo mode has no Host check.** `install_desktop_session`
With no token, the function returns early and the Host check is skipped. A repo backend started without `--desktop-owned` is then exposed to DNS rebinding from any website. This only affects development.

**9. `buildAudioUrl` breaks paths that already have a query string.** It always appends `?session=`, so such a path ends up with two `?`. Minor.

**10. The bridge script may run on pages outside the app (depends on config not shown).** `main.rs` injects `initialization_script(bridge)` into the main window whatever page it's on. If that window can navigate to a remote page, the token is injected there too. Check that navigation is restricted (`on_navigation`) or that CSP prevents it.

## Release certificate / acceptance gates (not bugs)

- **Release helper signing is only partly tested.** The library-validation exception is correctly limited to internal builds. But a release-signed helper (hardened runtime, library validation on, no entitlements) only gets the build-time `--self-test`, which doesn't run ASR or the spawned workers. Run `test_macos_delivery.py` against the actual release DMG.
- **Microphone entitlement.** `packaging/entitlements.plist` wasn't included. With the hardened runtime, the main app needs `com.apple.security.device.audio-input` for microphone access. This plus TCC is the interactive microphone gate.
- **The desktop app is never launched.** The harness never runs `vostavo-desktop`, so the webview, the bridge, CORS/preflight and owner shutdown from the Rust side are all untested (this would also catch #1).
- **Gatekeeper and notarization.** Acceptance with a quarantined download on a clean machine is still a manual gate. Also, if `notarytool` exits non-zero on an `Invalid` result, `output()` raises before you get a submission ID to fetch the log.


## Disposition

Accepted: enable the production custom-protocol feature (confirmed in installed Tauri macro source), separate independent media token, atomic readiness write, independent launch-directory nonce, locked npm ci with lifecycle scripts disabled and credentials scrubbed, early Python version check, RPATH/fat64/legacy deployment-target audit, query-preserving audio URLs, and navigation restriction to the app origin. Audio-input entitlement is present. Fixed Uvicorn signal replay preventing final readiness cleanup; actual owner EOF/SIGTERM tests pass.

Upload validation is called inside the spawned assessment worker's main thread (`app_backend/jobs.py`), so the signal handler is appropriate and does not block the API event loop. Repository mode remains the existing development contract; packaged and desktop-owned mode require host/session authorization. Release-signed ASR, notarization, quarantined launch and physical microphone remain external acceptance gates. Native app launch is being tested separately; helper-only tests are explicitly labeled.
