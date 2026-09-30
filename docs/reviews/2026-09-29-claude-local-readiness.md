# Code review: bounded uploads, one-at-a-time assessments and local launcher

I reviewed only the diff and the new files you pasted. I didn't have shell or file access in this session, so I didn't run tests or read unchanged code. Findings that depend on unseen code say **verify**. Line numbers are approximate, taken from the diff hunks and the pasted files.

## P1: fix before merging

**1. Uploads can be falsely rejected when disk space is moderately low.** `app_backend/jobs.py` ~307 (`register_upload_stream`)

- **Problem:** `limit = upload_limits(self.uploads_dir)["available_bytes"]` is calculated *after* Starlette has already written the multipart spool to disk. If the temp folder and app data share a volume (the normal case on macOS), the spool has already used some free space, and the formula divides by 2 again.
- **Example:** 612 MB free and a 512 MB reserve gives about 50 MB available in `parse_upload`. A 40 MB file passes there. After spooling, `register_upload_stream` recalculates about 30 MB and rejects the file with **413 validation / "exceeds the available upload limit"**. That message blames the file, not the disk.
- **Fix:** pass the `available` value from `parse_upload` into `register_upload_stream(limit=...)`. Inside the copy loop, only check the real free space against `DISK_RESERVE_BYTES`, without halving it again. Add a test that reduces the mocked `disk_usage` by the spooled size between the parse and the copy.

**2. Every retry leaves another copy of the audio on disk.** `frontend/src/routes/SpeakRoute.tsx` ~381–390 and `jobs.py` `submit`

- **Problem:** `handleSubmit` always uploads first and only then calls create-assessment. If creation fails, the stored upload is never used. That happens with the new **409 busy** response, the low-disk `OSError`, or a network error. Each retry uploads the same file again (up to 100 MB each time).
- **Why it matters:** this works against the low-disk handling and the one-at-a-time rule.
- **Fix:** cache `{file, audio_id}` in `SpeakRoute` and reuse the `audio_id` while the same `attachedFile` is attached. Alternatively, have the frontend check whether an assessment is running before uploading.

**3. The launcher's fresh build on every start will break tabs left open from an earlier session.** `scripts/start_practice.py` ~60–75 and `frontend/src/App.tsx`

- **Problem:** each launch picks a new backend port (`allocate_local_port`), bakes it into a new Vite build in a new temp folder, and serves it on the same fixed port, 4173. A tab still open from a previous session will:
  - request lazy route chunks that no longer exist, so the `lazy()` import fails and, with no error boundary, the app goes blank;
  - send API calls to the old backend port, which is dead.
- **Fix:**
  - Use a stable backend port, for example a fixed default or one saved in app data. This also avoids a full `vite build` on every launch.
  - Add an error boundary around the `Suspense` that offers a reload when a chunk fails to load.
  - Consider loading `SpeakRoute` eagerly.

**4. Assessments may now depend on the `ffmpeg` command being on `PATH` (verify).** `uploads.py` ~25 and `jobs.py` `_job_worker`

- **Problem:** `validate_audio_duration` runs the `ffmpeg` command in every assessment worker. If the current pipeline doesn't already need that command (faster-whisper decodes through PyAV), this is a new hard dependency.
- **Where it breaks:** macOS apps opened from Finder or the DMG get a minimal `PATH` without `/opt/homebrew/bin`. `subprocess.run` then raises `FileNotFoundError`, which isn't a `ValueError`, so every assessment fails with a raw error.
- **Fix:** catch `FileNotFoundError` and show a clear "ffmpeg not found" message, or use the same decoder the ASR step already uses.

## P2: usability and robustness

**5. Upload error messages are English-only and start with "Error: ".** `SpeakRoute.tsx` ~386 and ~431, `client.ts` XHR handlers

- **Problem:** the new errors are plain `Error` objects, so `String(error)` shows "Error: Upload connection lost…". The XHR `onerror`/`ontimeout` texts are hard-coded English, so Italian users see English.
- **Fix:** use `error.message`, and translate the messages through i18n keys (for example `speak.upload_connection_lost` and `speak.upload_timeout`).

**6. The whole app shell disappears while a route loads.** `App.tsx` ~120

- **Problem:** `<Suspense>` wraps all of `<Routes>`, so the first visit to each lazy route swaps the whole app, including navigation, for the fallback. The fallback also uses `home.setup_guide_loading`, which says "setup guide" on every route.
- **Fix:** move the `Suspense` inside the layout around `<Outlet/>` and use a generic loading key. The `const … = lazy()` lines placed between `import` statements will also trip `import/first` if that lint rule is on.

**7. The `.command` launcher fails without explanation in some setups.** `Start Vostavo.command` line 7 and `start_practice.py` ~50 and ~57

- **Missing Node 24:** with `set -e`, a failing `nvm use 24` exits before the `|| { echo "Press Return…"; read; }` guard. The user sees "[Process completed]" with no guidance. Use `nvm use 24 || true` and let the Python check report the problem.
- **Launching twice:** `probe.bind` fails with a raw `[Errno 48] Address already in use`. Detect that the app is already running on 4173 and just open the browser, or print a clear message.
- **Node version:** the check requires exactly Node 24, so Node 25 or 26 is rejected. That's fine only if `engines` requires exactly 24.
- **File mode:** make sure the untracked `.command` file is committed as executable (100755), or double-clicking it will fail.

**8. Closing the Terminal window skips cleanup.** `start_practice.py` ~70

- **Problem:** only `SIGTERM` is handled. Closing the Terminal window sends `SIGHUP`, which kills the launcher without running `finally`. The temp build folder is left behind and child processes may be orphaned.
- **Fix:** register the same handler for `signal.SIGHUP`.

**9. The upload request can hang forever.** `client.ts` ~272 (`xhr.onload = async …`)

- **Problem:** if `parseErrorResponse` throws (for example on a non-JSON 5xx from a proxy), the promise returned by `onload` rejects with no handler. The outer promise never settles and `isSubmitting` stays true.
- **Fix:** wrap it in try/catch and reject with a fallback `ApiClientError`.

**10. The XHR upload path may skip what `requestJson` does (verify).** `client.ts`

- **Problem:** the new XHR upload bypasses `requestJson` entirely.
- **Check:** if `requestJson` adds headers (a desktop runtime token, for example) or special base-URL or fetch handling, uploads with `onProgress` won't get them. `SpeakRoute` always passes `onProgress`, so this applies to every upload.

**11. Saved recordings have no file extension.** `RecorderPanel.tsx` ~637

- **Problem:** `download="practice-recording"` has no extension, and Safari often saves it that way. The Finder file then won't open, which defeats the purpose of keeping audio for retries.
- **Fix:** derive the extension from the blob or file type (`.webm`, `.m4a`/`.mp4`, `.wav`), or use `attachedFile.name`.

**12. Starting a new attempt right after cancelling may be refused.** `jobs.py` `submit` ~341

- **Problem:** `is_alive()` stays true briefly after a cancel while the worker is still terminating. An immediate retry then gets a 409 busy response.
- **Fix:** skip processes whose job metadata is already terminal, or `join(timeout)` inside the cancel path.

**13. Cancelling during analysis can leave `ffmpeg` running.** `uploads.py` ~25

- **Problem:** if a cancel terminates the worker during duration validation, the child `ffmpeg` process is orphaned for up to 90 s.
- **Fix:** start it with `start_new_session=False` and kill its process group on cancel, or accept this and note it.

**14. A client disconnect mid-upload logs as a server error.** `app.py` `upload_audio`

- **Problem:** the spool is closed correctly, but `ClientDisconnect` isn't caught, so Starlette logs a traceback or 500.
- **Fix:** catch it and return quietly.

**15. `BoundedMultipartParser` depends on a private Starlette attribute.** `uploads.py` ~59

- **Problem:** it uses `_files_to_close_on_error`. If Starlette renames it, the `except BaseException` block raises `AttributeError` and hides the original error.
- **Fix:** use `getattr(self, "_files_to_close_on_error", [])` and pin or test the Starlette version.

## Test gaps

- **XHR upload path in `client.ts`:** completely untested, because `SpeakRoute` mocks `uploadAudio`. Add unit tests for progress, abort before and during send, 507 error parsing, `onerror`, timeout, and a non-JSON error body.
- **`SpeakRoute` behaviour not covered:**
  - cancelling during upload shows `upload_cancelled` and keeps the attachment;
  - 507 and 409 responses keep the attachment, and retrying works;
  - files that are 0 bytes or over 100 MB are rejected;
  - the disk-limit message is used when `available_bytes < max_bytes`.
- **Existing `SpeakRoute` assertions:** check any assertion like `uploadAudio` toHaveBeenCalledWith(file). The call now passes a second argument.
- **`beforeEach` mock order:** `getUploadLimits.mockResolvedValue` is set *before* `vi.clearAllMocks()`. That works in current Vitest, but it breaks if `mockReset`/`restoreMocks` is turned on. Move it after the clear.
- **Backend tests to add:**
  - the HTTP 409 mapping for `/v1/assessments`;
  - a busy slot is freed once the process has exited;
  - a disconnect mid-stream leaves no files in `uploads_dir` or the temp folder;
  - the false-rejection case from finding 1;
  - missing `ffmpeg` (finding 4).
- **LM Studio:** nothing asserts that `llm_inference_profile == "lmstudio_bounded_v1"` is recorded. Add it to the unit tests or to the live spec when `provider === "lmstudio"`. In `ollamaBilingual.spec.ts`, the `ollama` and `OLLAMA_E2E_WHISPER` names are now misleading.
- **`webkitPractice.spec.ts`:**
  - `test.use({ browserName: "webkit" })` forces WebKit in every configured project, which may run the test twice;
  - `toHaveCount(1)` on the progress circles fails on a Playwright retry or with reused app data (use a unique speaker ID per run);
  - it never tests keeping the audio after a failed upload.
- **`start_practice.py`:** no tests. At least test `wait_ready` and the Node, `ffmpeg` and port checks with mocked `shutil.which` and `subprocess` calls.

Also worth a quick check: all five locale files add the same new keys (`speak.download_recording`, `upload_size_error`, `upload_disk_error`, `upload_cancelled`, `upload_progress`), `ErrorCode.STORAGE` has the value `"storage_error"`, and the backend's CORS settings allow `http://127.0.0.1:4173`.
