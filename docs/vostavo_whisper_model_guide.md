# Vostavo – Local Whisper Model Guide

> **Architecture**: Whisper runs on the user's machine → transcript + metadata sent to Vostavo service → CEFR rubric returned.
> Audio never leaves the user's device.

---

## Quick Decision: Can My Machine Run Whisper?

| Hardware | Verdict | Recommended Model |
|---|---|---|
| Apple Silicon Mac (M1–M4, any RAM) | ✅ Excellent | `large-v3` |
| Windows/Linux + NVIDIA GPU ≥ 8 GB VRAM | ✅ Excellent | `large-v3` |
| Windows/Linux + NVIDIA GPU 6 GB VRAM | ✅ Good | `medium` |
| Windows/Linux + NVIDIA GPU 4 GB VRAM | ⚠️ Limited | `small` (quality reduced) |
| Any machine, CPU only | ❌ Too slow | Not recommended |
| iPhone 15+ / recent iPad | ✅ Supported | Native iOS path; use quality tier guidance |
| Older smartphone / Chromebook | ⚠️ Limited or unsupported | Not recommended for full CEFR scoring |

---

## Language → Model Mapping

Whisper quality varies significantly by language, accent, audio conditions, and CEFR level. Use this table to select the minimum acceptable model for local practice, and the full-quality model for scored assessment.

| Language | Minimum for practice | Full-quality CEFR scoring | Notes |
|---|---|---|---|
| English | `small` | `large-v3` for B2/C1 | `small` is usable for rough EN feedback; scored results still benefit from `large-v3`. |
| German | `medium` | `large-v3` | `small` has materially higher WER on varied public benchmarks; use `large-v3` for case endings, compounds, and B2/C1 detail. |
| Italian | `medium` | `large-v3` | `medium` is the minimum for practice; `large-v3` is the baseline for B2/C1 grammar and function-word sensitivity. |
| French | `medium` | `large-v3` | Public benchmarks show a large jump from `small` to `medium`/`large`; use `large-v3` for scored work. |
| Spanish | `medium` | `large-v3` | Spanish benchmarks are comparatively strong, but `large-v3` is still the scoring baseline for dialect, accent, and B2/C1 detail. |
| Portuguese | `medium` | `large-v3` | Use `large-v3` for scored European/Brazilian Portuguese because dialect and accent variance affect WER. |
| Dutch | `medium` | `large-v3` | `medium` can support practice; use `large-v3` for full scoring and function-word reliability. |
| Polish / Czech | `large-v3` | `large-v3` | Morphology and benchmark WER make smaller models too risky for scoring. |
| Slovak / lower-resource targets | Not recommended until calibrated | `large-v3` plus validation | Do not promise full scoring until Vostavo has language-specific benchmark evidence. |
| Other languages | `large-v3` for trials | `large-v3` plus validation | Default to `large-v3`, then calibrate before claiming reliable CEFR scoring. |

> **Rule of thumb**: For any scored CEFR assessment at B2 level or above, use `large-v3`. `medium` is acceptable for practice in several major European languages, but reduced-quality transcripts can corrupt grammar, cohesion, and range scoring.

---

## Model Size & Download Info

Users need to download the model once; it is cached locally afterward.

| Model | Download Size | RAM Required | Relative Speed |
|---|---|---|---|
| `tiny` | ~75 MB | ~1 GB | Very fast (not for CEFR) |
| `base` | ~145 MB | ~1 GB | Fast (not for CEFR) |
| `small` | ~460 MB | ~2 GB | Fast |
| `medium` | ~1.5 GB | ~5 GB | Moderate |
| `large-v3-turbo` | ~1.6 GB | ~6 GB | Fast, near-`large-v3` quality |
| `large-v3` | ~3.0 GB | ~10 GB | Slower, highest accuracy |

> ℹ️ The model is downloaded automatically on first use and stored in the Vostavo cache directory. Subsequent runs use the local cache.

---

## ⚠️ Warnings to Show in the UI

These warnings should be surfaced in **Runtime Setup** before the user starts a session.

### CPU-only machine detected
> **⚠️ Local Whisper is not recommended on this machine.**
> No compatible GPU was found. Transcribing a 2-minute recording on CPU only may take 5–15 minutes, which will block your session.
> **Options:**
> – Switch to audio upload mode (your recording is sent to the Vostavo server for transcription).
> – Continue locally and expect long wait times.

### GPU with less than 6 GB VRAM
> **⚠️ Your GPU has limited memory.**
> Only the `small` Whisper model is recommended. For languages other than English, transcription quality may be reduced, and CEFR scores at B2 and above may be less reliable.
> Consider using audio upload mode for better accuracy.

### Language selected requires `large-v3`, but user chose a smaller model
> **⚠️ The selected Whisper model may not be accurate enough for [Italian / German / Polish / …].**
> `large-v3` is recommended for this language to ensure reliable CEFR scoring.
> Your current model is `[model]`. Transcription may contain errors that affect your score.

### Model not yet downloaded
> **ℹ️ The `large-v3` model (~3.0 GB) needs to be downloaded before your first session.**
> This happens automatically and is stored locally. You will not need to download it again.
> Ensure you have a stable internet connection and ~3 GB of free disk space.

---

## Throughput Reference (for 2-minute recordings)

| Hardware | Model | Transcription Time | Suitable for Vostavo? |
|---|---|---|---|
| M1 Mac (8 GB) | `large-v3` | ~15–25 s | ✅ Yes |
| M4 Max (128 GB) | `large-v3` | ~8–15 s | ✅ Yes |
| RTX 3060 (12 GB) | `large-v3` | ~10–20 s | ✅ Yes |
| GTX 1060 (6 GB) | `medium` | ~20–40 s | ✅ Acceptable |
| RTX 3050 (4 GB) | `small` | ~15–30 s | ⚠️ Quality limited |
| iPhone 15+ | Native iOS model | Device/model dependent | ✅ Supported with quality tier caveats |
| CPU only (any) | `medium` | 5–15 min | ❌ Not suitable |

---

## Architecture Note

The local Whisper path keeps audio private and reduces server load. The service receives only:

- Plain-text transcript
- Detected language
- Objective metrics (WPM, pause count, filler words, duration)
- Session metadata (CEFR target, topic/theme, speaker ID)

The CEFR rubric scoring runs server-side and is returned as structured JSON.

---

## Evidence Basis

- OpenAI's Whisper paper shows that multilingual transcription improves with model size and reports materially higher WER for `small` than `medium`/`large` on datasets such as MLS, Common Voice, and FLEURS.
- The `large-v3` model card reports broad multilingual improvements over `large-v2`, but also warns that performance is uneven across languages, accents, and dialects.
- Vostavo's stricter rule is product-specific: CEFR scoring depends on grammar, function words, cohesion markers, and lexical range, so a transcript that is "readable" may still be too noisy for reliable B2/C1 scoring.

Sources:
- https://cdn.openai.com/papers/whisper.pdf
- https://huggingface.co/openai/whisper-large-v3
- https://github.com/openai/whisper/blob/main/model-card.md

---

*Last updated: April 2026 — Vostavo / assess\_speaking*
