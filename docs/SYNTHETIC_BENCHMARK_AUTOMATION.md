# Synthetic Benchmark Automation

Last updated: 2026-03-14
Status: Agreed starting point for macOS `say` + `ffmpeg`

## Goal

Automate fine-grained synthetic spoken benchmark generation without coupling the
test corpus to one evaluation model or provider.

This is for engineering regression and scorer tuning, not for claiming external
validity.

## Consensus

Consensus across local investigation, PAL Gemini, PAL Opus, and PAL Qwen:

1. Use a manifest-driven pipeline.
2. Keep transcript generation separate from evaluation.
3. Store test contracts and recipes in git; avoid storing large derived audio by default.
4. Use provider/model adapters so OpenRouter, Ollama, or future models all normalize into the same contract.
5. Start with a small deterministic corpus before building a large variation matrix.

## Practical Tooling On This Mac

`say` on this macOS install supports:

1. voice selection via `-v`
2. rate control via `-r`
3. AIFF/WAVE/CAF/M4A output via `-o`
4. inline silence tags like `[[slnc 200]]`

`ffmpeg` should be used for:

1. format normalization
2. noise injection
3. pace variation
4. silence insertion and timing shifts
5. bandwidth and degradation filters

## Do Not Overfit To One Model

The benchmark corpus should not encode Gemini-specific expectations.

Recommended rule:

1. transcripts, expected CEFR, and expected dimension ranges are source-of-truth
2. prompt versions and response parsing live in an explicit contract
3. each provider/model is normalized through an adapter
4. adapters output the same schema before scoring/assertions

In repo terms, that means:

1. benchmark suites keep `llm_contract`
2. evaluators should log provider/model separately from benchmark expectations
3. benchmark assertions should target normalized rubric fields and scorer outputs, not raw model prose

## Artifacts

Store in git:

1. seed texts
2. benchmark manifests
3. generation recipes
4. prompt versions
5. parser/schema contracts
6. a very small golden audio subset if needed for CI sanity checks

Regenerate locally or in CI:

1. bulk synthetic audio
2. derived noisy variants
3. transient intermediate files

Why:

1. bulk audio will bloat the repo
2. `say` voices can drift across macOS releases, so generation recipes matter more than storing every derivative
3. a tiny golden subset is enough when bit-for-bit checks are needed

## Recommended Data Model

Use three layers.

### 1. Seed Manifest

One record per canonical transcript.

Suggested fields:

1. `seed_id`
2. `language_code`
3. `task_family`
4. `topic_tag`
5. `target_cefr`
6. `transcript`
7. `source_type`
8. `notes`

### 2. Render Recipe

One record per audio rendering variant.

Suggested fields:

1. `recipe_id`
2. `seed_id`
3. `voice`
4. `rate_wpm`
5. `inline_pause_profile`
6. `noise_profile`
7. `ffmpeg_filters`
8. `expected_language`
9. `render_platform`
10. `voice_fingerprint`

### 3. Evaluation Contract

One record per evaluated artifact.

Suggested fields:

1. `case_id`
2. `audio_path`
3. `ground_truth_transcript`
4. `expected_dimensions`
5. `expected_cefr`
6. `llm_contract`
7. `adapter_name`
8. `normalization_version`
9. `tags`

## Variation Strategy

Do not generate the full Cartesian explosion at first.

Start with a small matrix over:

1. pace
2. pauses
3. noise
4. topic mismatch
5. language mismatch

Recommended first levels:

1. pace: `slow`, `baseline`, `fast`
2. pauses: `none`, `sentence_200ms`, `sentence_800ms`
3. noise: `none`, `white_-20db`, `babble_-15db`
4. mismatch: `none`, `topic_off`, `language_off`

Sampling rule:

1. use a hand-picked minimal matrix first
2. move to Latin-square or similar partial sampling before considering full combinations

## Suggested macOS Generation Flow

### Clean base render

```bash
say -v "Samantha" -r 175 -o out/base.aiff --file-format=AIFF "Hello world"
ffmpeg -y -i out/base.aiff -ac 1 -ar 16000 out/base.wav
```

### Inline pauses

```bash
say -v "Samantha" -r 175 -o out/paused.aiff "Hello [[slnc 400]] world"
```

### White noise example

```bash
ffmpeg -y \
  -i out/base.wav \
  -f lavfi -i "anoisesrc=color=white:amplitude=0.02" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first" \
  -ac 1 -ar 16000 out/base_white_noise.wav
```

### Pace change after render

```bash
ffmpeg -y -i out/base.wav -filter:a "atempo=1.15" -ac 1 -ar 16000 out/base_fast.wav
```

## What To Implement First In This Repo

### Phase 1

1. add a seed-manifest format for English
2. add a small generator script that reads seeds and produces clean base audio via `say`
3. normalize output to mono 16 kHz WAV with `ffmpeg`
4. emit render metadata next to audio

### Phase 2

1. add one noise profile
2. add one pause profile
3. generate a second manifest layer for rendered variants
4. connect generated artifacts to the benchmark suite format already in `tests/fixtures/benchmarks`

### Phase 3

1. add provider adapters for rubric generation
2. normalize raw model responses with the existing parser/schema path
3. compare normalized rubric output against benchmark expectations

## Important Caveats

1. `say` is deterministic enough for engineering use, but voices may shift slightly across macOS updates.
2. WPM is acceptable for English and Italian to start, but it should not be treated as the long-term universal fluency measure.
3. Synthetic TTS is useful for regression and stress tests, but it must not replace real human learner audio for later calibration.

## Current Repo Implications

The benchmark harness already supports:

1. multiple discovered suites
2. suite filtering by `suite_type` and `tags`
3. explicit `llm_contract` metadata
4. per-case expected dimension ranges

That is the right base to build a controlled automation pipeline on top of.
