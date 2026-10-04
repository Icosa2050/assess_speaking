# Meaning-preserving feedback: diagnosis, changes and evaluation

Date: 2026-10-04. English and Italian speaking practice. B1/B2/C1 remain practice goals.

## What changed in the app

- Coaching receives the original transcript as a JSON string, alongside the rubric. Previously it could only see rubric-selected evidence. Historical/helper calls without a transcript still work with explicit `null`.
- Corrected a conflicting instruction: coaching must address each rubric finding, but may now qualify a doubtful diagnosis instead of prescribing a correction. Exact evidence coverage remains mandatory; a regression demonstrates that a qualified quotation is accepted.
- Coaching instructions require meaning preservation: people, objects, timing, negation, quantity, modality and causal relations. Related vocabulary is not interchangeable. Corrections need a supported language rule.
- Coaching uses spoken practice goals when there are fewer than three supported corrections. These are generation instructions, not a new deterministic semantic guarantee or a new schema constraint.
- The experimental rubric changes (v6/v7, including the reference note) were **withdrawn before commit** because they failed the semantic checks below. The app retains rubric v5 and ships coaching v7. Model defaults and validation remain unchanged.
- Prompt versions separate subsequent progress comparisons from older analysis conditions. Existing reports remain readable.

The experimental reviewer is **not called in the app**, does not change scores, and adds no production request. No model default, dependency, validation threshold or upload behavior changed. Filtering questionable diagnoses after scoring would leave possibly unsupported scores; this experiment therefore measures whole-candidate review and explicitly counts good components lost in mixed feedback.

## Reviewer experiment

Committed authored proposals cover English modals, Italian subject/agreement, meaning changes, invented travel context, correct sentences, and sound spoken practice goals. Expected labels and rationales are never sent to the model. Claude reviewed the source/test corpus; these are regression examples, not a representative learner benchmark or independent human assessment.

| Local reviewer | Protocol/corpus | Repeats | Harmful proposals accepted | Useful proposals retained |
| --- | --- | --- | --- | --- |
| Ollama `qwen3.5:4b` | v1 / 20 cases | 2 | 2/20 | 14/20 |
| LM Studio `qwen2.5-7b-instruct` | v1 / 20 cases | 2 | 4/20 | 12/20 |
| LM Studio `qwen2.5-7b-instruct` | v2 / 26 cases | 1 | 5/14 | 11/12 |

The first two rows each have **20 distinct cases and 40 observations**, not 40 independent examples. One Ollama observation was a truncated-response error; it is counted as withheld, not as correct rejection. The v2 total includes four framing variants and two mixed candidates. Framing variants share item groups with their source cases; final summaries distinguish rows, distinct case IDs and distinct item groups. These are counts, not population accuracy estimates.

Protocol v2 explicitly distinguishes reviewing a correction from judging whether the original is grammatical, and asks for the reason before the verdict. This reduced false rejection in the observed run, but missed obligation/recommendation changes and an unsupported causal link. Neutral framing did not fix those misses. Rejecting the two mixed candidates would also discard two sound coaching components.

Claude found one v2 framing control had accidentally dropped an emphasis caveat. Corpus v2.1 restores it and records item groups. The original v2 inputs/results remain snapshotted; the changed control was accepted in its separate local check. Do not present this as a rerun of all 26 cases on v2.1.

**Decision:** do not make this second model pass a production gate. It still approves harmful advice and can invent grammar rules while rejecting useful corrections. No cloud model or learner recordings were used in these experiments.

## Generation comparison

Six fixed cases from `bilingual_v2.json`: English modal clean/error, Italian person-agreement clean/error, and two style-only transcripts. Ollama `qwen3.5:4b`, same provider/validator code, 90-second call timeout and one validation repair. Authored text has no audio measurements. The original prompt reference is `043613b`.

| Prompt | Rubric schema/evidence accepted | Coaching accepted | Interpretation |
| --- | --- | --- | --- |
| Original rubric v5 / coaching v6 | 3/6 | 1/6 | Accepted output still changed balanced system into flexible policy; modal explanation included a contradictory agreement claim |
| Candidate rubric v6 / coaching v7 | 6/6 | 4/6 | More output, but the Italian seeded `noi partano` was falsely called correct; not a semantic quality pass |
| Reference-note rubric v7 / coaching v7 | 5/6 | 4/6 | Italian clean control failed quote grounding; seeded agreement error still falsely accepted and repeated in coaching |
| Shipped rubric v5 / coaching v7 | 4/6 | 2/6 | Narrower context/conflict fix only; accepted reports still contain inaccurate explanations |

Both candidate seeded-error coaching outputs were withheld by unchanged validation (English exact quote coverage; Italian unsupported delivery claim). The candidate English modal finding explained the base-form rule correctly, while its accuracy comment still misnamed the construction as subject agreement. The Italian clean report gave useful retry goals but contained language defects in its own feedback. The optional-style candidates supplied no rubric rewrites, while coaching continued to propose richer vocabulary that needs appropriate context. Neither score accuracy nor complete meaning preservation is established. In the final shipped-code replay, the modal correction itself was sound, but the rubric called can a singular subject and coaching still mislabelled the focus as subject agreement. The unchanged rubric v5 varies between runs; its 4/6 versus 3/6 acceptance cannot be attributed to the coaching change. These observations do not establish an end-to-end quality improvement.

A focused Italian reference-note follow-up checks that subject/person agreement survives subjunctive selection, without changing the actual subject. Primary sources: [Treccani on agreement](https://www.treccani.it/enciclopedia/concordanza_(La-grammatica-italiana)/), [Treccani on present subjunctive](https://www.treccani.it/enciclopedia/congiuntivo-presente_(La-grammatica-italiana)/). English reference: [Cambridge on modal + base form](https://dictionaryblog.cambridge.org/2016/11/23/modal-verbs-the-basics/).

## Reproduce locally

These runners accept only the installed local Ollama/LM Studio providers and committed authored cases. They do not accept arbitrary learner files or cloud keys. Choose a fresh output directory; existing evidence is never overwritten.

```sh
.venv/bin/python scripts/evaluate_feedback_review.py --provider ollama --model qwen3.5:4b --repeats 2 --output frontend/output/live-journeys/review-example
.venv/bin/python scripts/evaluate_feedback_generation.py --provider ollama --model qwen3.5:4b --output frontend/output/live-journeys/generation-current
.venv/bin/python scripts/evaluate_feedback_generation.py --provider ollama --model qwen3.5:4b --prompt-ref 043613b --output frontend/output/live-journeys/generation-baseline
```

`--prompt-ref` executes the Python prompt module from a **trusted local Git commit**, while using current generation/validation code; it is not a whole-application historical replay. Manifests and snapshots preserve the loaded inputs. Generation outputs need inspection beyond `accepted`; the runner never labels them grammatically correct.

## Evidence and review

Ignored local evidence under `frontend/output/live-journeys/`:

- `20261004-semantic-review-ollama`, `20261004-semantic-review-lmstudio`: v1, two repeats.
- `20261004-semantic-review-lmstudio-v2`: clarified protocol and framing/mixed controls.
- `20261004-semantic-review-framing-fix`: changed v2.1 control only.
- `20261004-semantic-generation`: six baseline cases. The superseded candidate half was intentionally interrupted before reuse; retained results are not silently overwritten.
- `20261004-semantic-generation-final`: completed candidate v6/v7 comparison.
- `20261004-semantic-generation-reference`: withdrawn reference-note follow-up.
- `20261004-semantic-thinking-italian`: experimental reasoning enabled; both cases timed out.
- `20261004-semantic-shipped-coaching`: six cases on the final shipped rubric v5/coaching v7 sources.

Source snapshots/hashes distinguish each run. Early v1 manifests lacked completion counters; all 40 expected rows were checked explicitly. The final runner freezes corpus bytes once and records planned/completed observations. Summary failures and uncertain verdicts remain visible by expected class. No auto-rejection is counted as a correct language diagnosis.

Claude CLI reviewed selected source and authored test fixtures twice. Fixed the coaching contradiction, framing-control caveat, distinct-case accounting, completion metadata, frozen corpus reads and runner prompt capture. The initially attempted CLI bare mode did not find authentication; normal logged-in mode succeeded. OCR delegate provided local selection/rules and host review; the accompanying coverage file accounts for unrelated untracked files and manually reviewed tests. CodeRabbit was not rerun because the prior separate approval block remains unresolved.


## Final verification and remaining limits

- Backend: **925 passed, 17 opt-in skips** in the current local checkout, including pre-existing untracked sample-workflow tests. The first sandboxed run had one port-binding PermissionError; rerunning with localhost permission passed. One existing Starlette/anyio deprecation remains.
- Frontend: **149 passed** and TypeScript passed.
- Final bilingual browser smoke: **2 passed**, covering English and Italian upload, feedback, microphone retry, History and playback; isolated `/v1/health` and Vite startup passed. Fixture inference tests workflow/provenance, not language quality.
- Quality checker and `git diff --check`: passed.
- CLI regression tests cover hidden labels, historical prompt signatures, request overrides preserving strict schemas, partial/error accounting, immutable evidence directories and restoring the wrapped client.

The reference note did not fix the Italian failure: the model still praised `noi partano` and instructed repetition of it. The candidate rubric prompts are not promoted. **Reliable grammar diagnosis and meaning preservation remain unresolved**; the narrower shipped coaching change is not claimed to solve them. Small local-model results are not a release quality pass. No model/provider default was promoted on this evidence. Do not interpret improved schema acceptance as better language advice.

A further experimental check enables Ollama reasoning (`low`, mapping to `true` on this installed boolean-capability model) with an 8192-token ceiling. The production profile still uses `none`/4096. `/api/show` confirmed support and default `true`; [Ollama documents this OpenAI-compatibility mapping](https://docs.ollama.com/api/openai-compatibility). This checks a candidate setting, not an isolated causal claim (the token budget also differs). Both the clean and seeded Italian cases timed out at 90 seconds (2/2 unavailable). No reasoning text is consumed as final feedback; a final answer must still pass the original validators.


Final disposition: commit the source-context/contradiction fix, authored evaluation cases and reproducible runners. Keep rubric v5, model defaults and validation gates; retain the failed rubric/reference/reasoning experiments as evidence. Further model-quality work is required before calling grammar correction dependable. The app's measured practice/progress and fallback exercise flows remain usable while that limitation is explicit.
