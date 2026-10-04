#!/usr/bin/env python3
"""Opt-in local generation comparison using six fixed, authored bilingual cases.

Use --prompt-ref to load prompts from a TRUSTED local Git commit while retaining
current provider/validation code. This executes that commit's Python prompt
module. Run each prompt version into a fresh output directory; inspect accepted
outputs for semantic quality separately. Expected labels never enter requests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assessment_runtime import assessment_prompts, llm_client

CASE_IDS = {'en_modal_can_clean', 'en_modal_can_seeded',
            'it_person_agreement_subj_clause_clean', 'it_person_agreement_subj_clause_seeded',
            'en_optional_style', 'it_optional_style'}
CORPUS = ROOT / 'tests/fixtures/feedback_quality/bilingual_v2.json'


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['ollama', 'lmstudio'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-ref', help='Trusted local Git commit/ref for historical prompt code')
    parser.add_argument('--case', action='append', dest='case_ids', help='Select fixed case ID (repeatable)')
    parser.add_argument('--ollama-reasoning', choices=['none', 'low'], help='Experimental override; no production setting change')
    parser.add_argument('--completion-token-limit', type=int, choices=[4096,8192], default=4096)
    args = parser.parse_args()
    if args.ollama_reasoning is not None and args.provider != 'ollama':
        parser.error('--ollama-reasoning requires Ollama')
    args.output.mkdir(parents=True, exist_ok=False)
    prompt_source = (ROOT / 'assessment_runtime/assessment_prompts.py').read_text(encoding='utf-8')
    prompts = assessment_prompts
    prompt_commit = None
    if args.prompt_ref:
        prompt_commit = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options',
                                                 args.prompt_ref+'^{commit}'], cwd=ROOT, text=True).strip()
        prompt_source = subprocess.check_output(['git', 'show', prompt_commit+':assessment_runtime/assessment_prompts.py'],
                                                cwd=ROOT, text=True)
        prompts = types.ModuleType('historical_assessment_prompts')
        exec(compile(prompt_source, 'historical_assessment_prompts', 'exec'), prompts.__dict__)
    (args.output / 'prompts.py').write_text(prompt_source, encoding='utf-8')
    cases = [c for c in json.loads(CORPUS.read_text(encoding='utf-8'))['cases'] if c['case_id'] in CASE_IDS]
    if {c['case_id'] for c in cases} != CASE_IDS or len(cases) != len(CASE_IDS):
        raise ValueError('Fixed generation cases missing or duplicated')
    if args.case_ids:
        if not set(args.case_ids) <= CASE_IDS:
            raise ValueError('Unknown case ID')
        cases = [c for c in cases if c['case_id'] in args.case_ids]
    paths = [CORPUS, Path(__file__), *(ROOT / path for path in (
        'assessment_runtime/llm_client.py', 'assessment_runtime/output_validation.py',
        'assessment_runtime/style_validation.py', 'assessment_runtime/feedback_claims.py', 'assess_core/schemas.py'))]
    for source in paths:
        destination = args.output / 'sources' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), provider=args.provider, model=args.model,
                    prompt_commit=prompt_commit, prompt_sha256=hashlib.sha256(prompt_source.encode()).hexdigest(),
                    rubric_prompt_version=prompts.RUBRIC_PROMPT_VERSION, coaching_prompt_version=prompts.COACHING_PROMPT_VERSION,
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    planned_case_ids=[c['case_id'] for c in cases], completed_cases=0, timeout_sec=90, validation_retries=1,
                    ollama_reasoning_override=args.ollama_reasoning, completion_token_limit=args.completion_token_limit,
                    limitation='Authored text; schema acceptance is not semantic accuracy. Scores are not level labels.')
    save(args.output / 'manifest.json', manifest)
    original_chat = llm_client._chat_completion
    calls = []

    def recorded_chat(*params, **kwargs):
        payload = dict(kwargs.get('extra_payload') or {})
        payload['max_tokens'] = args.completion_token_limit
        if args.ollama_reasoning is not None:
            payload['reasoning_effort'] = args.ollama_reasoning
        kwargs['extra_payload'] = payload
        raw = original_chat(*params, **kwargs)
        calls.append(dict(prompt=kwargs.get('prompt', params[2] if len(params)>2 else None), raw=raw))
        return raw

    llm_client._chat_completion = recorded_chat
    try:
        for case in cases:
            calls.clear()
            row = dict(case_id=case['case_id'], transcript=case['transcript'], variant=case['variant'],
                       expected=case['expected_grammar_errors'], semantic_verdict='requires output inspection')
            metrics = {'word_count':len(case['transcript'].split()), 'duration_sec':'unknown (authored text)'}
            note = '\nSpontaneous everyday speech. Authored text only, no audio. Do not invent audio measurements.\n'
            started = time.monotonic()
            try:
                rubric, _ = llm_client.generate_rubric(args.provider, args.model,
                    prompts.rubric_prompt(case['transcript'], metrics, case['theme'],
                        expected_language=case['language_code'], feedback_language=case['language_code'])+note,
                    timeout_sec=90, transcript=case['transcript'])
                row.update(rubric=rubric.to_dict(), rubric_contract='accepted')
                # Historical prompts predate the transcript parameter.
                extra = {'transcript':case['transcript']} if 'transcript' in inspect.signature(prompts.coaching_prompt).parameters else {}
                try:
                    coaching, _ = llm_client.generate_coaching_summary(args.provider, args.model,
                        prompts.coaching_prompt(metrics, row['rubric'], case['theme'], 90,
                            expected_language=case['language_code'], feedback_language=case['language_code'],
                            checks={'duration_pass':None}, **extra)+note,
                        timeout_sec=90, target_duration_sec=90, rubric=row['rubric'])
                    row.update(coaching=coaching.to_dict(), coaching_contract='accepted')
                except llm_client.LLMClientError as exc:
                    row.update(coaching_contract='failed', coaching_error=str(exc))
            except llm_client.LLMClientError as exc:
                row.update(rubric_contract='failed', rubric_error=str(exc))
            row.update(calls=list(calls), elapsed_sec=round(time.monotonic()-started, 3))
            save(args.output / (case['case_id']+'.json'), row)
            manifest['completed_cases'] += 1
            save(args.output / 'manifest.json', manifest)
            print(case['case_id'], row['rubric_contract'], row.get('coaching_contract', 'skipped'), flush=True)
    finally:
        llm_client._chat_completion = original_chat


if __name__ == '__main__':
    main()
