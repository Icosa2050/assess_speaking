#!/usr/bin/env python3
"""Opt-in, local-only evaluation on committed authored feedback proposals.

No learner input, credentials or cloud provider support. Expected labels never
enter requests. Keeps individual outcomes including errors and unknowns; does
not call model acceptance grammatical truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assessment_runtime import llm_client
from assessment_runtime.semantic_review import REVIEW_SCHEMA, SEMANTIC_REVIEW_VERSION, parse_review, review_prompt

TEMPERATURE = 0.2
TIMEOUT_SEC = 90
MAX_TOKENS = 768

CORPUS = ROOT / 'tests/fixtures/feedback_quality/semantic_review_v2.json'


def summarize(rows: list[dict]) -> dict:
    result = {}
    for split in ('all', 'regression', 'transfer', 'framing', 'mixed'):
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        bad = [r for r in selected if r['expected'] == 'reject']
        good = [r for r in selected if r['expected'] == 'accept']
        ids = {r['case_id'] for r in selected}
        result[split] = dict(
            rows=len(selected), unique_cases=len(ids),
            unique_item_groups=len({r.get("item_group", r["case_id"]) for r in selected}),
            harmful_proposals=len(bad), useful_proposals=len(good),
            harmful_accepted=sum(r['verdict'] == 'accept' for r in bad),
            harmful_rejected=sum(r['verdict'] == 'reject' for r in bad),
            useful_retained=sum(r['verdict'] == 'accept' for r in good),
            useful_withheld=sum(r['verdict'] != 'accept' for r in good),
            harmful_outcomes=dict(Counter(r['verdict'] for r in bad)),
            useful_outcomes=dict(Counter(r['verdict'] for r in good)),
            outcomes=dict(Counter(r['verdict'] for r in selected)),
            harmful_cases_accepted_in_any_repeat=len({r['case_id'] for r in bad if r['verdict']=='accept'}),
            useful_cases_withheld_in_any_repeat=len({r['case_id'] for r in good if r['verdict']!='accept'}),
            cases_with_inconsistent_verdicts=sum(len({r['verdict'] for r in selected if r['case_id']==case_id})>1 for case_id in ids),
            known_good_components_lost=sum(r.get('good_components', 0) for r in bad if r['verdict']!='accept'),
        )
    return result


def source_paths() -> list[Path]:
    return [CORPUS, Path(__file__), ROOT / 'assessment_runtime/semantic_review.py',
            ROOT / 'assessment_runtime/llm_client.py']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['ollama', 'lmstudio'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', dest='case_ids', help='Select case ID (repeatable)')
    parser.add_argument('--repeats', type=int, default=1, choices=range(1, 4))
    args = parser.parse_args()
    # Never overwrite retained evidence from another run.
    args.output.mkdir(parents=True, exist_ok=False)
    source_bytes = {p: p.read_bytes() for p in source_paths()}
    cases = json.loads(source_bytes[CORPUS])['cases']
    if args.case_ids:
        requested = set(args.case_ids)
        if not requested <= {c['case_id'] for c in cases}:
            raise ValueError('Unknown case ID')
        cases = [c for c in cases if c['case_id'] in requested]
    sources = list(source_bytes)
    for source in sources:
        target = args.output / 'sources' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_bytes[source])
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), provider=args.provider,
        model=args.model, repeats=args.repeats, review_version=SEMANTIC_REVIEW_VERSION,
        temperature=TEMPERATURE, timeout_sec=TIMEOUT_SEC, max_tokens=MAX_TOKENS, validation_retries=0,
        planned_rows=len(cases)*args.repeats, completed_rows=0, completed=False,
        planned_case_ids=[c["case_id"] for c in cases],
        hypothetical_policy="Only accept releases the whole candidate; error/uncertain withhold it. No production gating.",
        corpus_sha256=hashlib.sha256(source_bytes[CORPUS]).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(source_bytes[p]).hexdigest() for p in sources},
        limitation='Authored candidate review; not end-to-end generation quality or learner level.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows=[]
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            row = dict(item_group=case.get('item_group',case['case_id']), case_id=case['case_id'], split=case['split'], expected=case['expected'], repeat=repeat, good_components=case.get('good_components', 0))
            started=time.monotonic()
            try:
                raw=llm_client._chat_completion(args.provider,args.model,
                    review_prompt(case['transcript'],case['feedback'],case['language']),TIMEOUT_SEC,None,
                    require_json_object=True, extra_payload={'temperature':TEMPERATURE,'max_tokens':MAX_TOKENS,'response_format':{
                        'type':'json_schema','json_schema':{'name':'feedback_review','strict':True,'schema':REVIEW_SCHEMA}}})
                row['raw']=raw
                row.update(parse_review(llm_client.extract_json_object(raw)))
            except (llm_client.LLMClientError, ValueError) as exc:
                row.update(verdict='error',error=str(exc))
            row['elapsed_sec']=round(time.monotonic()-started,3)
            rows.append(row)
            manifest.update(completed_rows=len(rows), completed=len(rows)==manifest['planned_rows'])
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
            (args.output / f"{repeat:02}-{case['case_id']}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            (args.output / 'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')
            print(f"{repeat}/{case['case_id']}: expected={case['expected']} observed={row['verdict']} ({row['elapsed_sec']}s)",flush=True)
    print(json.dumps(summarize(rows),indent=2))


if __name__ == '__main__':
    main()
