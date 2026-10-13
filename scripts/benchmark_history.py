"""Bounded synthetic History service benchmark; never reads the learner journal."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_core.services import history_rows


def benchmark(count):
    with tempfile.TemporaryDirectory(prefix='vostavo-history-benchmark-') as directory:
        root = Path(directory)
        fields = ['timestamp', 'session_id', 'speaker_id', 'learning_language', 'task_family', 'theme', 'report_path', 'duration_sec', 'word_count', 'wpm', 'final_score', 'band']
        with (root / 'history.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
            for index in range(count):
                path = root / f'fixture-{index}.json'
                report = {'session_id': f'fixture-{index}', 'checks': {'min_words_pass': True, 'duration_pass': True, 'topic_pass': True, 'language_pass': True, 'content_validity_pass': True}, 'scores': {'final': 3, 'band': 3}}
                path.write_text(json.dumps({'report': report, 'metrics': {'duration_sec': 90, 'word_count': 150}, 'meta': {}}))
                writer.writerow(dict(timestamp='2026-10-08T00:00:00+00:00', session_id=f'fixture-{index}', speaker_id=f'fixture-{index % 20}', learning_language='it' if index % 2 else 'en', task_family='free_monologue', theme='synthetic', report_path=path, duration_sec=90, word_count=150, wpm=100, final_score=3, band=3))
        durations = []
        for _ in range(3):
            started = time.perf_counter(); rows = history_rows(root); durations.append((time.perf_counter() - started) * 1000)
            assert len(rows) == count and all(row['eligibility']['state'] == 'assessable' for row in rows)
        return {'attempts': count, 'load_ms': durations, 'median_ms': statistics.median(durations),
                'json_bytes': len(json.dumps(rows).encode()), 'limit': 'backend service only; warm filesystem, no HTTP/browser rendering or accessibility measurement'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    result = {'date': '2026-10-08', 'system': platform.system(), 'architecture': platform.machine(), 'logical_cpus': os.cpu_count(), 'python': platform.python_version(), 'datasets': [benchmark(count) for count in (1000, 10000)]}
    args.output.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))
