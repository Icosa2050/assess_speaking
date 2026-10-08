#!/usr/bin/env python3
"""Evaluate sample B2/C1 metric profiles against CEFR baselines."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import assess_speaking

DATA_FILE = PROJECT_ROOT / "tests" / "data" / "sample_metrics.json"


def main() -> None:
    samples = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    results = []
    for sample in samples:
        evaluation = assess_speaking.evaluate_baseline(sample["level"], sample["metrics"])
        results.append({
            "label": sample["label"],
            "level": sample["level"],
            "notes": sample.get("notes", ""),
            "passed": evaluation.get("passed") if evaluation else False,
            "details": evaluation,
        })
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
