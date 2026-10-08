import json
from pathlib import Path

import assess_speaking

DATA_FILE = Path(__file__).resolve().parent / "data" / "sample_metrics.json"


def load_samples():
    payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    assert payload, "sample_metrics.json is empty"
    return payload


def test_baseline_samples_pass():
    for sample in load_samples():
        result = assess_speaking.evaluate_baseline(sample["level"], sample["metrics"])
        assert result is not None, f"No baseline evaluation for {sample['label']}"
        assert result["passed"], f"Baseline failed for {sample['label']}"


def test_sample_metrics_are_in_range():
    for sample in load_samples():
        metr = sample["metrics"]
        assert metr["wpm"] >= 80
        assert metr["fillers"] >= 0
        assert metr["cohesion_markers"] >= 0
        assert metr["complexity_index"] >= 0
