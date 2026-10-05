#!/usr/bin/env python3
"""Summarize actual recordings and reports for human review, without calling an evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np


def audio_facts(path: Path) -> dict:
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        blocks = [output.to_ndarray().reshape(-1) for frame in container.decode(audio=0)
                  for output in resampler.resample(frame)]
        blocks += [output.to_ndarray().reshape(-1) for output in resampler.resample(None)]
    if not blocks:
        raise ValueError(f"Empty recording: {path}")
    samples = np.concatenate(blocks)
    # These are descriptive measurements, not CEFR or coaching-quality grades.
    frames = samples[:len(samples) // 480 * 480].reshape(-1, 480)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return {"path": str(path), "decoded_duration_sec": round(len(samples) / 16000, 3),
            "rms": round(float(np.sqrt(np.mean(samples * samples))), 5),
            "peak": round(float(np.max(np.abs(samples))), 5),
            "quiet_frame_fraction_at_0_008": round(float(np.mean(rms < .008)), 4)}


def inspect(root: Path) -> dict:
    rows = []
    for path in sorted(root.rglob("attempt-*-status.json")):
        if "attachments" in path.parts:
            continue
        status = json.loads(path.read_text())
        payload = status.get("payload") or {}
        report = payload.get("report") or {}
        prefix = path.name.removesuffix("-status.json")
        recordings = list(path.parent.glob(prefix + "-saved-audio.*"))
        if not recordings:
            recordings = list(path.parent.glob(prefix + "-input-*"))
        recordings = [audio_facts(audio) for audio in recordings]
        metrics = report.get("metrics") or {}
        for audio in recordings:
            audio["report_duration_delta_sec"] = round(abs(audio["decoded_duration_sec"] - metrics.get("duration_sec", 0)), 3)
        rows.append({"status_file": str(path), "status": status.get("status"),
            "session_id": report.get("session_id"), "input": report.get("input"),
            "practice": (payload.get("meta") or {}).get("practice"),
            "transcript": payload.get("transcript_full"), "metrics": metrics,
            "scores": report.get("scores"), "checks": report.get("checks"),
            "rubric": report.get("rubric"), "coaching": report.get("coaching"),
            "warnings": report.get("warnings"), "errors": report.get("errors"), "recordings": recordings,
            "screenshots": [str(p) for p in sorted(path.parent.glob(prefix + "-*.png"))],
            "human_quality_verdict": "unreviewed"})
    return {"description": "Measurements and retained report content for human review; no model-quality oracle", "assessments": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Summarized {len(result['assessments'])} assessments in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
