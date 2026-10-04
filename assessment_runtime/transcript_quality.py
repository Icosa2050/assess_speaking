"""Conservative observed ASR uncertainty, independent of language detection probability."""
from __future__ import annotations

import math

TRANSCRIPT_QUALITY_POLICY = "observed_asr_uncertainty_v2"


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def transcript_quality(result: dict) -> dict:
    probabilities = [value for word in result.get("words", [])
                     if (value := finite_number(word.get("probability"))) is not None and 0 <= value <= 1]
    low_count = sum(value < 0.5 for value in probabilities)
    low_ratio = low_count / len(probabilities) if probabilities else None
    segments = result.get("segment_diagnostics") or []
    weak_segments = sum(
        (finite_number(segment.get("avg_logprob")) is not None and segment["avg_logprob"] < -1.0)
        or (finite_number(segment.get("no_speech_prob")) is not None and segment["no_speech_prob"] > 0.6)
        for segment in segments
    )
    # These are diagnostic heuristics, not calibrated accuracy probabilities.
    weak_segment_ratio = weak_segments / len(segments) if segments else None
    uncertain = (len(probabilities) >= 3 and low_ratio >= 0.25) or (weak_segment_ratio is not None and weak_segment_ratio >= 0.25)
    observed = bool(probabilities or segments)
    return {
        "policy": TRANSCRIPT_QUALITY_POLICY,
        "status": "uncertain" if uncertain else "no_review_trigger" if observed else "unknown",
        "observed_word_count": len(probabilities),
        "low_confidence_word_count": low_count,
        "low_confidence_word_ratio": round(low_ratio, 4) if low_ratio is not None else None,
        "weak_segment_count": weak_segments,
        "observed_segment_count": len(segments),
        "weak_segment_ratio": round(weak_segment_ratio, 4) if weak_segment_ratio is not None else None,
        "low_confidence_spans": [
            {key: word.get(key) for key in ("text", "t0", "t1", "probability")}
            for word in result.get("words", [])
            if (value := finite_number(word.get("probability"))) is not None and 0 <= value < 0.5
        ],
    }
