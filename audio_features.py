"""Audio pause feature extraction."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import parselmouth  # type: ignore
    from parselmouth.praat import call  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime for CLI ergonomics
    parselmouth = None  # type: ignore
    call = None  # type: ignore


def load_audio_features(wav_path: Path, threshold_offset_db: float = -10.0) -> dict:
    if parselmouth is None or call is None:
        raise RuntimeError(
            "praat-parselmouth is not available. Install dependencies via `python -m pip install -r requirements.txt`."
        )
    snd = parselmouth.Sound(str(wav_path))
    intensity = snd.to_intensity()
    threshold = call(intensity, "Get mean", 0, 0) + threshold_offset_db
    step = 0.01
    t = 0.0
    pauses = []
    in_pause = False
    pause_start = 0.0
    total_duration = snd.get_total_duration()
    while t < total_duration:
        value = call(intensity, "Get value at time", t, "Cubic")
        if math.isnan(value) or value < threshold:
            if not in_pause:
                in_pause = True
                pause_start = t
        else:
            if in_pause:
                duration = t - pause_start
                if duration >= 0.3:
                    pauses.append((pause_start, t, duration))
                in_pause = False
        t += step
    if in_pause:
        duration = total_duration - pause_start
        if duration >= 0.3:
            pauses.append((pause_start, total_duration, duration))
    return {"duration_sec": total_duration, "pauses": pauses}
