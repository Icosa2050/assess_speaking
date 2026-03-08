"""Deterministic metric extraction from transcript tokens and pauses."""

from __future__ import annotations

import re

FILLERS = {"eh", "ehm", "mmm", "cioè", "allora", "dunque", "tipo", "insomma"}
COHESION = {
    "inoltre",
    "per quanto riguarda",
    "tuttavia",
    "ciò nonostante",
    "in definitiva",
    "da un lato",
    "dall’altro",
    "a mio avviso",
    "tenuto conto di",
    "a quanto pare",
    "presumibilmente",
    "parrebbe che",
    "pertanto",
    "quindi",
    "invece",
    "comunque",
}


def metrics_from(words: list[dict], audio_feats: dict) -> dict:
    duration = audio_feats["duration_sec"]
    pause_total = sum(p[2] for p in audio_feats["pauses"])
    speaking_time = max(0.001, duration - pause_total)
    tokens = [re.sub(r"[^a-zà-ù’']", "", w["text"]) for w in words]
    tokens = [token for token in tokens if token]
    word_count = len(tokens)
    wpm = word_count / (speaking_time / 60.0)
    fillers = sum(1 for token in tokens if token in FILLERS)
    text = " " + re.sub(r"\s+", " ", " ".join(tokens)) + " "
    cohesion_hits = 0
    for marker in COHESION:
        parts = [re.escape(part) for part in marker.split()]
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
        cohesion_hits += len(re.findall(pattern, text, flags=re.IGNORECASE))
    rel_markers = len(re.findall(r"\bche\b|\bcui\b|\bnella quale\b|\bnei quali\b", text))
    cond_markers = len(re.findall(r"\bse\b|\bqualora\b", text))
    complexity = rel_markers + cond_markers
    return {
        "duration_sec": round(duration, 2),
        "pause_count": len(audio_feats["pauses"]),
        "pause_total_sec": round(pause_total, 2),
        "speaking_time_sec": round(speaking_time, 2),
        "word_count": word_count,
        "wpm": round(wpm, 1),
        "fillers": fillers,
        "cohesion_markers": cohesion_hits,
        "complexity_index": complexity,
    }
