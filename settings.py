"""Runtime settings for speaking assessment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    provider: str = "openrouter"
    openrouter_model: str = "google/gemini-3.1-pro-preview"
    ollama_model: str = "llama3.1"
    expected_language: str = "it"
    asr_compute_type: str = "default"
    asr_fallback_compute_type: str | None = "int8"
    pause_threshold_offset_db: float = -10.0
    llm_timeout_sec: float = 30.0
    min_word_count: int = 5
    duration_pass_ratio: float = 0.8
    topic_fail_cap_score: float = 2.5

    @classmethod
    def from_env(cls) -> "Settings":
        asr_fallback = os.getenv("ASR_FALLBACK_COMPUTE_TYPE", cls.asr_fallback_compute_type or "")
        if asr_fallback.lower() in {"", "none", "null"}:
            fallback_compute_type = None
        else:
            fallback_compute_type = asr_fallback

        try:
            pause_threshold_offset_db = float(os.getenv("PAUSE_THRESHOLD_OFFSET_DB", str(cls.pause_threshold_offset_db)))
        except ValueError:
            pause_threshold_offset_db = cls.pause_threshold_offset_db
        try:
            llm_timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", str(cls.llm_timeout_sec)))
        except ValueError:
            llm_timeout_sec = cls.llm_timeout_sec
        try:
            min_word_count = int(os.getenv("MIN_WORD_COUNT", str(cls.min_word_count)))
        except ValueError:
            min_word_count = cls.min_word_count
        try:
            duration_pass_ratio = float(os.getenv("DURATION_PASS_RATIO", str(cls.duration_pass_ratio)))
        except ValueError:
            duration_pass_ratio = cls.duration_pass_ratio
        try:
            topic_fail_cap_score = float(os.getenv("TOPIC_FAIL_CAP_SCORE", str(cls.topic_fail_cap_score)))
        except ValueError:
            topic_fail_cap_score = cls.topic_fail_cap_score

        return cls(
            provider=os.getenv("ASSESS_PROVIDER", cls.provider),
            openrouter_model=os.getenv("OPENROUTER_MODEL", cls.openrouter_model),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            expected_language=os.getenv("EXPECTED_LANGUAGE", cls.expected_language),
            asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", cls.asr_compute_type),
            asr_fallback_compute_type=fallback_compute_type,
            pause_threshold_offset_db=pause_threshold_offset_db,
            llm_timeout_sec=llm_timeout_sec,
            min_word_count=min_word_count,
            duration_pass_ratio=duration_pass_ratio,
            topic_fail_cap_score=topic_fail_cap_score,
        )
