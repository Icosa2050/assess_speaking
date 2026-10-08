"""Minimum input required for a real speaking review."""

MIN_REVIEW_SECONDS = 30.0


class RecordingTooShortError(ValueError):
    def __init__(self) -> None:
        super().__init__(
            "Record at least 30 seconds before requesting a review. "
            "Use connected sentences about the topic, then record again."
        )


def require_review_duration(duration_sec: float) -> None:
    if duration_sec < MIN_REVIEW_SECONDS:
        raise RecordingTooShortError()
