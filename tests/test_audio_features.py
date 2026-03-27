import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from assessment_runtime import audio_features


class _FakeSound:
    def __init__(self, total_duration: float):
        self._total_duration = total_duration
        self.intensity = object()

    def to_intensity(self):
        return self.intensity

    def get_total_duration(self):
        return self._total_duration


def _build_audio_patches(total_duration: float, value_at_time, *, mean: float = 10.0):
    sound = _FakeSound(total_duration)
    sound_ctor = mock.Mock(return_value=sound)

    def fake_call(target, method, *args):
        if method == "Get mean":
            return mean
        if method == "Get value at time":
            return value_at_time(args[0])
        raise AssertionError(f"Unexpected Praat call: {method}")

    return (
        sound_ctor,
        mock.patch.object(audio_features, "parselmouth", SimpleNamespace(Sound=sound_ctor)),
        mock.patch.object(audio_features, "call", side_effect=fake_call),
    )


class AudioFeaturesTests(unittest.TestCase):
    def test_load_audio_features_detects_middle_and_trailing_pauses(self):
        def value_at_time(t: float) -> float:
            if 0.20 <= t < 0.60:
                return -1.0
            if t >= 0.69:
                return -1.0
            return 1.0

        sound_ctor, parselmouth_patch, call_patch = _build_audio_patches(1.0, value_at_time)

        with parselmouth_patch, call_patch:
            result = audio_features.load_audio_features(Path("sample.wav"))

        sound_ctor.assert_called_once_with("sample.wav")
        self.assertEqual(result["duration_sec"], 1.0)
        self.assertEqual(len(result["pauses"]), 2)

        first_pause, second_pause = result["pauses"]
        self.assertAlmostEqual(first_pause[0], 0.20, places=2)
        self.assertAlmostEqual(first_pause[1], 0.60, places=2)
        self.assertAlmostEqual(first_pause[2], 0.40, places=2)
        self.assertAlmostEqual(second_pause[0], 0.69, places=2)
        self.assertAlmostEqual(second_pause[1], 1.00, places=2)
        self.assertAlmostEqual(second_pause[2], 0.31, places=2)

    def test_load_audio_features_treats_nan_as_silence_and_ignores_short_pause(self):
        def value_at_time(t: float) -> float:
            if 0.10 <= t < 0.45:
                return math.nan
            if 0.50 <= t < 0.70:
                return -1.0
            return 1.0

        _, parselmouth_patch, call_patch = _build_audio_patches(0.80, value_at_time)

        with parselmouth_patch, call_patch:
            result = audio_features.load_audio_features(Path("nan_pause.wav"))

        self.assertEqual(len(result["pauses"]), 1)
        start, end, duration = result["pauses"][0]
        self.assertAlmostEqual(start, 0.11, places=2)
        self.assertAlmostEqual(end, 0.45, places=2)
        self.assertAlmostEqual(duration, 0.34, places=2)


if __name__ == "__main__":
    unittest.main()
