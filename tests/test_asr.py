import unittest
from pathlib import Path
from unittest import mock

import asr


class AsrTests(unittest.TestCase):
    def test_transcribe_uses_fallback_compute_type(self):
        class DummyWord:
            def __init__(self, word, start, end):
                self.word = word
                self.start = start
                self.end = end

        class DummySegment:
            def __init__(self, text, words):
                self.text = text
                self.words = words

        class DummyInfo:
            language = "it"
            language_probability = 0.95

        class DummyModel:
            calls = []

            def __init__(self, model_size, compute_type="default"):
                self.calls.append(compute_type)
                if compute_type == "default":
                    raise RuntimeError("unsupported")

            def transcribe(self, path, **kwargs):
                return [DummySegment(" Ciao ", [DummyWord("Ciao", 0.0, 0.5)])], DummyInfo()

        with mock.patch.object(asr, "WhisperModel", DummyModel):
            result = asr.transcribe(Path("sample.wav"), compute_type="default", fallback_compute_type="int8")
        self.assertEqual(result["compute_type_used"], "int8")
        self.assertTrue(result["compute_fallback_used"])


if __name__ == "__main__":
    unittest.main()
