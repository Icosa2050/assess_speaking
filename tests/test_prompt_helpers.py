import json
import os
import wave
import tempfile
from pathlib import Path

import numpy as np
import unittest

from scripts import interactive_dashboard as dashboard


class PromptHelperTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {
            "id": "b2_remote_work",
            "audio_path": "/tmp/audio.wav",
            "response_seconds": 90,
            "max_playbacks": 1,
            "cefr_target": "B2",
        }

    def test_create_prompt_attempt_initialises_fields(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=100.0)
        self.assertEqual(attempt["id"], self.prompt["id"])
        self.assertEqual(attempt["start"], 100.0)
        self.assertEqual(attempt["deadline"], 190.0)
        self.assertEqual(attempt["plays_remaining"], 1)
        self.assertEqual(attempt["audio"], self.prompt["audio_path"])
        self.assertEqual(attempt["cefr"], "B2")
        self.assertEqual(attempt["chunks"], [])

    def test_remaining_time_and_decrement(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=10.0)
        remaining = dashboard.remaining_time(attempt, now=55.0)
        self.assertAlmostEqual(remaining, 45.0)
        self.assertTrue(dashboard.can_play_prompt(attempt))
        dashboard.decrement_playback(attempt)
        self.assertEqual(attempt["plays_remaining"], 0)
        self.assertFalse(dashboard.can_play_prompt(attempt))
        with self.assertRaises(ValueError):
            dashboard.decrement_playback(attempt)

    def test_append_and_write_audio(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=0.0)
        # Generate a short sine wave chunk
        sample_rate = 16000
        t = np.linspace(0, 0.01, int(sample_rate * 0.01), endpoint=False)
        data = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        dashboard.append_audio_bytes(attempt, pcm, sample_rate, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.wav")
            dashboard.write_attempt_audio(attempt, Path(out_path))
            with wave.open(out_path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), sample_rate)
                frames = wf.readframes(wf.getnframes())
                self.assertEqual(frames, pcm)

    def test_write_attempt_audio_no_chunks_raises(self):
        attempt = dashboard.create_prompt_attempt(self.prompt, now=0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "empty.wav"
            with self.assertRaises(ValueError):
                dashboard.write_attempt_audio(attempt, out_path)

    def test_parse_cli_json_variants(self):
        payload = {"metrics": {"wpm": 100}}
        json_body = json.dumps(payload)
        direct = dashboard.parse_cli_json(json_body)
        self.assertEqual(direct, payload)
        wrapped = dashboard.parse_cli_json("INFO\n" + json_body)
        self.assertEqual(wrapped, payload)

    def test_load_prompts_resolves_relative_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_path = Path(tmpdir) / "prompts.json"
            prompts_path.write_text('[{"id":"p","audio":"relative.wav","response_seconds":30,"max_playbacks":1,"cefr_target":"B1"}]')
            loaded = dashboard.load_prompts(prompts_path)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0]["audio_path"].endswith("relative.wav"))


if __name__ == "__main__":
    unittest.main()
