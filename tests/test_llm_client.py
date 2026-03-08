import unittest
from unittest import mock
import socket

import llm_client
from llm_client import LLMClientError


VALID_JSON = """
{
  "fluency": 4,
  "cohesion": 4,
  "accuracy": 3,
  "range": 4,
  "overall": 4,
  "comments_fluency": "ok",
  "comments_cohesion": "ok",
  "comments_accuracy": "ok",
  "comments_range": "ok",
  "overall_comment": "ok",
  "on_topic": true
}
"""


class LlmClientTests(unittest.TestCase):
    def test_extract_json_object_accepts_fenced_json(self):
        payload = f"```json\n{VALID_JSON}\n```"
        parsed = llm_client.extract_json_object(payload)
        self.assertEqual(parsed["overall"], 4)

    def test_extract_json_object_handles_trailing_text_with_braces(self):
        payload = f"prefix {VALID_JSON} suffix }}"
        parsed = llm_client.extract_json_object(payload)
        self.assertEqual(parsed["comments_fluency"], "ok")

    @mock.patch("llm_client._chat_completion", return_value=VALID_JSON)
    def test_generate_rubric_returns_valid_rubric(self, _mock_chat):
        rubric, raw = llm_client.generate_rubric(provider="ollama", model="x", prompt="p")
        self.assertEqual(rubric.overall, 4)
        self.assertIn('"overall": 4', raw)

    def test_generate_rubric_retries_then_fails(self):
        with mock.patch("llm_client._chat_completion", return_value='{"overall": 4}'):
            with self.assertRaises(LLMClientError):
                llm_client.generate_rubric(provider="ollama", model="x", prompt="p", max_validation_retries=1)

    @mock.patch("llm_client.request.urlopen", side_effect=socket.timeout("slow"))
    def test_post_json_maps_timeout_to_llm_client_error(self, _mock_urlopen):
        with self.assertRaises(LLMClientError) as ctx:
            llm_client._post_json(
                "https://example.invalid",
                {"x": 1},
                {"Content-Type": "application/json"},
                timeout_sec=3.0,
            )
        self.assertIn("timed out", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
