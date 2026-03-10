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
  "on_topic": true,
  "topic_relevance_score": 4,
  "language_ok": true,
  "recurring_grammar_errors": [],
  "coherence_issues": [],
  "lexical_gaps": [],
  "evidence_quotes": ["ok"],
  "confidence": "medium"
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

    @mock.patch("llm_client._post_json")
    def test_chat_completion_openrouter_requests_json_object(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": VALID_JSON}}]}
        llm_client._chat_completion(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            prompt="p",
            timeout_sec=3.0,
            openrouter_api_key="key",
        )
        payload = mock_post.call_args.args[1]
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    @mock.patch("llm_client._post_json")
    def test_chat_completion_openrouter_falls_back_without_response_format(self, mock_post):
        mock_post.side_effect = [
            LLMClientError("HTTP 400: response_format is not supported"),
            {"choices": [{"message": {"content": VALID_JSON}}]},
        ]
        result = llm_client._chat_completion(
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            prompt="p",
            timeout_sec=3.0,
            openrouter_api_key="key",
        )
        self.assertIn('"overall"', result)
        first_payload = mock_post.call_args_list[0].args[1]
        second_payload = mock_post.call_args_list[1].args[1]
        self.assertIn("response_format", first_payload)
        self.assertNotIn("response_format", second_payload)

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
