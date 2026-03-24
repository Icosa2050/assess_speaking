import unittest
from unittest import mock
import socket

from assessment_runtime import llm_client
from assessment_runtime.llm_client import LLMClientError


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

    @mock.patch("assessment_runtime.llm_client._chat_completion", return_value=VALID_JSON)
    def test_generate_rubric_returns_valid_rubric(self, _mock_chat):
        rubric, raw = llm_client.generate_rubric(provider="ollama", model="x", prompt="p")
        self.assertEqual(rubric.overall, 4)
        self.assertIn('"overall": 4', raw)

    @mock.patch("assessment_runtime.llm_client._post_json")
    def test_chat_completion_openrouter_requests_json_object(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": VALID_JSON}}]}
        with mock.patch.dict(
            llm_client.os.environ,
            {
                "OPENROUTER_HTTP_REFERER": "http://localhost:8503",
                "OPENROUTER_APP_TITLE": "Speaking Studio",
            },
            clear=False,
        ):
            llm_client._chat_completion(
                provider="openrouter",
                model="google/gemini-3.1-pro-preview",
                prompt="p",
                timeout_sec=3.0,
                openrouter_api_key="key",
            )
        payload = mock_post.call_args.args[1]
        headers = mock_post.call_args.args[2]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(headers["Authorization"], "Bearer key")
        self.assertEqual(headers["HTTP-Referer"], "http://localhost:8503")
        self.assertEqual(headers["X-OpenRouter-Title"], "Speaking Studio")

    @mock.patch("assessment_runtime.llm_client._post_json")
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
        with mock.patch("assessment_runtime.llm_client._chat_completion", return_value='{"overall": 4}'):
            with self.assertRaises(LLMClientError):
                llm_client.generate_rubric(provider="ollama", model="x", prompt="p", max_validation_retries=1)

    @mock.patch("assessment_runtime.llm_client.request.urlopen", side_effect=socket.timeout("slow"))
    def test_post_json_maps_timeout_to_llm_client_error(self, _mock_urlopen):
        with self.assertRaises(LLMClientError) as ctx:
            llm_client._post_json(
                "https://example.invalid",
                {"x": 1},
                {"Content-Type": "application/json"},
                timeout_sec=3.0,
            )
        self.assertIn("timed out", str(ctx.exception))

    @mock.patch("assessment_runtime.llm_client._get_json")
    def test_list_models_supports_lmstudio_with_optional_bearer_token(self, mock_get):
        mock_get.return_value = {"data": [{"id": "qwen2.5"}]}
        payload = llm_client.list_models(
            provider="lmstudio",
            base_url="http://localhost:1234/v1",
            api_key="token-123",
        )
        self.assertEqual(payload["data"][0]["id"], "qwen2.5")
        self.assertEqual(mock_get.call_args.args[0], "http://localhost:1234/v1/models")
        headers = mock_get.call_args.args[1]
        self.assertEqual(headers["Authorization"], "Bearer token-123")

    @mock.patch("assessment_runtime.llm_client._get_json")
    def test_health_check_uses_native_ollama_tags_endpoint(self, mock_get):
        mock_get.return_value = {"models": [{"name": "llama3"}]}
        payload = llm_client.health_check(
            provider="ollama",
            base_url="http://localhost:11434",
        )
        self.assertEqual(payload["endpoint"], "http://localhost:11434/api/tags")
        self.assertEqual(payload["payload"]["models"][0]["name"], "llama3")
        self.assertEqual(mock_get.call_args.args[0], "http://localhost:11434/api/tags")

    @mock.patch("assessment_runtime.llm_client._chat_completion", return_value='{"ok": true}')
    def test_test_connection_reports_preview(self, mock_chat):
        payload = llm_client.test_connection(
            provider="openai_compatible",
            model="local-model",
            base_url="http://localhost:9999/v1",
            api_key="abc",
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["base_url"], "http://localhost:9999/v1")
        self.assertIn('"ok": true', payload["content_preview"])
        self.assertEqual(mock_chat.call_args.kwargs["api_key"], "abc")


if __name__ == "__main__":
    unittest.main()
