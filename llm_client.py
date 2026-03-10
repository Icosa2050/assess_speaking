"""LLM provider access for OpenRouter and local Ollama."""

from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib import error, request

from schemas import CoachingSummary, RubricResult, SchemaValidationError


class LLMClientError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: float) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMClientError(f"Request timed out after {timeout_sec:.1f}s") from exc
    except error.URLError as exc:
        raise LLMClientError(f"Network error: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"Invalid JSON response: {raw[:300]}") from exc


def extract_json_object(content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise SchemaValidationError("LLM content is neither string nor JSON object")
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    if not content:
        raise SchemaValidationError("LLM content is empty")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(content):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = content[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None
    raise SchemaValidationError("Could not find valid JSON object in LLM content")


def _chat_completion(
    provider: str,
    model: str,
    prompt: str,
    timeout_sec: float,
    openrouter_api_key: str | None,
) -> str:
    headers = {"Content-Type": "application/json"}
    if provider == "openrouter":
        if not openrouter_api_key:
            raise LLMClientError("OPENROUTER_API_KEY is not set.")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers["Authorization"] = f"Bearer {openrouter_api_key}"
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "https://local.assess-speaking")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "assess_speaking")
    elif provider == "ollama":
        url = "http://localhost:11434/v1/chat/completions"
    else:
        raise LLMClientError(f"Unsupported provider '{provider}'.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if provider == "openrouter":
        payload["response_format"] = {"type": "json_object"}
    try:
        result = _post_json(url, payload, headers, timeout_sec)
    except LLMClientError as exc:
        if provider != "openrouter":
            raise
        message = str(exc).lower()
        if "response_format" not in message and "json_object" not in message:
            raise
        fallback_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        result = _post_json(url, fallback_payload, headers, timeout_sec)
    try:
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMClientError(f"Unexpected chat completion payload: {result}") from exc


def generate_rubric(
    provider: str,
    model: str,
    prompt: str,
    timeout_sec: float = 30.0,
    openrouter_api_key: str | None = None,
    max_validation_retries: int = 1,
) -> tuple[RubricResult, str]:
    attempt_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(max_validation_retries + 1):
        raw = _chat_completion(provider, model, attempt_prompt, timeout_sec, openrouter_api_key)
        try:
            payload = extract_json_object(raw)
            rubric = RubricResult.from_dict(payload)
            return rubric, raw
        except (SchemaValidationError, LLMClientError) as exc:
            last_error = exc
            if attempt < max_validation_retries:
                attempt_prompt = (
                    prompt
                    + "\n\nATTENZIONE: la risposta precedente non rispettava lo schema JSON."
                    + f"\nErrore: {exc}\nRispondi SOLO con JSON valido nello schema richiesto."
                )
    raise LLMClientError(f"Failed to produce valid rubric output: {last_error}")


def generate_coaching_summary(
    provider: str,
    model: str,
    prompt: str,
    timeout_sec: float = 30.0,
    openrouter_api_key: str | None = None,
    max_validation_retries: int = 1,
) -> tuple[CoachingSummary, str]:
    attempt_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(max_validation_retries + 1):
        raw = _chat_completion(provider, model, attempt_prompt, timeout_sec, openrouter_api_key)
        try:
            payload = extract_json_object(raw)
            coaching = CoachingSummary.from_dict(payload)
            return coaching, raw
        except (SchemaValidationError, LLMClientError) as exc:
            last_error = exc
            if attempt < max_validation_retries:
                attempt_prompt = (
                    prompt
                    + "\n\nATTENZIONE: la risposta precedente non rispettava lo schema JSON."
                    + f"\nErrore: {exc}\nRispondi SOLO con JSON valido nello schema richiesto."
                )
    raise LLMClientError(f"Failed to produce valid coaching output: {last_error}")


def list_ollama_models(timeout_sec: float = 10.0) -> str:
    req = request.Request("http://localhost:11434/api/tags", method="GET")
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            return response.read().decode("utf-8")
    except Exception as exc:
        return json.dumps({"error": "ollama_tags_failed", "detail": str(exc)})
