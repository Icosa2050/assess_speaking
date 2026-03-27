"""LLM provider access for OpenRouter, Ollama, LM Studio, and compatible APIs."""

from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime
from typing import Any
from urllib import error, request

from assess_core.schemas import CoachingSummary, RubricResult, SchemaValidationError
from app_shell.runtime_providers import default_base_url, normalize_provider, resolved_base_url, runtime_base_url, service_base_url


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
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"Invalid JSON response: {raw[:300]}") from exc
    if not isinstance(parsed, dict):
        typename = type(parsed).__name__
        raise LLMClientError(f"Unexpected JSON response type: expected object, got {typename}")
    return parsed


def _get_json(url: str, headers: dict[str, str], timeout_sec: float) -> dict[str, Any]:
    req = request.Request(url, headers=headers, method="GET")
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
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"Invalid JSON response: {raw[:300]}") from exc
    if not isinstance(parsed, dict):
        typename = type(parsed).__name__
        raise LLMClientError(f"Unexpected JSON response type: expected object, got {typename}")
    return parsed


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


def _coerce_message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    fragments: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item:
                fragments.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("text", "content"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate:
                fragments.append(candidate)
                break
    return "".join(fragments)


def _extract_assistant_message_text(result: dict[str, Any]) -> str:
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except Exception as exc:
        raise LLMClientError(f"Unexpected chat completion payload: {result}") from exc
    if not isinstance(message, dict):
        raise LLMClientError(f"Unexpected chat completion payload: {result}")
    for key in ("content", "refusal", "reasoning", "reasoning_content"):
        candidate = _coerce_message_text(message.get(key))
        if candidate.strip():
            return candidate
    raise LLMClientError(f"Chat completion returned no assistant text: {result}")


def _chat_completion(
    provider: str,
    model: str,
    prompt: str,
    timeout_sec: float,
    openrouter_api_key: str | None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
    extra_payload: dict[str, Any] | None = None,
    require_json_object: bool = False,
) -> str:
    normalized_provider = normalize_provider(provider)
    headers = {"Content-Type": "application/json"}
    resolved_url = runtime_base_url(normalized_provider, base_url)
    resolved_api_key = api_key or openrouter_api_key
    if normalized_provider == "openrouter":
        if not resolved_api_key:
            raise LLMClientError("OPENROUTER_API_KEY is not set.")
        headers["Authorization"] = f"Bearer {resolved_api_key}"
        referer = openrouter_http_referer or os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8503")
        title = openrouter_app_title or os.getenv("OPENROUTER_APP_TITLE", "Speaking Studio")
        headers["HTTP-Referer"] = referer
        headers["X-OpenRouter-Title"] = title
        headers["X-Title"] = title
    elif normalized_provider in {"ollama", "lmstudio", "openai_compatible"}:
        if resolved_api_key:
            headers["Authorization"] = f"Bearer {resolved_api_key}"
    else:
        raise LLMClientError(f"Unsupported provider '{provider}'.")

    url = f"{resolved_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if extra_payload:
        payload.update(extra_payload)
    if normalized_provider == "openrouter" and require_json_object:
        payload["response_format"] = {"type": "json_object"}
    try:
        result = _post_json(url, payload, headers, timeout_sec)
    except LLMClientError as exc:
        if normalized_provider != "openrouter":
            raise
        message = str(exc).lower()
        if (not require_json_object) or ("response_format" not in message and "json_object" not in message):
            raise
        fallback_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        if extra_payload:
            fallback_payload.update(extra_payload)
        result = _post_json(url, fallback_payload, headers, timeout_sec)
    return _extract_assistant_message_text(result)


def generate_rubric(
    provider: str,
    model: str,
    prompt: str,
    timeout_sec: float = 30.0,
    openrouter_api_key: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
    max_validation_retries: int = 1,
) -> tuple[RubricResult, str]:
    attempt_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(max_validation_retries + 1):
        raw = _chat_completion(
            provider,
            model,
            attempt_prompt,
            timeout_sec,
            openrouter_api_key,
            base_url=base_url,
            api_key=api_key,
            openrouter_http_referer=openrouter_http_referer,
            openrouter_app_title=openrouter_app_title,
            require_json_object=True,
        )
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
    base_url: str | None = None,
    api_key: str | None = None,
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
    max_validation_retries: int = 1,
) -> tuple[CoachingSummary, str]:
    attempt_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(max_validation_retries + 1):
        raw = _chat_completion(
            provider,
            model,
            attempt_prompt,
            timeout_sec,
            openrouter_api_key,
            base_url=base_url,
            api_key=api_key,
            openrouter_http_referer=openrouter_http_referer,
            openrouter_app_title=openrouter_app_title,
            require_json_object=True,
        )
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
    try:
        return json.dumps(list_models(provider="ollama", timeout_sec=timeout_sec), ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": "ollama_tags_failed", "detail": str(exc)})


def list_models(
    *,
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: float = 10.0,
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
) -> dict[str, Any]:
    normalized_provider = normalize_provider(provider)
    resolved_url = runtime_base_url(normalized_provider, base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif normalized_provider == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
    if normalized_provider == "openrouter":
        headers["HTTP-Referer"] = openrouter_http_referer or os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8503")
        title = openrouter_app_title or os.getenv("OPENROUTER_APP_TITLE", "Speaking Studio")
        headers["X-OpenRouter-Title"] = title
        headers["X-Title"] = title
    return _get_json(f"{resolved_url}/models", headers, timeout_sec)


def health_check(
    *,
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: float = 10.0,
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
) -> dict[str, Any]:
    normalized_provider = normalize_provider(provider)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif normalized_provider == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
    if normalized_provider == "openrouter":
        headers["HTTP-Referer"] = openrouter_http_referer or os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8503")
        title = openrouter_app_title or os.getenv("OPENROUTER_APP_TITLE", "Speaking Studio")
        headers["X-OpenRouter-Title"] = title
        headers["X-Title"] = title
    if normalized_provider == "ollama":
        endpoint = f"{service_base_url(normalized_provider, base_url)}/api/tags"
    else:
        endpoint = f"{runtime_base_url(normalized_provider, base_url)}/models"
    payload = _get_json(endpoint, headers, timeout_sec)
    return {
        "provider": normalized_provider,
        "endpoint": endpoint,
        "payload": payload,
    }


def test_connection(
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: float = 10.0,
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
) -> dict[str, Any]:
    normalized_provider = normalize_provider(provider)
    content = _chat_completion(
        normalized_provider,
        model,
        "Reply with OK.",
        timeout_sec,
        api_key if normalized_provider == "openrouter" else None,
        base_url=base_url,
        api_key=api_key,
        openrouter_http_referer=openrouter_http_referer,
        openrouter_app_title=openrouter_app_title,
        extra_payload={"temperature": 0, "max_tokens": 8},
    )
    if not isinstance(content, str):
        typename = type(content).__name__
        raise LLMClientError(f"Unexpected chat completion content type: expected text, got {typename}")
    return {
        "provider": normalized_provider,
        "base_url": runtime_base_url(normalized_provider, base_url),
        "model": model,
        "ok": True,
        "tested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "content_preview": content[:120],
    }
