from __future__ import annotations

from typing import Literal, TypedDict


class OpenRouterMetadata(TypedDict, total=False):
    http_referer: str
    app_title: str


class OllamaMetadata(TypedDict, total=False):
    deployment: Literal["local", "cloud"]


class LMStudioMetadata(TypedDict, total=False):
    deployment: Literal["local"]
    token_optional: bool


class OpenAICompatibleMetadata(TypedDict, total=False):
    deployment: Literal["custom"]

