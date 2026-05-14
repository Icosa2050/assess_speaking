"""Persistent language/theme catalog and workspace preferences."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

_SESSION_SETUP_CONTENT_PATH = Path(__file__).with_name("data") / "session_setup_content.json"
logger = logging.getLogger(__name__)
_REQUIRED_SESSION_SETUP_KEYS = ("default_theme_library", "practice_brief_templates")
_FALLBACK_SESSION_SETUP_CONTENT = {
    "default_theme_library": {
        "en": {"label": "English", "themes": []},
        "it": {"label": "Italiano", "themes": []},
    },
    "practice_brief_templates": {
        "en": {
            "travel_narrative": "Speak about '{theme}' in a clear sequence.",
            "personal_experience": "Explain '{theme}' as a personal experience.",
            "opinion_monologue": "Give your opinion about '{theme}'.",
            "free_monologue": "Speak in English about '{theme}'.",
            "picture_description": "Describe '{theme}'.",
            "default_duration_minutes": "Aim to speak for about {minutes} minutes.",
            "default_duration_seconds": "Aim to speak for about {seconds} seconds.",
            "success_focus": [],
        }
    },
}


def _load_session_setup_content(path: Path = _SESSION_SETUP_CONTENT_PATH) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load session setup content from %s: %s", path, exc)
        return deepcopy(_FALLBACK_SESSION_SETUP_CONTENT)
    if not isinstance(payload, dict):
        logger.warning("Session setup content from %s is not a JSON object.", path)
        return deepcopy(_FALLBACK_SESSION_SETUP_CONTENT)
    normalized = deepcopy(_FALLBACK_SESSION_SETUP_CONTENT)
    for key in _REQUIRED_SESSION_SETUP_KEYS:
        if isinstance(payload.get(key), dict):
            normalized[key] = payload[key]
        else:
            logger.warning("Session setup content from %s is missing key %s.", path, key)
    return normalized


_SESSION_SETUP_CONTENT = _load_session_setup_content()
DEFAULT_THEME_LIBRARY = deepcopy(_SESSION_SETUP_CONTENT["default_theme_library"])


def theme_library_path(log_dir: Path) -> Path:
    return log_dir / "theme_library.json"


def workspace_prefs_path(log_dir: Path) -> Path:
    return log_dir / "workspace_prefs.json"


def practice_brief_templates() -> dict:
    return deepcopy(_SESSION_SETUP_CONTENT["practice_brief_templates"])


def _normalize_theme_library(data: dict | None) -> dict:
    normalized = deepcopy(DEFAULT_THEME_LIBRARY)
    if not isinstance(data, dict):
        return normalized
    for language_code, payload in data.items():
        if not isinstance(payload, dict):
            continue
        label = str(payload.get("label") or language_code).strip() or language_code
        themes = normalized.setdefault(language_code, {"label": label, "themes": []})["themes"]
        normalized[language_code]["label"] = label
        for theme in payload.get("themes") or []:
            if not isinstance(theme, dict):
                continue
            title = str(theme.get("title") or "").strip()
            if not title:
                continue
            entry = {
                "title": title,
                "level": str(theme.get("level") or "B1").upper(),
                "task_family": str(theme.get("task_family") or "free_monologue"),
            }
            if entry not in themes:
                themes.append(entry)
    return normalized


def load_theme_library(log_dir: Path) -> dict:
    path = theme_library_path(log_dir)
    if not path.exists():
        return deepcopy(DEFAULT_THEME_LIBRARY)
    try:
        return _normalize_theme_library(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return deepcopy(DEFAULT_THEME_LIBRARY)


def save_theme_library(log_dir: Path, library: dict) -> None:
    path = theme_library_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_theme_library(library), ensure_ascii=False, indent=2), encoding="utf-8")


def add_theme(
    library: dict,
    *,
    language_code: str,
    language_label: str,
    title: str,
    level: str,
    task_family: str,
) -> dict:
    updated = _normalize_theme_library(library)
    code = language_code.strip().lower()
    label = language_label.strip() or code
    title = title.strip()
    if not code or not title:
        raise ValueError("Language code and theme title are required.")
    payload = updated.setdefault(code, {"label": label, "themes": []})
    payload["label"] = label
    entry = {"title": title, "level": level.upper(), "task_family": task_family}
    if entry not in payload["themes"]:
        payload["themes"].append(entry)
    return updated


def load_workspace_prefs(log_dir: Path) -> dict:
    path = workspace_prefs_path(log_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_workspace_prefs(log_dir: Path, prefs: dict) -> None:
    path = workspace_prefs_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")


def language_options(library: dict) -> list[str]:
    return sorted(library)


def language_label(library: dict, language_code: str) -> str:
    payload = library.get(language_code) or {}
    return str(payload.get("label") or language_code)


def themes_for_language_and_level(library: dict, language_code: str, level: str) -> list[dict]:
    payload = library.get(language_code) or {}
    themes = payload.get("themes") or []
    return [theme for theme in themes if str(theme.get("level") or "").upper() == level.upper()]
