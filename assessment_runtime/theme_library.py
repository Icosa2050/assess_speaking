"""Persistent language/theme catalog for the Streamlit dashboard."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

DEFAULT_THEME_LIBRARY = {
    "it": {
        "label": "Italiano",
        "themes": [
            {"title": "Il mio ultimo viaggio all'estero", "level": "B1", "task_family": "travel_narrative"},
            {"title": "Una festa o un evento importante a cui ho partecipato", "level": "B1", "task_family": "personal_experience"},
            {"title": "Una giornata che ricordo molto bene", "level": "B1", "task_family": "personal_experience"},
            {"title": "I vantaggi e gli svantaggi del lavoro da casa", "level": "B2", "task_family": "opinion_monologue"},
            {"title": "Come il turismo è cambiato negli ultimi anni", "level": "B2", "task_family": "opinion_monologue"},
            {"title": "Un'esperienza che mi ha cambiato il modo di vedere le cose", "level": "B2", "task_family": "personal_experience"},
            {"title": "Il rapporto tra tecnologia e vita quotidiana", "level": "C1", "task_family": "opinion_monologue"},
            {"title": "Il ruolo dei social media nel dibattito pubblico", "level": "C1", "task_family": "opinion_monologue"},
            {"title": "Come conciliare libertà personale e responsabilità sociale", "level": "C1", "task_family": "opinion_monologue"},
        ],
    },
    "en": {
        "label": "English",
        "themes": [
            {"title": "My last trip abroad", "level": "B1", "task_family": "travel_narrative"},
            {"title": "An event or celebration I still remember well", "level": "B1", "task_family": "personal_experience"},
            {"title": "A typical day in my life", "level": "B1", "task_family": "personal_experience"},
            {"title": "The pros and cons of working from home", "level": "B2", "task_family": "opinion_monologue"},
            {"title": "How travel habits have changed in recent years", "level": "B2", "task_family": "opinion_monologue"},
            {"title": "An experience that changed the way I think", "level": "B2", "task_family": "personal_experience"},
            {"title": "Technology and the quality of daily life", "level": "C1", "task_family": "opinion_monologue"},
            {"title": "The influence of social media on public debate", "level": "C1", "task_family": "opinion_monologue"},
            {"title": "How to balance personal freedom with social responsibility", "level": "C1", "task_family": "opinion_monologue"},
        ],
    },
}


def theme_library_path(log_dir: Path) -> Path:
    return log_dir / "theme_library.json"


def dashboard_prefs_path(log_dir: Path) -> Path:
    return log_dir / "dashboard_prefs.json"


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


def load_dashboard_prefs(log_dir: Path) -> dict:
    path = dashboard_prefs_path(log_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_dashboard_prefs(log_dir: Path, prefs: dict) -> None:
    path = dashboard_prefs_path(log_dir)
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
