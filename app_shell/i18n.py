from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import streamlit as st

from app_shell.state import APP_SHELL_STATE_KEY, DEFAULT_UI_LOCALE, SUPPORTED_UI_LOCALES

LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"


@lru_cache(maxsize=None)
def load_locale(locale: str) -> dict:
    path = LOCALES_DIR / f"{locale}.json"
    if not path.exists():
        path = LOCALES_DIR / f"{DEFAULT_UI_LOCALE}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def current_locale() -> str:
    state = st.session_state.get(APP_SHELL_STATE_KEY)
    if state is None:
        return DEFAULT_UI_LOCALE
    return state.prefs.ui_locale


def t(key: str, locale: str | None = None, **kwargs) -> str:
    locale = locale or current_locale()
    strings = load_locale(locale)
    value: object = strings
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return f"[{key}]"
        value = value[part]
    if not isinstance(value, str):
        return f"[{key}]"
    if kwargs:
        return value.format(**kwargs)
    return value


def flatten_keys(mapping: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.update(flatten_keys(value, prefix=full_key))
        else:
            keys.add(full_key)
    return keys


def locale_key_map() -> dict[str, set[str]]:
    return {locale: flatten_keys(load_locale(locale)) for locale in SUPPORTED_UI_LOCALES}
