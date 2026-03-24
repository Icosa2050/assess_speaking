from __future__ import annotations

import os

import streamlit as st
from streamlit.errors import StreamlitAPIException

from app_shell.i18n import t
from app_shell.services import hydrate_state_from_storage, load_dashboard_prefs, resolve_log_dir
from app_shell.state import (
    APP_NAME,
    AppShellState,
    DEFAULT_UI_LOCALE,
    SUPPORTED_UI_LOCALES,
    get_app_state,
    set_current_page,
    set_return_to,
)
from app_shell.visual_system import inject_visual_system


def resolve_page_title_locale(log_dir: str | os.PathLike[str] | None = None) -> str:
    resolved_log_dir = resolve_log_dir(log_dir)
    prefs = load_dashboard_prefs(resolved_log_dir)
    nested_log_dir = str(prefs.get("log_dir") or "").strip()
    if nested_log_dir:
        nested_resolved = resolve_log_dir(nested_log_dir)
        if nested_resolved != resolved_log_dir:
            nested_prefs = load_dashboard_prefs(nested_resolved)
            nested_locale = str(nested_prefs.get("ui_locale") or "").strip().lower()
            if nested_locale in SUPPORTED_UI_LOCALES:
                return nested_locale
    locale = str(prefs.get("ui_locale") or "").strip().lower()
    if locale in SUPPORTED_UI_LOCALES:
        return locale
    return DEFAULT_UI_LOCALE


def configure_page(page_id: str, title_key: str, *, icon: str) -> AppShellState:
    page_title_locale = resolve_page_title_locale()
    st.set_page_config(
        page_title=f"{APP_NAME} · {t(title_key, locale=page_title_locale)}",
        page_icon=icon,
        layout="wide",
    )
    inject_visual_system()
    state = set_current_page(page_id)
    return hydrate_state_from_storage(state)


def render_page_intro(title_key: str, body_key: str | None = None) -> None:
    st.title(t(title_key))
    if body_key:
        st.caption(t(body_key))


def render_shell_summary(state: AppShellState) -> None:
    st.caption(
        t(
            "common.session_summary",
            session_id=state.draft.session_id,
            ui_locale=state.prefs.ui_locale,
            learning_language=state.draft.learning_language_label,
            cefr=state.draft.cefr_level,
            speaker_id=state.draft.speaker_id or "—",
        )
    )


def go_to(page_path: str, *, return_to: str | None = None) -> None:
    state = get_app_state()
    set_return_to(return_to or state.nav.current_page)
    if os.environ.get("APP_SHELL_SKIP_BOOTSTRAP") == "1":
        st.session_state["_next_page"] = page_path
        st.stop()
    candidates = [page_path]
    if page_path.startswith("pages/"):
        candidates.append(page_path.split("/", 1)[1])
    elif page_path.endswith(".py") and page_path != "streamlit_app.py":
        candidates.append(f"pages/{page_path}")
    last_error: StreamlitAPIException | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            st.switch_page(candidate)
            return
        except StreamlitAPIException as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def render_guard(message_key: str, action_label_key: str, target_page: str) -> None:
    st.warning(t(message_key))
    if st.button(t(action_label_key), key=f"guard::{target_page}"):
        go_to(target_page)
    st.stop()
