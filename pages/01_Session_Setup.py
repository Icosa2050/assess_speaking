from __future__ import annotations

import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, go_to, render_page_intro, render_shell_summary
from app_shell.services import (
    build_practice_brief,
    language_codes,
    language_label,
    load_theme_library,
    save_state_preferences,
    theme_entry_id,
    themes_for_language_and_level,
)
from app_shell.state import CEFR_LEVELS, DURATION_OPTIONS, apply_setup
from app_shell.visual_system import render_checklist, render_detail_grid, render_inline_note, render_kicker, render_quote


def _safe_index(options: list[str], value: str, default: int = 0) -> int:
    if value in options:
        return options.index(value)
    return default


state = configure_page("setup", "nav.setup", icon="🧭")

render_page_intro("setup.title", "setup.body")
render_shell_summary(state)

library = load_theme_library(state.prefs.log_dir)
available_languages = language_codes(library)
if not available_languages:
    st.error(t("setup.no_languages"))
    st.stop()

default_language = state.draft.learning_language if state.draft.learning_language in available_languages else available_languages[0]
available_themes = themes_for_language_and_level(library, default_language, state.draft.cefr_level)
theme_labels = [item["title"] for item in available_themes]
current_theme_option = (
    st.session_state.get("setup_theme_select")
    or (state.draft.theme_label if state.draft.theme_label in theme_labels else "")
    or (theme_labels[0] if theme_labels else t("setup.custom_theme"))
)

form_col, preview_col = st.columns([1.15, 0.95], gap="large")

with form_col:
    with st.container(border=True):
        render_kicker(t("nav.setup"))
        speaker_id = st.text_input(
            t("setup.speaker_id"),
            value=state.draft.speaker_id,
            key="setup_speaker_id",
        ).strip()
        selected_language = st.selectbox(
            t("setup.learning_language"),
            options=available_languages,
            index=_safe_index(available_languages, default_language),
            format_func=lambda code: language_label(library, code),
            key="setup_learning_language",
        )
        selected_cefr = st.selectbox(
            t("setup.cefr"),
            options=list(CEFR_LEVELS),
            index=_safe_index(list(CEFR_LEVELS), state.draft.cefr_level),
            key="setup_cefr",
        )
        available_themes = themes_for_language_and_level(library, selected_language, selected_cefr)
        theme_labels = [item["title"] for item in available_themes]
        theme_select_options = theme_labels + [t("setup.custom_theme")]
        if "setup_theme_select" in st.session_state and st.session_state["setup_theme_select"] not in theme_select_options:
            del st.session_state["setup_theme_select"]
        active_theme_option = st.session_state.get("setup_theme_select", current_theme_option)
        if active_theme_option not in theme_select_options:
            active_theme_option = theme_labels[0] if theme_labels else t("setup.custom_theme")
        selected_theme_option = st.selectbox(
            t("setup.theme"),
            options=theme_select_options,
            index=_safe_index(theme_select_options, active_theme_option),
            key="setup_theme_select",
        )
        custom_theme = st.text_input(
            t("setup.custom_theme_label"),
            value=state.draft.theme_label if selected_theme_option == t("setup.custom_theme") else "",
            key="setup_custom_theme",
            disabled=selected_theme_option != t("setup.custom_theme"),
        ).strip()
        selected_duration = st.select_slider(
            t("setup.duration"),
            options=list(DURATION_OPTIONS),
            value=state.draft.duration_sec if state.draft.duration_sec in DURATION_OPTIONS else list(DURATION_OPTIONS)[0],
            key="setup_duration",
        )
        submitted = st.button(t("setup.continue"), key="setup_continue", width="stretch")

selected_theme_entry = next(
    (item for item in available_themes if item["title"] == selected_theme_option),
    None,
)
resolved_theme_label = custom_theme if selected_theme_option == t("setup.custom_theme") else (selected_theme_entry or {}).get("title", "")
resolved_task_family = (
    (selected_theme_entry or {}).get("task_family")
    or state.draft.task_family
    or "free_monologue"
)
brief = build_practice_brief(
    task_family=resolved_task_family,
    theme=resolved_theme_label,
    target_duration_sec=int(selected_duration),
    language_code=selected_language,
)
selected_language_label = language_label(library, selected_language)
task_family_label = t(f"task_family.{resolved_task_family}")
if task_family_label.startswith("["):
    task_family_label = resolved_task_family.replace("_", " ")

prompt_text = str(brief.get("prompt") or "")
success_focus = brief.get("success_focus") if isinstance(brief.get("success_focus"), list) else []

with preview_col:
    with st.container(border=True):
        render_kicker(t("setup.preview_title"))
        st.subheader(resolved_theme_label or t("setup.preview_title"))
        if resolved_theme_label and prompt_text:
            render_quote(prompt_text)
            if success_focus:
                st.caption(t("setup.success_focus_title"))
                render_checklist([str(item) for item in success_focus])
        else:
            render_quote(t("setup.preview_placeholder"), empty=True)

    with st.container(border=True):
        render_kicker(t("setup.selection_title"))
        render_detail_grid(
            [
                (t("setup.speaker_id"), speaker_id or "-"),
                (t("setup.learning_language"), selected_language_label),
                (t("setup.cefr"), selected_cefr),
                (t("setup.theme"), resolved_theme_label or "-"),
                (t("setup.duration"), f"{int(selected_duration)} s"),
                (t("history.task_family_name"), task_family_label),
            ]
        )
        render_inline_note(t("setup.flow_hint"))

if submitted:
    errors: list[str] = []
    if not speaker_id:
        errors.append(t("setup.error_speaker_id"))
    if not resolved_theme_label:
        errors.append(t("setup.error_theme"))
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    apply_setup(
        speaker_id=speaker_id,
        learning_language=selected_language,
        learning_language_label=selected_language_label,
        cefr_level=selected_cefr,
        theme_id=theme_entry_id(
            selected_theme_entry or {"title": resolved_theme_label, "level": selected_cefr}
        ),
        theme_label=resolved_theme_label,
        task_family=resolved_task_family,
        duration_sec=int(selected_duration),
        prompt_text=prompt_text,
    )
    save_state_preferences(state)
    go_to("pages/02_Speak.py", return_to="setup")
