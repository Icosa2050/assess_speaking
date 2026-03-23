from __future__ import annotations

import pandas as pd
import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, render_page_intro, render_shell_summary
from app_shell.services import (
    NEW_LANGUAGE_OPTION,
    add_theme,
    language_codes,
    language_label,
    load_theme_library,
    save_theme_library,
    theme_option_label,
    validate_theme_submission,
)


def _safe_index(options: list[str], value: str, default: int = 0) -> int:
    if value in options:
        return options.index(value)
    return default


def _task_family_label(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    localized = t(f"task_family.{raw}")
    if localized.startswith("[") and localized.endswith("]"):
        return raw.replace("_", " ")
    return localized


state = configure_page("library", "nav.library", icon="📚")

render_page_intro("library.title", "library.body")
render_shell_summary(state)

pending_filter_language = st.session_state.pop("library_pending_filter_language", None)
if pending_filter_language:
    st.session_state["library_filter_language"] = pending_filter_language
pending_manage_language = st.session_state.pop("library_pending_manage_language", None)
if pending_manage_language:
    st.session_state["library_manage_language"] = pending_manage_language
pending_theme_title = st.session_state.pop("library_pending_theme_title", None)
if pending_theme_title is not None:
    st.session_state["library_theme_title"] = pending_theme_title
pending_language_code = st.session_state.pop("library_pending_language_code", None)
if pending_language_code is not None:
    st.session_state["library_language_code"] = pending_language_code
pending_language_label = st.session_state.pop("library_pending_language_label", None)
if pending_language_label is not None:
    st.session_state["library_language_label"] = pending_language_label

library = load_theme_library(state.prefs.log_dir)
codes = language_codes(library)

if "library_filter_language" in st.session_state and st.session_state["library_filter_language"] not in codes:
    del st.session_state["library_filter_language"]
selected_language = ""
if codes:
    filter_select_kwargs = {}
    if "library_filter_language" not in st.session_state:
        filter_select_kwargs["index"] = _safe_index(codes, state.draft.learning_language)
    selected_language = st.selectbox(
        t("library.language_filter"),
        options=codes,
        format_func=lambda code: language_label(library, code),
        key="library_filter_language",
        **filter_select_kwargs,
    )
else:
    st.info(t("library.empty_library"))

theme_rows = library.get(selected_language, {}).get("themes", [])
with st.container(border=True):
    st.subheader(t("library.existing_title"))
    if theme_rows:
        frame = pd.DataFrame(theme_rows).rename(
            columns={
                "title": t("library.table_title"),
                "level": t("library.table_level"),
                "task_family": t("library.table_task_family"),
            }
        )
        if t("library.table_task_family") in frame.columns:
            frame[t("library.table_task_family")] = frame[t("library.table_task_family")].map(
                _task_family_label
            )
        st.dataframe(frame, width="stretch", hide_index=True)
    elif selected_language:
        st.info(t("library.empty_language"))

errors = st.session_state.get("library_form_errors", {})
success_message = st.session_state.pop("library_success", "")

with st.container(border=True):
    st.subheader(t("library.add_title"))
    if success_message:
        st.success(success_message)
    manage_options = codes + [NEW_LANGUAGE_OPTION]
    if "library_manage_language" in st.session_state and st.session_state["library_manage_language"] not in manage_options:
        del st.session_state["library_manage_language"]
    manage_select_kwargs = {}
    if "library_manage_language" not in st.session_state:
        manage_select_kwargs["index"] = _safe_index(
            manage_options,
            selected_language or NEW_LANGUAGE_OPTION,
        )
    manage_mode = st.selectbox(
        t("library.manage_language"),
        options=manage_options,
        format_func=lambda code: t("library.new_language") if code == NEW_LANGUAGE_OPTION else language_label(library, code),
        key="library_manage_language",
        **manage_select_kwargs,
    )
    if manage_mode == NEW_LANGUAGE_OPTION:
        manage_language_code = st.text_input(t("library.language_code"), key="library_language_code")
        if errors.get("language_code"):
            st.error(t("library.error_language_code"))
        manage_language_label = st.text_input(t("library.language_label"), key="library_language_label")
        if errors.get("language_label"):
            st.error(t("library.error_language_label"))
    else:
        manage_language_code = manage_mode
        manage_language_label = language_label(library, manage_mode)
        st.caption(t("library.saving_under", code=manage_language_code, label=manage_language_label))
    new_theme_title = st.text_input(t("library.theme_title"), key="library_theme_title")
    if errors.get("theme_title"):
        st.error(t("library.error_theme_title"))
    new_theme_level = st.selectbox(t("library.theme_level"), options=["B1", "B2", "C1"], key="library_theme_level")
    new_theme_family = st.selectbox(
        t("library.theme_family"),
        options=[
            "travel_narrative",
            "personal_experience",
            "opinion_monologue",
            "picture_description",
            "free_monologue",
        ],
        format_func=_task_family_label,
        key="library_theme_family",
    )
    submitted = st.button(t("library.save_theme"), key="library_save_theme", width="stretch")

if submitted:
    normalized_language_code = manage_language_code.strip().lower()
    normalized_language_label = manage_language_label.strip()
    normalized_theme_title = new_theme_title.strip()
    validation_errors = validate_theme_submission(
        manage_mode=manage_mode,
        language_code=normalized_language_code,
        language_label_text=normalized_language_label,
        theme_title=normalized_theme_title,
    )
    if validation_errors:
        st.session_state["library_form_errors"] = validation_errors
        st.rerun()
    updated_library = add_theme(
        library,
        language_code=normalized_language_code,
        language_label=normalized_language_label,
        title=normalized_theme_title,
        level=new_theme_level,
        task_family=new_theme_family,
    )
    save_theme_library(state.prefs.log_dir, updated_library)
    st.session_state["library_form_errors"] = {}
    st.session_state["library_success"] = t("library.saved_success", theme=normalized_theme_title)
    st.session_state["library_pending_filter_language"] = normalized_language_code
    st.session_state["library_pending_manage_language"] = normalized_language_code
    st.session_state["library_pending_theme_title"] = ""
    st.session_state["library_pending_language_code"] = ""
    st.session_state["library_pending_language_label"] = ""
    st.rerun()
