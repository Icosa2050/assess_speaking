from __future__ import annotations

import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, go_to, render_page_intro, render_shell_summary
from app_shell.services import needs_runtime_setup
from app_shell.state import begin_new_session, get_app_state, has_setup, has_review, serialize_state

state = configure_page("home", "nav.home", icon="🎙️")

render_page_intro("home.title", "home.body")
render_shell_summary(state)

if needs_runtime_setup(state):
    with st.container(border=True):
        st.subheader(t("home.runtime_setup_title"))
        st.write(t("home.runtime_setup_body"))
        if st.button(t("home.runtime_setup_button"), key="home_runtime_setup", width="stretch"):
            go_to("pages/00_Setup.py", return_to="home")

with st.container(border=True):
    st.subheader(t("home.primary_title"))
    st.write(t("home.primary_body"))
    start_col, resume_col = st.columns(2)
    with start_col:
        if st.button(t("home.start_new"), key="home_start_new", width="stretch"):
            begin_new_session()
            go_to("pages/01_Session_Setup.py", return_to="home")
    with resume_col:
        if st.button(
            t("home.resume"),
            key="home_resume",
            width="stretch",
            disabled=not has_setup(state),
        ):
            target = "pages/03_Review.py" if has_review(state) else "pages/02_Speak.py"
            go_to(target, return_to="home")

with st.container(border=True):
    st.subheader(t("home.secondary_title"))
    st.write(t("home.secondary_body"))
    history_col, library_col, guide_col, settings_col = st.columns(4)
    with history_col:
        if st.button(t("nav.history"), key="home_history", width="stretch"):
            go_to("pages/04_History.py", return_to="home")
    with library_col:
        if st.button(t("nav.library"), key="home_library", width="stretch"):
            go_to("pages/05_Library.py", return_to="home")
    with guide_col:
        if st.button(t("nav.guide"), key="home_guide", width="stretch"):
            go_to("pages/07_Scoring_Guide.py", return_to="home")
    with settings_col:
        if st.button(t("nav.settings"), key="home_settings", width="stretch"):
            go_to("pages/06_Settings.py", return_to="home")

with st.expander(t("home.debug_title"), expanded=False):
    st.json(serialize_state())
