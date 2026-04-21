from __future__ import annotations

import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, go_to, render_guard, render_page_intro, render_shell_summary
from app_shell.review_components import render_report_panels
from app_shell.services import poll_assessment_request, review_summary
from app_shell.state import RecordingStatus, apply_review_payload, clear_attempt, get_app_state, has_review, set_recording_error, set_recording_job


def _apply_review_payload_in_place(payload: dict | None) -> None:
    summary = review_summary(payload)
    apply_review_payload(
        payload=payload or {},
        report_id=str(summary.get("report_id") or ""),
        transcript=str(summary.get("transcript") or ""),
        score_overall=summary.get("score_overall"),
        band=str(summary.get("band") or ""),
        summary=str(summary.get("coach_summary") or ""),
    )

state = configure_page("review", "nav.review", icon="🧾")

if not has_review(state) and state.recording.job.assessment_id:
    job_state, payload, error = poll_assessment_request(
        state.recording.job.assessment_id,
        log_dir=state.prefs.log_dir,
    )
    if job_state is not None:
        if error and not job_state.error:
            job_state.error = error
        set_recording_job(job_state)
        state = get_app_state()
    elif error:
        set_recording_error(error)
        state = get_app_state()
    if payload is not None:
        _apply_review_payload_in_place(payload)
        state = get_app_state()

if not has_review(state):
    if state.recording.status == RecordingStatus.ASSESSING and state.recording.job.assessment_id:
        render_guard("review.guard_still_assessing", "review.go_back_speak", "pages/02_Speak.py")
    render_guard("review.guard_missing_review", "review.go_speak", "pages/02_Speak.py")

render_page_intro("review.title", "review.body")
render_shell_summary(state)

summary = review_summary(state.review.payload)
render_report_panels(
    summary,
    transcript=str(state.review.transcript or summary.get("transcript") or ""),
    notes=str(summary.get("notes") or ""),
    key_prefix="review",
)

retry_col, setup_col, history_col = st.columns(3)
with retry_col:
    if st.button(t("review.try_again"), key="review_try_again", width="stretch"):
        clear_attempt(keep_setup=True)
        go_to("pages/02_Speak.py", return_to="review")
with setup_col:
    if st.button(t("review.new_setup"), key="review_new_setup", width="stretch"):
        clear_attempt(keep_setup=False)
        go_to("pages/01_Session_Setup.py", return_to="review")
with history_col:
    if st.button(t("review.view_history"), key="review_view_history", width="stretch"):
        go_to("pages/04_History.py", return_to="review")
