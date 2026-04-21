from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_shell.i18n import t
from app_shell.page_helpers import configure_page, go_to, render_guard, render_page_intro, render_shell_summary
from app_shell.runtime_providers import requires_api_key
from app_shell.runtime_resolver import active_connection, resolve_connection_runtime
from app_shell.runtime_status import job_status_message
from app_shell.services import (
    cancel_assessment_request,
    cleanup_temp_audio,
    create_assessment_request,
    poll_assessment_request,
    prime_assessment_request_status,
    review_summary,
    store_uploaded_audio,
    submit_assessment_request,
)
from app_shell.state import (
    RecordingStatus,
    apply_review_payload,
    clear_recording,
    get_app_state,
    has_recording,
    has_setup,
    set_recording_assessing,
    set_recording_error,
    set_recording_job,
    update_recording,
    update_recording_inputs,
)
from app_shell.visual_system import render_detail_grid, render_inline_note, render_kicker, render_quote

AUTO_PROGRESS_REFRESH_SEC = 4.0
PENDING_REVIEW_PAYLOAD_KEY = "speak_pending_review_payload"


def _recording_file_exists(audio_path: str) -> bool:
    return bool(audio_path and Path(audio_path).exists())


def _task_family_label(task_family: str) -> str:
    label = t(f"task_family.{task_family}")
    if label.startswith("["):
        return task_family.replace("_", " ")
    return label


def _clear_attached_recording(log_dir: str) -> None:
    state = get_app_state()
    if state.recording.audio_path:
        cleanup_temp_audio(state.recording.audio_path, allowed_root=log_dir)
    clear_recording()
    for key in ("speak_audio_input", "speak_upload"):
        st.session_state.pop(key, None)


def _apply_review_and_go(payload: dict | None) -> None:
    summary = review_summary(payload)
    apply_review_payload(
        payload=payload or {},
        report_id=str(summary.get("report_id") or ""),
        transcript=str(summary.get("transcript") or ""),
        score_overall=summary.get("score_overall"),
        band=str(summary.get("band") or ""),
        summary=str(summary.get("coach_summary") or ""),
    )
    go_to("pages/03_Review.py", return_to="speak")


def _queue_completed_review_payload(payload: dict | None) -> None:
    if isinstance(payload, dict):
        st.session_state[PENDING_REVIEW_PAYLOAD_KEY] = payload


def _consume_completed_review_payload() -> None:
    payload = st.session_state.pop(PENDING_REVIEW_PAYLOAD_KEY, None)
    if isinstance(payload, dict):
        _apply_review_and_go(payload)


def _sync_assessment_job(log_dir: str) -> tuple[object, dict | None]:
    state = get_app_state()
    if state.recording.status != RecordingStatus.ASSESSING or not state.recording.job.assessment_id:
        return state, None
    job_state, payload, error = poll_assessment_request(
        state.recording.job.assessment_id,
        log_dir=log_dir,
    )
    if job_state is not None:
        if error and not job_state.error:
            job_state.error = error
        set_recording_job(job_state)
        state = get_app_state()
    elif error:
        set_recording_error(error)
        state = get_app_state()
    return state, payload


def _job_phase_message(phase: str) -> str:
    key = f"speak.job_phase_{phase or 'running'}"
    translated = t(key)
    if translated.startswith("["):
        return phase.replace("_", " ").strip() or t("speak.job_phase_running")
    return translated


state = configure_page("speak", "nav.speak", icon="🎤")

if not has_setup(state):
    render_guard("speak.guard_missing_setup", "speak.go_setup", "pages/01_Session_Setup.py")

runtime_connection = active_connection(state.prefs)
if runtime_connection is None:
    render_guard("home.runtime_setup_body", "home.runtime_setup_button", "pages/00_Setup.py")
runtime_config = resolve_connection_runtime(runtime_connection)

_consume_completed_review_payload()

if state.recording.status == RecordingStatus.ASSESSING and state.recording.job.assessment_id:
    state, payload = _sync_assessment_job(state.prefs.log_dir)
    if payload is not None:
        _apply_review_and_go(payload)

render_page_intro("speak.title")
render_shell_summary(state)

with st.container(border=True):
    render_kicker(t("speak.prompt_title"))
    st.subheader(state.draft.theme_label or t("speak.prompt_title"))
    if state.draft.prompt_text:
        render_quote(state.draft.prompt_text)
    else:
        render_quote(t("speak.prompt_placeholder"), empty=True)
    render_detail_grid(
        [
            (t("setup.speaker_id"), state.draft.speaker_id or "-"),
            (t("setup.learning_language"), state.draft.learning_language_label or "-"),
            (t("setup.cefr"), state.draft.cefr_level or "-"),
            (t("setup.duration"), f"{int(state.draft.duration_sec)} s" if state.draft.duration_sec else "-"),
            (t("history.task_family_name"), _task_family_label(state.draft.task_family or "free_monologue")),
            (t("setup.theme"), state.draft.theme_label or "-"),
        ]
    )

record_col, action_col = st.columns([1.08, 0.92], gap="large")

with record_col:
    with st.container(border=True):
        render_kicker(t("speak.recording_title"))
        st.write(t("speak.recording_body"))
        input_method = st.radio(
            t("speak.input_method"),
            options=["record", "upload"],
            format_func=lambda value: t(f"speak.input_method_{value}"),
            horizontal=True,
            key="speak_input_method",
        )
        audio_input = None
        uploaded_file = None
        if input_method == "record":
            audio_input = st.audio_input(t("speak.audio_input"), key="speak_audio_input")
        else:
            uploaded_file = st.file_uploader(
                t("speak.upload"),
                type=["wav", "mp3", "m4a", "flac", "ogg"],
                key="speak_upload",
            )
            render_inline_note(t("speak.upload_help"))
        new_path: Path | None = None
        new_digest = ""
        try:
            if audio_input is not None:
                new_path, new_digest = store_uploaded_audio(
                    audio_input,
                    target_dir=Path(state.prefs.log_dir) / "recordings",
                    filename=getattr(audio_input, "name", "recording.wav"),
                    previous_digest=state.recording.input_digest,
                    previous_path=state.recording.audio_path,
                )
            elif uploaded_file is not None:
                new_path, new_digest = store_uploaded_audio(
                    uploaded_file,
                    target_dir=Path(state.prefs.log_dir) / "uploads",
                    filename=getattr(uploaded_file, "name", "upload.wav"),
                    previous_digest=state.recording.input_digest,
                    previous_path=state.recording.audio_path,
                )
        except Exception as exc:  # pragma: no cover - filesystem / streamlit boundary  # quality: allow[broad-except] upload boundary should become localized recording feedback
            set_recording_error(str(exc))
            st.error(t("speak.ingest_error", detail=str(exc)))

        state = get_app_state()
        if state.recording.audio_path and state.recording.input_method and state.recording.input_method != input_method:
            _clear_attached_recording(state.prefs.log_dir)
            state = get_app_state()
        if new_path and new_digest != state.recording.input_digest:
            if state.recording.audio_path and state.recording.audio_path != str(new_path):
                cleanup_temp_audio(state.recording.audio_path, allowed_root=state.prefs.log_dir)
            update_recording(audio_path=str(new_path), input_digest=new_digest, input_method=input_method)
            state = get_app_state()

        recording_exists = _recording_file_exists(state.recording.audio_path)
        if state.recording.error:
            st.error(state.recording.error)
        elif has_recording(state) and not recording_exists:
            st.warning(t("speak.status_missing_file"))
        elif has_recording(state):
            st.success(t("speak.status_ready"))
            audio_path = Path(state.recording.audio_path)
            if audio_path.exists() and state.recording.input_method == "upload":
                try:
                    st.audio(str(audio_path))
                except OSError as exc:  # pragma: no cover - filesystem boundary
                    st.error(t("speak.preview_error", detail=str(exc)))
        else:
            st.info(t("speak.status_idle"))
        if has_recording(state):
            if st.button(t("speak.remove_recording"), key="speak_remove_recording", width="content"):
                _clear_attached_recording(state.prefs.log_dir)
                st.rerun()

with action_col:
    with st.container(border=True):
        render_kicker(t("speak.assessment_title"))
        st.subheader(t("speak.submit"))
        st.caption(
            t(
                "speak.assessment_caption",
                provider=runtime_config.provider,
                model=runtime_config.model,
                whisper=state.prefs.whisper_model,
            )
        )
        render_detail_grid(
            [
                (t("setup.speaker_id"), state.draft.speaker_id or "-"),
                (t("setup.learning_language"), state.draft.learning_language_label or "-"),
                (t("setup.cefr"), state.draft.cefr_level or "-"),
                (t("setup.duration"), f"{int(state.draft.duration_sec)} s" if state.draft.duration_sec else "-"),
            ]
        )
        effective_llm_api_key = runtime_config.api_key
        if requires_api_key(runtime_config.provider) and not effective_llm_api_key:
            st.warning(t("speak.openrouter_missing_key"))
        if "speak_label" not in st.session_state:
            st.session_state["speak_label"] = state.recording.label_input
        if "speak_notes" not in st.session_state:
            st.session_state["speak_notes"] = state.recording.notes_input
        label = st.text_input(t("speak.label"), key="speak_label")
        notes = st.text_area(t("speak.notes"), key="speak_notes", height=120)
        if label != state.recording.label_input or notes != state.recording.notes_input:
            update_recording_inputs(label_input=label, notes_input=notes)
            state = get_app_state()
        is_assessing = state.recording.status == RecordingStatus.ASSESSING
        submit_disabled = not recording_exists or not state.draft.speaker_id or bool(state.recording.error) or is_assessing

        @st.fragment(run_every=AUTO_PROGRESS_REFRESH_SEC if is_assessing else None)
        def _render_assessment_status_panel() -> None:
            current_state = get_app_state()
            if current_state.recording.status == RecordingStatus.ASSESSING and current_state.recording.job.assessment_id:
                current_state, payload = _sync_assessment_job(current_state.prefs.log_dir)
                if payload is not None:
                    _queue_completed_review_payload(payload)
                    st.rerun()
                if current_state.recording.status != RecordingStatus.ASSESSING:
                    st.rerun()
                current_connection = active_connection(current_state.prefs)
                current_runtime = resolve_connection_runtime(current_connection) if current_connection is not None else runtime_config
                with st.status(job_status_message(current_state.recording.job.status, current_runtime), state="running", expanded=True):
                    if current_state.recording.job.phase:
                        st.write(t("speak.job_phase", phase=_job_phase_message(current_state.recording.job.phase)))
                    st.caption(t("speak.job_auto_refresh"))
                if st.button(t("speak.cancel_assessment"), key="speak_cancel_assessment", width="stretch"):
                    cancelled_job, error = cancel_assessment_request(
                        current_state.recording.job.assessment_id,
                        log_dir=current_state.prefs.log_dir,
                    )
                    if error:
                        set_recording_error(error)
                    elif cancelled_job is not None:
                        if not cancelled_job.error:
                            cancelled_job.error = t("speak.job_status_cancelled")
                        set_recording_job(cancelled_job)
                    st.rerun()
                return
            render_inline_note(t("speak.recording_body"))

        _render_assessment_status_panel()
        if st.button(t("speak.submit"), key="speak_submit", width="stretch", disabled=submit_disabled):
            request = create_assessment_request(
                audio_path=Path(state.recording.audio_path),
                log_dir=state.prefs.log_dir,
                whisper=state.prefs.whisper_model,
                provider=runtime_config.provider,
                llm_model=runtime_config.model,
                llm_base_url=runtime_config.base_url,
                expected_language=state.draft.learning_language,
                feedback_language=state.prefs.ui_locale,
                llm_api_key=effective_llm_api_key,
                openrouter_http_referer=runtime_config.extra_headers.get("HTTP-Referer", ""),
                openrouter_app_title=runtime_config.extra_headers.get("X-Title", ""),
                speaker_id=state.draft.speaker_id,
                task_family=state.draft.task_family,
                theme=state.draft.theme_label,
                target_duration_sec=state.draft.duration_sec,
                label=label.strip(),
                notes=notes.strip(),
                target_cefr=state.draft.cefr_level,
            )
            created_job, error = submit_assessment_request(request)
            if error:
                set_recording_error(error)
            elif created_job is not None:
                set_recording_assessing(created_job)
                primed_job, payload, priming_error = prime_assessment_request_status(
                    created_job.assessment_id,
                    log_dir=state.prefs.log_dir,
                )
                if primed_job is not None:
                    if priming_error and not primed_job.error:
                        primed_job.error = priming_error
                    set_recording_job(primed_job)
                elif priming_error:
                    set_recording_error(priming_error)
                if payload is not None:
                    _apply_review_and_go(payload)
            st.rerun()
