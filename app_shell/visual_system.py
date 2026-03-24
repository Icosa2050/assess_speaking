from __future__ import annotations

from html import escape

import streamlit as st

APP_CSS = """
<style>
:root {
    --as-bg: #f8f4ec;
    --as-surface: rgba(255, 255, 255, 0.8);
    --as-surface-strong: #fffdf8;
    --as-ink: #1f2430;
    --as-muted: #5f6775;
    --as-line: rgba(82, 91, 118, 0.16);
    --as-accent: #4d5e8b;
    --as-accent-soft: rgba(77, 94, 139, 0.12);
    --as-warm: rgba(215, 199, 171, 0.18);
    --as-shadow: 0 18px 45px rgba(31, 36, 48, 0.08);
    --as-radius: 22px;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(77, 94, 139, 0.10), transparent 38%),
        radial-gradient(circle at bottom right, rgba(215, 199, 171, 0.18), transparent 28%),
        linear-gradient(180deg, #fcfaf6 0%, var(--as-bg) 100%);
    color: var(--as-ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

.stApp .block-container {
    max-width: 1180px;
    padding-top: 2.9rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: var(--as-ink);
    font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    letter-spacing: -0.03em;
    line-height: 0.98;
}

h1 {
    font-size: clamp(2.6rem, 4.2vw, 4.6rem);
    margin-bottom: 0.6rem;
}

h2 {
    font-size: clamp(1.55rem, 2.6vw, 2.2rem);
}

h3 {
    font-size: clamp(1.15rem, 2vw, 1.45rem);
}

p, li, label, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {
    color: var(--as-ink);
}

[data-testid="stCaptionContainer"] {
    color: var(--as-muted);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--as-surface);
    border: 1px solid var(--as-line);
    border-radius: var(--as-radius);
    box-shadow: var(--as-shadow);
}

[data-testid="stMetric"] {
    background: var(--as-surface-strong);
    border: 1px solid var(--as-line);
    border-radius: 18px;
    padding: 0.8rem 0.95rem;
}

.stButton > button,
[data-testid="stBaseButton-secondary"] > button,
[data-testid="stBaseButton-primary"] > button {
    min-height: 3rem;
    border-radius: 999px;
    border: 1px solid rgba(77, 94, 139, 0.18);
    background: linear-gradient(135deg, #51638f 0%, #41527f 100%);
    color: #fbf9f4;
    font-weight: 600;
    box-shadow: 0 12px 30px rgba(65, 82, 127, 0.18);
}

.stButton > button[kind="secondary"] {
    background: var(--as-surface-strong);
    color: var(--as-ink);
    box-shadow: none;
}

.stButton > button:disabled {
    background: rgba(77, 94, 139, 0.2);
    color: rgba(31, 36, 48, 0.55);
    box-shadow: none;
}

.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] > div,
[data-testid="stSelectSlider"] [role="slider"],
[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px;
}

.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {
    border: 1px solid rgba(82, 91, 118, 0.12);
    background: rgba(255, 255, 255, 0.72);
}

.as-kicker {
    margin: 0 0 0.4rem;
    color: var(--as-accent);
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.as-detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 0.5rem;
}

.as-detail-card {
    background: var(--as-surface-strong);
    border: 1px solid var(--as-line);
    border-radius: 18px;
    padding: 0.95rem 1rem;
}

.as-detail-label {
    display: block;
    margin-bottom: 0.3rem;
    color: var(--as-muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.as-detail-value {
    color: var(--as-ink);
    font-size: 1rem;
    line-height: 1.35;
}

.as-quote {
    margin-top: 0.5rem;
    padding: 1.25rem 1.35rem;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(249, 246, 239, 0.92));
    border: 1px solid rgba(82, 91, 118, 0.14);
    color: var(--as-ink);
    font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    font-size: 1.2rem;
    line-height: 1.42;
}

.as-quote-empty {
    font-family: inherit;
    font-size: 1rem;
    color: var(--as-muted);
}

.as-checklist {
    list-style: none;
    margin: 0.9rem 0 0;
    padding: 0;
}

.as-checklist li {
    position: relative;
    margin: 0 0 0.7rem;
    padding-left: 1.35rem;
    color: var(--as-ink);
    line-height: 1.5;
}

.as-checklist li::before {
    content: "";
    position: absolute;
    left: 0.05rem;
    top: 0.48rem;
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 999px;
    background: var(--as-accent);
    box-shadow: 0 0 0 6px var(--as-accent-soft);
}

.as-inline-note {
    margin-top: 0.8rem;
    padding: 0.75rem 0.95rem;
    border-radius: 16px;
    background: var(--as-warm);
    color: var(--as-muted);
    font-size: 0.94rem;
    line-height: 1.45;
}

@media (max-width: 900px) {
    .stApp .block-container {
        padding-top: 2.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .as-detail-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def inject_visual_system() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_kicker(text: str) -> None:
    st.markdown(f'<p class="as-kicker">{escape(text)}</p>', unsafe_allow_html=True)


def render_detail_grid(items: list[tuple[str, str]]) -> None:
    cards = []
    for label, value in items:
        cards.append(
            '<div class="as-detail-card">'
            f'<span class="as-detail-label">{escape(label)}</span>'
            f'<span class="as-detail-value">{escape(value or "-")}</span>'
            "</div>"
        )
    st.markdown(f'<div class="as-detail-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_quote(text: str, *, empty: bool = False) -> None:
    quote_class = "as-quote as-quote-empty" if empty else "as-quote"
    escaped = escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="{quote_class}">{escaped}</div>', unsafe_allow_html=True)


def render_checklist(items: list[str]) -> None:
    entries = "".join(f"<li>{escape(item)}</li>" for item in items)
    st.markdown(f'<ul class="as-checklist">{entries}</ul>', unsafe_allow_html=True)


def render_inline_note(text: str) -> None:
    st.markdown(f'<div class="as-inline-note">{escape(text)}</div>', unsafe_allow_html=True)
