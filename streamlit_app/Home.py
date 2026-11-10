import sys
from pathlib import Path

# Streamlit puts this script's own directory on sys.path, not the project root,
# so `agent`/`onesearch`/`pipeline` aren't importable without this (confirmed
# live 2026-09-29: ModuleNotFoundError: No module named 'agent').
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agent.cli import HISTORY_TURNS, build_default_graph
from agent.compose import extract_shown_results, wants_citations
from agent.llm import LLMConfigError
from agent.models import ROLES
from analytics.auth import app_auth_configured, check_app_access
from analytics.log_writer import log_turn
from streamlit_app.formatting import (
    block_markdown,
    card_detail_markdown,
    card_header_markdown,
    parse_answer,
)

# UB brand colors: #005BBB is confirmed from library.buffalo.edu's own site CSS; the
# rest are pixel-sampled from the reference mockup (docs/design/ub-ai-assistant-reference.png).
# Native widget colors live in .streamlit/config.toml; this CSS only covers what that
# can't reach (the header band, the hero, and the chip buttons).
THEME_CSS = """
<style>
.ub-header {
    background: linear-gradient(90deg, #044098 0%, #0054C6 55%, #155EC9 100%);
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}
.ub-header .ub-mark {
    width: 44px;
    height: 44px;
    border-radius: 0.6rem;
    background: rgba(255, 255, 255, 0.16);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    flex-shrink: 0;
}
.ub-header .ub-title {
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1.2rem;
    line-height: 1.25;
}
.ub-header .ub-subtitle {
    color: #CFE0FA;
    font-size: 0.88rem;
    font-weight: 400;
}
.ub-hero {
    background: linear-gradient(135deg, #F4F8FD 0%, #E3EFFC 100%);
    border: 1px solid #D7E9FE;
    border-radius: 0.75rem;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.ub-hero h2 {
    color: #002163;
    font-weight: 800;
    font-size: 1.5rem;
    margin: 0 0 0.35rem 0;
}
.ub-hero p {
    color: #37507D;
    font-size: 0.95rem;
    margin: 0;
}
div[data-testid="stChatMessage"] {
    border-radius: 0.75rem;
}
.ub-sidebar-about {
    font-size: 0.85rem;
    color: #37507D;
    line-height: 1.45;
}
.ub-sidebar-about ul {
    margin: 0.3rem 0 0.5rem 0;
    padding-left: 1.1rem;
}
.ub-sidebar-about li {
    margin-bottom: 0.2rem;
}
.stButton > button {
    background: #FFFFFF;
    border: 1px solid #D7E9FE;
    color: #002163;
    text-align: left;
    font-weight: 500;
    height: auto;
}
.stButton > button p {
    white-space: normal !important;
}
.stButton > button:hover {
    border-color: #005BBB;
    color: #005BBB;
}
</style>
"""


@st.cache_resource
def _get_graph():
    """One graph (model + OneSearch client + RAG collection) shared across every
    visitor on this server process, not rebuilt per browser session: OneSearch's
    session-mint and query cache are meant to be shared (CLAUDE.md, "quiet,
    well-behaved caller"; never mint a Playwright session per user search)."""
    from dotenv import load_dotenv

    load_dotenv()
    return build_default_graph()


ABOUT_HTML = """
<div class="ub-sidebar-about">
<strong>What I can do</strong>
<ul>
<li>Find books, articles, journals, dissertations and videos in OneSearch</li>
<li>Name your subject librarian and relevant help guides</li>
<li>Suggest databases for your topic</li>
<li>Format APA, MLA or Chicago citations (just ask)</li>
<li>Connect you to live chat, with your question ready to paste</li>
</ul>
I point you to the right place; I don't write your assignments.
Usage is counted anonymously. Your questions are never stored.
</div>
"""

PROMPT_CHIPS = [
    "Peer-reviewed journals on mental health",
    "Who's the subject librarian for chemistry?",
    "Browse databases for computer science",
]

st.set_page_config(page_title="UB Library Navigator", page_icon="📚")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="ub-header">
        <div class="ub-mark">UB</div>
        <div>
            <div class="ub-title">University at Buffalo</div>
            <div class="ub-subtitle">Library Navigator</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Gated only when APP_ACCESS_METHOD/APP_ACCESS_PASSWORD are actually set (e.g.
# once the app is reachable over a public cloud URL, not just localhost) -
# unconfigured means local/dev use is unaffected, same fail-closed-but-visible
# pattern as the staff dashboard's own gate (analytics/auth.py).
if app_auth_configured():
    if "app_authed" not in st.session_state:
        st.session_state.app_authed = False
    if not st.session_state.app_authed:
        st.info("This app is password-protected. Ask whoever shared the link for the password.")
        with st.form("app_login"):
            secret = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter")
        if submitted:
            if check_app_access(secret):
                st.session_state.app_authed = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

if "role" not in st.session_state:
    st.session_state.role = None
if "history" not in st.session_state:
    st.session_state.history = []  # last few turns only: this is what the LLM sees
if "shown_results" not in st.session_state:
    st.session_state.shown_results = []  # details of the results last shown, for follow-ups
if "transcript" not in st.session_state:
    st.session_state.transcript = []  # every turn, for display only
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

with st.sidebar:
    st.subheader("Who are you?")
    chosen = st.selectbox(
        "Role", [""] + list(ROLES), index=0, label_visibility="collapsed"
    )
    if chosen:
        st.session_state.role = chosen
    st.markdown(ABOUT_HTML, unsafe_allow_html=True)
    st.image(str(Path(__file__).parent / "assets" / "ub-bull-mascot.png"), use_container_width=True)
    debug = st.checkbox("Show debug info")

if not st.session_state.role:
    st.info("Pick a role in the sidebar to start chatting.")
    st.stop()

try:
    graph = _get_graph()
except LLMConfigError as exc:
    st.error(f"Cannot start: {exc}")
    st.stop()

if not st.session_state.transcript:
    st.markdown(
        """
        <div class="ub-hero">
            <h2>Hello there 👋</h2>
            <p>I'm your UB Library Navigator. Ask me about resources, databases, or who to
            contact, and I'll point you to the right place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(len(PROMPT_CHIPS))
    for col, prompt in zip(chip_cols, PROMPT_CHIPS):
        if col.button(prompt, use_container_width=True):
            st.session_state.pending_message = prompt

def render_answer(entry: dict, show_debug: bool) -> None:
    """One assistant turn, laid out as: intro, one bordered card per result (title and
    byline, why it fits, then labeled links), then the closing notes and any live-chat
    handoff. Same function for new and replayed turns so they always look identical."""
    parsed = parse_answer(entry["answer"])
    for block in block_markdown(parsed.intro):
        st.markdown(block)
    for card in parsed.cards:
        with st.container(border=True):
            st.markdown(card_header_markdown(card))
            if card.why:
                st.markdown(f"*{card.why}*")
            for label, value in card.details:
                st.markdown(card_detail_markdown(label, value))
    if parsed.outro:
        if parsed.cards:
            st.divider()
        for block in block_markdown(parsed.outro):
            st.markdown(block)
    if parsed.escalation:
        with st.container(border=True):
            st.markdown(f"**Talk to a librarian**  \n<{parsed.escalation.chat_url}>")
            st.caption("Paste this into the chat so they know what you need:")
            st.code(parsed.escalation.context, language=None, wrap_lines=True)
    for title, styles in entry.get("citations", []):
        with st.expander(f"Cite: {title}"):
            for style_name, text in styles.items():
                st.markdown(f"**{style_name.upper()}**")
                st.markdown(text)
    if show_debug:
        for line in entry.get("debug", []):
            st.caption(line)


for entry in st.session_state.transcript:
    st.chat_message("user", avatar="🧑").write(entry["user"])
    with st.chat_message("assistant", avatar="🦬"):
        render_answer(entry, debug)

message = st.chat_input("Ask a question")
if not message and st.session_state.pending_message:
    message = st.session_state.pending_message
    st.session_state.pending_message = None

if message:
    st.chat_message("user", avatar="🧑").write(message)
    with st.spinner("Searching..."):
        state = graph.invoke(
            {
                "role": st.session_state.role,
                "message": message,
                "history": st.session_state.history,
                "shown_results": st.session_state.shown_results,
            }
        )
    log_turn(st.session_state.role, state)
    answer = state["answer"]
    # Citations are computed every turn regardless (cheap, local, no extra API call)
    # but only kept for display when the user's own message asked for them - showing
    # them unasked was a real usability complaint (see agent/compose.py).
    # state["search_result"] is absent whenever the turn ended before search_node ran
    # (a clarifying question, a fast path) - confirmed live 2026-10-22 - so guard it.
    citations = []
    if wants_citations(message):
        search_result = state.get("search_result")
        titles_by_id = {c.record_id: c.title for c in search_result.candidates} if search_result else {}
        citations = [
            (titles_by_id.get(record_id, record_id), styles)
            for record_id, styles in state.get("citations", {}).items()
        ]
    debug_lines = []
    if "plan" in state:  # absent when the human-request fast path short-circuits
        debug_lines.append(f"plan: {state['plan'].model_dump()}")
    debug_lines.append(f"timings: {state.get('timings', {})}")
    debug_lines.extend(f"note: {note}" for note in state.get("notes", []))
    entry = {"user": message, "answer": answer, "citations": citations, "debug": debug_lines}
    with st.chat_message("assistant", avatar="🦬"):
        render_answer(entry, debug)
    # The transcript keeps every turn on screen; only the LLM's own context is trimmed.
    st.session_state.transcript.append(entry)
    # Kept across follow-up turns: only a turn that shows new results replaces it.
    st.session_state.shown_results = extract_shown_results(state) or st.session_state.shown_results
    st.session_state.history.append({"user": message, "assistant": answer})
    st.session_state.history = st.session_state.history[-HISTORY_TURNS:]
