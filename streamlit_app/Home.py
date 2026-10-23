import sys
from pathlib import Path

# Streamlit puts this script's own directory on sys.path, not the project root,
# so `agent`/`onesearch`/`pipeline` aren't importable without this (confirmed
# live 2026-09-29: ModuleNotFoundError: No module named 'agent').
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agent.cli import HISTORY_TURNS, build_default_graph
from agent.llm import LLMConfigError
from agent.models import ROLES
from analytics.log_writer import log_turn

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

if "role" not in st.session_state:
    st.session_state.role = None
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

with st.sidebar:
    st.subheader("Who are you?")
    chosen = st.selectbox(
        "Role", [""] + list(ROLES), index=0, label_visibility="collapsed"
    )
    if chosen:
        st.session_state.role = chosen
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

if not st.session_state.history:
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

for turn in st.session_state.history:
    st.chat_message("user", avatar="🧑").write(turn["user"])
    st.chat_message("assistant", avatar="🦬").write(turn["assistant"])

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
            }
        )
    log_turn(st.session_state.role, state)
    answer = state["answer"]
    with st.chat_message("assistant", avatar="🦬"):
        st.write(answer)
        if debug:
            if "plan" in state:  # absent when the human-request fast path short-circuits
                st.caption(f"plan: {state['plan'].model_dump()}")
            st.caption(f"timings: {state.get('timings', {})}")
            for note in state.get("notes", []):
                st.caption(f"note: {note}")
    st.session_state.history.append({"user": message, "assistant": answer})
    st.session_state.history = st.session_state.history[-HISTORY_TURNS:]
