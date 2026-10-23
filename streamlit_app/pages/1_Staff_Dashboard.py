import sys
from pathlib import Path

# Same fix as Home.py: Streamlit puts this script's own directory on sys.path,
# not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from analytics.auth import check_staff_access, staff_auth_configured
from analytics.dashboard_queries import (
    escalation_rate,
    total_turns,
    volume_by_role,
    volume_by_topic,
)
from analytics.db import get_engine, get_session

st.set_page_config(page_title="Staff Dashboard | UB Library Navigator", page_icon="📊")
st.title("📊 Staff usage dashboard")
st.caption(
    "Aggregated, anonymized usage only. No query text, no session or user "
    "identifiers are ever logged. Small groups are suppressed to avoid "
    "re-identifying a rare role/topic combination."
)

if not staff_auth_configured():
    st.warning(
        "Staff dashboard access isn't configured yet. Set "
        "`STAFF_DASHBOARD_AUTH_METHOD=password` and `STAFF_DASHBOARD_PASSWORD` "
        "in the environment to enable this page."
    )
    st.stop()

if "staff_authed" not in st.session_state:
    st.session_state.staff_authed = False

if not st.session_state.staff_authed:
    with st.form("staff_login"):
        secret = st.text_input("Staff password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if check_staff_access(secret):
            st.session_state.staff_authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

from dotenv import load_dotenv

load_dotenv()
session = get_session(get_engine())
try:
    turns = total_turns(session)
    if turns == 0:
        st.info("No usage has been logged yet.")
    else:
        st.metric("Total turns logged", turns)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Query volume by role")
            st.bar_chart(volume_by_role(session))
        with col2:
            st.subheader("Topic category breakdown")
            st.bar_chart(volume_by_topic(session))

        st.subheader("Escalation rate")
        st.metric("Turns escalated to a human librarian", f"{escalation_rate(session)}%")
        st.caption(
            "Shown as one overall rate, not broken down by role or topic — a "
            "small cross-tab (e.g. one faculty member in one topic who "
            "escalated) could otherwise be identifying."
        )
finally:
    session.close()

if st.button("Sign out"):
    st.session_state.staff_authed = False
    st.rerun()
