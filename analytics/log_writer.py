from agent.escalation import LIVE_CHAT_URL
from analytics.db import LogEntry, get_engine, get_session

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_engine()
    return _ENGINE


def log_turn(role: str, state: dict, session=None) -> None:
    """Writes one anonymized log row for a completed turn. Covers all three paths
    uniformly (human-request fast path, clarification short-circuit, normal flow)
    by reading only what `graph.invoke()`'s returned state dict actually has —
    `plan` is absent for the fast path, so every field below degrades to a safe
    default rather than raising. Never raises: a logging failure must not break
    the chat turn that already succeeded."""
    try:
        plan = state.get("plan")
        answer = state.get("answer") or ""
        entry = LogEntry(
            role=role,
            course=None,  # no UI captures this yet; the column exists for when one does
            topic_category=plan.topic_category if plan else "other",
            resource_type=plan.resource_type if plan else None,
            escalated=LIVE_CHAT_URL in answer,
            had_results=bool(state.get("selection")),
        )
        owns_session = session is None
        session = session or get_session(_engine())
        try:
            session.add(entry)
            session.commit()
        finally:
            if owns_session:
                session.close()
    except Exception:
        pass
