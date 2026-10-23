from agent.escalation import LIVE_CHAT_URL
from agent.models import QueryPlan
from analytics.db import LogEntry, get_engine, get_session
from analytics.log_writer import log_turn


def _session(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    return get_session(engine)


def test_logs_a_normal_turn_with_plan_fields(tmp_path):
    session = _session(tmp_path)
    state = {
        "plan": QueryPlan(keywords="chemistry", resource_type="articles", topic_category="sciences"),
        "answer": "Here is what I found...",
        "selection": [object()],
    }

    log_turn("graduate", state, session=session)

    row = session.query(LogEntry).one()
    assert row.role == "graduate"
    assert row.topic_category == "sciences"
    assert row.resource_type == "articles"
    assert row.escalated is False
    assert row.had_results is True


def test_logs_a_human_request_fast_path_turn_with_no_plan(tmp_path):
    session = _session(tmp_path)
    state = {"answer": f"I'll connect you with a librarian.\n\nTalk to a librarian: {LIVE_CHAT_URL}"}

    log_turn("undergraduate", state, session=session)

    row = session.query(LogEntry).one()
    assert row.topic_category == "other"  # no plan: safe default, never a crash
    assert row.resource_type is None
    assert row.escalated is True
    assert row.had_results is False


def test_logs_a_clarification_turn(tmp_path):
    session = _session(tmp_path)
    state = {
        "plan": QueryPlan(keywords="", needs_clarification="What topic?", topic_category="other"),
        "answer": "What topic?",
    }

    log_turn("faculty", state, session=session)

    row = session.query(LogEntry).one()
    assert row.escalated is False
    assert row.had_results is False


def test_logs_an_escalated_low_confidence_turn(tmp_path):
    session = _session(tmp_path)
    state = {
        "plan": QueryPlan(keywords="x", topic_category="other"),
        "answer": f"I am not confident...\n\nTalk to a librarian: {LIVE_CHAT_URL}",
        "selection": [],
    }

    log_turn("graduate", state, session=session)

    row = session.query(LogEntry).one()
    assert row.escalated is True
    assert row.had_results is False


def test_course_is_always_none():
    # No UI captures course info yet; documented gap, not a crash.
    import inspect
    assert "course" not in inspect.signature(log_turn).parameters


def test_never_raises_even_if_the_session_is_broken():
    class BrokenSession:
        def add(self, entry):
            raise RuntimeError("db is down")

    log_turn("graduate", {"answer": "x"}, session=BrokenSession())  # must not raise
