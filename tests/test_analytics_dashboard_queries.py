import pytest

from analytics.dashboard_queries import (
    SUPPRESSED_LABEL,
    escalation_rate,
    total_turns,
    volume_by_role,
    volume_by_topic,
)
from analytics.db import LogEntry, get_engine, get_session


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    s = get_session(engine)
    yield s
    s.close()


def _seed(session, rows):
    for role, topic, escalated in rows:
        session.add(LogEntry(role=role, topic_category=topic, escalated=escalated))
    session.commit()


def test_total_turns_counts_all_rows(session):
    _seed(session, [("student", "databases", False)] * 3)
    assert total_turns(session) == 3


def test_total_turns_zero_on_empty_db(session):
    assert total_turns(session) == 0


def test_volume_by_role_groups_at_or_above_threshold_shown_individually(session):
    _seed(session, [("student", "databases", False)] * 6 + [("faculty", "databases", False)] * 5)
    result = volume_by_role(session)
    assert result == {"student": 6, "faculty": 5}


def test_volume_by_role_suppresses_small_groups(session):
    # "staff" appears only twice, below MIN_GROUP_COUNT=5, so it must not
    # appear under its own label.
    _seed(
        session,
        [("student", "databases", False)] * 6
        + [("staff", "databases", False)] * 2,
    )
    result = volume_by_role(session)
    assert "staff" not in result
    assert result[SUPPRESSED_LABEL] == 2
    assert result["student"] == 6


def test_volume_by_topic_suppresses_small_groups(session):
    _seed(
        session,
        [("student", "databases", False)] * 6
        + [("student", "citations", False)] * 1,
    )
    result = volume_by_topic(session)
    assert "citations" not in result
    assert result[SUPPRESSED_LABEL] == 1


def test_escalation_rate_computes_percentage(session):
    _seed(
        session,
        [("student", "databases", True)] * 2 + [("student", "databases", False)] * 8,
    )
    assert escalation_rate(session) == 20.0


def test_escalation_rate_zero_on_empty_db(session):
    assert escalation_rate(session) == 0.0


def test_escalation_rate_never_broken_down_by_group():
    # Documentation-as-test: dashboard_queries exposes no function that
    # cross-tabulates escalation with role or topic.
    import analytics.dashboard_queries as module

    public_fns = [name for name in dir(module) if not name.startswith("_")]
    for name in public_fns:
        assert "by_role" not in name or "escalat" not in name.lower()
        assert "by_topic" not in name or "escalat" not in name.lower()
