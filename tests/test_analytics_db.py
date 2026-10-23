from analytics.db import LogEntry, get_engine, get_session


def test_get_engine_creates_the_table(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    with get_session(engine) as session:
        session.add(LogEntry(role="graduate", topic_category="sciences"))
        session.commit()
        rows = session.query(LogEntry).all()
        assert len(rows) == 1
        assert rows[0].role == "graduate"
        assert rows[0].course is None
        assert rows[0].escalated is False


def test_log_entry_has_no_raw_query_text_column():
    columns = {c.name for c in LogEntry.__table__.columns}
    assert "message" not in columns
    assert "query" not in columns
    assert "question" not in columns


def test_log_entry_has_no_identifying_column():
    columns = {c.name for c in LogEntry.__table__.columns}
    for forbidden in ("user_id", "session_id", "name", "email", "ip_address"):
        assert forbidden not in columns
