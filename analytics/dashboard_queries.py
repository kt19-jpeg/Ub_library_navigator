from sqlalchemy import func

from analytics.db import LogEntry

# CLAUDE.md "known risks": "a rare combination of course, role, and topic
# could be identifiable... suppress any breakdown below a small count
# threshold." Any group with fewer rows than this is folded into "Other
# (suppressed)" instead of shown on its own.
MIN_GROUP_COUNT = 5
SUPPRESSED_LABEL = "Other (suppressed)"


def _suppress_small_groups(counts: dict[str, int]) -> dict[str, int]:
    """Groups counted below MIN_GROUP_COUNT are folded into one suppressed
    bucket rather than dropped, so the total still adds up to the real total."""
    visible: dict[str, int] = {}
    suppressed_total = 0
    for label, count in counts.items():
        if count >= MIN_GROUP_COUNT:
            visible[label] = count
        else:
            suppressed_total += count
    if suppressed_total:
        visible[SUPPRESSED_LABEL] = suppressed_total
    return visible


def total_turns(session) -> int:
    return session.query(func.count(LogEntry.id)).scalar() or 0


def volume_by_role(session) -> dict[str, int]:
    rows = session.query(LogEntry.role, func.count(LogEntry.id)).group_by(LogEntry.role).all()
    return _suppress_small_groups({role: count for role, count in rows})


def volume_by_topic(session) -> dict[str, int]:
    rows = (
        session.query(LogEntry.topic_category, func.count(LogEntry.id))
        .group_by(LogEntry.topic_category)
        .all()
    )
    return _suppress_small_groups({topic: count for topic, count in rows})


def escalation_rate(session) -> float:
    """One aggregate percentage across all rows (0.0-100.0), never broken
    down by role or topic — CLAUDE.md's design note: a small cross-tab like
    "1 faculty, law, escalated" is more identifying than either dimension
    alone, so this intentionally has no group-by."""
    total = total_turns(session)
    if total == 0:
        return 0.0
    escalated = session.query(func.count(LogEntry.id)).filter(LogEntry.escalated.is_(True)).scalar() or 0
    return round(100.0 * escalated / total, 1)
