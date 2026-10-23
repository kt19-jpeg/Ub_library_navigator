# Anonymized Conversation Logging (Layer 2, Slice 7)

Date: 2026-10-05. Status: built autonomously (user away), decisions documented here for review.

## Goal

CLAUDE.md capability 7 (`git show dd388a3:CLAUDE.md`): "Every session is logged in fully
anonymous form, capturing role, course if provided, and a topic category rather than raw question
text. No name or identifier is ever attached to a log entry." Layer 2's orchestration section
reiterates: "Every turn, regardless of which path it took, writes one anonymized log entry (role,
course if given, topic category — never raw query text or an identifier)."

## Decisions made without a round-trip (user unavailable), with reasoning

**Topic category comes from the planner, at zero extra cost.** A genuine topic category needs
semantic understanding of the message — a keyword-bucket classifier would be crude and brittle.
Rather than add a third LLM call per turn, `QueryPlan` gains one more field,
`topic_category: Literal[...]` (a fixed 10-value enum — see below), filled by the *same* planning
call that already produces `keywords`/`resource_type`/`peer_reviewed`. This matches CLAUDE.md's
own stated principle for the planner: "This call can share a prompt with the fast-path
classification so the turn costs one LLM round trip, not two." Fallback on any planning failure:
`"other"`, same spirit as the existing `fallback_plan()`.

**The category list** (coarse, academic-domain buckets, chosen to stay broad enough that a single
category alone can't narrow down to an individual, per the original brief's own re-identification
concern): `sciences`, `health_and_medicine`, `engineering_and_technology`, `social_sciences`,
`business_and_economics`, `humanities_and_arts`, `law`, `education`, `library_services` (for
meta/service questions — "who's my librarian," citation help, printing — not an academic subject),
`other`.

**`course` is always `None` for now.** Nothing in the system today asks for or captures a course
— no UI field exists. The schema has the column (matching the brief's field list exactly), but
it's genuinely empty until a future slice adds a way to capture it. Documented as a known gap, not
silently dropped from the brief.

**`escalated` and `had_results` are added beyond the brief's literal field list.** Not in capability
7's text, but required by capability 8 (staff dashboard: "escalation rate") and by the dashboard
slice that follows this one. Both are booleans, not identifying.

**No session/conversation identifier of any kind.** Each log entry is independent — nothing ties
multiple turns from the same person together, which the brief's own anonymity requirement implies
even though it isn't stated in so many words: a linkable session ID would let someone reconstruct
one person's whole conversation even without a name attached.

**Where logging happens: the caller, not the graph.** `agent/graph.py` stays a pure
orchestration/reasoning layer with no disk I/O, matching how `compose.py` already renders without
writing anywhere. `agent/cli.py`'s `run_chat()` and `streamlit_app/Home.py`'s turn handler each
call `analytics.log_writer.log_turn(role, state)` right after `graph.invoke()` returns — both
already have the full `state` dict in hand. This covers every path (the human-request fast path,
the clarification short-circuit, and the normal flow) uniformly, since all three return through
the same `graph.invoke()` call site in each caller.

## Schema

`analytics/db.py`, SQLAlchemy + SQLite (`DATABASE_URL` already in `.env.example`):

```
LogEntry: id, timestamp, role, course (nullable), topic_category, resource_type (nullable),
          escalated (bool), had_results (bool)
```

## Testing

Offline: `analytics/db.py` against a temp SQLite file (not the real `analytics/sessions.db`);
`analytics/log_writer.py`'s field derivation from constructed `state` dicts covering all three
paths (fast-path escalation, clarification, normal); `agent/planner.py`'s new field via the
existing `FakeChatModel` pattern. Live: one real run confirming a log row actually lands in the
real SQLite file with no raw query text in any column.
