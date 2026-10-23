# Staff Usage Dashboard (Layer 2, Slice 8)

Date: 2026-10-05. Status: built autonomously (user away), decisions documented here for review.

## Goal

CLAUDE.md capability 8: "Library-staff-only view summarizing query volume by role, topic category
breakdown, and escalation rate." Reads the logging DB (slice 7).

## Decisions made without a round-trip, with reasoning

**Auth: a simple, swappable shared-password gate — not real SSO.** The original brief says this
explicitly: "The method (UB SSO versus a shared staff login) is still pending confirmation from
the library... build this behind a simple, swappable interface for now rather than hardcoding a
specific auth method." `STAFF_DASHBOARD_AUTH_METHOD` already existed as a placeholder in
`.env.example`; `analytics/auth.py` implements one method (`"password"`, checked against a new
`STAFF_DASHBOARD_PASSWORD` secret) behind a small registry so swapping in real SSO later means
adding a method to that registry, not rewriting the page. An unconfigured or unrecognized method
fails the gate closed (no access), never open.

**Small-count suppression, per the brief's own documented risk.** "Known risks" in the original
brief: "a rare combination of course, role, and topic could be identifiable... suppress any
breakdown below a small count threshold." Any role or topic group with fewer than
`MIN_GROUP_COUNT = 5` rows is folded into an "Other (suppressed)" bucket rather than shown with
its own count. Escalation rate is shown only as one aggregate percentage across all rows, never
cross-tabulated with a small role/topic breakdown — a combination like "1 faculty, law,
escalated" would be far more identifying than either dimension alone.

**No raw rows, no timestalmps-as-a-list, no per-entry browsing.** The dashboard only ever shows
pre-aggregated counts. A page that let staff scroll through individual log rows would defeat the
anonymity the logging schema itself already enforces (no query text, no identifier) by potentially
letting someone pattern-match a sequence of role+topic+timestamp rows back to a specific
conversation.

## What the page shows

1. Total turns logged (one number).
2. Query volume by role (bar/table, small groups suppressed).
3. Topic category breakdown (bar/table, small groups suppressed).
4. Escalation rate (one aggregate percentage, not broken down).

## Testing

Offline: `analytics/auth.py`'s gate (correct password, wrong password, unconfigured, unknown
method — all against env vars, not a real session); `analytics/dashboard_queries.py`'s
aggregation and suppression logic against a temp SQLite DB seeded with known rows, including a
case that specifically exercises the suppression threshold. Live: one real check that the gate
actually blocks without the password and admits with it, and that the aggregated numbers match
what's really in the DB from the logging slice's own live check.
