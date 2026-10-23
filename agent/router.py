import re

# Deterministic, no LLM call (CLAUDE.md, "Two deterministic fast paths"): a short,
# conservative list of patterns for a message that is clearly and explicitly asking
# for a human, not a message that merely mentions a person/topic in passing. Kept
# narrow on purpose — a real request phrased unusually just falls through to the
# normal plan/search/compose flow instead of triggering a false positive, which is
# the safer failure mode for a fast path that skips ranking entirely.
_HUMAN_REQUEST_PATTERNS = [
    r"\b(talk|speak) (to|with) (a |an )?(real )?(human|person|someone|librarian)\b",
    r"\bconnect me (to|with) (a |an )?(human|person|someone|librarian)\b",
    r"\b(chat|talk) with (a )?(live|real) (person|librarian|human)\b",
    r"\bI (want|need|'d like) to (talk|speak) to (a )?(human|person|someone|librarian)\b",
    r"^\s*(a |an )?(real )?(human|person)\s*[.!?]*\s*$",  # just "human." or "a person" alone
    r"\btalk to a librarian\b",
]
_HUMAN_REQUEST_RE = re.compile("|".join(_HUMAN_REQUEST_PATTERNS), re.IGNORECASE)


def is_human_request(message: str) -> bool:
    """True only for a message that's clearly, explicitly asking for a human —
    not one that merely contains a word like "person" as part of something else."""
    return bool(_HUMAN_REQUEST_RE.search(message))
