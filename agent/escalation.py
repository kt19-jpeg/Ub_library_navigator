LIVE_CHAT_URL = "https://library.buffalo.edu/help-support/askalibrarian/chat/instant.html"


def build_escalation_block(
    role: str, message: str, notes: list[str], history: list[dict] | None = None
) -> str:
    """A link to UB's live chat plus a short, copy-paste-ready summary of the
    question. Confirmed live 2026-09-29: the chat widget (LibraryH3lp) is a plain
    "type here to chat" box built by UB's own JavaScript, with no documented way
    to pre-fill a message via URL and no API of any kind, so this cannot inject
    the context automatically - the user pastes it in themselves once the chat
    opens. Always link to LIVE_CHAT_URL directly, never construct an h3lp iframe
    URL: the queue id it resolves to is an internal detail that could change.

    `history` (the same [{"user", "assistant"}, ...] list the graph already
    carries) is optional and, when given with at least one prior turn, numbers
    every question the user asked this conversation, ending with the current
    `message` - a librarian picking this up cold gets the real arc of what was
    asked, not just the last sentence (confirmed live 2026-10-22: a user who'd
    already asked a resource question and then asked for help got a paste block
    quoting only "can someone help me directly?" with no sign they'd been
    asking about mental health journals a moment earlier). With no prior turns,
    the block stays a single line, unchanged from before."""
    lines = [f"Talk to a librarian: {LIVE_CHAT_URL}", "", "Context to paste into the chat:"]
    history = history or []
    if history:
        questions = [turn["user"] for turn in history] + [message]
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
        context = f'"As a {role}, I asked:\n{numbered}"'
    else:
        context = f'"As a {role}, I asked: {message}"'
    if notes:
        context += " (" + " ".join(notes) + ")"
    lines.append(context)
    return "\n".join(lines)
