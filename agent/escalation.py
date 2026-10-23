LIVE_CHAT_URL = "https://library.buffalo.edu/help-support/askalibrarian/chat/instant.html"


def build_escalation_block(
    role: str,
    message: str,
    notes: list[str],
    history: list[dict] | None = None,
    include_current: bool = True,
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
    every question the user asked this conversation - a librarian picking this
    up cold gets the real arc of what was asked, not just the last sentence
    (confirmed live 2026-10-22: a user who'd already asked a resource question
    and then asked for help got a paste block quoting only "can someone help me
    directly?" with no sign they'd been asking about mental health journals a
    moment earlier). With no prior turns, the block stays a single line.

    `include_current` controls whether `message` itself is appended to that
    list. Callers whose `message` IS the real question (an ordinary search that
    returned nothing, or came back low-confidence) leave it True, the default -
    that text is exactly what a librarian needs. Callers whose `message` is
    just the ask-for-a-human trigger itself ("can you bring in a human", "talk
    to a human") pass False, so the paste block shows the real prior topic
    instead of a redundant line quoting the human-request message (confirmed
    live 2026-10-22: the user didn't want "I asked: can you bring in human in
    loop i want more info" cluttering the context - that phrase tells a
    librarian nothing about what the patron actually needs help with)."""
    lines = [f"Talk to a librarian: {LIVE_CHAT_URL}", "", "Context to paste into the chat:"]
    questions = [turn["user"] for turn in (history or [])]
    if include_current:
        questions.append(message)
    if not questions:
        context = f'"As a {role}, I would like assistance."'
    elif len(questions) == 1:
        context = f'"As a {role}, I asked: {questions[0]}"'
    else:
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
        context = f'"As a {role}, I asked:\n{numbered}"'
    if notes:
        context += " (" + " ".join(notes) + ")"
    lines.append(context)
    return "\n".join(lines)
