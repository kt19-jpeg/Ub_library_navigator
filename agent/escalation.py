LIVE_CHAT_URL = "https://library.buffalo.edu/help-support/askalibrarian/chat/instant.html"


def build_escalation_block(role: str, message: str, notes: list[str]) -> str:
    """A link to UB's live chat plus a short, copy-paste-ready summary of the
    question. Confirmed live 2026-09-29: the chat widget (LibraryH3lp) is a plain
    "type here to chat" box built by UB's own JavaScript, with no documented way
    to pre-fill a message via URL and no API of any kind, so this cannot inject
    the context automatically - the user pastes it in themselves once the chat
    opens. Always link to LIVE_CHAT_URL directly, never construct an h3lp iframe
    URL: the queue id it resolves to is an internal detail that could change."""
    lines = [f"Talk to a librarian: {LIVE_CHAT_URL}", "", "Context to paste into the chat:"]
    context = f'"As a {role}, I asked: {message}"'
    if notes:
        context += " (" + " ".join(notes) + ")"
    lines.append(context)
    return "\n".join(lines)
