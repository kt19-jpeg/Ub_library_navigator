"""Turns the agent's plain-text answer into structured pieces the Streamlit UI can lay
out as separate cards. Pure functions, no Streamlit import, so they are unit-tested.

Why parse text at all: `agent.compose` renders answers as plain text (the CLI prints
it as-is, and links/locations must stay code-rendered, never LLM output). Streamlit's
Markdown collapses single newlines, so that text used to render as one run-on
paragraph - results ran together with no visible distinction. Parsing it back into
cards fixes the UI without changing the agent."""

import re
from dataclasses import dataclass, field

ESCALATION_MARKER = "Talk to a librarian: "
CONTEXT_MARKER = "Context to paste into the chat:"
_ITEM_RE = re.compile(r"^(\d+)\. (.+)$")
_FIELD_RE = re.compile(r"^ {3}([A-Za-z ]+): (.+)$")
_KIND_RE = re.compile(r"^(.*) \[([^\]]+)\]$")
# Lazy, stopping before trailing punctuation: a profile URL followed by ", " must not
# swallow the comma, but commas INSIDE a URL (OneSearch queries) must survive.
_URL_RE = re.compile(r"https?://\S*?(?=[,.;:)]*(?:\s|$))")


@dataclass
class ResultCard:
    number: int
    title: str
    byline: str = ""
    kind: str = ""
    why: str = ""
    # (label, value) in display order, e.g. ("Online", "https://... (via X)")
    details: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Escalation:
    chat_url: str
    context: str  # the exact text to paste, quotes and numbering included


@dataclass
class ParsedAnswer:
    intro: list[str] = field(default_factory=list)
    cards: list[ResultCard] = field(default_factory=list)
    outro: list[str] = field(default_factory=list)  # grouped lines; "" separates groups
    escalation: Escalation | None = None


def escape_markdown(text: str) -> str:
    """`$` starts LaTeX in Streamlit and `:name[` starts colored-text syntax; neither
    should ever fire on library titles."""
    return re.sub(r":(?=[A-Za-z-]+\[)", r"\\:", text.replace("$", r"\$"))


def _split_escalation(lines: list[str]) -> tuple[list[str], Escalation | None]:
    for index, line in enumerate(lines):
        if line.startswith(ESCALATION_MARKER):
            chat_url = line[len(ESCALATION_MARKER):].strip()
            rest = lines[index + 1:]
            if CONTEXT_MARKER in rest:
                rest = rest[rest.index(CONTEXT_MARKER) + 1:]
            context = "\n".join(rest).strip()
            return lines[:index], Escalation(chat_url=chat_url, context=context)
    return lines, None


def _parse_title_line(number: int, text: str) -> ResultCard:
    kind = ""
    match = _KIND_RE.match(text)
    if match:
        text, kind = match.group(1), match.group(2)
    title, sep, byline = text.rpartition(" - ")
    if not sep:
        title, byline = text, ""
    return ResultCard(number=number, title=title.strip(), byline=byline.strip(), kind=kind)


def parse_answer(text: str) -> ParsedAnswer:
    lines, escalation = _split_escalation(text.split("\n"))
    parsed = ParsedAnswer(escalation=escalation)
    current: ResultCard | None = None
    seen_card = False
    for line in lines:
        item = _ITEM_RE.match(line)
        if item and not (current is None and seen_card):
            current = _parse_title_line(int(item.group(1)), item.group(2))
            parsed.cards.append(current)
            seen_card = True
            continue
        if current is not None:
            detail = _FIELD_RE.match(line)
            if detail:
                label, value = detail.group(1), detail.group(2)
                if label == "Why":
                    current.why = value
                else:
                    current.details.append((label, value))
                continue
            if not line.strip():
                continue  # blank line after a card
            current = None  # first non-card line ends the card list
            parsed.outro.append(line)
            continue
        (parsed.outro if seen_card else parsed.intro).append(line)
    parsed.intro = _trim(parsed.intro)
    parsed.outro = _trim(parsed.outro)
    return parsed


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def linkify(value: str) -> str:
    """Wraps bare URLs in <...> so they render as clean links even next to
    punctuation or "(via ...)" text; leaves everything else untouched."""
    return _URL_RE.sub(lambda m: f"<{m.group(0)}>", value)


def card_header_markdown(card: ResultCard) -> str:
    meta = " · ".join(p for p in (card.byline, card.kind) if p)
    head = f"**{card.number}. {escape_markdown(card.title)}**"
    return head + (f"  \n{escape_markdown(meta)}" if meta else "")


LINK_LABELS = {"Online": "Open online", "Record": "View in OneSearch"}


def card_detail_markdown(label: str, value: str) -> str:
    """Long catalog URLs read as noise, so Online/Record show a short link label
    (any trailing "(via ...)" text is kept). Other fields render as-is."""
    if label in LINK_LABELS:
        match = _URL_RE.match(value)
        if match:
            rest = escape_markdown(value[match.end():])
            return f"**{label}:** [{LINK_LABELS[label]}]({match.group(0)}){rest}"
    return f"**{label}:** {linkify(escape_markdown(value))}"


def block_markdown(lines: list[str]) -> list[str]:
    """Groups lines (blank-line separated) into markdown paragraphs whose lines stay on
    separate rows - Streamlit collapses single newlines, so join with a hard break."""
    blocks, group = [], []
    for line in lines + [""]:
        if line.strip():
            group.append(linkify(escape_markdown(line)))
        elif group:
            blocks.append("  \n".join(group))
            group = []
    return blocks
