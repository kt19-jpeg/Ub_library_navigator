from streamlit_app.formatting import (
    block_markdown,
    card_detail_markdown,
    card_header_markdown,
    escape_markdown,
    linkify,
    parse_answer,
)

ANSWER = """Here is what I found for "mental health":
Note: something

1. Mental health - a guide - Doe, Jane, 2020 [book]
   Why: Covers the topic.
   Online: https://search/x (via Ebook Central)
   Location: Lockwood - call number RC1 (available)
   Record: https://rec/1

2. Second Title - Roe, 2019 [article]
   Record: https://rec/2

Some online resources ask you to log in with your UB credentials.
More results: https://more/mh

Subject librarian: Doe, Jane - https://staff/doe
Helpful pages:
- Citing Sources: https://research/cite
Browse databases in Psychology: https://research/az.php?s=1"""


def test_parses_intro_cards_and_outro_separately():
    parsed = parse_answer(ANSWER)

    assert parsed.intro == ['Here is what I found for "mental health":', "Note: something"]
    assert [c.number for c in parsed.cards] == [1, 2]
    first = parsed.cards[0]
    assert first.title == "Mental health - a guide"  # only the LAST " - " splits off the byline
    assert first.byline == "Doe, Jane, 2020"
    assert first.kind == "book"
    assert first.why == "Covers the topic."
    assert [label for label, _ in first.details] == ["Online", "Location", "Record"]
    assert parsed.cards[1].why == ""
    assert parsed.outro[0].startswith("Some online resources")
    assert "Subject librarian: Doe, Jane - https://staff/doe" in parsed.outro


def test_escalation_block_is_split_out_with_the_exact_paste_text():
    text = (
        "I am not confident I found a good match for that.\n"
        "You can browse the full OneSearch results here: https://more/x\n\n"
        "Talk to a librarian: https://chat/url\n\n"
        "Context to paste into the chat:\n"
        '"As a graduate, I asked:\n1. first q\n2. second q"'
    )
    parsed = parse_answer(text)

    assert parsed.cards == []  # the numbered questions are NOT result cards
    assert parsed.escalation.chat_url == "https://chat/url"
    assert parsed.escalation.context == '"As a graduate, I asked:\n1. first q\n2. second q"'
    assert parsed.intro[0].startswith("I am not confident")


def test_answer_without_cards_or_escalation_is_all_intro():
    parsed = parse_answer("Just a sentence.")
    assert parsed.intro == ["Just a sentence."]
    assert parsed.cards == [] and parsed.outro == [] and parsed.escalation is None


def test_title_without_byline_or_kind():
    card = parse_answer("1. Plain Title\n   Record: https://r").cards[0]
    assert (card.title, card.byline, card.kind) == ("Plain Title", "", "")


def test_dollar_signs_and_color_syntax_are_escaped():
    assert escape_markdown("Costs $5 and $10") == r"Costs \$5 and \$10"
    assert escape_markdown("see :red[x]") == r"see \:red[x]"
    assert escape_markdown("Plain: text") == "Plain: text"


def test_urls_are_wrapped_so_trailing_text_does_not_break_them():
    assert linkify("https://a/b (via X)") == "<https://a/b> (via X)"


def test_header_and_detail_markdown():
    card = parse_answer(ANSWER).cards[0]
    assert card_header_markdown(card) == "**1. Mental health - a guide**  \nDoe, Jane, 2020 · book"
    assert card_detail_markdown("Record", "https://rec/1") == "**Record:** [View in OneSearch](https://rec/1)"
    assert (
        card_detail_markdown("Online", "https://o/x (via Ebook Central)")
        == "**Online:** [Open online](https://o/x) (via Ebook Central)"
    )
    assert card_detail_markdown("Location", "Lockwood - RC1 (available)") == "**Location:** Lockwood - RC1 (available)"


def test_block_markdown_keeps_lines_separate_within_a_group():
    blocks = block_markdown(["a: https://x", "- b", "", "c"])
    assert blocks == ["a: <https://x>  \n- b", "c"]


def test_trailing_commas_stay_out_of_urls_but_inner_commas_stay_in():
    assert linkify("A - https://s/a, https://s/b") == "A - <https://s/a>, <https://s/b>"
    assert linkify("https://x/search?query=any,contains,cat&vid=1") == "<https://x/search?query=any,contains,cat&vid=1>"
    assert linkify("see https://x/y.") == "see <https://x/y>."
