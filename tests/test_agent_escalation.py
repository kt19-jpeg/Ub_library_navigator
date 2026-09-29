from agent.escalation import LIVE_CHAT_URL, build_escalation_block


def test_includes_the_real_live_chat_url():
    block = build_escalation_block("graduate", "mental health journals", [])
    assert LIVE_CHAT_URL in block
    assert LIVE_CHAT_URL == "https://library.buffalo.edu/help-support/askalibrarian/chat/instant.html"


def test_includes_a_copyable_context_line_with_role_and_message():
    block = build_escalation_block("undergraduate", "help with citations", [])
    assert "undergraduate" in block
    assert "help with citations" in block


def test_includes_notes_when_present():
    block = build_escalation_block("faculty", "rare manuscripts", ["OneSearch was slow to respond."])
    assert "OneSearch was slow to respond." in block


def test_omits_notes_section_when_there_are_none():
    block = build_escalation_block("faculty", "rare manuscripts", [])
    assert "OneSearch" not in block


def test_labels_the_block_so_the_user_knows_to_paste_it():
    block = build_escalation_block("graduate", "x", [])
    assert "paste" in block.lower()
