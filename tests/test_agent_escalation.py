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


def test_no_history_still_renders_a_single_line_question():
    block = build_escalation_block("graduate", "mental health journals", [])
    assert '"As a graduate, I asked: mental health journals"' in block


def test_with_history_numbers_every_question_ending_with_the_current_one():
    # Regression: confirmed live 2026-10-22 — a user who'd already asked a
    # resource question and then asked for human help got a paste block
    # quoting only the final message, with no sign they'd asked about mental
    # health journals a moment earlier.
    history = [
        {"user": "find me peer reviewed journals on mental health", "assistant": "Here is what I found..."},
        {"user": "what about books instead?", "assistant": "Here is what I found..."},
    ]
    block = build_escalation_block(
        "graduate", "these results are not what I need, can someone help me directly?", [], history
    )
    assert "1. find me peer reviewed journals on mental health" in block
    assert "2. what about books instead?" in block
    assert "3. these results are not what I need, can someone help me directly?" in block


def test_empty_history_list_behaves_like_no_history():
    block = build_escalation_block("graduate", "mental health journals", [], [])
    assert '"As a graduate, I asked: mental health journals"' in block
    assert "1." not in block


def test_notes_still_appended_after_a_multi_turn_context():
    history = [{"user": "find me journals on mental health", "assistant": "Here is what I found..."}]
    block = build_escalation_block("graduate", "can someone help me directly?", ["A note."], history)
    assert block.rstrip().endswith("(A note.)")


def test_include_current_false_omits_the_trigger_message_with_prior_history():
    # The real complaint: a paste block quoting "I asked: can you bring in
    # human in loop i want more info" tells a librarian nothing useful - the
    # real topic was the prior turn.
    history = [{"user": "find me peer reviewed journals on mental health", "assistant": "Here is what I found..."}]
    block = build_escalation_block(
        "graduate", "can you bring in human in loop i want more info", [], history, include_current=False
    )
    assert '"As a graduate, I asked: find me peer reviewed journals on mental health"' in block
    assert "human in loop" not in block


def test_include_current_false_with_multiple_prior_turns_still_numbers_them():
    history = [
        {"user": "find me peer reviewed journals on mental health", "assistant": "Here is what I found..."},
        {"user": "what about books instead?", "assistant": "Here is what I found..."},
    ]
    block = build_escalation_block(
        "graduate", "can someone help me directly?", [], history, include_current=False
    )
    assert "1. find me peer reviewed journals on mental health" in block
    assert "2. what about books instead?" in block
    assert "can someone help me directly" not in block


def test_include_current_false_with_no_history_falls_back_to_a_generic_line():
    # First-turn human request, nothing substantive asked yet: no topic to
    # quote, so don't quote the human-request phrasing either.
    block = build_escalation_block("graduate", "I want to talk to a human", [], include_current=False)
    assert '"As a graduate, I would like assistance."' in block
    assert "talk to a human" not in block
