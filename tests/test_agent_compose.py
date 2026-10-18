from unittest.mock import MagicMock

from agent.compose import (
    DESCRIPTION_CHARS,
    MAX_SHOWN,
    compose,
    is_library_help_request,
    select_results,
    validate_selection,
)
from agent.models import QueryPlan, RagContext, Selection, SelectionItem
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult
from tests.fakes import FakeChatModel


def _candidate(record_id, title="A Title", **overrides):
    fields = dict(
        record_id=record_id, title=title, author="Doe, Jane", resource_type="book",
        format=None, subject="Topic", description="About the topic.", date="2020",
        source="Alma", delivery_category="Alma-E", online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )
    fields.update(overrides)
    return SearchCandidate(**fields)


def _detail(record_id, **overrides):
    fields = dict(
        record_id=record_id, online_link=None, online_package=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )
    fields.update(overrides)
    return ResourceDetail(**fields)


def _client(details=None):
    client = MagicMock()
    client.fetch_detail.return_value = details or {}
    client.record_link.side_effect = lambda rid: f"https://rec/{rid}"
    client.fallback_link.side_effect = lambda q: f"https://more/{q}"
    return client


def _state(candidates, **extra):
    state = {
        "role": "graduate",
        "message": "find me peer reviewed journals on mental health",
        "plan": QueryPlan(keywords="mental health"),
        "search_result": SearchResult(candidates=candidates, timed_out=False, fallback_link=None),
        "rag_context": RagContext(),
        "notes": [],
    }
    state.update(extra)
    return state


def test_is_library_help_request_matches_explicit_service_questions():
    for message in [
        "Who is the subject librarian for chemistry?",
        "How do I contact a librarian?",
        "How do I cite a journal article in APA?",
        "How do I print from the library computers?",
        "Can you help me browse the databases for biology?",
    ]:
        assert is_library_help_request(message), message


def test_is_library_help_request_does_not_match_ordinary_topic_searches():
    for message in [
        "find me peer reviewed journals on mental health",
        "books about the history of chemistry",
        "I need articles on climate change databases in Africa",
    ]:
        assert not is_library_help_request(message), message


def test_validate_selection_drops_unknown_duplicates_and_caps():
    candidates = [_candidate(str(i)) for i in range(6)]
    items = [SelectionItem(record_id=i, reason="r") for i in ["9", "1", "1", "2", "3", "4"]]

    kept = validate_selection(items, candidates)

    assert [i.record_id for i in kept] == ["1", "2", "3"]
    assert len(kept) == MAX_SHOWN


def test_select_results_sends_truncated_candidate_json_and_role():
    long = "x" * (DESCRIPTION_CHARS + 100)
    model = FakeChatModel(Selection(items=[], confidence="low"))

    select_results(model, "faculty", "the request", [_candidate("a1", description=long)], RagContext())

    human = model.calls[0][1][1]
    assert "role: faculty" in human and "request: the request" in human
    assert '"record_id": "a1"' in human
    assert "x" * (DESCRIPTION_CHARS + 1) not in human
    assert model.schemas == [Selection]


def test_compose_fetches_details_only_for_valid_selected_ids_and_renders_them():
    candidates = [_candidate("e1", title="Online Journal"), _candidate("p1", title="Print Book"),
                  _candidate("z9", title="Not Picked")]
    model = FakeChatModel(Selection(
        items=[SelectionItem(record_id="e1", reason="Covers mental health research."),
               SelectionItem(record_id="bogus", reason="invented"),
               SelectionItem(record_id="p1", reason="A core textbook.")],
        confidence="high"))
    client = _client({
        "e1": _detail("e1", online_link="https://search.lib.buffalo.edu/view/x", online_package="SpringerLink"),
        "p1": _detail("p1", call_number="RC454 .M46 2020", library="Lockwood Library",
                      sub_location="General Collection", availability_status="unavailable"),
    })

    out = compose(_state(candidates), model, client)

    client.fetch_detail.assert_called_once_with(["e1", "p1"])
    answer = out["answer"]
    assert "1. Online Journal - Doe, Jane, 2020 [book]" in answer
    assert "Why: Covers mental health research." in answer
    assert "Online: https://search.lib.buffalo.edu/view/x (via SpringerLink)" in answer
    assert "Lockwood Library / General Collection - call number RC454 .M46 2020 (currently unavailable)" in answer
    assert "Record: https://rec/e1" in answer
    assert "bogus" not in answer and "Not Picked" not in answer
    assert "log in with your UB credentials" in answer
    assert "More results: https://more/mental health" in answer
    assert [i.record_id for i in out["selection"]] == ["e1", "p1"]


def test_compose_low_confidence_is_honest_and_skips_detail_fetch():
    model = FakeChatModel(Selection(items=[], confidence="low"))
    client = _client()

    out = compose(_state([_candidate("a")]), model, client)

    client.fetch_detail.assert_not_called()
    assert "not confident" in out["answer"]
    assert "https://more/mental health" in out["answer"]
    assert out["selection"] == []


def test_compose_low_confidence_includes_the_live_chat_escalation():
    from agent.escalation import LIVE_CHAT_URL

    model = FakeChatModel(Selection(items=[], confidence="low"))
    client = _client()

    out = compose(_state([_candidate("a")]), model, client)

    assert LIVE_CHAT_URL in out["answer"]
    assert "graduate" in out["answer"]
    assert "find me peer reviewed journals on mental health" in out["answer"]


def test_compose_llm_failure_falls_back_to_top_three_without_reasons():
    candidates = [_candidate(str(i), title=f"T{i}") for i in range(5)]
    model = FakeChatModel(RuntimeError("provider down"))
    client = _client()

    out = compose(_state(candidates), model, client)

    client.fetch_detail.assert_called_once_with(["0", "1", "2"])
    assert "T0" in out["answer"] and "T2" in out["answer"] and "T3" not in out["answer"]
    assert "Why:" not in out["answer"]
    assert "could not rank" in out["answer"]


def test_compose_survives_detail_fetch_failure_using_search_fields():
    candidates = [_candidate("p1", call_number="QA1 .B2", library="Lockwood Library",
                             availability_status="available", delivery_category="Alma-P")]
    model = FakeChatModel(Selection(items=[SelectionItem(record_id="p1", reason="Fits.")],
                                    confidence="medium"))
    client = _client()
    client.fetch_detail.side_effect = RuntimeError("HTTP 500")

    out = compose(_state(candidates), model, client)

    assert "call number QA1 .B2 (available)" in out["answer"]
    assert "could not load the full access details" in out["answer"]


def test_compose_with_no_candidates_skips_the_model_and_gives_the_link():
    model = FakeChatModel()
    state = _state([], notes=["OneSearch was slow to respond."])
    state["search_result"] = SearchResult(candidates=[], timed_out=True, fallback_link="https://direct/link")

    out = compose(state, model, _client())

    assert model.calls == []
    assert "could not find results" in out["answer"]
    assert "https://direct/link" in out["answer"]
    assert "OneSearch was slow to respond." in out["answer"]


def test_compose_with_no_candidates_also_includes_the_live_chat_escalation():
    from agent.escalation import LIVE_CHAT_URL

    out = compose(_state([]), FakeChatModel(), _client())

    assert LIVE_CHAT_URL in out["answer"]
    assert "Context to paste into the chat:" in out["answer"]


def _full_rag_context():
    from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record

    return RagContext(
        librarians=[Record(id="librarian:mh", type=LIBRARIAN_SUBJECT, text="", metadata={
            "subject": "Mental Health", "librarians": [{"name": "Doe, Jane", "profile_url": "https://staff/doe"}]
        })],
        help_topics=[Record(id="help:citing", type=HELP_TOPIC, text="", metadata={
            "title": "Citing Sources", "url": "https://research/citingsources"
        })],
        databases=[Record(id="db:mh", type=DATABASE_SUBJECT, text="", metadata={
            "subject": "Psychology", "count": 12, "browse_url": "https://research/az.php"
        })],
    )


def test_compose_offers_rag_help_instead_of_dumping_it_for_an_ordinary_resource_request():
    # Regression: a plain resource request used to get the full librarian/help/
    # database footer appended unasked on every answer. Now it just gets a
    # one-line offer, since the request ("...on mental health", see _state())
    # didn't actually ask about library services.
    candidates = [_candidate("e1")]
    model = FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="Fits.")], confidence="high"))
    client = _client()

    out = compose(_state(candidates, rag_context=_full_rag_context()), model, client)

    answer = out["answer"]
    assert "Subject librarian" not in answer
    assert "Helpful pages" not in answer
    assert "Browse databases" not in answer
    assert "Want a subject librarian, related help guides, or database suggestions" in answer


def test_compose_shows_the_full_rag_footer_for_an_explicit_library_services_request():
    candidates = [_candidate("e1")]
    model = FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="Fits.")], confidence="high"))
    client = _client()
    state = _state(
        candidates,
        rag_context=_full_rag_context(),
        message="who is the subject librarian for mental health?",
    )

    out = compose(state, model, client)

    answer = out["answer"]
    assert "Subject librarian: Doe, Jane - https://staff/doe" in answer
    assert "Helpful pages:" in answer and "- Citing Sources: https://research/citingsources" in answer
    assert "Browse databases in Psychology: https://research/az.php" in answer
    assert "Want a subject librarian" not in answer


def test_compose_appends_rag_sections_even_with_no_candidates():
    from pipeline.records import HELP_TOPIC, Record

    rag_context = RagContext(help_topics=[
        Record(id="help:a", type=HELP_TOPIC, text="", metadata={"title": "Ask A Librarian", "url": "https://x"})
    ])
    state = _state([], rag_context=rag_context)

    out = compose(state, FakeChatModel(), _client())

    assert "Helpful pages:" in out["answer"]
    assert "- Ask A Librarian: https://x" in out["answer"]


def test_compose_prepends_out_of_scope_note_ahead_of_the_normal_answer():
    candidates = [_candidate("e1", title="Algorithms Textbook")]
    model = FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="Covers it.")], confidence="high"))
    note = "I can't prepare your midterm notes for you, but here are some resources on algorithms."
    state = _state(
        candidates,
        plan=QueryPlan(keywords="algorithms", out_of_scope_note=note),
        message="can you help me prepare notes for my algorithms midterm from books available at UB",
    )

    out = compose(state, model, _client())

    assert out["answer"].startswith(note)
    assert "Algorithms Textbook" in out["answer"]


def test_compose_prepends_out_of_scope_note_even_with_no_candidates():
    note = "I can't write your essay for you, but here are some resources on climate policy."
    state = _state([], plan=QueryPlan(keywords="climate policy", out_of_scope_note=note))

    out = compose(state, FakeChatModel(), _client())

    assert out["answer"].startswith(note)
    assert "could not find results" in out["answer"]


def test_compose_without_out_of_scope_note_does_not_add_a_prefix():
    out = compose(_state([_candidate("e1")]),
                   FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="ok")], confidence="high")),
                   _client())
    assert not out["answer"].startswith("I can't")


def test_compose_omits_rag_footer_when_context_is_empty():
    out = compose(_state([_candidate("e1")], rag_context=RagContext()),
                   FakeChatModel(Selection(items=[SelectionItem(record_id="e1", reason="ok")], confidence="high")),
                   _client())
    assert "Subject librarian" not in out["answer"]
    assert "Helpful pages" not in out["answer"]
    assert "Browse databases" not in out["answer"]
