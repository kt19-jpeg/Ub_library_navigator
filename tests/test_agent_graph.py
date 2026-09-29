from unittest.mock import MagicMock

from agent.graph import build_graph
from agent.models import QueryPlan, Selection, SelectionItem
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult
from tests.fakes import FakeChatModel


def _candidate(record_id):
    return SearchCandidate(
        record_id=record_id, title="Community Mental Health Journal", author=None,
        resource_type="journal", format=None, subject=None, description=None, date="2020",
        source="Alma", delivery_category="Alma-E", online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )


def _client():
    client = MagicMock()
    client.search_with_fallback.return_value = SearchResult(
        candidates=[_candidate("j1"), _candidate("j2")], timed_out=False, fallback_link=None
    )
    client.fetch_detail.return_value = {
        "j1": ResourceDetail(
            record_id="j1", online_link="https://search.lib.buffalo.edu/view/j1",
            online_package="SpringerLink", call_number=None, library=None,
            sub_location=None, availability_status=None,
        )
    }
    client.record_link.side_effect = lambda rid: f"https://rec/{rid}"
    client.fallback_link.side_effect = lambda q: f"https://more/{q}"
    return client


def test_full_turn_plans_searches_and_composes():
    model = FakeChatModel(
        QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True),
        Selection(items=[SelectionItem(record_id="j1", reason="Core journal in the field.")],
                  confidence="high"),
    )
    client = _client()
    graph = build_graph(model, client)

    out = graph.invoke({
        "role": "graduate",
        "message": "find me peer reviewed journals on mental health",
        "history": [],
    })

    client.search_with_fallback.assert_called_once_with(
        "mental health", resource_type="journals", peer_reviewed=True
    )
    client.fetch_detail.assert_called_once_with(["j1"])
    assert "https://search.lib.buffalo.edu/view/j1" in out["answer"]
    assert "Core journal in the field." in out["answer"]
    assert out["plan"].resource_type == "journals"
    assert set(out["timings"]) == {"plan", "search", "rag", "compose"}
    assert out["rag_context"] is None  # no collection passed to build_graph in this test


def test_clarifying_question_ends_the_turn_before_searching():
    model = FakeChatModel(QueryPlan(keywords="", needs_clarification="What topic are you researching?"))
    client = _client()
    graph = build_graph(model, client)

    out = graph.invoke({"role": "graduate", "message": "find me something", "history": []})

    assert out["answer"] == "What topic are you researching?"
    client.search_with_fallback.assert_not_called()
    client.fetch_detail.assert_not_called()


def test_planner_failure_still_produces_an_answer():
    model = FakeChatModel(
        RuntimeError("provider down"),
        Selection(items=[SelectionItem(record_id="j2", reason="Relevant.")], confidence="medium"),
    )
    client = _client()
    graph = build_graph(model, client)

    out = graph.invoke({"role": "undergraduate", "message": "mental health help", "history": []})

    client.search_with_fallback.assert_called_once_with(
        "mental health help", resource_type=None, peer_reviewed=False
    )
    assert "Community Mental Health Journal" in out["answer"]


def test_rag_hits_flow_through_to_the_final_answer(tmp_path):
    import chromadb

    from pipeline.embed import COLLECTION_NAME, sync_collection
    from pipeline.records import LIBRARIAN_SUBJECT, Record
    from tests.fake_embeddings import FakeEmbeddingFunction

    client_ = chromadb.PersistentClient(path=str(tmp_path))
    collection = client_.get_or_create_collection(COLLECTION_NAME, embedding_function=FakeEmbeddingFunction())
    sync_collection(
        [Record(id="librarian:mental-health", type=LIBRARIAN_SUBJECT, text="mental health", metadata={
            "subject": "Mental Health", "librarians": [{"name": "Doe, Jane", "profile_url": "https://staff/doe"}]
        })],
        collection,
    )
    import agent.rag as rag_module
    for t in rag_module.RAG_DISTANCE_CUTOFF:
        rag_module.RAG_DISTANCE_CUTOFF[t] = 999.0  # fake embeddings aren't semantic; see tests/test_agent_rag.py

    model = FakeChatModel(
        QueryPlan(keywords="mental health", resource_type="journals", peer_reviewed=True),
        Selection(items=[SelectionItem(record_id="j1", reason="Core journal in the field.")], confidence="high"),
    )
    graph = build_graph(model, _client(), collection=collection)

    out = graph.invoke({"role": "graduate", "message": "mental health journals", "history": []})

    assert out["rag_context"].librarians[0].metadata["subject"] == "Mental Health"
    assert "Subject librarian: Doe, Jane - https://staff/doe" in out["answer"]
