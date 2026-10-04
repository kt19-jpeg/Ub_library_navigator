import json

from agent.escalation import LIVE_CHAT_URL
from agent.models import QueryPlan, RagContext, SelectionItem
from evaluation.faithfulness import FaithfulnessVerdict
from evaluation.run_eval import evaluate, print_summary
from onesearch.models import SearchCandidate, SearchResult
from pipeline.records import LIBRARIAN_SUBJECT, Record
from tests.fakes import FakeChatModel


def _candidate(record_id, **overrides):
    fields = dict(
        record_id=record_id, title="A Title", author="Doe, Jane", resource_type="book",
        format=None, subject="Topic", description="About the topic.", date="2020",
        source="Alma", delivery_category="Alma-E", online_link=None, call_number=None,
        library=None, sub_location=None, availability_status=None,
    )
    fields.update(overrides)
    return SearchCandidate(**fields)


class FakeGraph:
    """Scripted per-question states, matched by message so test order doesn't matter."""

    def __init__(self, states_by_message):
        self._states = states_by_message
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        return self._states[state["message"]]


def _answer_state(plan, candidates, selection, rag_context=None, answer="Here is what I found..."):
    return {
        "plan": plan,
        "answer": answer,
        "search_result": SearchResult(candidates=candidates, timed_out=False, fallback_link=None),
        "selection": selection,
        "rag_context": rag_context or RagContext(),
    }


def test_evaluate_scores_every_question_and_reports_faithfulness():
    questions = [
        {
            "id": "q1", "role": "graduate", "message": "chemistry resources",
            "expected_resource_type": "articles", "expected_peer_reviewed": True,
            "should_escalate": False,
        },
        {
            "id": "q2", "role": "graduate", "message": "nonsense query",
            "expected_resource_type": None, "expected_peer_reviewed": False,
            "should_escalate": True,
        },
    ]
    candidate = _candidate("c1")
    states = {
        "chemistry resources": _answer_state(
            QueryPlan(keywords="chemistry", resource_type="articles", peer_reviewed=True),
            [candidate], [SelectionItem(record_id="c1", reason="Covers the topic.")],
        ),
        "nonsense query": _answer_state(
            QueryPlan(keywords="nonsense"), [], [],
            answer=f"Not confident.\nTalk to a librarian: {LIVE_CHAT_URL}",
        ),
    }
    graph = FakeGraph(states)
    model = FakeChatModel(FaithfulnessVerdict(grounded=True))

    report = evaluate(graph, model, questions)

    assert len(graph.calls) == 2
    assert report["summary"]["planning_accuracy"] == 1.0
    assert report["summary"]["escalation_true_positives"] == 1
    assert report["faithfulness_accuracy"] == 1.0
    assert report["questions"][0]["faithfulness"] == [True]
    assert report["questions"][0]["plan"]["resource_type"] == "articles"
    assert json.dumps(report)  # the whole report must be JSON-serializable


def test_evaluate_skips_faithfulness_for_clarification_questions():
    questions = [{"id": "q1", "role": "graduate", "message": "help", "expect_clarification": True}]
    states = {
        "help": _answer_state(
            QueryPlan(keywords="", needs_clarification="What topic?"), [], [],
            answer="What topic?",
        )
    }
    report = evaluate(FakeGraph(states), FakeChatModel(), questions)

    assert report["questions"][0]["faithfulness"] == []
    assert report["faithfulness_accuracy"] is None


def test_evaluate_includes_rag_hits_in_the_report():
    questions = [{
        "id": "q1", "role": "graduate", "message": "chemistry librarian",
        "expected_resource_type": None, "expected_peer_reviewed": False,
        "expected_rag_type": LIBRARIAN_SUBJECT,
    }]
    rag_context = RagContext(librarians=[Record(id="librarian:chemistry", type=LIBRARIAN_SUBJECT, text="", metadata={})])
    states = {"chemistry librarian": _answer_state(QueryPlan(keywords="chemistry"), [], [], rag_context=rag_context)}

    report = evaluate(FakeGraph(states), FakeChatModel(), questions)

    assert report["summary"]["rag_hit_accuracy"] == 1.0


def test_print_summary_prints_every_metric():
    report = {
        "summary": {
            "planning_accuracy": 0.8, "clarification_accuracy": 1.0, "rag_hit_accuracy": 0.5,
            "escalation_true_positives": 1, "escalation_false_positives": 0,
            "escalation_true_negatives": 5, "escalation_false_negatives": 0,
        },
        "faithfulness_accuracy": 0.9,
    }
    lines = []
    print_summary(report, write=lines.append)
    output = "\n".join(lines)
    assert "0.8" in output and "0.9" in output and "TP=1" in output
