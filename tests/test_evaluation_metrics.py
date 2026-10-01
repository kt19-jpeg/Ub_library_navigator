from agent.escalation import LIVE_CHAT_URL
from agent.models import QueryPlan, RagContext
from evaluation.metrics import QuestionResult, score_question, summarize
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT, Record


def _state(plan, answer, rag_context=None):
    return {"plan": plan, "answer": answer, "rag_context": rag_context or RagContext()}


def test_scores_correct_planning_and_no_escalation():
    question = {
        "id": "q1", "expected_resource_type": "journals", "expected_peer_reviewed": True,
        "should_escalate": False,
    }
    plan = QueryPlan(keywords="x", resource_type="journals", peer_reviewed=True)
    result = score_question(question, _state(plan, "Here is what I found..."))

    assert result.planning_ok is True
    assert result.escalation_expected is False
    assert result.escalation_actual is False
    assert result.clarification_ok is None
    assert result.rag_hit is None


def test_scores_wrong_resource_type_as_not_ok():
    question = {"id": "q2", "expected_resource_type": "books", "expected_peer_reviewed": False}
    plan = QueryPlan(keywords="x", resource_type="articles", peer_reviewed=False)
    result = score_question(question, _state(plan, "answer"))
    assert result.planning_ok is False


def test_detects_escalation_from_the_live_chat_link():
    question = {"id": "q3", "expected_resource_type": None, "expected_peer_reviewed": False, "should_escalate": True}
    plan = QueryPlan(keywords="x")
    answer = f"I am not confident...\nTalk to a librarian: {LIVE_CHAT_URL}"
    result = score_question(question, _state(plan, answer))
    assert result.escalation_actual is True
    assert result.escalation_expected is True


def test_clarification_question_only_scores_clarification():
    question = {"id": "q4", "expect_clarification": True}
    plan = QueryPlan(keywords="", needs_clarification="What topic are you researching?")
    result = score_question(question, _state(plan, "What topic are you researching?"))

    assert result.clarification_ok is True
    assert result.planning_ok is None
    assert result.rag_hit is None


def test_clarification_question_fails_if_plan_did_not_actually_ask():
    question = {"id": "q5", "expect_clarification": True}
    plan = QueryPlan(keywords="x")  # no needs_clarification set
    result = score_question(question, _state(plan, "Here is what I found..."))
    assert result.clarification_ok is False


def test_rag_hit_true_when_the_expected_type_has_a_hit():
    question = {"id": "q6", "expected_resource_type": None, "expected_peer_reviewed": False, "expected_rag_type": LIBRARIAN_SUBJECT}
    plan = QueryPlan(keywords="chemistry")
    rag = RagContext(librarians=[Record(id="librarian:chemistry", type=LIBRARIAN_SUBJECT, text="", metadata={})])
    result = score_question(question, _state(plan, "answer", rag))
    assert result.rag_hit is True


def test_rag_hit_false_when_the_expected_type_has_no_hit():
    question = {"id": "q7", "expected_resource_type": None, "expected_peer_reviewed": False, "expected_rag_type": HELP_TOPIC}
    plan = QueryPlan(keywords="x")
    result = score_question(question, _state(plan, "answer", RagContext()))
    assert result.rag_hit is False


def test_rag_hit_is_none_when_not_expected():
    question = {"id": "q8", "expected_resource_type": "books", "expected_peer_reviewed": False}
    plan = QueryPlan(keywords="x", resource_type="books")
    result = score_question(question, _state(plan, "answer"))
    assert result.rag_hit is None


def test_summarize_computes_rates_and_confusion_matrix():
    results = [
        QuestionResult("a", None, True, False, False, None),
        QuestionResult("b", None, False, False, False, None),
        QuestionResult("c", True, None, False, False, None),
        QuestionResult("d", None, None, True, True, None),   # escalation TP
        QuestionResult("e", None, None, False, True, None),  # escalation FP
        QuestionResult("f", None, None, True, False, None),  # escalation FN
        QuestionResult("g", None, True, False, False, True),
        QuestionResult("h", None, True, False, False, False),
    ]
    summary = summarize(results)

    assert summary.planning_accuracy == 0.75  # a, g, h True; b False -> 3 of 4
    assert summary.clarification_accuracy == 1.0
    assert summary.rag_hit_accuracy == 0.5
    assert (summary.escalation_true_positives, summary.escalation_false_positives,
            summary.escalation_true_negatives, summary.escalation_false_negatives) == (1, 1, 5, 1)


def test_summarize_handles_no_applicable_questions_for_a_metric():
    results = [QuestionResult("a", None, True, False, False, None)]
    summary = summarize(results)
    assert summary.clarification_accuracy is None
    assert summary.rag_hit_accuracy is None


def test_check_planning_false_skips_the_planning_metric():
    question = {"id": "q9", "check_planning": False, "expected_resource_type": "books"}
    plan = QueryPlan(keywords="x", resource_type="articles")  # would otherwise be "wrong"
    result = score_question(question, _state(plan, "answer"))
    assert result.planning_ok is None
