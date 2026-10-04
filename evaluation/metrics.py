from dataclasses import dataclass

from agent.escalation import LIVE_CHAT_URL
from pipeline.records import DATABASE_SUBJECT, HELP_TOPIC, LIBRARIAN_SUBJECT

RAG_TYPE_FIELDS = {
    LIBRARIAN_SUBJECT: "librarians",
    HELP_TOPIC: "help_topics",
    DATABASE_SUBJECT: "databases",
}


@dataclass(frozen=True)
class QuestionResult:
    id: str
    clarification_ok: bool | None  # None unless the question expects clarification
    planning_ok: bool | None  # None only for a clarification question (planning never ran)
    escalation_expected: bool
    escalation_actual: bool
    rag_hit: bool | None  # None when the question has no expected_rag_type


def score_question(question: dict, state: dict) -> QuestionResult:
    """Scores one golden question against the real graph state it produced."""
    plan = state["plan"]
    escalation_actual = LIVE_CHAT_URL in state["answer"]

    if question.get("expect_clarification"):
        clarification_ok = bool(plan.needs_clarification) and state["answer"] == plan.needs_clarification
        return QuestionResult(
            id=question["id"],
            clarification_ok=clarification_ok,
            planning_ok=None,
            escalation_expected=False,
            escalation_actual=escalation_actual,
            rag_hit=None,
        )

    planning_ok = None
    if question.get("check_planning", True):
        planning_ok = (
            plan.resource_type == question.get("expected_resource_type")
            and plan.peer_reviewed == question.get("expected_peer_reviewed", False)
        )

    rag_hit = None
    expected_rag_type = question.get("expected_rag_type")
    if expected_rag_type:
        rag_context = state.get("rag_context")
        field = RAG_TYPE_FIELDS[expected_rag_type]
        rag_hit = bool(rag_context is not None and getattr(rag_context, field))

    return QuestionResult(
        id=question["id"],
        clarification_ok=None,
        planning_ok=planning_ok,
        escalation_expected=question.get("should_escalate", False),
        escalation_actual=escalation_actual,
        rag_hit=rag_hit,
    )


@dataclass(frozen=True)
class Summary:
    planning_accuracy: float | None
    clarification_accuracy: float | None
    rag_hit_accuracy: float | None
    escalation_true_positives: int
    escalation_false_positives: int
    escalation_true_negatives: int
    escalation_false_negatives: int


def _rate(values) -> float | None:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def summarize(results: list[QuestionResult]) -> Summary:
    return Summary(
        planning_accuracy=_rate(r.planning_ok for r in results),
        clarification_accuracy=_rate(r.clarification_ok for r in results),
        rag_hit_accuracy=_rate(r.rag_hit for r in results),
        escalation_true_positives=sum(1 for r in results if r.escalation_expected and r.escalation_actual),
        escalation_false_positives=sum(1 for r in results if not r.escalation_expected and r.escalation_actual),
        escalation_true_negatives=sum(1 for r in results if not r.escalation_expected and not r.escalation_actual),
        escalation_false_negatives=sum(1 for r in results if r.escalation_expected and not r.escalation_actual),
    )
