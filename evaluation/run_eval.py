import json
import sys
from pathlib import Path

from evaluation.faithfulness import judge_faithfulness
from evaluation.metrics import score_question, summarize

GOLDEN_SET_PATH = Path("evaluation/golden_test_set.json")
REPORT_PATH = Path("evaluation/last_run.json")


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _judge_selection_faithfulness(model, state: dict) -> list[bool]:
    selection = state.get("selection") or []
    if not selection:
        return []
    candidates_by_id = {c.record_id: c for c in state["search_result"].candidates}
    judgments = []
    for item in selection:
        candidate = candidates_by_id.get(item.record_id)
        if candidate is None:
            continue
        view = {
            "title": candidate.title,
            "author": candidate.author,
            "subject": candidate.subject,
            "description": candidate.description,
        }
        judgments.append(judge_faithfulness(model, view, item.reason))
    return judgments


def evaluate(graph, model, questions: list[dict]) -> dict:
    """Runs one real graph.invoke() per golden question, scores it, and judges faithfulness
    for whatever it selected. Returns a JSON-serializable report."""
    results = []
    all_faithfulness: list[bool] = []
    per_question = []

    for question in questions:
        state = graph.invoke({"role": question["role"], "message": question["message"], "history": []})
        result = score_question(question, state)
        results.append(result)

        faithfulness = [] if question.get("expect_clarification") else _judge_selection_faithfulness(model, state)
        all_faithfulness.extend(faithfulness)

        per_question.append(
            {
                "id": question["id"],
                "role": question["role"],
                "message": question["message"],
                "plan": state["plan"].model_dump(),
                "answer": state.get("answer"),
                "result": vars(result),
                "faithfulness": faithfulness,
            }
        )

    summary = summarize(results)
    faithfulness_accuracy = (sum(all_faithfulness) / len(all_faithfulness)) if all_faithfulness else None
    return {
        "summary": vars(summary),
        "faithfulness_accuracy": faithfulness_accuracy,
        "questions": per_question,
    }


def print_summary(report: dict, write=print) -> None:
    s = report["summary"]
    write(f"Planning accuracy:      {s['planning_accuracy']}")
    write(f"Clarification accuracy: {s['clarification_accuracy']}")
    write(f"RAG hit accuracy:       {s['rag_hit_accuracy']}")
    write(f"Faithfulness accuracy:  {report['faithfulness_accuracy']}")
    write(
        "Escalation: "
        f"TP={s['escalation_true_positives']} FP={s['escalation_false_positives']} "
        f"TN={s['escalation_true_negatives']} FN={s['escalation_false_negatives']}"
    )


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    from agent.cli import build_default_graph
    from agent.llm import LLMConfigError, get_chat_model
    from evaluation.langsmith_config import tracing_configured

    if not tracing_configured():
        print("Note: LangSmith tracing is not configured; continuing without it.")

    try:
        graph = build_default_graph()
        model = get_chat_model()
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2

    report = evaluate(graph, model, load_golden_set())
    print_summary(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Full report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
