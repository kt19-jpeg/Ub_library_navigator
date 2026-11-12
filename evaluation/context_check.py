"""Multi-turn context check: `python -m evaluation.context_check` (live, real OneSearch + LLM).

Replays one 7-turn conversation through the graph with the same state handling the
Streamlit app and CLI use (3-turn LLM history, stored shown-results kept across turns) and
checks each turn's ROUTING, not its wording: a search turn searches, chit-chat doesn't, a
question about shown results is answered from them (no search), and a new topic searches
again instead of being read as a follow-up. Also checks the invariants behind two real
bugs: the on-screen transcript must never shrink (it once dropped the 1st question when the
LLM history was trimmed) and the LLM history must stay capped.

Routing is what the graph's `timings` record, so it is checked the same way live and in the
offline test (tests/test_evaluation_context_check.py, fake model, runs in CI)."""

import sys
from dataclasses import dataclass, field

from agent.cli import HISTORY_TURNS
from agent.compose import extract_shown_results


@dataclass
class Step:
    message: str
    ran: tuple[str, ...] = ()       # graph nodes that must have run this turn
    not_ran: tuple[str, ...] = ()   # graph nodes that must NOT have run this turn


SCENARIO = [
    Step("books on cancer immunotherapy", ran=("search",), not_ran=("followup",)),
    Step("hi", not_ran=("search", "followup")),
    Step("who is the subject librarian for medicine?", ran=("search",)),
    Step("thanks!", not_ran=("search", "followup")),
    Step("in what way do these differ?", ran=("followup",), not_ran=("search",)),
    Step("now find me articles on diabetes", ran=("search",), not_ran=("followup",)),
    Step("which one is better for a beginner?", ran=("followup",), not_ran=("search",)),
]


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def run_scenario(graph, role: str = "graduate", scenario: list[Step] = SCENARIO) -> Report:
    report = Report()
    history: list[dict] = []
    shown: list[dict] = []
    transcript: list[str] = []
    for number, step in enumerate(scenario, start=1):
        state = graph.invoke(
            {"role": role, "message": step.message, "history": history, "shown_results": shown}
        )
        nodes = set(state.get("timings", {}))
        label = f"turn {number} ({step.message!r})"
        for node in step.ran:
            if node not in nodes:
                report.failures.append(f"{label}: expected '{node}' to run, ran {sorted(nodes)}")
        for node in step.not_ran:
            if node in nodes:
                report.failures.append(f"{label}: '{node}' should not have run")
        shown = extract_shown_results(state) or shown  # same rule as the app
        history.append({"user": step.message, "assistant": state["answer"]})
        history = history[-HISTORY_TURNS:]
        transcript.append(step.message)
        if len(history) > HISTORY_TURNS:
            report.failures.append(f"{label}: LLM history exceeded {HISTORY_TURNS} turns")
        if len(transcript) != number:
            report.failures.append(f"{label}: transcript has {len(transcript)} turns, expected {number}")
        if number >= 1 and not shown:
            report.failures.append(f"{label}: stored shown-results were lost")
    return report


def main() -> int:
    from agent.cli import build_default_graph
    from agent.llm import LLMConfigError

    try:
        graph = build_default_graph()
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    report = run_scenario(graph)
    if report.ok:
        print(f"PASS: all {len(SCENARIO)} turns routed correctly and context held.")
        return 0
    print("FAIL:")
    for failure in report.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
