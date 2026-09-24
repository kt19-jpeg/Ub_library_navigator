import time

from langgraph.graph import END, StateGraph

from agent.compose import compose
from agent.models import AgentState
from agent.onesearch_tool import run_onesearch
from agent.planner import plan_query


def _timed(name: str, fn):
    """Wrap a node so its wall time lands in state["timings"] (shown by the CLI's --debug)."""

    def run(state: AgentState) -> dict:
        start = time.monotonic()
        update = dict(fn(state))
        update["timings"] = {
            **state.get("timings", {}),
            name: round(time.monotonic() - start, 3),
        }
        return update

    return run


def build_graph(model, client):
    """plan -> search -> rag (stub) -> compose. A clarifying question from the
    planner ends the turn early with that question as the answer."""

    def plan_node(state: AgentState) -> dict:
        plan = plan_query(model, state["role"], state["message"], state.get("history", []))
        return {"plan": plan, "answer": plan.needs_clarification}

    def search_node(state: AgentState) -> dict:
        outcome = run_onesearch(state["plan"], client)
        return {"search_result": outcome.result, "notes": outcome.notes}

    def rag_node(state: AgentState) -> dict:
        # Slice 1 stub: real LibGuides / librarian-directory retrieval arrives with the data pipeline.
        return {"rag_context": []}

    def compose_node(state: AgentState) -> dict:
        return compose(state, model, client)

    graph = StateGraph(AgentState)
    graph.add_node("plan", _timed("plan", plan_node))
    graph.add_node("search", _timed("search", search_node))
    graph.add_node("rag", _timed("rag", rag_node))
    graph.add_node("compose", _timed("compose", compose_node))
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        lambda state: "clarify" if state.get("answer") else "search",
        {"clarify": END, "search": "search"},
    )
    graph.add_edge("search", "rag")
    graph.add_edge("rag", "compose")
    graph.add_edge("compose", END)
    return graph.compile()
