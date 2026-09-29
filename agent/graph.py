import time

from langgraph.graph import END, StateGraph

from agent.compose import compose
from agent.models import AgentState
from agent.onesearch_tool import run_onesearch
from agent.planner import plan_query
from agent.rag import query_rag


def _timed(name: str, fn):
    """Wrap a node so its own wall time is added to state["timings"] (merged via
    AgentState's reducer, safe even when nodes run in parallel)."""

    def run(state: AgentState) -> dict:
        start = time.monotonic()
        update = dict(fn(state))
        update["timings"] = {name: round(time.monotonic() - start, 3)}
        return update

    return run


def build_graph(model, client, collection=None):
    """plan -> {search, rag} -> compose, search and rag run in parallel. A
    clarifying question from the planner ends the turn early with that question
    as the answer. `collection` is the Chroma collection for RAG; None means "no
    library-guide data yet" (rag always returns an empty RagContext)."""

    def plan_node(state: AgentState) -> dict:
        plan = plan_query(model, state["role"], state["message"], state.get("history", []))
        return {"plan": plan, "answer": plan.needs_clarification}

    def search_node(state: AgentState) -> dict:
        outcome = run_onesearch(state["plan"], client)
        return {"search_result": outcome.result, "notes": outcome.notes}

    def rag_node(state: AgentState) -> dict:
        if collection is None:
            return {"rag_context": None, "notes": []}
        rag_context, notes = query_rag(state["plan"].keywords, collection)
        return {"rag_context": rag_context, "notes": notes}

    def compose_node(state: AgentState) -> dict:
        return compose(state, model, client)

    graph = StateGraph(AgentState)
    graph.add_node("plan", _timed("plan", plan_node))
    graph.add_node("search", _timed("search", search_node))
    graph.add_node("rag", _timed("rag", rag_node))
    graph.add_node("compose", _timed("compose", compose_node))
    graph.set_entry_point("plan")

    def after_plan(state: AgentState):
        # A clarifying question ends the turn right away (no search, no rag
        # lookup: there are no real keywords to look anything up with yet).
        # Otherwise fan out to both branches in the same step; AgentState's
        # reducers on "notes"/"timings" merge their updates.
        return "clarify" if state.get("answer") else ["run_search", "run_rag"]

    graph.add_conditional_edges(
        "plan", after_plan, {"clarify": END, "run_search": "search", "run_rag": "rag"}
    )
    graph.add_edge("search", "compose")
    graph.add_edge("rag", "compose")
    graph.add_edge("compose", END)
    return graph.compile()
