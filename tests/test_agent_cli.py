from agent.cli import HISTORY_TURNS, choose_role, main, run_chat
from agent.models import QueryPlan


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state):
        self.calls.append({**state, "history": list(state["history"])})
        return {
            "answer": f"answer to {state['message']}",
            "plan": QueryPlan(keywords=state["message"]),
            "timings": {"plan": 0.1},
            "notes": ["a note"],
            "citations": {"r1": {"apa": "APA citation.", "mla": "MLA citation.", "chicago": "Chicago citation."}},
        }


def _io(*inputs):
    feed = iter(inputs)
    written = []
    return (lambda prompt="": next(feed)), written.append, written


def test_choose_role_accepts_name_or_number_and_reprompts():
    read, write, written = _io("dean", "2")
    assert choose_role(read, write) == "graduate"
    assert any("Please enter one of" in line for line in written)


def test_chat_passes_role_message_and_history_to_the_graph():
    read, write, written = _io("faculty", "first question", "second question", "/quit")
    graph = FakeGraph()

    run_chat(graph, read=read, write=write)

    assert graph.calls[0] == {"role": "faculty", "message": "first question", "history": []}
    assert graph.calls[1]["history"] == [
        {"user": "first question", "assistant": "answer to first question"}
    ]
    assert "answer to second question" in written


def test_history_is_capped():
    inputs = ["undergraduate"] + [f"q{i}" for i in range(6)] + ["/quit"]
    read, write, _ = _io(*inputs)
    graph = FakeGraph()

    run_chat(graph, read=read, write=write)

    assert len(graph.calls[-1]["history"]) == HISTORY_TURNS
    assert graph.calls[-1]["history"][-1]["user"] == "q4"


def test_role_command_changes_role_mid_conversation():
    read, write, _ = _io("undergraduate", "q1", "/role", "faculty", "q2", "/quit")
    graph = FakeGraph()

    run_chat(graph, read=read, write=write)

    assert [c["role"] for c in graph.calls] == ["undergraduate", "faculty"]


def test_debug_prints_plan_timings_and_notes():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write, debug=True)

    assert any(line.startswith("[plan]") for line in written)
    assert any(line.startswith("[timings]") for line in written)
    assert "[note] a note" in written


def test_eof_ends_the_chat_cleanly():
    feed = iter(["graduate"])

    def read(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    run_chat(FakeGraph(), read=read, write=lambda _: None)


def test_main_reports_missing_key_and_exits_nonzero(monkeypatch, capsys):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    assert main([]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_citations_flag_prints_citations_for_each_result():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write, citations=True)

    assert any("[citations for r1]" in line for line in written)
    assert any("APA: APA citation." in line for line in written)


def test_citations_hidden_by_default():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write)

    assert not any("citations for" in line for line in written)


def test_debug_does_not_crash_when_plan_is_absent():
    # Regression: confirmed live 2026-10-05 — the human-request fast path ends a
    # turn before plan_node ever runs, so state["plan"] doesn't exist; --debug
    # used to assume it always did and crashed with KeyError.
    class NoPlanGraph:
        def invoke(self, state):
            return {"answer": "I'll connect you with a librarian.", "timings": {"entry": 0.01}, "notes": []}

    read, write, written = _io("graduate", "talk to a human", "/quit")

    run_chat(NoPlanGraph(), read=read, write=write, debug=True)

    assert any("connect you with a librarian" in line for line in written)
