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

    run_chat(graph, read=read, write=write, log_fn=lambda role, state: None)

    assert graph.calls[0] == {"role": "faculty", "message": "first question", "history": []}
    assert graph.calls[1]["history"] == [
        {"user": "first question", "assistant": "answer to first question"}
    ]
    assert "answer to second question" in written


def test_history_is_capped():
    inputs = ["undergraduate"] + [f"q{i}" for i in range(6)] + ["/quit"]
    read, write, _ = _io(*inputs)
    graph = FakeGraph()

    run_chat(graph, read=read, write=write, log_fn=lambda role, state: None)

    assert len(graph.calls[-1]["history"]) == HISTORY_TURNS
    assert graph.calls[-1]["history"][-1]["user"] == "q4"


def test_role_command_changes_role_mid_conversation():
    read, write, _ = _io("undergraduate", "q1", "/role", "faculty", "q2", "/quit")
    graph = FakeGraph()

    run_chat(graph, read=read, write=write, log_fn=lambda role, state: None)

    assert [c["role"] for c in graph.calls] == ["undergraduate", "faculty"]


def test_debug_prints_plan_timings_and_notes():
    read, write, written = _io("graduate", "q1", "/quit")

    run_chat(FakeGraph(), read=read, write=write, debug=True, log_fn=lambda role, state: None)

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

    run_chat(FakeGraph(), read=read, write=lambda _: None, log_fn=lambda role, state: None)


def test_main_reports_missing_key_and_exits_nonzero(monkeypatch, capsys):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    assert main([]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_debug_does_not_crash_when_plan_is_absent():
    # Regression: confirmed live 2026-10-05 — the human-request fast path ends a
    # turn before plan_node ever runs, so state["plan"] doesn't exist; --debug
    # used to assume it always did and crashed with KeyError.
    class NoPlanGraph:
        def invoke(self, state):
            return {"answer": "I'll connect you with a librarian.", "timings": {"entry": 0.01}, "notes": []}

    read, write, written = _io("graduate", "talk to a human", "/quit")

    run_chat(NoPlanGraph(), read=read, write=write, debug=True, log_fn=lambda role, state: None)

    assert any("connect you with a librarian" in line for line in written)


def test_run_chat_calls_log_fn_with_role_and_state():
    read, write, _ = _io("graduate", "q1", "/quit")
    calls = []

    run_chat(FakeGraph(), read=read, write=write, log_fn=lambda role, state: calls.append((role, state)))

    assert len(calls) == 1
    assert calls[0][0] == "graduate"
    assert calls[0][1]["answer"] == "answer to q1"


def test_default_log_fn_never_touches_a_real_db_during_this_test_run():
    # Regression guard: confirmed live 2026-10-05 that omitting log_fn in a test
    # silently wrote a real analytics/sessions.db file. Every other test in this
    # file now injects a no-op log_fn; this one checks the default wiring exists
    # without actually invoking it against a real database.
    import inspect
    assert "log_fn" in inspect.signature(run_chat).parameters
