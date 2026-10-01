from evaluation.langsmith_config import tracing_configured


def test_false_when_nothing_is_set(monkeypatch):
    for name in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert tracing_configured() is False


def test_true_when_langsmith_vars_are_set(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    assert tracing_configured() is True


def test_true_with_the_older_langchain_names(monkeypatch):
    for name in ("LANGSMITH_TRACING_V2", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-test")
    assert tracing_configured() is True


def test_false_when_enabled_but_no_api_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert tracing_configured() is False


def test_false_when_tracing_flag_is_not_literally_true(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    assert tracing_configured() is False
