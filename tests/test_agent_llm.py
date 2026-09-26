import sys
import types

import pytest

from agent.llm import DEFAULT_MODELS, LLMConfigError, get_chat_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_to_openai_and_needs_a_key():
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        get_chat_model()


def test_openai_uses_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = get_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == DEFAULT_MODELS["openai"]


def test_env_vars_select_provider_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    assert get_chat_model().model_name == "gpt-4o"


def test_arguments_override_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    assert get_chat_model(model="from-arg").model_name == "from-arg"


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
        get_chat_model(provider="bogus")


def test_anthropic_needs_key(monkeypatch):
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        get_chat_model(provider="anthropic")


def test_anthropic_missing_package_is_explained(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)  # makes import fail
    with pytest.raises(LLMConfigError, match="pip install langchain-anthropic"):
        get_chat_model(provider="anthropic")


def test_anthropic_builds_chat_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    fake = types.ModuleType("langchain_anthropic")

    class ChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake.ChatAnthropic = ChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake)

    model = get_chat_model(provider="anthropic")
    assert model.kwargs["model"] == DEFAULT_MODELS["anthropic"]
    assert model.kwargs["api_key"] == "k"


def test_local_with_base_url_uses_openai_compatible_client():
    model = get_chat_model(provider="local", model="qwen2.5", base_url="http://localhost:1234/v1")
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "qwen2.5"
    assert model.openai_api_base == "http://localhost:1234/v1"


def test_local_without_base_url_uses_ollama(monkeypatch):
    fake = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake.ChatOllama = ChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake)

    model = get_chat_model(provider="local")
    assert model.kwargs["model"] == DEFAULT_MODELS["local"]


def test_local_missing_ollama_package_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    with pytest.raises(LLMConfigError, match="langchain-ollama"):
        get_chat_model(provider="local")
