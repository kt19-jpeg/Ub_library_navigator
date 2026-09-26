import os

from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("openai", "anthropic", "local")
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "local": "llama3.1",
}


class LLMConfigError(Exception):
    """The requested model cannot be built: bad provider, missing key, or missing package."""


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Build the chat model. Arguments override the LLM_PROVIDER, LLM_MODEL and
    LLM_BASE_URL env vars; the provider defaults to openai."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Unknown LLM provider {provider!r}; expected one of {list(PROVIDERS)}")
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    base_url = base_url or os.getenv("LLM_BASE_URL")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigError("OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY is not set")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LLMConfigError(
                "The anthropic provider needs langchain-anthropic: pip install langchain-anthropic"
            ) from exc
        return ChatAnthropic(model=model, api_key=api_key, temperature=0)

    # local: any OpenAI-compatible server (LM Studio, llama.cpp, vLLM, Ollama's
    # /v1 endpoint) when LLM_BASE_URL is set, otherwise Ollama's native API.
    if base_url:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY") or "not-needed",
            temperature=0,
        )
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise LLMConfigError(
            "The local provider needs langchain-ollama (pip install langchain-ollama) "
            "or set LLM_BASE_URL to an OpenAI-compatible server"
        ) from exc
    return ChatOllama(model=model, temperature=0)
