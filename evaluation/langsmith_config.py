import os


def tracing_configured() -> bool:
    """True if LangSmith tracing is actually switched on. Tracing itself needs no code
    changes anywhere else: LangChain/LangGraph trace automatically through a global
    callback manager once these env vars are set (confirmed live 2026-09-30 against the
    installed langsmith 0.14.0, which checks LANGSMITH_* first, falling back to the
    older LANGCHAIN_* names) - this function only decides whether to print a heads-up
    that tracing is off, not whether tracing can work."""
    enabled = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return bool(enabled and enabled.lower() == "true" and api_key)
