import argparse
import sys

from agent.llm import LLMConfigError, get_chat_model
from agent.models import ROLES

HISTORY_TURNS = 3
HELP = "Type a question. /role changes your role, /quit exits."


def choose_role(read, write) -> str:
    """Ask for a role until the answer is a role name or its number."""
    write("Who are you? " + ", ".join(f"{i}) {r}" for i, r in enumerate(ROLES, start=1)))
    while True:
        answer = read("role> ").strip().lower()
        if answer in ROLES:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(ROLES):
            return ROLES[int(answer) - 1]
        write(f"Please enter one of: {', '.join(ROLES)} (or 1-{len(ROLES)}).")


def run_chat(graph, read=input, write=print, debug: bool = False, citations: bool = False) -> None:
    role = choose_role(read, write)
    write(f"Role set to {role}. {HELP}")
    history: list[dict] = []
    while True:
        try:
            message = read("> ").strip()
        except EOFError:
            return
        if not message:
            continue
        if message == "/quit":
            return
        if message == "/role":
            role = choose_role(read, write)
            write(f"Role set to {role}.")
            continue
        state = graph.invoke({"role": role, "message": message, "history": history})
        answer = state["answer"]
        if debug:
            write(f"[plan] {state['plan'].model_dump()}")
            write(f"[timings] {state.get('timings', {})}")
            for note in state.get("notes", []):
                write(f"[note] {note}")
        write(answer)
        if citations:
            for record_id, styles in state.get("citations", {}).items():
                write(f"[citations for {record_id}]")
                for style, text in styles.items():
                    write(f"  {style.upper()}: {text}")
        history.append({"user": message, "assistant": answer})
        history = history[-HISTORY_TURNS:]


def build_default_graph(provider: str | None = None, model_name: str | None = None):
    """Real model + real Layer 1 client. Raises LLMConfigError if the model can't be built."""
    from dotenv import load_dotenv

    load_dotenv()
    model = get_chat_model(provider=provider, model=model_name)

    from agent.graph import build_graph
    from onesearch.cache import QueryCache
    from onesearch.client import OneSearchClient
    from onesearch.session import SessionManager
    from pipeline.embed import open_rag_collection

    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))
    # None (no OPENAI_API_KEY, or `python pipeline/refresh.py` hasn't been run yet)
    # is fine: build_graph then always returns an empty RagContext for this chat.
    collection = open_rag_collection()
    return build_graph(model, client, collection=collection)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UB Library Navigator (Layer 2, slice 1)")
    parser.add_argument("--provider", help="openai | anthropic | local (overrides LLM_PROVIDER)")
    parser.add_argument("--model", help="model name (overrides LLM_MODEL)")
    parser.add_argument("--debug", action="store_true", help="show plan and per-step timings")
    parser.add_argument("--citations", action="store_true", help="show APA/MLA/Chicago citations for each result")
    args = parser.parse_args(argv)

    try:
        graph = build_default_graph(provider=args.provider, model_name=args.model)
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    run_chat(graph, debug=args.debug, citations=args.citations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
