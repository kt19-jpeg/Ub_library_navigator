"""Manual end-to-end check against real UB OneSearch and a real model. Run locally, not in CI.

Usage: python -m agent.live_check [--provider P] [--model M] ["<question>" ...]
"""
import argparse
import sys
import time

from agent.cli import build_default_graph
from agent.llm import LLMConfigError

DEFAULT_QUESTIONS = [
    "find me peer reviewed journals on mental health",
    "books about the history of Buffalo",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--role", default="graduate")
    parser.add_argument("questions", nargs="*", default=DEFAULT_QUESTIONS)
    args = parser.parse_args(argv)

    try:
        graph = build_default_graph(provider=args.provider, model_name=args.model)
    except LLMConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2

    for question in args.questions:
        print("=" * 78)
        print(f"ROLE: {args.role}\nQUESTION: {question}\n")
        start = time.monotonic()
        state = graph.invoke({"role": args.role, "message": question, "history": []})
        total = time.monotonic() - start
        print(f"PLAN: {state['plan'].model_dump()}")
        print(f"NOTES: {state.get('notes', [])}")
        print(f"TIMINGS: {state.get('timings', {})} (total {total:.2f}s)\n")
        print(state["answer"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
