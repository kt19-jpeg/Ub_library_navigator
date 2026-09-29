"""Manual check of the library-guide retrieval against a real, built Chroma store.
Run `python pipeline/refresh.py` first so there is data to query. Not run in CI.

Usage: python -m agent.rag_live_check ["<query>" ...]
"""
import sys

from agent.rag import query_rag
from pipeline.embed import open_rag_collection

DEFAULT_QUERIES = [
    "mental health",
    "citing sources",
    "chemistry",
    "anatomy",
]


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    queries = argv if argv else DEFAULT_QUERIES

    collection = open_rag_collection()
    if collection is None:
        print(
            "No usable library-guide data: either OPENAI_API_KEY is not set, or "
            "`python pipeline/refresh.py` has not been run yet.",
            file=sys.stderr,
        )
        return 2

    for query in queries:
        print("=" * 78)
        print(f"QUERY: {query}\n")
        context, notes = query_rag(query, collection)
        for label, hits in (
            ("librarians", context.librarians),
            ("help_topics", context.help_topics),
            ("databases", context.databases),
        ):
            print(f"{label}:")
            if not hits:
                print("  (no hit under the current distance cutoff)")
            for hit in hits:
                print(f"  {hit.id}: {hit.metadata}")
        if notes:
            print(f"notes: {notes}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
