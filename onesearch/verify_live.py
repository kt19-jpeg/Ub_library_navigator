"""Manual verification against real UB data. Run locally, not in CI.

Usage: python -m onesearch.verify_live "<query>" [<query> ...]

For each query: searches, prints every result with the fields Layer 2 needs
(identity, description for ranking, and access info), fetches detail for the
results in ONE batched call, and checks that each online link actually
redirects to a resource.

Note: fetching detail for the whole result set is done here only to verify
and time the batched call. Production code (Layer 2) must still call
fetch_detail() only for the 1-3 items it decides to show.
"""
import concurrent.futures
import sys
import textwrap
import time

import requests

from onesearch.cache import QueryCache
from onesearch.client import OneSearchClient
from onesearch.session import ONESEARCH_USER_AGENT, SessionManager

DESCRIPTION_CHARS = 220


def _check_redirect(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            allow_redirects=False,
            timeout=10,
        )
    except requests.RequestException as exc:
        return f"ERROR {type(exc).__name__}"
    location = response.headers.get("location")
    return f"{response.status_code} -> {location}" if location else str(response.status_code)


def _print_result(index: int, candidate, detail, redirect: str | None) -> None:
    print(f"[{index}] {candidate.title}")
    print(f"    record_id:  {candidate.record_id}")
    print(f"    author:     {candidate.author}")
    print(f"    date/type:  {candidate.date} | {candidate.resource_type} | {candidate.format}")
    print(f"    subject:    {candidate.subject}")
    if candidate.description:
        description = textwrap.shorten(candidate.description, DESCRIPTION_CHARS, placeholder="...")
        print(f"    description: {description}")
    print(f"    access:     {candidate.delivery_category}")
    if candidate.call_number or candidate.library:
        print(
            f"    location:   {candidate.library} / {candidate.sub_location} | "
            f"call number: {candidate.call_number} | "
            f"status: {candidate.availability_status}"
        )
    if detail.online_link:
        print(f"    online:     {detail.online_link}")
        print(f"    package:    {detail.online_package}")
        print(f"    redirect:   {redirect}")
    print()


def run(client: OneSearchClient, query: str) -> None:
    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)

    start = time.monotonic()
    candidates = client.search(query)
    print(f"search: {time.monotonic() - start:.2f}s, {len(candidates)} results\n")

    start = time.monotonic()
    details = client.fetch_detail([c.record_id for c in candidates])
    print(f"detail: {time.monotonic() - start:.2f}s, one batched call for {len(details)} ids\n")

    online_urls = {rid: d.online_link for rid, d in details.items() if d.online_link}
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        redirects = dict(zip(online_urls, pool.map(_check_redirect, online_urls.values())))
    print(f"redirect checks: {time.monotonic() - start:.2f}s for {len(online_urls)} online links\n")

    for index, candidate in enumerate(candidates, start=1):
        _print_result(index, candidate, details[candidate.record_id], redirects.get(candidate.record_id))

    online = sum(1 for c in candidates if c.delivery_category and "Alma-E" in c.delivery_category)
    physical = sum(1 for c in candidates if c.call_number)
    resolved = sum(1 for r in redirects.values() if r.startswith("30"))
    print(
        f"SUMMARY: {len(candidates)} results | {online} online-category | "
        f"{len(online_urls)} with resolved link ({resolved} redirect OK) | "
        f"{physical} with call number\n"
    )


def main():
    queries = sys.argv[1:] or ["machine learning", "buffalo history"]
    client = OneSearchClient(session_manager=SessionManager(), cache=QueryCache(ttl_seconds=300))

    start = time.monotonic()
    client._session_manager.get_cookies()
    print(f"session mint (cold): {time.monotonic() - start:.2f}s\n")

    for query in queries:
        run(client, query)


if __name__ == "__main__":
    main()
