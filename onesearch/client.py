import concurrent.futures
import urllib.parse

import requests

from onesearch.cache import QueryCache
from onesearch.models import ResourceDetail, SearchCandidate, SearchResult
from onesearch.parser import parse_brief_response, parse_detail_response
from onesearch.session import ONESEARCH_USER_AGENT, SessionManager

BASE_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/pnxs"
DETAIL_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/discovery/fulldisplay"


def _build_detail_url(record_id: str, vid: str) -> str:
    params = {"docid": record_id, "context": "L", "vid": vid}
    query_string = urllib.parse.urlencode(
        params, safe=",", quote_via=urllib.parse.quote
    )
    return f"{DETAIL_URL}?{query_string}"


def _build_search_url(query: str, vid: str) -> str:
    params = {
        "q": f"any,contains,{query}",
        "vid": vid,
        "skipDelivery": "N",
        "pcAvailability": "true",
        "scope": "MyInst_and_CI",
        "tab": "Everything",
        "offset": "0",
        "limit": "10",
    }
    # requests' params= dict percent-encodes commas (%2C), which the
    # backend rejects. Build the query string manually and keep commas
    # literal (CLAUDE.md, confirmed finding 1).
    query_string = urllib.parse.urlencode(
        params, safe=",", quote_via=urllib.parse.quote
    )
    return f"{BASE_URL}?{query_string}"


class OneSearchClient:
    def __init__(
        self,
        session_manager: SessionManager,
        cache: QueryCache,
        http_get=requests.get,
        vid: str = "01SUNY_BUF:UB",
    ):
        self._session_manager = session_manager
        self._cache = cache
        self._http_get = http_get
        self._vid = vid

    def search(self, query: str) -> list[SearchCandidate]:
        cached = self._cache.get(query)
        if cached is not None:
            return cached

        cookies = self._session_manager.get_cookies()
        url = _build_search_url(query, self._vid)
        response = self._http_get(
            url,
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            cookies={
                "JSESSIONID": cookies.jsessionid,
                "urm_st": cookies.urm_st,
                "urm_se": cookies.urm_se,
                "institute": cookies.institute,
            },
            timeout=10,
        )
        response.raise_for_status()
        candidates = parse_brief_response(response.json())
        self._cache.set(query, candidates)
        return candidates

    def _fetch_one_detail(self, record_id: str) -> ResourceDetail:
        cookies = self._session_manager.get_cookies()
        url = _build_detail_url(record_id, self._vid)
        response = self._http_get(
            url,
            headers={"User-Agent": ONESEARCH_USER_AGENT},
            cookies={
                "JSESSIONID": cookies.jsessionid,
                "urm_st": cookies.urm_st,
                "urm_se": cookies.urm_se,
                "institute": cookies.institute,
            },
            timeout=10,
        )
        response.raise_for_status()
        return parse_detail_response(record_id, response.json())

    def fetch_detail(self, record_ids: list[str]) -> dict[str, ResourceDetail]:
        # Bounded by definition: callers (Layer 2) pass only the 1-3 ids
        # they decided to show. Never call this with a full result set
        # (CLAUDE.md: "never loop fetch_detail() over every search result").
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(record_ids) or 1) as pool:
            futures = {
                pool.submit(self._fetch_one_detail, rid): rid for rid in record_ids
            }
            return {futures[future]: future.result() for future in futures}

    def search_with_fallback(
        self, query: str, timeout_seconds: float = 9.0
    ) -> SearchResult:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.search, query)
            try:
                candidates = future.result(timeout=timeout_seconds)
                return SearchResult(candidates=candidates, timed_out=False, fallback_link=None)
            except concurrent.futures.TimeoutError:
                fallback_link = (
                    "https://search.lib.buffalo.edu/discovery/search?"
                    f"query=any,contains,{urllib.parse.quote(query)}&vid={self._vid}"
                )
                return SearchResult(candidates=[], timed_out=True, fallback_link=fallback_link)
