import urllib.parse

import requests

from onesearch.cache import QueryCache
from onesearch.models import SearchCandidate
from onesearch.parser import parse_brief_response
from onesearch.session import ONESEARCH_USER_AGENT, SessionManager

BASE_URL = "https://search.lib.buffalo.edu/primaws/rest/pub/pnxs"


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
