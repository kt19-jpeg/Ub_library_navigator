import time
import urllib.parse

import requests

USER_AGENT = "UBLibraryNavigator/1.0 (+contact: kavyansht21@gmail.com)"
MIN_SECONDS_BETWEEN_REQUESTS = 10
TIMEOUT = 20

_last_request_at: dict[str, float] = {}
_robots_rules_cache: dict[str, list[str]] = {}


class FetchError(Exception):
    """A page could not be fetched (network error, non-200, or robots.txt disallows it)."""


def _parse_disallow_rules(robots_text: str) -> list[str]:
    """Returns the Disallow rule strings from the `User-agent: *` group, verbatim
    (no URL round-trip). Not using `urllib.robotparser`: it reconstructs each rule
    with `urlparse`/`urlunparse`, which silently drops a bare trailing "?" (e.g.
    "Disallow: /az.php?" becomes the rule "/az.php" instead of "/az.php?") -
    confirmed live 2026-09-27 against research.lib.buffalo.edu/robots.txt, whose
    intent is to block only the querystring form ("/az.php?...") and allow the
    bare page ("/az.php") we actually need. A plain prefix match on the rule's
    exact text (per the robots.txt convention) gets this right."""
    rules = []
    in_wildcard_group = False
    for raw_line in robots_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, _, value = line.partition(":")
        field, value = field.strip().lower(), value.strip()
        if field == "user-agent":
            in_wildcard_group = value == "*"
        elif field == "disallow" and in_wildcard_group and value:
            rules.append(value)
    return rules


def _path_and_query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def _robots_rules_for(host: str, get=requests.get) -> list[str]:
    if host not in _robots_rules_cache:
        try:
            response = get(f"https://{host}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            _robots_rules_cache[host] = _parse_disallow_rules(response.text) if response.status_code == 200 else []
        except requests.RequestException:
            _robots_rules_cache[host] = []
    return _robots_rules_cache[host]


def _wait_for_host(host: str, sleep=time.sleep, now=time.monotonic) -> None:
    last = _last_request_at.get(host)
    if last is not None:
        elapsed = now() - last
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at[host] = now()


def fetch(url: str, get=requests.get, sleep=time.sleep, now=time.monotonic) -> str:
    """Fetch one page: obeys robots.txt, waits at least MIN_SECONDS_BETWEEN_REQUESTS
    between requests to the same host, sends an identifying User-Agent. One attempt,
    no retry (politeness over completeness). Raises FetchError on any failure."""
    host = urllib.parse.urlparse(url).netloc
    rules = _robots_rules_for(host, get)
    if any(_path_and_query(url).startswith(rule) for rule in rules):
        raise FetchError(f"robots.txt disallows fetching {url}")
    _wait_for_host(host, sleep, now)
    try:
        response = get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc
    if response.status_code != 200:
        raise FetchError(f"{url} returned HTTP {response.status_code}")
    return response.text
