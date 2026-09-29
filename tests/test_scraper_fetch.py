from unittest.mock import MagicMock

import pytest

from scraper.fetch import USER_AGENT, FetchError, _last_request_at, _robots_rules_cache, fetch

# Captured live from https://research.lib.buffalo.edu/robots.txt on 2026-09-27.
REAL_ROBOTS_TXT = """User-agent: *
Crawl-delay: 10
    Disallow: /az.php?
    Disallow: /az/databases?
    Disallow: /er.php
"""


@pytest.fixture(autouse=True)
def _clean_caches():
    _robots_rules_cache.clear()
    _last_request_at.clear()
    yield
    _robots_rules_cache.clear()
    _last_request_at.clear()


def _response(status=200, text="ok"):
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


def test_fetches_with_identifying_user_agent_when_no_robots_txt():
    robots_response = _response(status=404)
    page_response = _response(text="<html>hi</html>")
    get = MagicMock(side_effect=[robots_response, page_response])

    result = fetch("https://example.org/page", get=get, sleep=MagicMock(), now=lambda: 0.0)

    assert result == "<html>hi</html>"
    assert get.call_args_list[1].kwargs["headers"]["User-Agent"] == USER_AGENT


def test_a_bare_path_is_allowed_when_only_its_query_form_is_disallowed():
    # Regression test for a real bug: urllib.robotparser silently turns
    # "Disallow: /az.php?" into the rule "/az.php" (it round-trips the rule
    # through urlparse/urlunparse, which drops a bare trailing "?"), which
    # would wrongly block research.lib.buffalo.edu/az.php itself. A correct
    # robots.txt reader only blocks the querystring form.
    robots_response = _response(text=REAL_ROBOTS_TXT)
    page_response = _response(text="<html>the real page</html>")
    get = MagicMock(side_effect=[robots_response, page_response])

    result = fetch(
        "https://research.lib.buffalo.edu/az.php", get=get, sleep=MagicMock(), now=lambda: 0.0
    )

    assert result == "<html>the real page</html>"


def test_the_querystring_form_is_still_disallowed():
    robots_response = _response(text=REAL_ROBOTS_TXT)
    get = MagicMock(return_value=robots_response)

    with pytest.raises(FetchError, match="robots.txt disallows"):
        fetch(
            "https://research.lib.buffalo.edu/az.php?subject=94213",
            get=get, sleep=MagicMock(), now=lambda: 0.0,
        )


def test_a_bare_path_rule_with_no_trailing_question_mark_is_fully_disallowed():
    robots_response = _response(text=REAL_ROBOTS_TXT)
    get = MagicMock(return_value=robots_response)

    with pytest.raises(FetchError, match="robots.txt disallows"):
        fetch("https://research.lib.buffalo.edu/er.php", get=get, sleep=MagicMock(), now=lambda: 0.0)


def test_disallow_rules_only_apply_to_the_wildcard_user_agent_group():
    robots_text = "User-agent: SomeBot\nDisallow: /az.php\n\nUser-agent: *\nDisallow:\n"
    robots_response = _response(text=robots_text)
    page_response = _response(text="ok")
    get = MagicMock(side_effect=[robots_response, page_response])

    result = fetch("https://example.org/az.php", get=get, sleep=MagicMock(), now=lambda: 0.0)

    assert result == "ok"


def test_waits_at_least_ten_seconds_between_requests_to_the_same_host():
    robots_response = _response(status=404)
    page_response = _response()
    get = MagicMock(side_effect=[robots_response, page_response, page_response])
    sleep = MagicMock()
    # now() is called: once to record the first request's time (0.0), once to
    # check elapsed time before the second request (3.0), once to record the
    # second request's time (value unused by this test).
    clock = iter([0.0, 3.0, 3.0])

    fetch("https://example.org/a", get=get, sleep=sleep, now=lambda: next(clock))
    fetch("https://example.org/b", get=get, sleep=sleep, now=lambda: next(clock))

    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(7.0)  # 10s minimum - 3s already elapsed


def test_raises_on_non_200():
    get = MagicMock(side_effect=[_response(status=404), _response(status=500)])
    with pytest.raises(FetchError, match="HTTP 500"):
        fetch("https://example.org/x", get=get, sleep=MagicMock(), now=lambda: 0.0)


def test_raises_on_network_error():
    import requests
    get = MagicMock(side_effect=[_response(status=404), requests.ConnectionError("boom")])
    with pytest.raises(FetchError, match="could not fetch"):
        fetch("https://example.org/x", get=get, sleep=MagicMock(), now=lambda: 0.0)
