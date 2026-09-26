from onesearch.session import SessionCookies, SessionManager

COOKIES_A = SessionCookies(jsessionid="A", urm_st="1000", urm_se="901000", institute="UB")
COOKIES_B = SessionCookies(jsessionid="B", urm_st="2000", urm_se="902000", institute="UB")


def test_mints_session_on_first_call():
    calls = []
    manager = SessionManager(mint_fn=lambda: calls.append(1) or COOKIES_A)
    cookies = manager.get_cookies()
    assert cookies == COOKIES_A
    assert len(calls) == 1


def test_reuses_session_within_window():
    calls = [COOKIES_A, COOKIES_B]
    manager = SessionManager(mint_fn=lambda: calls.pop(0))
    first = manager.get_cookies()
    second = manager.get_cookies()
    assert first == second == COOKIES_A


def test_remints_after_window_expires():
    calls = [COOKIES_A, COOKIES_B]
    manager = SessionManager(mint_fn=lambda: calls.pop(0), session_window_seconds=0)
    first = manager.get_cookies()
    second = manager.get_cookies()
    assert first == COOKIES_A
    assert second == COOKIES_B
